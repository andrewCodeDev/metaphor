//! LLaMA LoRA Model Module
//!
//! Reusable LLaMA model with LoRA (Low-Rank Adaptation) support.
//! Extracted from llama_generate.c3 for use in training pipelines.
//!
//! Architecture:
//!   token -> Embedding -> N x LoRALlamaLayer -> RMSNorm -> LM Head -> logits
//!
//! Each LoRALlamaLayer:
//!   x -> RMSNorm -> Q/K/V(+LoRA) -> RoPE -> GQA Attention -> O_proj(+LoRA) -> residual
//!   -> RMSNorm -> Gate(+LoRA)*Up(+LoRA) -> Down(+LoRA) -> residual
//!
//! LoRA targets (configurable): q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

module metaphor::examples::llama;
import metaphor::device::host_device;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::device::common::operations;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::rtti;
import metaphor::utils::safetensors;
import std::core::mem;
import std::math;
import std::io;
import libc;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias TransferDirection = device_reference::TransferDirection;
alias HipDevice = hip_device::HipDevice;
alias Datatype = rtti::Datatype;
alias SliceRange = operations::SliceRange;
alias Optimizer = optimizer::Optimizer;

// ============================================================================
// Constants
// ============================================================================

const usz MAX_LAYERS = 32;
const usz MAX_LORA_DESCS = 512;  // 32 layers * 7 projs * 2 (A+B) + headroom

// ============================================================================
// Configuration
// ============================================================================

struct LlamaConfig
{
	ulong d_model;
	ulong n_heads;
	ulong n_kv_heads;
	ulong head_dim;
	ulong intermediate;
	ulong vocab_size;
	ulong n_layers;
	ulong max_seq_len;
	float eps;
	float rope_theta;
	ulong batch_size;
	ulong seq_len;        // training sequence length
	Datatype dtype;
}

fn ulong LlamaConfig.heads_per_kv_group(&self)
{
	return self.n_heads / self.n_kv_heads;
}

// LLaMA 3.1 8B Instruct
fn LlamaConfig llama_31_8b()
{
	return {
		.d_model = 4096,
		.n_heads = 32,
		.n_kv_heads = 8,
		.head_dim = 128,
		.intermediate = 14336,
		.vocab_size = 128256,
		.n_layers = 32,
		.max_seq_len = 2048,
		.eps = 1e-5,
		.rope_theta = 500000.0,
		.batch_size = 1,
		.seq_len = 512,
		.dtype = Datatype.BF16,
	};
}

// ============================================================================
// LoRA Configuration & Adapter
// ============================================================================

struct LoRAConfig
{
	ulong rank;
	float alpha;
	bool adapt_q_proj;
	bool adapt_k_proj;
	bool adapt_v_proj;
	bool adapt_o_proj;
	bool adapt_gate_proj;
	bool adapt_up_proj;
	bool adapt_down_proj;
}

fn LoRAConfig default_lora_config()
{
	return {
		.rank = 16,
		.alpha = 16.0,
		.adapt_q_proj = true,
		.adapt_k_proj = false,
		.adapt_v_proj = true,
		.adapt_o_proj = true,
		.adapt_gate_proj = false,
		.adapt_up_proj = false,
		.adapt_down_proj = false,
	};
}

struct LoRAAdapter
{
	Tensor lora_a;  // Down-projection: (rank, in_features), Kaiming init
	Tensor lora_b;  // Up-projection: (out_features, rank), zero init
	float scale;    // alpha / rank
	ulong rank;
}

fn LoRAAdapter? lora_init(
	ulong in_features,
	ulong out_features,
	ulong rank,
	float alpha,
	Datatype dtype,
	DeviceReference dev,
)
{
	Tensor lora_a = tensor::random(dtype, dev, { rank, in_features },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)in_features })!!.enable_grad();

	Tensor lora_b = tensor::zeros(dtype, dev, { out_features, rank })!!.enable_grad();

	return {
		.lora_a = lora_a,
		.lora_b = lora_b,
		.scale = alpha / (float)rank,
		.rank = rank,
	};
}

/// Apply LoRA adaptation to a linear layer.
/// Computes: output = x @ W^T + scale * (x @ A^T @ B^T)
fn Tensor? lora_forward(LoRAAdapter* self, Tensor x, Tensor base_weight, DeviceReference dev)
{
	Tensor base_out = functional::linear(x, base_weight)!!;

	Tensor lora_down = functional::linear(x, self.lora_a)!!;
	Tensor lora_up = functional::linear(lora_down, self.lora_b)!!;

	Tensor scale_t = lora_up.scalar_like((double)self.scale)!!;
	Tensor lora_out = lora_up.mul(scale_t)!!;

	return base_out.add(lora_out)!!;
}

/// Apply LoRA adaptation with an einsum-style base projection.
/// base_expr: e.g., "bsd,hkd->bshk" for Q/K/V projections
fn Tensor? lora_forward_einsum(LoRAAdapter* self, Tensor x, Tensor base_weight, char[] base_expr)
{
	Tensor base_out = x.einsum(base_weight, base_expr)!!;

	// LoRA path works on the last dim: x [..., d_model] -> [..., rank] -> [..., out_features]
	// Then reshape to match base_out
	Tensor lora_down = functional::linear(x, self.lora_a)!!;
	Tensor lora_up = functional::linear(lora_down, self.lora_b)!!;

	Tensor scale_t = lora_up.scalar_like((double)self.scale)!!;
	Tensor lora_out = lora_up.mul(scale_t)!!;

	// Reshape lora_out to match base_out shape
	ulong[] base_shape = base_out.core().shape().slice();
	lora_out = lora_out.reshape(base_shape)!!;

	return base_out.add(lora_out)!!;
}

// ============================================================================
// RoPE Cache
// ============================================================================

struct RopeCache
{
	Tensor cos_cache;   // [max_seq_len, head_dim/2] F32
	Tensor sin_cache;   // [max_seq_len, head_dim/2] F32
}

fn RopeCache? build_rope_cache(LlamaConfig* cfg, DeviceReference dev)
{
	ulong half_dim = cfg.head_dim / 2;
	ulong max_seq = cfg.max_seq_len;
	usz total = (usz)(max_seq * half_dim);
	usz nbytes = total * float.sizeof;

	float* cos_buf = (float*)mem::malloc(nbytes);
	float* sin_buf = (float*)mem::malloc(nbytes);
	assert(cos_buf != null && sin_buf != null, "Failed to allocate RoPE buffers");

	for (usz i = 0; i < (usz)half_dim; i++)
	{
		double exp = 2.0 * (double)i / (double)cfg.head_dim;
		double freq = 1.0 / math::pow((double)cfg.rope_theta, exp);

		for (usz pos = 0; pos < (usz)max_seq; pos++)
		{
			double angle = (double)pos * freq;
			usz idx = pos * (usz)half_dim + i;
			cos_buf[idx] = (float)math::cos(angle);
			sin_buf[idx] = (float)math::sin(angle);
		}
	}

	ulong[2] cache_shape = { max_seq, half_dim };

	RopeCache cache;
	cache.cos_cache = tensor::empty(Datatype.F32, dev, &cache_shape)!;
	cache.sin_cache = tensor::empty(Datatype.F32, dev, &cache_shape)!;

	cache.cos_cache.set(cos_buf[:total]);
	cache.sin_cache.set(sin_buf[:total]);

	mem::free(cos_buf);
	mem::free(sin_buf);

	return cache;
}

// ============================================================================
// RoPE Application (batched — for training full-sequence forward)
// ============================================================================

/// Apply rotary position embeddings (batched).
/// x: [B, S, H, head_dim], cos_pos: [1, S, 1, head_dim/2], sin_pos: [1, S, 1, head_dim/2]
fn Tensor? apply_rope_batched(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(3) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 3)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 3)!;

	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	return tensor::cat({ rot1, rot2 }, 3);
}

// ============================================================================
// LoRA LLaMA Layer
// ============================================================================

struct LoRALlamaLayer
{
	// Base weights (frozen)
	Tensor input_ln;      // [d_model]
	Tensor post_ln;       // [d_model]
	Tensor q_proj;        // [n_heads * head_dim, d_model]
	Tensor k_proj;        // [n_kv_heads * head_dim, d_model]
	Tensor v_proj;        // [n_kv_heads * head_dim, d_model]
	Tensor o_proj;        // [d_model, n_heads * head_dim]
	Tensor gate_proj;     // [intermediate, d_model]
	Tensor up_proj;       // [intermediate, d_model]
	Tensor down_proj;     // [d_model, intermediate]

	// LoRA adapters (optional per projection)
	LoRAAdapter q_lora;
	LoRAAdapter k_lora;
	LoRAAdapter v_lora;
	LoRAAdapter o_lora;
	LoRAAdapter gate_lora;
	LoRAAdapter up_lora;
	LoRAAdapter down_lora;

	bool has_q_lora;
	bool has_k_lora;
	bool has_v_lora;
	bool has_o_lora;
	bool has_gate_lora;
	bool has_up_lora;
	bool has_down_lora;

	DeviceReference dev;
}

fn LoRALlamaLayer? init_layer(LlamaConfig* cfg, LoRAConfig* lcfg, DeviceReference dev, usz layer_idx)
{
	Datatype dtype = cfg.dtype;
	ulong qkv_out = cfg.n_heads * cfg.head_dim;
	ulong kv_out = cfg.n_kv_heads * cfg.head_dim;

	LoRALlamaLayer layer = {
		.dev = dev,
		.input_ln = tensor::empty(dtype, dev, { cfg.d_model })!,
		.post_ln = tensor::empty(dtype, dev, { cfg.d_model })!,
		.q_proj = tensor::empty(dtype, dev, { qkv_out, cfg.d_model })!,
		.k_proj = tensor::empty(dtype, dev, { kv_out, cfg.d_model })!,
		.v_proj = tensor::empty(dtype, dev, { kv_out, cfg.d_model })!,
		.o_proj = tensor::empty(dtype, dev, { cfg.d_model, qkv_out })!,
		.gate_proj = tensor::empty(dtype, dev, { cfg.intermediate, cfg.d_model })!,
		.up_proj = tensor::empty(dtype, dev, { cfg.intermediate, cfg.d_model })!,
		.down_proj = tensor::empty(dtype, dev, { cfg.d_model, cfg.intermediate })!,
	};

	// Initialize LoRA adapters for enabled projections
	if (lcfg.adapt_q_proj)
	{
		layer.q_lora = lora_init(cfg.d_model, qkv_out, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_q_lora = true;
	}
	if (lcfg.adapt_k_proj)
	{
		layer.k_lora = lora_init(cfg.d_model, kv_out, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_k_lora = true;
	}
	if (lcfg.adapt_v_proj)
	{
		layer.v_lora = lora_init(cfg.d_model, kv_out, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_v_lora = true;
	}
	if (lcfg.adapt_o_proj)
	{
		layer.o_lora = lora_init(qkv_out, cfg.d_model, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_o_lora = true;
	}
	if (lcfg.adapt_gate_proj)
	{
		layer.gate_lora = lora_init(cfg.d_model, cfg.intermediate, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_gate_lora = true;
	}
	if (lcfg.adapt_up_proj)
	{
		layer.up_lora = lora_init(cfg.d_model, cfg.intermediate, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_up_lora = true;
	}
	if (lcfg.adapt_down_proj)
	{
		layer.down_lora = lora_init(cfg.intermediate, cfg.d_model, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!;
		layer.has_down_lora = true;
	}

	return layer;
}

/// Count LoRA parameters in a single layer.
fn ulong layer_lora_param_count(LoRALlamaLayer* l)
{
	ulong count = 0;
	if (l.has_q_lora)
	{
		count += l.q_lora.lora_a.core().shape().size() + l.q_lora.lora_b.core().shape().size();
	}
	if (l.has_k_lora)
	{
		count += l.k_lora.lora_a.core().shape().size() + l.k_lora.lora_b.core().shape().size();
	}
	if (l.has_v_lora)
	{
		count += l.v_lora.lora_a.core().shape().size() + l.v_lora.lora_b.core().shape().size();
	}
	if (l.has_o_lora)
	{
		count += l.o_lora.lora_a.core().shape().size() + l.o_lora.lora_b.core().shape().size();
	}
	if (l.has_gate_lora)
	{
		count += l.gate_lora.lora_a.core().shape().size() + l.gate_lora.lora_b.core().shape().size();
	}
	if (l.has_up_lora)
	{
		count += l.up_lora.lora_a.core().shape().size() + l.up_lora.lora_b.core().shape().size();
	}
	if (l.has_down_lora)
	{
		count += l.down_lora.lora_a.core().shape().size() + l.down_lora.lora_b.core().shape().size();
	}
	return count;
}

/// Training forward pass for a single layer (full sequence, no KV cache).
/// x: [B, S, d_model]
/// cos_pos: [1, S, 1, head_dim/2], sin_pos: [1, S, 1, head_dim/2]
/// causal_mask: [S, S]
/// scale_tensor: [head_dim]
/// Returns: [B, S, d_model]
fn Tensor? layer_forward_train(LoRALlamaLayer* self, Tensor x, LlamaConfig* cfg,
	Tensor cos_pos, Tensor sin_pos,
	Tensor causal_mask, Tensor scale_tensor)
{
	Tensor normed = functional::rms_norm(x, self.input_ln, cfg.eps)!;

	// Q/K/V projections — use LoRA where enabled
	Tensor q;
	Tensor k;
	Tensor v;

	if (self.has_q_lora)
	{
		q = lora_forward(&self.q_lora, normed, self.q_proj, self.dev)!;
		q = q.reshape({ cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.head_dim })!;
	}
	else
	{
		q = normed.einsum(self.q_proj, "bsd,od->bso")!;
		q = q.reshape({ cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.head_dim })!;
	}

	if (self.has_k_lora)
	{
		k = lora_forward(&self.k_lora, normed, self.k_proj, self.dev)!;
		k = k.reshape({ cfg.batch_size, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim })!;
	}
	else
	{
		k = normed.einsum(self.k_proj, "bsd,od->bso")!;
		k = k.reshape({ cfg.batch_size, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim })!;
	}

	if (self.has_v_lora)
	{
		v = lora_forward(&self.v_lora, normed, self.v_proj, self.dev)!;
		v = v.reshape({ cfg.batch_size, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim })!;
	}
	else
	{
		v = normed.einsum(self.v_proj, "bsd,od->bso")!;
		v = v.reshape({ cfg.batch_size, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim })!;
	}

	// RoPE
	q = apply_rope_batched(q, cos_pos, sin_pos)!;
	k = apply_rope_batched(k, cos_pos, sin_pos)!;

	// GQA via transpose + reshape
	ulong n = cfg.seq_len;
	ulong bg = cfg.batch_size * cfg.n_kv_heads;
	ulong r = cfg.heads_per_kv_group();

	Tensor q_t = q.transpose(1, 2)!.reshape({ bg, r, n, cfg.head_dim })!;
	Tensor k_t = k.transpose(1, 2)!.reshape({ bg, n, cfg.head_dim })!;
	Tensor v_t = v.transpose(1, 2)!.reshape({ bg, n, cfg.head_dim })!;

	Tensor q_scaled = q_t * scale_tensor;

	// Scores: [B*g, r, S, hd] @ [B*g, T, hd] -> [B*g, r, S, T]
	Tensor scores = q_scaled.einsum(k_t, "brsk,btk->brst")!;

	// Causal mask: [S, T]
	Tensor masked = scores.einsum(causal_mask, "brst,st->brst", inner: operations::BinaryOp.ADD)!;

	// Softmax over T dimension
	Tensor weights = functional::softmax(masked, 3)!;

	// Weighted sum: [B*g, r, S, T] @ [B*g, T, hd] -> [B*g, r, S, hd]
	Tensor attn_out = weights.einsum(v_t, "brst,btk->brsk")!;

	// Reshape back: [B*g, r, S, hd] -> [B, n_heads, S, hd] -> [B, S, n_heads, hd] -> [B, S, n_heads*hd]
	attn_out = attn_out.reshape({ cfg.batch_size, cfg.n_heads, n, cfg.head_dim })!
		.transpose(1, 2)!
		.reshape({ cfg.batch_size, n, cfg.n_heads * cfg.head_dim })!.stable();

	// Output projection + residual
	Tensor projected;
	if (self.has_o_lora)
	{
		projected = lora_forward(&self.o_lora, attn_out, self.o_proj, self.dev)!;
	}
	else
	{
		projected = functional::linear(attn_out, self.o_proj)!;
	}
	Tensor after_attn = (x + projected).stable();

	// MLP with gated SiLU
	Tensor normed2 = functional::rms_norm(after_attn, self.post_ln, cfg.eps)!;

	Tensor gate;
	if (self.has_gate_lora)
	{
		gate = lora_forward(&self.gate_lora, normed2, self.gate_proj, self.dev)!.silu()!;
	}
	else
	{
		gate = functional::linear(normed2, self.gate_proj)!.silu()!;
	}

	Tensor up;
	if (self.has_up_lora)
	{
		up = lora_forward(&self.up_lora, normed2, self.up_proj, self.dev)!;
	}
	else
	{
		up = functional::linear(normed2, self.up_proj)!;
	}

	Tensor mlp_in = (gate * up).stable();
	Tensor mlp_out;
	if (self.has_down_lora)
	{
		mlp_out = lora_forward(&self.down_lora, mlp_in, self.down_proj, self.dev)!;
	}
	else
	{
		mlp_out = functional::linear(mlp_in, self.down_proj)!;
	}

	return (after_attn + mlp_out).stable();
}

// ============================================================================
// LoRA LLaMA Model
// ============================================================================

struct LoRALlamaModel
{
	LlamaConfig config;
	LoRAConfig lora_config;

	// Global weights (frozen)
	Tensor embedding;    // [vocab_size, d_model]
	Tensor norm_f;       // [d_model]
	Tensor lm_head;      // [vocab_size, d_model]

	// Precomputed
	RopeCache rope;
	Tensor scale_tensor; // [head_dim] — 1/sqrt(head_dim)
	Tensor causal_mask;  // [seq_len, seq_len]

	// Layers
	LoRALlamaLayer[MAX_LAYERS] layers;
	usz num_layers;

	// Pipeline I/O
	Tensor tokens;       // [batch_size, seq_len] U64
	Tensor logits;       // [batch_size, seq_len, vocab_size]
}

fn LoRALlamaModel? model_init(LlamaConfig cfg, LoRAConfig lcfg, DeviceReference dev, int train_layers = 0)
{
	assert(cfg.n_layers <= MAX_LAYERS, "Too many layers");

	LoRALlamaModel model = {
		.config = cfg,
		.lora_config = lcfg,
		.num_layers = (usz)cfg.n_layers,
	};

	// Embedding + lm_head (frozen)
	model.embedding = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;
	model.norm_f = tensor::empty(cfg.dtype, dev, { cfg.d_model })!;
	model.lm_head = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;

	// Precomputed
	model.rope = build_rope_cache(&cfg, dev)!;
	float scale_val = (float)(1.0 / math::sqrt((double)cfg.head_dim));
	model.scale_tensor = tensor::constant(cfg.dtype, dev, { cfg.head_dim }, scale_val)!;

	// Causal mask: [seq_len, seq_len]
	{
		model.causal_mask = tensor::empty(cfg.dtype, dev, { cfg.seq_len, cfg.seq_len })!;
		model.causal_mask.core().alloc_data();
		usz n = (usz)cfg.seq_len;
		usz mask_elems = n * n;

		if (cfg.dtype == Datatype.BF16)
		{
			float* host_f32 = (float*)mem::malloc(mask_elems * float.sizeof);
			ushort* host_bf16 = (ushort*)mem::malloc(mask_elems * ushort.sizeof);
			assert(host_f32 != null && host_bf16 != null, "Failed to allocate causal mask");
			for (usz i = 0; i < n; i++)
			{
				for (usz j = 0; j < n; j++)
				{
					host_f32[i * n + j] = j <= i ? 0.0 : -1e9;
				}
			}
			for (usz idx = 0; idx < mask_elems; idx++)
			{
				uint bits = *(uint*)&host_f32[idx];
				host_bf16[idx] = (ushort)(bits >> 16);
			}
			model.causal_mask.set(host_bf16[:mask_elems]);
			mem::free(host_f32);
			mem::free(host_bf16);
		}
		else
		{
			float* host_mask = (float*)mem::malloc(mask_elems * float.sizeof);
			assert(host_mask != null, "Failed to allocate causal mask");
			for (usz i = 0; i < n; i++)
			{
				for (usz j = 0; j < n; j++)
				{
					host_mask[i * n + j] = j <= i ? 0.0 : -1e9;
				}
			}
			model.causal_mask.set(host_mask[:mask_elems]);
			mem::free(host_mask);
		}
	}

	// Create layers — only last train_layers get LoRA (0 = all)
	usz lora_start = 0;
	if (train_layers > 0 && (usz)train_layers < model.num_layers)
	{
		lora_start = model.num_layers - (usz)train_layers;
	}

	LoRAConfig no_lora = {
		.rank = lcfg.rank,
		.alpha = lcfg.alpha,
		.adapt_q_proj = false,
		.adapt_k_proj = false,
		.adapt_v_proj = false,
		.adapt_o_proj = false,
		.adapt_gate_proj = false,
		.adapt_up_proj = false,
		.adapt_down_proj = false,
	};

	for (usz i = 0; i < model.num_layers; i++)
	{
		model.layers[i] = init_layer(&cfg, i >= lora_start ? &lcfg : &no_lora, dev, i)!;
		if (i >= lora_start)
		{
			LoRALlamaLayer* l = &model.layers[i];
			if (l.has_q_lora) { l.q_lora.lora_a.fmt_label("lora_L%d_q_a", (int)i); l.q_lora.lora_b.fmt_label("lora_L%d_q_b", (int)i); }
			if (l.has_k_lora) { l.k_lora.lora_a.fmt_label("lora_L%d_k_a", (int)i); l.k_lora.lora_b.fmt_label("lora_L%d_k_b", (int)i); }
			if (l.has_v_lora) { l.v_lora.lora_a.fmt_label("lora_L%d_v_a", (int)i); l.v_lora.lora_b.fmt_label("lora_L%d_v_b", (int)i); }
			if (l.has_o_lora) { l.o_lora.lora_a.fmt_label("lora_L%d_o_a", (int)i); l.o_lora.lora_b.fmt_label("lora_L%d_o_b", (int)i); }
			if (l.has_gate_lora) { l.gate_lora.lora_a.fmt_label("lora_L%d_gate_a", (int)i); l.gate_lora.lora_b.fmt_label("lora_L%d_gate_b", (int)i); }
			if (l.has_up_lora) { l.up_lora.lora_a.fmt_label("lora_L%d_up_a", (int)i); l.up_lora.lora_b.fmt_label("lora_L%d_up_b", (int)i); }
			if (l.has_down_lora) { l.down_lora.lora_a.fmt_label("lora_L%d_down_a", (int)i); l.down_lora.lora_b.fmt_label("lora_L%d_down_b", (int)i); }
		}
	}

	// Build forward graph — input tokens placeholder
	model.tokens = tensor::empty(Datatype.U64, dev, { cfg.batch_size, cfg.seq_len })!;

	// Embedding lookup: [batch, seq_len] -> [batch, seq_len, d_model]
	Tensor x = model.embedding.gather(0, model.tokens)!;
	x = x.stable();
	x.set_label("embed_out");

	// Build position tensor for RoPE: [0, 1, ..., seq_len-1]
	Tensor pos_tensor = tensor::empty(Datatype.U64, dev, { cfg.seq_len })!;
	pos_tensor.core().alloc_data();
	ulong* host_pos = (ulong*)mem::malloc((usz)cfg.seq_len * ulong.sizeof);
	assert(host_pos != null, "Failed to allocate position buffer");
	for (usz i = 0; i < (usz)cfg.seq_len; i++)
	{
		host_pos[i] = (ulong)i;
	}
	pos_tensor.set(host_pos[:(usz)cfg.seq_len]);
	mem::free(host_pos);

	ulong half_dim = cfg.head_dim / 2;
	Tensor cos_pos = model.rope.cos_cache.gather(0, pos_tensor)!
		.reshape({ 1, cfg.seq_len, 1, half_dim })!.cast(cfg.dtype)!;
	Tensor sin_pos = model.rope.sin_cache.gather(0, pos_tensor)!
		.reshape({ 1, cfg.seq_len, 1, half_dim })!.cast(cfg.dtype)!;

	// Forward through all layers
	for (usz i = 0; i < model.num_layers; i++)
	{
		x = layer_forward_train(&model.layers[i], x, &cfg,
			cos_pos, sin_pos,
			model.causal_mask, model.scale_tensor)!;
	}

	// Final norm + lm_head
	x = functional::rms_norm(x, model.norm_f, cfg.eps)!;
	x.set_label("final_norm");

	model.logits = functional::linear(x, model.lm_head)!.stable();
	model.logits.set_label("logits");

	return model;
}

// ============================================================================
// Parameter Counting
// ============================================================================

fn ulong total_base_params(LlamaConfig cfg)
{
	ulong qkv_out = cfg.n_heads * cfg.head_dim;
	ulong kv_out = cfg.n_kv_heads * cfg.head_dim;

	ulong count = cfg.vocab_size * cfg.d_model;  // embedding
	count += cfg.d_model;                          // norm_f
	count += cfg.vocab_size * cfg.d_model;         // lm_head

	ulong per_layer = cfg.d_model                             // input_ln
		+ cfg.d_model                                          // post_ln
		+ qkv_out * cfg.d_model                                // q_proj
		+ kv_out * cfg.d_model                                 // k_proj
		+ kv_out * cfg.d_model                                 // v_proj
		+ cfg.d_model * qkv_out                                // o_proj
		+ cfg.intermediate * cfg.d_model                       // gate_proj
		+ cfg.intermediate * cfg.d_model                       // up_proj
		+ cfg.d_model * cfg.intermediate;                      // down_proj

	count += per_layer * cfg.n_layers;
	return count;
}

fn ulong total_lora_params(LoRALlamaModel* model)
{
	ulong count = 0;
	for (usz i = 0; i < model.num_layers; i++)
	{
		count += layer_lora_param_count(&model.layers[i]);
	}
	return count;
}

// ============================================================================
// Optimizer Registration
// ============================================================================

fn usz register_lora_params(LoRALlamaModel* model, Optimizer* opt, AdamOptions adam_opts = {
	...optimizer::DEFAULT_ADAM_OPTIONS, .clip_grad = 1.0,
})
{
	usz count = 0;

	// Register layers in reverse order (last layer = closest to loss)
	for (usz ri = 0; ri < model.num_layers; ri++)
	{
		usz i = model.num_layers - 1 - ri;
		LoRALlamaLayer* l = &model.layers[i];

		if (l.has_q_lora)
		{
			opt.adam(&&(Tensor[2]){ l.q_lora.lora_a, l.q_lora.lora_b }, adam_opts);
			count += 2;
		}
		if (l.has_k_lora)
		{
			opt.adam(&&(Tensor[2]){ l.k_lora.lora_a, l.k_lora.lora_b }, adam_opts);
			count += 2;
		}
		if (l.has_v_lora)
		{
			opt.adam(&&(Tensor[2]){ l.v_lora.lora_a, l.v_lora.lora_b }, adam_opts);
			count += 2;
		}
		if (l.has_o_lora)
		{
			opt.adam(&&(Tensor[2]){ l.o_lora.lora_a, l.o_lora.lora_b }, adam_opts);
			count += 2;
		}
		if (l.has_gate_lora)
		{
			opt.adam(&&(Tensor[2]){ l.gate_lora.lora_a, l.gate_lora.lora_b }, adam_opts);
			count += 2;
		}
		if (l.has_up_lora)
		{
			opt.adam(&&(Tensor[2]){ l.up_lora.lora_a, l.up_lora.lora_b }, adam_opts);
			count += 2;
		}
		if (l.has_down_lora)
		{
			opt.adam(&&(Tensor[2]){ l.down_lora.lora_a, l.down_lora.lora_b }, adam_opts);
			count += 2;
		}
	}
	return count;
}

// ============================================================================
// Checkpoint Saving (LoRA weights only)
// ============================================================================

fn safetensors::TensorDescriptor make_lora_desc(char[] name, Tensor t)
{
	assert(t.core().has_data(), "make_lora_desc: tensor has no data");
	ulong[] shape_slice = t.core().shape().slice();
	char[] data_bytes = t.value_bytes();

	ZString dev_name = t.core().device.name();
	if (dev_name != "host")
	{
		char* host_buf = (char*)libc::malloc(data_bytes.len);
		assert(host_buf != null, "Failed to allocate host buffer for LoRA save");
		char[] host_bytes = host_buf[:data_bytes.len];
		t.get(host_bytes);
		data_bytes = host_bytes;
	}

	return {
		.name = name,
		.dtype = safetensors::dtype_from_rtti(t.dtype()),
		.shape = shape_slice,
		.data = data_bytes,
	};
}

fn void save_lora_checkpoint(LoRALlamaModel* model, char[] path)
{
	safetensors::TensorDescriptor[MAX_LORA_DESCS] descs;
	char[64][MAX_LORA_DESCS] name_bufs;
	usz d = 0;

	for (usz i = 0; i < model.num_layers; i++)
	{
		LoRALlamaLayer* l = &model.layers[i];

		if (l.has_q_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.q_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.q_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.q_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.q_lora.lora_b); d++;
		}
		if (l.has_k_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.k_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.k_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.k_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.k_lora.lora_b); d++;
		}
		if (l.has_v_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.v_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.v_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.v_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.v_lora.lora_b); d++;
		}
		if (l.has_o_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.o_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.o_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.o_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.o_lora.lora_b); d++;
		}
		if (l.has_gate_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.gate_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.gate_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.gate_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.gate_lora.lora_b); d++;
		}
		if (l.has_up_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.up_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.up_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.up_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.up_lora.lora_b); d++;
		}
		if (l.has_down_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.down_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.down_lora.lora_a); d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.down_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.down_lora.lora_b); d++;
		}
	}

	char[]? buf = safetensors::serialize_tensors(descs[:d]);
	if (catch err = buf)
	{
		io::printfn("  Serialization failed: %s", err);
		return;
	}

	char[] data = buf;

	if (try f = io::file::open((String)path, "wb"))
	{
		f.write(data)!!;
		f.close()!!;
	}
	else
	{
		io::printfn("  Failed to open %s for writing", (String)path);
		safetensors::free_serialized(data);
		return;
	}
	safetensors::free_serialized(data);

	io::printfn("  Saved checkpoint: %s (%d tensors, %d bytes)", (String)path, d, data.len);
}

// ============================================================================
// Checkpoint Loading (LoRA weights)
// ============================================================================

fn void? load_lora_checkpoint(LoRALlamaModel* model, char[] path, DeviceReference dev)
{
	io::printfn("Loading LoRA checkpoint from: %s", (ZString)path.ptr);

	String p = string::tformat("%s", (ZString)path.ptr);
	char[] file_data = io::file::load(mem, p)!;
	safetensors::SafeTensorsFile sf = safetensors::deserialize(file_data)!;

	io::printfn("  Parsed %d tensors from checkpoint", sf.len());

	usz loaded = 0;

	for (usz i = 0; i < model.num_layers; i++)
	{
		LoRALlamaLayer* l = &model.layers[i];
		char[128] name_buf;

		if (l.has_q_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.q_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.q_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.q_proj_lora_b", i)!!;
				copy_weight_from_sf(l.q_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_v_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.v_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.v_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.v_proj_lora_b", i)!!;
				copy_weight_from_sf(l.v_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_k_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.k_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.k_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.k_proj_lora_b", i)!!;
				copy_weight_from_sf(l.k_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_o_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.o_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.o_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.o_proj_lora_b", i)!!;
				copy_weight_from_sf(l.o_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_gate_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.gate_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.gate_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.gate_proj_lora_b", i)!!;
				copy_weight_from_sf(l.gate_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_up_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.up_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.up_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.up_proj_lora_b", i)!!;
				copy_weight_from_sf(l.up_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_down_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.down_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_weight_from_sf(l.down_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.down_proj_lora_b", i)!!;
				copy_weight_from_sf(l.down_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
	}

	io::printfn("  Loaded %d LoRA tensors", loaded);

	sf.free();
	mem::free(file_data.ptr);
}

// ============================================================================
// Pretrained Weight Loading
// ============================================================================

fn void? load_pretrained_weights(LoRALlamaModel* model, char[] path, DeviceReference dev)
{
	io::printfn("Loading pretrained weights from: %s", (ZString)path.ptr);

	if (safetensors::is_sharded_model(path))
	{
		load_pretrained_weights_sharded(model, path, dev)!;
	}
	else
	{
		load_pretrained_weights_single(model, path, dev)!;
	}
}

fn void? load_pretrained_weights_single(LoRALlamaModel* model, char[] path, DeviceReference dev)
{
	String file_path;
	if (path.len > 12 && path[path.len-12..] == ".safetensors")
	{
		file_path = string::tformat("%s", (ZString)path.ptr);
	}
	else
	{
		file_path = string::tformat("%s/model.safetensors", (ZString)path.ptr);
	}
	char[] file_data = safetensors::read_file(file_path)!;
	safetensors::SafeTensorsFile sf = safetensors::deserialize(file_data)!;

	io::printfn("  Parsed %d tensors from safetensors header", sf.len());

	usz loaded = 0;

	copy_weight_from_sf(model.embedding, &sf, "model.embed_tokens.weight", dev)!;
	loaded++;
	copy_weight_from_sf(model.norm_f, &sf, "model.norm.weight", dev)!;
	loaded++;
	copy_weight_from_sf(model.lm_head, &sf, "lm_head.weight", dev)!;
	loaded++;

	for (usz i = 0; i < model.num_layers; i++)
	{
		load_layer_base_weights(&model.layers[i], &sf, i, dev)!;
		loaded += 9;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sf.free();
	mem::free(file_data.ptr);
}

fn void? load_pretrained_weights_sharded(LoRALlamaModel* model, char[] path, DeviceReference dev)
{
	safetensors::ShardedSafeTensors* sharded = (safetensors::ShardedSafeTensors*)mem::malloc(safetensors::ShardedSafeTensors.sizeof);
	assert(sharded != null, "Failed to allocate ShardedSafeTensors");
	*sharded = safetensors::sharded_open(path)!;

	usz loaded = 0;

	copy_weight_from_sf_sharded(model.embedding, sharded, "model.embed_tokens.weight")!;
	loaded++;
	copy_weight_from_sf_sharded(model.norm_f, sharded, "model.norm.weight")!;
	loaded++;
	copy_weight_from_sf_sharded(model.lm_head, sharded, "lm_head.weight")!;
	loaded++;

	for (usz i = 0; i < model.num_layers; i++)
	{
		load_layer_base_weights_sharded(&model.layers[i], sharded, i)!;
		loaded += 9;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sharded.free();
	mem::free(sharded);
}

fn void? load_layer_base_weights(LoRALlamaLayer* layer, safetensors::SafeTensorsFile* sf, usz layer_idx, DeviceReference dev)
{
	char[128] name_buf;
	usz prefix_len = format_layer_prefix(&name_buf, layer_idx);

	copy_layer_weight(layer.input_ln, sf, &name_buf, prefix_len, "input_layernorm.weight", dev)!;
	copy_layer_weight(layer.q_proj, sf, &name_buf, prefix_len, "self_attn.q_proj.weight", dev)!;
	copy_layer_weight(layer.k_proj, sf, &name_buf, prefix_len, "self_attn.k_proj.weight", dev)!;
	copy_layer_weight(layer.v_proj, sf, &name_buf, prefix_len, "self_attn.v_proj.weight", dev)!;
	copy_layer_weight(layer.o_proj, sf, &name_buf, prefix_len, "self_attn.o_proj.weight", dev)!;
	copy_layer_weight(layer.post_ln, sf, &name_buf, prefix_len, "post_attention_layernorm.weight", dev)!;
	copy_layer_weight(layer.gate_proj, sf, &name_buf, prefix_len, "mlp.gate_proj.weight", dev)!;
	copy_layer_weight(layer.up_proj, sf, &name_buf, prefix_len, "mlp.up_proj.weight", dev)!;
	copy_layer_weight(layer.down_proj, sf, &name_buf, prefix_len, "mlp.down_proj.weight", dev)!;
}

fn void? load_layer_base_weights_sharded(LoRALlamaLayer* layer, safetensors::ShardedSafeTensors* st, usz layer_idx)
{
	char[128] name_buf;
	usz prefix_len = format_layer_prefix(&name_buf, layer_idx);

	copy_layer_weight_sharded(layer.input_ln, st, &name_buf, prefix_len, "input_layernorm.weight")!;
	copy_layer_weight_sharded(layer.q_proj, st, &name_buf, prefix_len, "self_attn.q_proj.weight")!;
	copy_layer_weight_sharded(layer.k_proj, st, &name_buf, prefix_len, "self_attn.k_proj.weight")!;
	copy_layer_weight_sharded(layer.v_proj, st, &name_buf, prefix_len, "self_attn.v_proj.weight")!;
	copy_layer_weight_sharded(layer.o_proj, st, &name_buf, prefix_len, "self_attn.o_proj.weight")!;
	copy_layer_weight_sharded(layer.post_ln, st, &name_buf, prefix_len, "post_attention_layernorm.weight")!;
	copy_layer_weight_sharded(layer.gate_proj, st, &name_buf, prefix_len, "mlp.gate_proj.weight")!;
	copy_layer_weight_sharded(layer.up_proj, st, &name_buf, prefix_len, "mlp.up_proj.weight")!;
	copy_layer_weight_sharded(layer.down_proj, st, &name_buf, prefix_len, "mlp.down_proj.weight")!;
}

fn usz format_layer_prefix(char[128]* buf, usz layer_idx)
{
	char[] prefix = "model.layers.";
	for (usz i = 0; i < prefix.len; i++) { (*buf)[i] = prefix[i]; }
	usz pos = prefix.len;

	if (layer_idx >= 10)
	{
		(*buf)[pos++] = (char)('0' + layer_idx / 10);
	}
	(*buf)[pos++] = (char)('0' + layer_idx % 10);
	(*buf)[pos++] = '.';

	return pos;
}

fn void? copy_layer_weight(Tensor dst, safetensors::SafeTensorsFile* sf, char[128]* buf, usz prefix_len, char[] suffix, DeviceReference dev)
{
	usz pos = prefix_len;
	for (usz i = 0; i < suffix.len; i++)
	{
		(*buf)[pos + i] = suffix[i];
	}
	char[] name = (*buf)[:pos + suffix.len];
	copy_weight_from_sf(dst, sf, name, dev)!;
}

fn void? copy_layer_weight_sharded(Tensor dst, safetensors::ShardedSafeTensors* st, char[128]* buf, usz prefix_len, char[] suffix)
{
	usz pos = prefix_len;
	for (usz i = 0; i < suffix.len; i++)
	{
		(*buf)[pos + i] = suffix[i];
	}
	char[] name = (*buf)[:pos + suffix.len];
	copy_weight_from_sf_sharded(dst, st, name)!;
}

fn void? copy_weight_from_sf(Tensor dst, safetensors::SafeTensorsFile* sf, char[] name, DeviceReference dev)
{
	safetensors::TensorView view = sf.get(name)!;

	char[] dst_bytes = dst.value_bytes();
	if (dst_bytes.len == 0)
	{
		dst.core().alloc_data();
		dst_bytes = dst.value_bytes();
	}

	usz expected = dst.core().shape().size() * dst.core().dtype().size_of();
	assert(view.data.len == expected, "Weight size mismatch for %s: got %d, expected %d",
		(ZString)name.ptr, view.data.len, expected);

	dev.mem_transfer(view.data, dst_bytes[:expected], TransferDirection.HTOD);
}

fn void? copy_weight_from_sf_sharded(Tensor dst, safetensors::ShardedSafeTensors* st, char[] name)
{
	safetensors::TensorView view = st.get(name)!;

	char[] dst_bytes = dst.value_bytes();
	if (dst_bytes.len == 0)
	{
		dst.core().alloc_data();
		dst_bytes = dst.value_bytes();
	}

	usz expected = dst.core().shape().size() * dst.core().dtype().size_of();
	assert(view.data.len == expected, "Weight size mismatch for %s", (ZString)name.ptr);

	dst.core().device.mem_transfer(view.data, dst_bytes[:expected], TransferDirection.HTOD);
}

// ============================================================================
// Loss Reading
// ============================================================================

fn float read_loss_gpu(Tensor loss_t)
{
	float val;
	loss_t.get(&val);
	return val;
}
