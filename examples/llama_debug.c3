//! LLaMA 3.1 8B Instruct — Text Generation Pipeline
//!
//! Step-based inference for pretrained LLaMA 3.1 models.
//! Loads weights from safetensors, tokenizes prompts via BPE,
//! and generates text one token at a time using greedy/sampled decoding.
//!
//! Key features:
//!   - Grouped Query Attention (GQA): 32 Q heads, 8 KV heads
//!   - RoPE positional embeddings (theta=500000)
//!   - Gated MLP: silu(gate) * up → down
//!   - Pre-allocated KV cache for incremental decoding
//!   - Separate lm_head (not tied to embedding)
//!
//! Two-phase execution per layer per step:
//!   Phase 1: Q/K/V projection + RoPE → collect K_new/V_new → write to cache
//!   Phase 2: Q attends to K/V cache directly → o_proj → MLP → residual
//!   GQA via reshape (no gather): Q reshaped to [B, g, r, 1, hd] groups
//!
//! Run: c3c build llama_generate && ./build/llama_generate

module metaphor::examples::llama_debug;
import metaphor::device::host_device;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::device::common::operations;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::utils::safetensors;
import metaphor::tokenizers::bpe;
import metaphor::rtti;
import metaphor::utils::sampling;
import metaphor::utils::cjson;
import metaphor::tensor::tensor_core;
import std::math;
import std::io;
import std::core::mem;
import std::time;
import std::time::clock;
import libc;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias TransferDirection = device_reference::TransferDirection;
alias HostDevice = host_device::HostDevice;
alias HipDevice = hip_device::HipDevice;
alias Datatype = rtti::Datatype;
alias SliceRange = operations::SliceRange;

// Global sampler — persists PRNG state across all sample calls
sampling::Sampler g_sampler;
float* g_host_logits;
usz g_host_logits_size;

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
	Datatype dtype;
}

fn ulong LlamaConfig.heads_per_kv_group(&self)
{
	return self.n_heads / self.n_kv_heads;
}

// LLaMA 3.1 8B Instruct
const LlamaConfig LLAMA_31_8B = {
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
	.dtype = Datatype.BF16,
};

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

	// freqs[i] = 1.0 / pow(theta, 2*i / head_dim)
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

	// Transfer to device as F32 for precision
	ulong[2] cache_shape = { max_seq, half_dim };

	RopeCache cache;
	cache.cos_cache = tensor::empty(Datatype.F32, dev, &cache_shape)!;
	cache.sin_cache = tensor::empty(Datatype.F32, dev, &cache_shape)!;

	char[] cos_dst = cache.cos_cache.value_bytes();
	char[] sin_dst = cache.sin_cache.value_bytes();
	dev.mem_transfer(((char*)cos_buf)[:nbytes], cos_dst[:nbytes], TransferDirection.HTOD);
	dev.mem_transfer(((char*)sin_buf)[:nbytes], sin_dst[:nbytes], TransferDirection.HTOD);

	mem::free(cos_buf);
	mem::free(sin_buf);

	return cache;
}

// ============================================================================
// RoPE Application
// ============================================================================

/// Apply rotary position embeddings to q or k.
/// x: [B, H, head_dim], cos_pos: [1, 1, head_dim/2], sin_pos: [1, 1, head_dim/2]
/// Returns [B, H, head_dim] with RoPE applied.
fn Tensor? apply_rope(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(2) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 2)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 2)!;

	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	return tensor::cat({ rot1, rot2 }, 2);
}

/// Apply rotary position embeddings to q or k (batched prefill).
/// x: [B, S, H, head_dim], cos_pos: [1, S, 1, head_dim/2], sin_pos: [1, S, 1, head_dim/2]
/// Returns [B, S, H, head_dim] with RoPE applied.
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
// LLaMA Layer
// ============================================================================

const usz MAX_LAYERS = 32;

struct LlamaLayer
{
	// Weights (9 tensors)
	Tensor input_ln;      // [d_model]
	Tensor post_ln;       // [d_model]
	Tensor q_proj;        // [n_heads, head_dim, d_model]
	Tensor k_proj;        // [n_kv_heads, head_dim, d_model]
	Tensor v_proj;        // [n_kv_heads, head_dim, d_model]
	Tensor o_proj;        // [d_model, n_heads, head_dim]
	Tensor gate_proj;     // [intermediate, d_model]
	Tensor up_proj;       // [intermediate, d_model]
	Tensor down_proj;     // [d_model, intermediate]

	// KV Cache — leaf tensors, pre-allocated zeros
	Tensor k_cache;       // [B, n_kv_heads, max_seq_len, head_dim]
	Tensor v_cache;       // [B, n_kv_heads, max_seq_len, head_dim]

	// Graph handles for K/V projections (for cache copy after collection)
	Tensor k_new;         // [B, n_kv_heads, head_dim] — K after RoPE (stable)
	Tensor v_new;         // [B, n_kv_heads, head_dim] — V projection (stable)
	Tensor q_ready;       // [B, n_kv_heads, heads_per_group, 1, head_dim] — Q scaled+reshaped (stable)

	DeviceReference dev;
}

fn LlamaLayer? init_layer(LlamaConfig* cfg, DeviceReference dev, usz layer_idx)
{
	Datatype dtype = cfg.dtype;
	ulong[1] d_model_shape = { cfg.d_model };
	ulong[4] kv_shape = { cfg.batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim };

	return {
		.dev = dev,
		.input_ln = tensor::empty(dtype, dev, &d_model_shape)!,
		.post_ln = tensor::empty(dtype, dev, &d_model_shape)!,
		.q_proj = tensor::empty(dtype, dev, { cfg.n_heads, cfg.head_dim, cfg.d_model })!,
		.k_proj = tensor::empty(dtype, dev, { cfg.n_kv_heads, cfg.head_dim, cfg.d_model })!,
		.v_proj = tensor::empty(dtype, dev, { cfg.n_kv_heads, cfg.head_dim, cfg.d_model })!,
		.o_proj = tensor::empty(dtype, dev, { cfg.d_model, cfg.n_heads, cfg.head_dim })!,
		.gate_proj = tensor::empty(dtype, dev, { cfg.intermediate, cfg.d_model })!,
		.up_proj = tensor::empty(dtype, dev, { cfg.intermediate, cfg.d_model })!,
		.down_proj = tensor::empty(dtype, dev, { cfg.d_model, cfg.intermediate })!,
		// KV Cache - zero initialized
		.k_cache = tensor::zeros(dtype, dev, &kv_shape)!,
		.v_cache = tensor::zeros(dtype, dev, &kv_shape)!,
	};
}

// ============================================================================
// Two-Phase Layer Forward (Decode)
// ============================================================================

/// Phase 1: RMS norm + Q/K/V projection + RoPE.
/// Materializes k_new, v_new (for cache copy) and q_ready (for phase 2).
///
/// x:          [B, d_model] hidden state
/// cos_pos:    [1, 1, head_dim/2] RoPE cosines for current position
/// sin_pos:    [1, 1, head_dim/2] RoPE sines for current position
/// scale:      [head_dim] = 1/sqrt(head_dim)
fn void? layer_forward_proj(LlamaLayer* self, Tensor x, LlamaConfig* cfg,
	Tensor cos_pos, Tensor sin_pos, Tensor scale_tensor)
{
	Tensor normed = functional::rms_norm(x, self.input_ln, cfg.eps)!;

	Tensor q = normed.einsum(self.q_proj, "bd,hkd->bhk")!;    // [B, n_heads, head_dim]
	Tensor k = normed.einsum(self.k_proj, "bd,hkd->bhk")!;    // [B, n_kv_heads, head_dim]
	Tensor v = normed.einsum(self.v_proj, "bd,hkd->bhk")!;    // [B, n_kv_heads, head_dim]

	q = apply_rope(q, cos_pos, sin_pos)!;
	k = apply_rope(k, cos_pos, sin_pos)!;

	self.k_new = k.stable();
	self.v_new = v.stable();

	// Scale Q and reshape for GQA: [B, 32, hd] → [B*g, r, 1, hd] (4D)
	self.q_ready = (q * scale_tensor)
		.reshape({ cfg.batch_size * cfg.n_kv_heads, cfg.heads_per_kv_group(), 1, cfg.head_dim })!
		.stable();
}

/// Phase 2: Attention using KV cache directly (no gather), output projection, MLP.
/// Reads k_cache/v_cache as leaves — caller must update cache between phases.
///
/// x:          [B, d_model] hidden state (same as phase 1 input, for residual)
/// attn_mask:  [max_seq] additive attention mask
///
/// Returns x_hidden [B, d_model] for next layer.
fn Tensor? layer_forward_attn(LlamaLayer* self, Tensor x, LlamaConfig* cfg,
	Tensor attn_mask)
{
	// Reshape cache to 3D: [B, g, S, hd] → [B*g, S, hd]
	Tensor k_cache_r = self.k_cache.reshape({ cfg.batch_size * cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim })!;
	Tensor v_cache_r = self.v_cache.reshape({ cfg.batch_size * cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim })!;

	// Q: [B*g, r, 1, hd] @ K: [B*g, S, hd] → [B*g, r, 1, S]
	Tensor scores = self.q_ready.einsum(k_cache_r, "brik,bsk->bris")!;

	// Add mask: [S] broadcast
	Tensor masked = scores.einsum(attn_mask, "bris,s->bris", inner: operations::BinaryOp.ADD)!;

	// Softmax over S dimension (dim 3)
	Tensor weights = functional::softmax(masked, 3)!;

	// Weighted sum: [B*g, r, 1, S] @ V: [B*g, S, hd] → [B*g, r, 1, hd]
	Tensor attn_out = weights.einsum(v_cache_r, "bris,bsk->brik")!;

	// [B*g, r, 1, hd] → [B, n_heads, hd]
	attn_out = attn_out.reshape({ cfg.batch_size, cfg.n_heads, cfg.head_dim })!;

	// Output projection + residual
	Tensor projected = attn_out.einsum(self.o_proj, "bhk,dhk->bd")!;
	Tensor after_attn = x + projected;

	// MLP with gated SiLU
	Tensor normed2 = functional::rms_norm(after_attn, self.post_ln, cfg.eps)!;
	Tensor gate = functional::linear(normed2, self.gate_proj)!.silu()!;
	Tensor up = functional::linear(normed2, self.up_proj)!;
	Tensor mlp_out = functional::linear(gate * up, self.down_proj)!;

	return after_attn + mlp_out;
}

/// Prefill layer forward pass — processes N tokens at once with self-attention.
/// No KV cache interaction in-graph; K/V are saved as stable for post-exec cache copy.
/// Uses reshape-Q for GQA (no gather).
///
/// x:          [B, S, d_model] hidden states for all positions
/// cos_pos:    [1, S, 1, head_dim/2] RoPE cosines for positions 0..S-1
/// sin_pos:    [1, S, 1, head_dim/2] RoPE sines for positions 0..S-1
/// causal_mask:[S, S] additive causal mask (0 where j<=i, -1e9 otherwise)
/// scale:      [head_dim] = 1/sqrt(head_dim)
///
/// Returns x_hidden [B, S, d_model] for next layer.
fn Tensor? prefill_layer_forward(LlamaLayer* self, Tensor x, LlamaConfig* cfg,
	Tensor cos_pos, Tensor sin_pos,
	Tensor causal_mask, Tensor scale_tensor)
{
	Tensor normed = functional::rms_norm(x, self.input_ln, cfg.eps)!;
	Tensor q = normed.einsum(self.q_proj, "bsd,hkd->bshk")!;  // [B, S, n_heads, hd]
	Tensor k = normed.einsum(self.k_proj, "bsd,hkd->bshk")!;  // [B, S, n_kv_heads, hd]
	Tensor v = normed.einsum(self.v_proj, "bsd,hkd->bshk")!;  // [B, S, n_kv_heads, hd]

	q = apply_rope_batched(q, cos_pos, sin_pos)!;
	k = apply_rope_batched(k, cos_pos, sin_pos)!;

	self.k_new = k.stable();
	self.v_new = v.stable();

	// GQA via transpose + reshape — merge B*g to stay 4D
	ulong n = (ulong)x.shape().get(1);
	ulong bg = cfg.batch_size * cfg.n_kv_heads;
	ulong r = cfg.heads_per_kv_group();

	// Transpose to [B, H, S, hd] then reshape to merge batch+group
	// Q: [B, S, 32, hd] → transpose → [B, 32, S, hd] → reshape [B*g, r, S, hd]
	Tensor q_t = q.transpose(1, 2)!.reshape({ bg, r, n, cfg.head_dim })!;
	// K: [B, S, 8, hd] → transpose → [B, 8, S, hd] → reshape [B*g, S, hd]
	Tensor k_t = k.transpose(1, 2)!.reshape({ bg, n, cfg.head_dim })!;
	// V: [B, S, 8, hd] → transpose → [B, 8, S, hd] → reshape [B*g, S, hd]
	Tensor v_t = v.transpose(1, 2)!.reshape({ bg, n, cfg.head_dim })!;

	// Scale Q
	Tensor q_scaled = q_t * scale_tensor;

	// Scores: Q[B*g, r, S, hd] @ K[B*g, T, hd] → [B*g, r, S, T]
	Tensor scores = q_scaled.einsum(k_t, "brsk,btk->brst")!;

	// Causal mask: [S, T]
	Tensor masked = scores.einsum(causal_mask, "brst,st->brst", inner: operations::BinaryOp.ADD)!;

	// Softmax over T dimension (dim 3)
	Tensor weights = functional::softmax(masked, 3)!;

	// Weighted sum: [B*g, r, S, T] @ V[B*g, T, hd] → [B*g, r, S, hd]
	Tensor attn_out = weights.einsum(v_t, "brst,btk->brsk")!;

	// Reshape back: [B*g, r, S, hd] → [B, 32, S, hd] → transpose → [B, S, 32, hd]
	attn_out = attn_out.reshape({ cfg.batch_size, cfg.n_heads, n, cfg.head_dim })!
		.transpose(1, 2)!;

	Tensor projected = attn_out.einsum(self.o_proj, "bshk,dhk->bsd")!;
	Tensor after_attn = x + projected;

	Tensor normed2 = functional::rms_norm(after_attn, self.post_ln, cfg.eps)!;
	Tensor gate = functional::linear(normed2, self.gate_proj)!.silu()!;
	Tensor up = functional::linear(normed2, self.up_proj)!;
	Tensor mlp_out = functional::linear(gate * up, self.down_proj)!;

	return after_attn + mlp_out;
}

/// Zero KV cache for new sequence.
fn void layer_reset(LlamaLayer* self)
{
	self.k_cache.fill(0.0);
	self.v_cache.fill(0.0);
}

// ============================================================================
// LLaMA Model
// ============================================================================

struct LlamaModel
{
	LlamaConfig config;
	DeviceReference dev;

	// Global weights
	Tensor embedding;    // [vocab_size, d_model]
	Tensor norm_f;       // [d_model]
	Tensor lm_head;      // [vocab_size, d_model]

	// Precomputed
	RopeCache rope;
	Tensor attn_mask;    // [max_seq_len] — additive mask
	Tensor scale_tensor; // [head_dim] — 1/sqrt(head_dim)
	Tensor pos_tensor;   // [1] U64 — current position

	// Layers
	LlamaLayer[MAX_LAYERS] layers;

	// Pipeline I/O
	Tensor tokens;       // [batch_size] U64
	Tensor logits;       // [batch_size, vocab_size]

	// State
	usz current_pos;

	// Debug
}

fn void? build_attn_mask(LlamaModel* model, LlamaConfig* cfg, DeviceReference dev)
{
	// 1D [max_seq_len] filled with -1e9 (masked)
	// Added to 4D scores via einsum("bhqs,s->bhqs", inner:ADD)
	model.attn_mask = tensor::empty(cfg.dtype, dev, { cfg.max_seq_len })!;
	model.attn_mask.fill(-1e9);
}

fn void? build_scale_tensor(LlamaModel* model, LlamaConfig* cfg, DeviceReference dev)
{
	// Shape [head_dim] filled with 1/sqrt(head_dim) — avoids elemwise broadcast bug
	float scale_val = (float)(1.0 / math::sqrt((double)cfg.head_dim));
	model.scale_tensor = tensor::constant(cfg.dtype, dev, { cfg.head_dim }, scale_val)!;
}

fn LlamaModel? model_init(LlamaConfig* cfg, DeviceReference dev)
{
	assert(cfg.n_layers <= MAX_LAYERS, "Too many layers");

	LlamaModel model;
	model.config = *cfg;
	model.dev = dev;
	model.current_pos = 0;

	// Embedding
	model.embedding = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;

	// Final norm
	model.norm_f = tensor::empty(cfg.dtype, dev, { cfg.d_model })!;

	// Separate lm_head
	model.lm_head = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;

	// Precomputed tensors
	model.rope = build_rope_cache(cfg, dev)!;
	build_attn_mask(&model, cfg, dev)!;
	build_scale_tensor(&model, cfg, dev)!;

	// Position tensor
	model.pos_tensor = tensor::empty(Datatype.U64, dev, { 1 })!;

	// Layers
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		model.layers[i] = init_layer(cfg, dev, i)!;
	}

	// Pipeline input
	model.tokens = tensor::empty(Datatype.U64, dev, { cfg.batch_size })!;

	// ---- Build forward graph ----

	// 1. Embed: gather + unsqueeze for batch dim
	Tensor x_embed = model.embedding.gather(0, model.tokens)!.unsqueeze(0)!;


	// 2. Get RoPE cos/sin for current position:
	//    pos_tensor [1] U64 → gather from cache [max_seq, half] → [1, half] → unsqueeze → [1, 1, half]
	Tensor cos_pos = model.rope.cos_cache.gather(0, model.pos_tensor)!.unsqueeze(0)!;
	Tensor sin_pos = model.rope.sin_cache.gather(0, model.pos_tensor)!.unsqueeze(0)!;

	// Cast to model dtype for the einsum ops
	cos_pos = cos_pos.cast(cfg.dtype)!;
	sin_pos = sin_pos.cast(cfg.dtype)!;

	// 3. Two-phase forward: projection then attention per layer
	Tensor x_hidden = x_embed;
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		layer_forward_proj(&model.layers[i], x_hidden, cfg,
			cos_pos, sin_pos, model.scale_tensor)!;
		x_hidden = layer_forward_attn(&model.layers[i], x_hidden, cfg,
			model.attn_mask)!;
	}

	// 4. Final norm
	x_hidden = functional::rms_norm(x_hidden, model.norm_f, cfg.eps)!;

	// 5. Project to vocab (separate lm_head)
	model.logits = functional::linear(x_hidden, model.lm_head)!.stable();

	return model;
}

// ============================================================================
// Model Step — Single-Phase Execution
// ============================================================================

/// Write current position to pos_tensor (host-to-device).
fn void write_pos(LlamaModel* self, usz pos)
{
	char[] pos_bytes = self.pos_tensor.value_bytes();
	if (pos_bytes.len == 0)
	{
		self.pos_tensor.core().alloc_data();
		pos_bytes = self.pos_tensor.value_bytes();
	}
	ulong[1] host_pos = { (ulong)pos };
	char[] host_buf = ((char*)&host_pos)[:ulong.sizeof];
	self.dev.mem_transfer(host_buf, pos_bytes[:ulong.sizeof], TransferDirection.HTOD);
}

/// Unmask position `pos` in the attention mask (set to 0.0).
fn void unmask_position(LlamaModel* self, usz pos)
{
	char[] mask_bytes = self.attn_mask.value_bytes();
	assert(mask_bytes.len > 0, "Attention mask not allocated");

	usz dtype_size = self.config.dtype.size_of();
	usz offset = pos * dtype_size;

	if (self.config.dtype == Datatype.BF16)
	{
		ushort[1] zero_bf16 = { 0 };
		char[] src = ((char*)&zero_bf16)[:2];
		self.dev.mem_transfer(src, mask_bytes[offset:2], TransferDirection.HTOD);
	}
	else
	{
		float[1] zero_f32 = { 0.0 };
		char[] src = ((char*)&zero_f32)[:4];
		self.dev.mem_transfer(src, mask_bytes[offset:4], TransferDirection.HTOD);
	}
}

/// Copy K/V new projections into the KV cache at position `pos` for one layer.
fn void copy_kv_to_cache(LlamaLayer* layer, LlamaConfig* cfg, usz pos, DeviceReference dev)
{
	usz head_dim = (usz)cfg.head_dim;
	usz max_seq = (usz)cfg.max_seq_len;
	usz dtype_size = cfg.dtype.size_of();
	usz head_bytes = head_dim * dtype_size;

	char[] k_src = layer.k_new.value_bytes();
	char[] v_src = layer.v_new.value_bytes();
	char[] k_dst = layer.k_cache.value_bytes();
	char[] v_dst = layer.v_cache.value_bytes();

	assert(k_src.len > 0, "k_new data not materialized");
	assert(v_src.len > 0, "v_new data not materialized");

	// k_new: [B, n_kv_heads, head_dim]
	// k_cache:  [B, n_kv_heads, max_seq_len, head_dim]
	for (usz b = 0; b < (usz)cfg.batch_size; b++)
	{
		for (usz h = 0; h < (usz)cfg.n_kv_heads; h++)
		{
			usz src_off = (b * (usz)cfg.n_kv_heads + h) * head_bytes;
			usz dst_off = ((b * (usz)cfg.n_kv_heads + h) * max_seq + pos) * head_bytes;

			dev.mem_copy(k_src[src_off:head_bytes], k_dst[dst_off:head_bytes]);
			dev.mem_copy(v_src[src_off:head_bytes], v_dst[dst_off:head_bytes]);
		}
	}
}

/// Copy S positions of K/V into the KV cache starting at `start_pos`.
/// k_new/v_new shape: [B, S, n_kv_heads, head_dim] (from prefill einsum output order).
/// k_cache shape: [B, n_kv_heads, max_seq_len, head_dim].
fn void copy_kv_to_cache_batched(LlamaLayer* layer, LlamaConfig* cfg,
	usz start_pos, usz n_tokens, DeviceReference dev)
{
	usz head_dim = (usz)cfg.head_dim;
	usz max_seq = (usz)cfg.max_seq_len;
	usz n_kv = (usz)cfg.n_kv_heads;
	usz dtype_size = cfg.dtype.size_of();
	usz head_bytes = head_dim * dtype_size;

	char[] k_src = layer.k_new.value_bytes();
	char[] v_src = layer.v_new.value_bytes();
	char[] k_dst = layer.k_cache.value_bytes();
	char[] v_dst = layer.v_cache.value_bytes();

	assert(k_src.len > 0, "k_new data not materialized");
	assert(v_src.len > 0, "v_new data not materialized");

	// k_new:   [B, S, n_kv_heads, head_dim] — stride: (S*n_kv*hd, n_kv*hd, hd, 1)
	// k_cache: [B, n_kv_heads, max_seq, head_dim] — stride: (n_kv*max_seq*hd, max_seq*hd, hd, 1)
	for (usz b = 0; b < (usz)cfg.batch_size; b++)
	{
		for (usz s = 0; s < n_tokens; s++)
		{
			for (usz h = 0; h < n_kv; h++)
			{
				usz src_off = ((b * n_tokens + s) * n_kv + h) * head_bytes;
				usz dst_off = ((b * n_kv + h) * max_seq + start_pos + s) * head_bytes;

				dev.mem_copy(k_src[src_off:head_bytes], k_dst[dst_off:head_bytes]);
				dev.mem_copy(v_src[src_off:head_bytes], v_dst[dst_off:head_bytes]);
			}
		}
	}
}

/// Batched prefill: process all prompt tokens in a single forward pass.
fn Tensor? model_prefill(LlamaModel* self, uint[] tokens)
{
	usz n_tokens = tokens.len;
	assert(n_tokens > 0, "Empty token list for prefill");
	assert(n_tokens <= (usz)self.config.max_seq_len,
		"Prefill length %d exceeds max_seq_len %d", n_tokens, self.config.max_seq_len);

	LlamaConfig* cfg = &self.config;
	DeviceReference dev = self.dev;
	ulong n = (ulong)n_tokens;

	// Save decode graph's tensor pointers — prefill_layer_forward overwrites
	// them, but model_step needs the original decode tensors.
	Tensor[MAX_LAYERS] saved_k_new;
	Tensor[MAX_LAYERS] saved_v_new;
	Tensor[MAX_LAYERS] saved_q_ready;
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		saved_k_new[i] = self.layers[i].k_new;
		saved_v_new[i] = self.layers[i].v_new;
		saved_q_ready[i] = self.layers[i].q_ready;
	}

	// Build and execute prefill in an isolated subgraph so it doesn't
	// interfere with the decode graph. Export last-token logits as a
	// leaf on the root graph for the first decode iteration.
	Tensor exported_logits;

	graph::@subgraph()
	{
		// 1. Create [N] token tensor and transfer all token IDs
		ulong[1] tok_shape = { n };
		Tensor tokens_tensor = tensor::empty(Datatype.U64, dev, &tok_shape)!;
		tokens_tensor.core().alloc_data();
		usz tok_bytes = n_tokens * ulong.sizeof;
		ulong* host_tokens = (ulong*)mem::malloc(tok_bytes);
		assert(host_tokens != null, "Failed to allocate prefill token buffer");
		for (usz i = 0; i < n_tokens; i++)
		{
			host_tokens[i] = (ulong)tokens[i];
		}
		dev.mem_transfer(((char*)host_tokens)[:tok_bytes],
			tokens_tensor.value_bytes()[:tok_bytes], TransferDirection.HTOD);
		mem::free(host_tokens);

		// 2. Create [N] position tensor with values [0, 1, ..., N-1]
		Tensor pos_tensor = tensor::empty(Datatype.U64, dev, &tok_shape)!;
		pos_tensor.core().alloc_data();
		ulong* host_pos = (ulong*)mem::malloc(tok_bytes);
		assert(host_pos != null, "Failed to allocate prefill position buffer");
		for (usz i = 0; i < n_tokens; i++)
		{
			host_pos[i] = (ulong)i;
		}
		dev.mem_transfer(((char*)host_pos)[:tok_bytes],
			pos_tensor.value_bytes()[:tok_bytes], TransferDirection.HTOD);
		mem::free(host_pos);

		// 3. Build prefill graph

		// Embed: gather all N tokens → [N, d_model] → unsqueeze → [1, N, d_model]
		Tensor x = self.embedding.gather(0, tokens_tensor)!.unsqueeze(0)!;

		// RoPE: gather cos/sin for positions 0..N-1
		ulong half_dim = cfg.head_dim / 2;
		Tensor cos_all = self.rope.cos_cache.gather(0, pos_tensor)!
			.reshape({ 1, n, 1, half_dim })!.cast(cfg.dtype)!;
		Tensor sin_all = self.rope.sin_cache.gather(0, pos_tensor)!
			.reshape({ 1, n, 1, half_dim })!.cast(cfg.dtype)!;

		// Causal mask: [N, N] — 0 where j <= i, -1e9 otherwise
		Tensor causal_mask = tensor::empty(cfg.dtype, dev, { n, n })!;
		causal_mask.core().alloc_data();
		usz mask_elems = n_tokens * n_tokens;
		usz mask_bytes = mask_elems * cfg.dtype.size_of();
		if (cfg.dtype == Datatype.BF16)
		{
			ushort* host_mask = (ushort*)mem::malloc(mask_elems * ushort.sizeof);
			assert(host_mask != null, "Failed to allocate causal mask");
			float* host_mask_f32 = (float*)mem::malloc(mask_elems * float.sizeof);
			assert(host_mask_f32 != null, "Failed to allocate causal mask f32");
			for (usz i = 0; i < n_tokens; i++)
			{
				for (usz j = 0; j < n_tokens; j++)
				{
					host_mask_f32[i * n_tokens + j] = j <= i ? 0.0 : -1e9;
				}
			}
			for (usz k = 0; k < mask_elems; k++)
			{
				uint bits = *(uint*)&host_mask_f32[k];
				host_mask[k] = (ushort)(bits >> 16);
			}
			dev.mem_transfer(((char*)host_mask)[:mask_bytes],
				causal_mask.value_bytes()[:mask_bytes], TransferDirection.HTOD);
			mem::free(host_mask_f32);
			mem::free(host_mask);
		}
		else
		{
			float* host_mask = (float*)mem::malloc(mask_bytes);
			assert(host_mask != null, "Failed to allocate causal mask");
			for (usz i = 0; i < n_tokens; i++)
			{
				for (usz j = 0; j < n_tokens; j++)
				{
					host_mask[i * n_tokens + j] = j <= i ? 0.0 : -1e9;
				}
			}
			dev.mem_transfer(((char*)host_mask)[:mask_bytes],
				causal_mask.value_bytes()[:mask_bytes], TransferDirection.HTOD);
			mem::free(host_mask);
		}

		// Forward through all layers
		for (usz i = 0; i < (usz)cfg.n_layers; i++)
		{
			x = prefill_layer_forward(&self.layers[i], x, cfg,
				cos_all, sin_all,
				causal_mask, self.scale_tensor)!;
		}

		// Final norm + lm_head
		x = functional::rms_norm(x, self.norm_f, cfg.eps)!;
		Tensor prefill_logits = functional::linear(x, self.lm_head)!.stable();

		// 4. Execute
		graph::Graph* g = graph::get();
		g.advance_generation();
		prefill_logits.collect()!;

		// 5. Copy KV cache BEFORE subgraph teardown frees the layer tensors
		for (usz i = 0; i < (usz)cfg.n_layers; i++)
		{
			copy_kv_to_cache_batched(&self.layers[i], cfg, 0, n_tokens, dev);
		}

		// 6. Export last-token logits as a leaf on the root graph.
		// Create a [vocab] tensor with just the last token's logits.
		usz vocab = (usz)cfg.vocab_size;
		usz logit_bytes = vocab * cfg.dtype.size_of();
		usz last_offset = (n_tokens - 1) * logit_bytes;
		char[] pl_bytes = prefill_logits.value_bytes();
		Tensor last_logits = tensor::empty(cfg.dtype, dev, { (ulong)vocab })!;
		last_logits.core().alloc_data();

		dev.mem_copy(pl_bytes[last_offset:logit_bytes], last_logits.value_bytes()[:logit_bytes]);
		exported_logits = graph::export(last_logits);
	};

	// Restore decode graph's tensors so model_step reads the right ones
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		self.layers[i].k_new = saved_k_new[i];
		self.layers[i].v_new = saved_v_new[i];
		self.layers[i].q_ready = saved_q_ready[i];
	}

	// Unmask positions and set current_pos for decode
	for (usz p = 0; p < n_tokens; p++)
	{
		unmask_position(self, p);
	}
	self.current_pos = n_tokens;

	return exported_logits;
}

/// Process a single token and produce logits.
/// Two-phase per layer: collect projections → write cache → attention uses updated cache.
fn void? model_step(LlamaModel* self, ulong token)
{
	assert(self.current_pos < (usz)self.config.max_seq_len,
		"Sequence length exceeded max_seq_len (%d)", self.config.max_seq_len);

	// 1. Write token input
	char[] token_bytes = self.tokens.value_bytes();
	if (token_bytes.len == 0) { self.tokens.core().alloc_data(); token_bytes = self.tokens.value_bytes(); }
	self.dev.mem_transfer(((char*)&token)[:ulong.sizeof], token_bytes[:ulong.sizeof], TransferDirection.HTOD);

	// 2. Write current position
	write_pos(self, self.current_pos);

	// 3. Unmask current position in attention mask
	unmask_position(self, self.current_pos);

	// 4. Advance generation so graph re-executes with new input
	graph::Graph* g = graph::get();
	g.advance_generation();

	// 5. Per-layer: collect phase 1 (projections), write cache, then phase 2 runs lazily
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		self.layers[i].k_new.collect()!;
		self.layers[i].v_new.collect()!;

		// Write K/V into cache at current position
		copy_kv_to_cache(&self.layers[i], &self.config, self.current_pos, self.dev);
	}

	// 6. Collect logits (triggers phase 2 of all layers lazily via dependency chain)
	self.logits.collect()!;

	// 7. Advance position
	self.current_pos++;
}

/// Reset all layer states for a new sequence.
fn void model_reset(LlamaModel* self)
{
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		layer_reset(&self.layers[i]);
	}
	self.current_pos = 0;
	// Refill attention mask with -1e9
	self.attn_mask.fill(-1e9);
}

// ============================================================================
// Weight Loading
// ============================================================================

fn void? load_weights(LlamaModel* self, char[] path)
{
	io::printfn("Loading weights from: %s", (ZString)path.ptr);

	if (safetensors::is_sharded_model(path))
	{
		load_weights_sharded(self, path)!;
	}
	else
	{
		load_weights_single(self, path)!;
	}
}

fn void? load_weights_single(LlamaModel* self, char[] path)
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

	copy_weight(self.embedding, &sf, "model.embed_tokens.weight")!;
	loaded++;
	copy_weight(self.norm_f, &sf, "model.norm.weight")!;
	loaded++;
	copy_weight(self.lm_head, &sf, "lm_head.weight")!;
	loaded++;

	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		load_layer_weights(&self.layers[i], &sf, i)!;
		loaded += 9;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sf.free();
	mem::free(file_data.ptr);
}

fn void? load_weights_sharded(LlamaModel* self, char[] path)
{
	safetensors::ShardedSafeTensors* sharded = (safetensors::ShardedSafeTensors*)mem::malloc(safetensors::ShardedSafeTensors.sizeof);
	assert(sharded != null, "Failed to allocate ShardedSafeTensors");
	*sharded = safetensors::sharded_open(path)!;

	usz loaded = 0;

	copy_weight_sharded(self.embedding, sharded, "model.embed_tokens.weight")!;
	loaded++;
	copy_weight_sharded(self.norm_f, sharded, "model.norm.weight")!;
	loaded++;
	copy_weight_sharded(self.lm_head, sharded, "lm_head.weight")!;
	loaded++;

	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		load_layer_weights_sharded(&self.layers[i], sharded, i)!;
		loaded += 9;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sharded.free();
	mem::free(sharded);
}

fn void? load_layer_weights(LlamaLayer* layer, safetensors::SafeTensorsFile* sf, usz layer_idx)
{
	char[128] name_buf;
	usz prefix_len = format_layer_prefix(&name_buf, layer_idx);

	copy_layer_weight(layer.input_ln, sf, &name_buf, prefix_len, "input_layernorm.weight")!;
	copy_layer_weight(layer.q_proj, sf, &name_buf, prefix_len, "self_attn.q_proj.weight")!;
	copy_layer_weight(layer.k_proj, sf, &name_buf, prefix_len, "self_attn.k_proj.weight")!;
	copy_layer_weight(layer.v_proj, sf, &name_buf, prefix_len, "self_attn.v_proj.weight")!;
	copy_layer_weight(layer.o_proj, sf, &name_buf, prefix_len, "self_attn.o_proj.weight")!;
	copy_layer_weight(layer.post_ln, sf, &name_buf, prefix_len, "post_attention_layernorm.weight")!;
	copy_layer_weight(layer.gate_proj, sf, &name_buf, prefix_len, "mlp.gate_proj.weight")!;
	copy_layer_weight(layer.up_proj, sf, &name_buf, prefix_len, "mlp.up_proj.weight")!;
	copy_layer_weight(layer.down_proj, sf, &name_buf, prefix_len, "mlp.down_proj.weight")!;
}

fn void? load_layer_weights_sharded(LlamaLayer* layer, safetensors::ShardedSafeTensors* st, usz layer_idx)
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

fn void? copy_layer_weight(Tensor dst, safetensors::SafeTensorsFile* sf, char[128]* buf, usz prefix_len, char[] suffix)
{
	usz pos = prefix_len;
	for (usz i = 0; i < suffix.len; i++)
	{
		(*buf)[pos + i] = suffix[i];
	}
	char[] name = (*buf)[:pos + suffix.len];
	copy_weight(dst, sf, name)!;
}

fn void? copy_layer_weight_sharded(Tensor dst, safetensors::ShardedSafeTensors* st, char[128]* buf, usz prefix_len, char[] suffix)
{
	usz pos = prefix_len;
	for (usz i = 0; i < suffix.len; i++)
	{
		(*buf)[pos + i] = suffix[i];
	}
	char[] name = (*buf)[:pos + suffix.len];
	copy_weight_sharded(dst, st, name)!;
}

fn void? copy_weight(Tensor dst, safetensors::SafeTensorsFile* sf, char[] name)
{
	safetensors::TensorView view = sf.get(name)!;

	char[] dst_bytes = dst.value_bytes();
	if (dst_bytes.len == 0)
	{
		dst.core().alloc_data();
		dst_bytes = dst.value_bytes();
	}

	usz expected = dst.core().shape().size() * dst.core().dtype().size_of();
	assert(view.data.len == expected, "Weight size mismatch");

	dst.core().device.mem_transfer(view.data, dst_bytes[:expected],
		TransferDirection.HTOD);
}

fn void? copy_weight_sharded(Tensor dst, safetensors::ShardedSafeTensors* st, char[] name)
{
	safetensors::TensorView view = st.get(name)!;

	char[] dst_bytes = dst.value_bytes();
	if (dst_bytes.len == 0)
	{
		dst.core().alloc_data();
		dst_bytes = dst.value_bytes();
	}

	usz expected = dst.core().shape().size() * dst.core().dtype().size_of();
	assert(view.data.len == expected, "Weight size mismatch");

	dst.core().device.mem_transfer(view.data, dst_bytes[:expected],
		TransferDirection.HTOD);
}

// ============================================================================
// Text Generation (streaming REPL)
// ============================================================================

const usz INITIAL_TOKEN_CAPACITY = 4096;

struct GenerationState {
	uint* all_tokens;
	usz num_tokens;
	usz capacity;
}

fn void GenerationState.init(&self)
{
	self.all_tokens = (uint*)mem::malloc(uint.sizeof * INITIAL_TOKEN_CAPACITY);
	assert(self.all_tokens != null, "Failed to allocate token buffer");
	self.num_tokens = 0;
	self.capacity = INITIAL_TOKEN_CAPACITY;
}

fn void GenerationState.deinit(&self)
{
	if (self.all_tokens != null)
	{
		mem::free(self.all_tokens);
		self.all_tokens = null;
	}
}

fn void GenerationState.reset(&self)
{
	self.num_tokens = 0;
}

fn void GenerationState.append(&self, uint token)
{
	if (self.num_tokens >= self.capacity)
	{
		usz new_cap = self.capacity * 2;
		uint* new_buf = (uint*)libc::realloc(self.all_tokens, uint.sizeof * new_cap);
		assert(new_buf != null, "Failed to grow token buffer");
		self.all_tokens = new_buf;
		self.capacity = new_cap;
	}
	self.all_tokens[self.num_tokens++] = token;
}

/// Generate a continuation from the given prompt, streaming tokens to stdout.
fn void? generate_continuation(LlamaModel* model, bpe::Bpe* tokenizer, GenerationState* state,
	char[] prompt, usz max_new_tokens, usz tokenizer_vocab_size, uint eos_token)
{
	// 1. Encode prompt
	uint[] prompt_ids = tokenizer.encode(prompt)!;

	// 2. Append prompt tokens to persistent history
	for (usz i = 0; i < prompt_ids.len; i++)
	{
		state.append(prompt_ids[i]);
	}

	// 3. Prefill
	Clock prefill_clock = clock::now();
	bool use_batched_prefill = true;
	Tensor prefill_logits_leaf;
	if (use_batched_prefill)
	{
		prefill_logits_leaf = model_prefill(model, prompt_ids)!;
	}
	else
	{
		for (usz i = 0; i < prompt_ids.len; i++)
		{
			model_step(model, (ulong)prompt_ids[i])!;
		}
	}
	NanoDuration prefill_ns = prefill_clock.mark();

	usz n_prompt = prompt_ids.len;
	mem::free(prompt_ids.ptr);

	// 4. Generate new tokens, streaming each to stdout
	Clock decode_clock = clock::now();
	usz limit = max_new_tokens > 0 ? max_new_tokens : 200;
	usz generated_count = 0;
	for (usz i = 0; i < limit; i++)
	{
		// On first iteration after batched prefill, use the exported leaf logits
		// (collect is a no-op since it's a leaf with data, no graph re-execution).
		// On subsequent iterations, use model.logits from the decode graph.
		Tensor logit_source;
		if (i == 0 && use_batched_prefill)
		{
			logit_source = prefill_logits_leaf;
		}
		else
		{
			logit_source = model.logits;
		}

		Tensor logits_f32 = logit_source.cast(F32)!;
		logits_f32.collect()!;
		char[] logits_bytes = logits_f32.value_bytes();
		assert(logits_bytes.len > 0, "No logits data");

		usz model_vocab = model.config.vocab_size;
		usz sample_vocab = tokenizer_vocab_size < model_vocab ? tokenizer_vocab_size : model_vocab;
		usz byte_size = sample_vocab * float.sizeof;
		model.dev.mem_transfer(logits_bytes[:byte_size],
			((char*)g_host_logits)[:byte_size], TransferDirection.DTOH);

		float[] flogits = g_host_logits[:sample_vocab];

		// Sample using persistent global sampler
		uint[] ctx = state.all_tokens[:state.num_tokens];
		uint next_token = g_sampler.sample_with_context(flogits, ctx);

		// Stop on EOS
		if (next_token == eos_token) { break; }
		// Stop on out-of-vocab
		if (next_token >= (uint)tokenizer_vocab_size) { break; }

		generated_count++;
		state.append(next_token);

		// Decode and stream this token immediately
		uint[1] single = { next_token };
		if (try text = tokenizer.decode(&single))
		{
			io::stdout().write(text)!!;
			io::stdout().flush()!!;
			mem::free(text.ptr);
		}

		model_step(model, (ulong)next_token)!;
	}

	NanoDuration decode_ns = decode_clock.mark();

	// Newline after generation + timing report
	io::printn("");
	double prefill_ms = (double)prefill_ns / 1_000_000.0;
	double decode_ms = (double)decode_ns / 1_000_000.0;
	double prefill_tps = (double)n_prompt / (prefill_ms / 1000.0);
	double decode_tps = (double)generated_count / (decode_ms / 1000.0);
	io::printfn("\n--- Performance ---");
	io::printfn("Prefill: %d tokens in %.1f ms (%.1f tok/s)", n_prompt, prefill_ms, prefill_tps);
	io::printfn("Decode:  %d tokens in %.1f ms (%.1f tok/s)", generated_count, decode_ms, decode_tps);
}

// ============================================================================
// Main
// ============================================================================

fn void main(String[] args)
{
	// --- Parse CLI args ---
	ZString weights_path = null;
	ZString tokenizer_path = null;
	ZString initial_prompt = null;
	usz max_tokens = 0;
	usz max_seq_len = 0;
	float temperature = 0.6;
	uint top_k = 40;
	float top_p = 0.9;
	float repetition_penalty = 1.0;
	uint ngram_block = 4;
	float frequency_penalty = 0.0;
	ulong seed = 0;

	usz i = 1;
	while (i < args.len)
	{
		String arg = args[i];
		if (arg == "--weights" && i + 1 < args.len)
		{
			weights_path = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--tokenizer" && i + 1 < args.len)
		{
			tokenizer_path = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--prompt" && i + 1 < args.len)
		{
			initial_prompt = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--max-tokens" && i + 1 < args.len)
		{
			max_tokens = (usz)(args[i + 1].to_int() ?? 150);
			i += 2;
		}
		else if (arg == "--max-seq-len" && i + 1 < args.len)
		{
			max_seq_len = (usz)(args[i + 1].to_int() ?? 2048);
			i += 2;
		}
		else if (arg == "--temperature" && i + 1 < args.len)
		{
			temperature = (float)(args[i + 1].to_double() ?? 0.6);
			i += 2;
		}
		else if (arg == "--top-k" && i + 1 < args.len)
		{
			top_k = (uint)(args[i + 1].to_int() ?? 40);
			i += 2;
		}
		else if (arg == "--top-p" && i + 1 < args.len)
		{
			top_p = (float)(args[i + 1].to_double() ?? 0.9);
			i += 2;
		}
		else if (arg == "--repetition-penalty" && i + 1 < args.len)
		{
			repetition_penalty = (float)(args[i + 1].to_double() ?? 1.1);
			i += 2;
		}
		else if (arg == "--ngram-block" && i + 1 < args.len)
		{
			ngram_block = (uint)(args[i + 1].to_int() ?? 4);
			i += 2;
		}
		else if (arg == "--frequency-penalty" && i + 1 < args.len)
		{
			frequency_penalty = (float)(args[i + 1].to_double() ?? 0.0);
			i += 2;
		}
		else if (arg == "--seed" && i + 1 < args.len)
		{
			seed = (ulong)(args[i + 1].to_long() ?? 0);
			i += 2;
		}
		else if (arg == "--help" || arg == "-h")
		{
			io::printfn("Usage: %s --weights <path> --tokenizer <path> [options]", args[0]);
			io::printfn("");
			io::printfn("Required:");
			io::printfn("  --weights <path>      LLaMA 3.1 8B Instruct weights dir");
			io::printfn("  --tokenizer <path>    HuggingFace tokenizer.json");
			io::printfn("");
			io::printfn("Optional:");
			io::printfn("  --prompt <text>       Initial prompt (generates, then enters REPL)");
			io::printfn("  --max-tokens <int>    Max tokens per generation, 0=unlimited (default: 0)");
			io::printfn("  --max-seq-len <int>   Max sequence length for KV cache (default: 2048)");
			io::printfn("  --temperature <float> Sampling temperature (default: 0.6)");
			io::printfn("  --top-k <int>         Top-k sampling, 0=disabled (default: 40)");
			io::printfn("  --top-p <float>       Nucleus sampling, 0=disabled (default: 0.9)");
			io::printfn("  --repetition-penalty <float>  Repetition penalty (default: 1.0)");
			io::printfn("  --ngram-block <int>   Block repeated n-grams, 0=disabled (default: 4)");
			io::printfn("  --frequency-penalty <float>  Per-occurrence penalty (default: 0.0)");
			io::printfn("  --seed <int>          Random seed, 0=random (default: 0)");
			io::printfn("");
			io::printfn("REPL commands:");
			io::printfn("  /tokenize <text>      Encode text and print token IDs");
			io::printfn("  /reset                Reset model state and token history");
			io::printfn("  /quit, /exit          Exit the REPL");
			io::printfn("  Ctrl+D                Exit the REPL");
			return;
		}
		else
		{
			io::printfn("Unknown argument: %s", arg);
			return;
		}
	}

	if (weights_path == null)
	{
		io::printfn("Error: --weights is required");
		return;
	}

	HipDevice gpu = hip_device::hip_device_create({ .device_id = 0 });
	defer gpu.deinit();

	DeviceReference dev = gpu.reference();

	graph::init();
	graph::set_default_persistence(STABLE);

	io::printfn("\n=== LLaMA 3.1 8B Instruct — Decode Benchmark ===\n");

	// 1. Init model
	LlamaConfig cfg = LLAMA_31_8B;
	if (max_seq_len > 0)
	{
		cfg.max_seq_len = (ulong)max_seq_len;
	}

	io::printfn("Config: d_model=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d, intermediate=%d, n_layers=%d, vocab=%d, max_seq=%d",
		cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.n_layers, cfg.vocab_size, cfg.max_seq_len);

	io::printfn("\nBuilding model graph...");
	LlamaModel model = model_init(&cfg, dev)!!;
	io::printfn("  Model graph built successfully");

	// 2. Load weights (only load layer 0 since n_layers=1)
	usz wlen = libc::strlen(weights_path);
	io::printfn("");
	io::printfn("Loading weights...");
	load_weights(&model, weights_path[:wlen])!!;
	io::printfn("Weights loaded.");

	// 3. Benchmark: skip tokenizer, just run decode steps with token 0
	usz bench_tokens = max_tokens > 0 ? max_tokens : 10;
	io::printfn("\nBenchmark: %d decode steps (no tokenizer)", bench_tokens);

	Clock decode_clock = clock::now();
	for (usz step = 0; step < bench_tokens; step++)
	{
		model_step(&model, 0)!!;
	}
	NanoDuration decode_ns = decode_clock.mark();

	double decode_ms = (double)decode_ns / 1_000_000.0;
	double decode_tps = (double)bench_tokens / (decode_ms / 1000.0);
	io::printfn("\n--- Performance ---");
	io::printfn("Decode:  %d tokens in %.1f ms (%.1f tok/s)", bench_tokens, decode_ms, decode_tps);
	io::printfn("Exec stats: executed=%d, skipped=%d, freed=%d",
		graph::execution_sequence::stats_executed, graph::execution_sequence::stats_skipped,
		graph::execution_sequence::stats_freed);

	io::printfn("\n=== Done ===");
}
