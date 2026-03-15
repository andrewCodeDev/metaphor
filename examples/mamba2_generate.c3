//! Mamba-2 (SSD) Text Generation Pipeline
//!
//! Step-based inference for pretrained Mamba-2 models (Codestral Mamba 7B).
//! Loads weights from safetensors, tokenizes prompts via BPE,
//! and generates text one token at a time using greedy/sampled decoding.
//!
//! Key differences from Mamba-1:
//!   - Multi-head SSM: nheads=128, head_dim=64, n_groups=8
//!   - No x_proj/dt_proj — in_proj outputs x, z, B, C, dt in one shot
//!   - A and D are per-head [H], not per-inner [E] or [E,N]
//!   - norm_before_gate: rms_norm(y) * silu(z) instead of y * silu(z)
//!   - Separate lm_head (not tied to embedding)
//!
//! Architecture (step-based, single token per step):
//!   token (B,) u64
//!     → gather(embedding, dim=0)
//!   embedding (B, d_model)
//!     → N × Mamba2Block (with residual + RMS norm)
//!   hidden (B, d_model)
//!     → rms_norm → linear(lm_head)
//!   logits (B, vocab_size)
//!
//! Run: c3c build mamba2_generate && ./build/mamba2_generate

module metaphor::examples::mamba2_generate;
import metaphor::device::host_device;
import metaphor::device::cuda_device;
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
alias CudaDevice = cuda_device::CudaDevice;
alias Datatype = rtti::Datatype;
alias SliceRange = operations::SliceRange;

// Global sampler — persists PRNG state across all sample calls
sampling::Sampler g_sampler;
float* g_host_logits;

usz g_host_logits_size;

// ============================================================================
// Configuration
// ============================================================================

struct Mamba2Config
{
	ulong d_model;
	ulong d_state;
	ulong d_conv;
	ulong expand;
	ulong nheads;
	ulong head_dim;
	ulong n_groups;
	ulong vocab_size;
	ulong n_layers;
	float eps;
	ulong batch_size;
	Datatype dtype;
}

fn ulong Mamba2Config.d_inner(&self)
{
	return self.d_model * self.expand;
}

fn ulong Mamba2Config.heads_per_group(&self)
{
	return self.nheads / self.n_groups;
}

// Codestral Mamba 7B
const Mamba2Config CODESTRAL_7B = {
	.d_model = 4096,
	.d_state = 128,
	.d_conv = 4,
	.expand = 2,
	.nheads = 128,
	.head_dim = 64,
	.n_groups = 8,
	.vocab_size = 32768,
	.n_layers = 64,
	.eps = 1e-5,
	.batch_size = 1,
	.dtype = Datatype.BF16,
};

// ============================================================================
// Mamba-2 Layer
// ============================================================================

struct Mamba2Layer
{
	// Weights (9 tensors)
	Tensor norm_weight;   // [d_model]
	Tensor in_proj;       // [E + conv_dim + H, d_model]  (z, xBC, dt)
	Tensor conv_weight;   // [conv_dim, 1, d_conv]  where conv_dim = E + 2*G*N
	Tensor conv_bias;     // [conv_dim]
	Tensor out_proj;      // [d_model, E]
	Tensor a_log;         // [H]
	Tensor d_vec;         // [H]
	Tensor dt_bias;       // [H]
	Tensor mixer_norm;    // [E]

	// Cached state (input) — rank 3 with batch dim
	Tensor h;             // [B, E, N]
	Tensor conv_state;    // [B, conv_dim, d_conv-1]

	// Output state (computed by step, copied back)
	Tensor new_h;
	Tensor new_conv_state;

	DeviceReference dev;
}

fn Mamba2Layer? init_layer(Mamba2Config* cfg, DeviceReference dev, usz layer_idx)
{
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong d_conv = cfg.d_conv;
	ulong nheads = cfg.nheads;
	ulong n_groups = cfg.n_groups;
	Datatype dtype = cfg.dtype;

	Mamba2Layer layer;

	// conv_dim = d_inner + 2*G*N (conv operates on x, B, C concatenated)
	ulong conv_dim = d_inner + n_groups * d_state * 2;
	// in_proj output: E (z) + conv_dim (xBC) + H (dt)
	ulong in_proj_out = d_inner + conv_dim + nheads;

	ulong[1] nheads_shape = { nheads };

	layer.norm_weight = tensor::empty(dtype, dev, { cfg.d_model })!;
	layer.in_proj = tensor::empty(dtype, dev, { in_proj_out, cfg.d_model })!;
	layer.conv_weight = tensor::empty(dtype, dev, { conv_dim, 1, d_conv })!;
	layer.conv_bias = tensor::empty(dtype, dev, { conv_dim })!;
	layer.out_proj = tensor::empty(dtype, dev, { cfg.d_model, d_inner })!;
	layer.a_log = tensor::empty(dtype, dev, &nheads_shape)!;
	layer.d_vec = tensor::empty(dtype, dev, &nheads_shape)!;
	layer.dt_bias = tensor::empty(dtype, dev, &nheads_shape)!;
	layer.mixer_norm = tensor::empty(dtype, dev, { d_inner })!;

	// State tensors — zero-initialized (with batch dim)
	layer.h = tensor::zeros(dtype, dev, { cfg.batch_size, d_inner, d_state })!;
	layer.conv_state = tensor::zeros(dtype, dev, { cfg.batch_size, conv_dim, d_conv - 1 })!;

	layer.new_h = layer.h;
	layer.new_conv_state = layer.conv_state;
	layer.dev = dev;

	return layer;
}

/// Single-token forward for Mamba-2 inference (with batch dim).
/// Input x: (B, D)       — rank 2
/// Output:  (B, D)       — rank 2
///
/// Uses gather with precomputed index tensors to expand per-head/per-group
/// values to the flat inner dimension, making the SSM step structurally
/// identical to Mamba-1.
fn Tensor? layer_step(Mamba2Layer* self, Tensor x, Mamba2Config* cfg,
	Tensor head_idx, Tensor bc_idx)
{
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong nheads = cfg.nheads;
	ulong n_groups = cfg.n_groups;
	ulong gn = n_groups * d_state;
	ulong conv_dim = d_inner + gn * 2;

	// 1. Input projection: (B, D) -> (B, E + conv_dim + H)
	Tensor proj = functional::linear(x, self.in_proj)!;

	// 2. Slice into z, xBC, dt (HF split order: z, xBC, dt)
	SliceRange[2] z_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = d_inner, .step = 1 },
	};
	Tensor z_branch = proj.slice(&z_ranges)!;

	SliceRange[2] xbc_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = d_inner, .end = d_inner + conv_dim, .step = 1 },
	};
	Tensor xbc = proj.slice(&xbc_ranges)!;

	SliceRange[2] dt_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = d_inner + conv_dim, .end = d_inner + conv_dim + nheads, .step = 1 },
	};
	Tensor dt_raw = proj.slice(&dt_ranges)!;

	// 3. Conv step on xBC (conv_dim = E + 2*G*N)
	Tensor xbc_unsqueezed = xbc.unsqueeze(2)!;
	Tensor[2] conv_cat = { self.conv_state, xbc_unsqueezed };
	Tensor conv_input = tensor::cat(&conv_cat, 2)!;

	// Update conv_state: shift left, append new xBC
	SliceRange[3] shift_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 1, .end = 0, .step = 1 },
	};
	Tensor shifted_state = self.conv_state.slice(&shift_ranges)!;
	Tensor[2] new_conv_cat = { shifted_state, xbc_unsqueezed };
	Tensor new_conv_state = tensor::cat(&new_conv_cat, 2)!;

	// Conv: squeeze weight, broadcast multiply, reduce, add bias, SiLU
	Tensor conv_weight_2d = self.conv_weight.reduce("eod->ed")!;
	Tensor conv_mul = conv_input.mul(conv_weight_2d)!;
	Tensor conv_sum = conv_mul.reduce("bed->be")!;
	Tensor conv_biased = conv_sum.add(self.conv_bias)!;
	Tensor xbc_silu = conv_biased.silu()!;

	// 4. Split conv output into x, B, C
	SliceRange[2] x_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = d_inner, .step = 1 },
	};
	Tensor x_silu = xbc_silu.slice(&x_ranges)!;

	SliceRange[2] b_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = d_inner, .end = d_inner + gn, .step = 1 },
	};
	Tensor b_proj = xbc_silu.slice(&b_ranges)!;

	SliceRange[2] c_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = d_inner + gn, .end = d_inner + gn * 2, .step = 1 },
	};
	Tensor c_proj = xbc_silu.slice(&c_ranges)!;

	// 5. SSM step (multi-head via gather expansion)

	// dt = softplus(dt_raw + dt_bias): (B, H)
	Tensor dt_biased = dt_raw.add(self.dt_bias)!;
	Tensor dt = dt_biased.softplus()!;

	// A = -exp(A_log): (H)
	Tensor a_neg = self.a_log.exp()!.neg()!;

	// A_bar_heads = exp(dt * A): (B, H)
	Tensor dt_a = dt.einsum(a_neg, "bh,h->bh")!;
	Tensor a_bar_heads = dt_a.exp()!;

	// Gather-expand to flat inner dimension using precomputed indices
	Tensor a_bar = a_bar_heads.gather(1, head_idx)!;  // (B, H) -> (B, E)
	Tensor dt_exp = dt.gather(1, head_idx)!;           // (B, H) -> (B, E)
	Tensor b_exp = b_proj.gather(1, bc_idx)!;          // (B, G*N) -> (B, E, N)
	Tensor c_exp = c_proj.gather(1, bc_idx)!;          // (B, G*N) -> (B, E, N)
	Tensor d_exp = self.d_vec.gather(0, head_idx)!;    // (H) -> (E)

	// SSM recurrence: h_new = A_bar * h + dt * x * B
	Tensor dt_x = dt_exp.mul(x_silu)!;                           // (B, E)
	Tensor b_bar_x = dt_x.einsum(b_exp, "be,ben->ben")!;       // (B, E, N)
	Tensor a_bar_h = a_bar.einsum(self.h, "be,ben->ben")!;     // (B, E, N)
	Tensor new_h = a_bar_h.add(b_bar_x)!;                       // (B, E, N)

	// Output: y = C @ h_new + D * x
	Tensor c_h = c_exp.mul(new_h)!;                              // (B, E, N)
	Tensor y_ssm = c_h.reduce("ben->be")!;                      // (B, E)
	Tensor d_x = x_silu.mul(d_exp)!;                             // (B, E)
	Tensor y = y_ssm.add(d_x)!;                                  // (B, E)

	// 5. Gated RMS norm: rms_norm(y * silu(z)) * weight
	// HF MambaRMSNormGated: gate first, then normalize
	Tensor z_silu = z_branch.silu()!;
	Tensor gated = y.mul(z_silu)!;
	Tensor normed = functional::rms_norm(gated, self.mixer_norm, cfg.eps)!;

	// 6. Output projection: (B, E) -> (B, D)
	Tensor output = functional::linear(normed, self.out_proj)!;

	// Store output state — mark stable for collection
	self.new_h = new_h.stable();
	self.new_conv_state = new_conv_state.stable();

	b_bar_x.stable();

	return output;
}

/// Zero h and conv_state for new sequence.
fn void layer_reset(Mamba2Layer* self)
{
	self.h.fill(0.0);
	self.conv_state.fill(0.0);
}

/// Copy new_h/new_conv_state bytes back to h/conv_state.
fn void layer_copy_state(Mamba2Layer* self)
{
	char[] h_src = self.new_h.value_bytes();
	char[] h_dst = self.h.value_bytes();
	if (h_src.len > 0 && h_dst.len > 0)
	{
		self.dev.mem_copy(h_src, h_dst);
	}

	char[] conv_src = self.new_conv_state.value_bytes();
	char[] conv_dst = self.conv_state.value_bytes();
	if (conv_src.len > 0 && conv_dst.len > 0)
	{
		self.dev.mem_copy(conv_src, conv_dst);
	}
}

// ============================================================================
// Mamba-2 Model
// ============================================================================

const usz MAX_LAYERS = 64;

struct Mamba2Model
{
	Mamba2Config config;
	DeviceReference dev;

	// Weights
	Tensor embedding;           // [vocab_size, d_model]
	Tensor norm_f;              // [d_model]
	Tensor lm_head;             // [vocab_size, d_model] — separate, NOT tied

	// Precomputed index tensors for multi-head gather expansion
	Tensor head_idx;            // [E] U64: maps e -> e / head_dim
	Tensor bc_idx;              // [E, N] U64: maps (e,n) -> group(e)*N + n

	// Layers
	Mamba2Layer[MAX_LAYERS] layers;

	// Pipeline I/O
	Tensor tokens;              // [batch_size] u64
	Tensor logits;              // [batch_size, vocab_size]
}

/// Build precomputed index tensors for multi-head gather expansion.
fn void? build_index_tensors(Mamba2Model* model, Mamba2Config* cfg, DeviceReference dev)
{
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong head_dim = cfg.head_dim;
	ulong hpg = cfg.heads_per_group();

	// head_idx: [E] -> maps each inner dim to its head index
	// head_idx[e] = e / head_dim
	model.head_idx = tensor::empty(Datatype.U64, dev, { d_inner })!;
	{
		usz e_sz = (usz)d_inner;
		usz nbytes = e_sz * ulong.sizeof;
		ulong* host_buf = (ulong*)mem::malloc(nbytes);
		assert(host_buf != null, "Failed to allocate head_idx buffer");
		for (usz e = 0; e < e_sz; e++)
		{
			host_buf[e] = (ulong)(e / (usz)head_dim);
		}
		char[] dst = model.head_idx.value_bytes();
		if (dst.len == 0) { model.head_idx.core().alloc_data(); dst = model.head_idx.value_bytes(); }
		dev.mem_transfer(((char*)host_buf)[:nbytes], dst[:nbytes], TransferDirection.HTOD);
		mem::free(host_buf);
	}

	// bc_idx: [E, N] -> maps (e, n) to flat index into B_proj/C_proj [G*N]
	// bc_idx[e][n] = group(e) * N + n, where group(e) = e / (head_dim * hpg)
	model.bc_idx = tensor::empty(Datatype.U64, dev, { d_inner, d_state })!;
	{
		usz e_sz = (usz)d_inner;
		usz n_sz = (usz)d_state;
		usz total = e_sz * n_sz;
		usz nbytes = total * ulong.sizeof;
		ulong* host_buf = (ulong*)mem::malloc(nbytes);
		assert(host_buf != null, "Failed to allocate bc_idx buffer");
		usz elems_per_group = (usz)(head_dim * hpg);
		for (usz e = 0; e < e_sz; e++)
		{
			usz group = e / elems_per_group;
			for (usz n = 0; n < n_sz; n++)
			{
				host_buf[e * n_sz + n] = (ulong)(group * n_sz + n);
			}
		}
		char[] dst = model.bc_idx.value_bytes();
		if (dst.len == 0) { model.bc_idx.core().alloc_data(); dst = model.bc_idx.value_bytes(); }
		dev.mem_transfer(((char*)host_buf)[:nbytes], dst[:nbytes], TransferDirection.HTOD);
		mem::free(host_buf);
	}
}

fn Mamba2Model? model_init(Mamba2Config* cfg, DeviceReference dev)
{
	assert(cfg.n_layers <= MAX_LAYERS, "Too many layers");

	Mamba2Model model;
	model.config = *cfg;
	model.dev = dev;

	// Embedding
	model.embedding = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;

	// Final norm
	model.norm_f = tensor::empty(cfg.dtype, dev, { cfg.d_model })!;

	// Separate lm_head (not tied to embedding)
	model.lm_head = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;

	// Precomputed index tensors for multi-head SSM
	build_index_tensors(&model, cfg, dev)!;

	// Layers
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		model.layers[i] = init_layer(cfg, dev, i)!;
	}

	// Pipeline input: single token per batch
	model.tokens = tensor::empty(Datatype.U64, dev, { cfg.batch_size })!;

	// Build forward graph
	// 1. Embed: gather + unsqueeze for batch dim
	Tensor x_embed = model.embedding.gather(0, model.tokens)!.unsqueeze(0)!;

	// 2. Mamba-2 layers with residual
	Tensor x_hidden = x_embed;
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		Tensor normed = functional::rms_norm(x_hidden, model.layers[i].norm_weight, cfg.eps)!;
		Tensor layer_out = layer_step(&model.layers[i], normed, cfg,
			model.head_idx, model.bc_idx)!;
		x_hidden = x_hidden.add(layer_out)!;
	}

	// 3. Final norm
	x_hidden = functional::rms_norm(x_hidden, model.norm_f, cfg.eps)!;

	// 4. Project to vocab (separate lm_head)
	model.logits = functional::linear(x_hidden, model.lm_head)!.stable();

	return model;
}

/// Process a single token and return logits.
fn void? model_step(Mamba2Model* self, ulong token)
{
	// Fill token input
	char[] token_bytes = self.tokens.value_bytes();
	if (token_bytes.len == 0)
	{
		self.tokens.core().alloc_data();
		token_bytes = self.tokens.value_bytes();
	}
	self.dev.mem_transfer(((char*)&token)[:ulong.sizeof], token_bytes[:ulong.sizeof], TransferDirection.HTOD);

	// Advance generation so graph re-executes with new input
	graph::Graph* g = graph::get();
	g.advance_generation();

	// Collect logits (runs full forward pass)
	self.logits.collect()!;

	// Collect new_conv_state for each layer (not on path to logits)
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		self.layers[i].new_conv_state.collect()!;
	}

	// Copy state back: new_h -> h, new_conv_state -> conv_state
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		layer_copy_state(&self.layers[i]);
	}
}

/// Reset all layer states for a new sequence.
fn void model_reset(Mamba2Model* self)
{
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		layer_reset(&self.layers[i]);
	}
}

// ============================================================================
// Weight Loading
// ============================================================================

fn void? load_weights(Mamba2Model* self, char[] path)
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

fn void? load_weights_single(Mamba2Model* self, char[] path)
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

	copy_weight(self.embedding, &sf, "backbone.embeddings.weight")!;
	loaded++;
	copy_weight(self.norm_f, &sf, "backbone.norm_f.weight")!;
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

fn void? load_weights_sharded(Mamba2Model* self, char[] path)
{
	safetensors::ShardedSafeTensors* sharded = (safetensors::ShardedSafeTensors*)mem::malloc(safetensors::ShardedSafeTensors.sizeof);
	assert(sharded != null, "Failed to allocate ShardedSafeTensors");
	*sharded = safetensors::sharded_open(path)!;

	usz loaded = 0;

	copy_weight_sharded(self.embedding, sharded, "backbone.embeddings.weight")!;
	loaded++;
	copy_weight_sharded(self.norm_f, sharded, "backbone.norm_f.weight")!;
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

fn void? load_layer_weights(Mamba2Layer* layer, safetensors::SafeTensorsFile* sf, usz layer_idx)
{
	char[128] name_buf;
	usz prefix_len = format_layer_prefix(&name_buf, layer_idx);

	copy_layer_weight(layer.norm_weight, sf, &name_buf, prefix_len, "norm.weight")!;
	copy_layer_weight(layer.in_proj, sf, &name_buf, prefix_len, "mixer.in_proj.weight")!;
	copy_layer_weight(layer.out_proj, sf, &name_buf, prefix_len, "mixer.out_proj.weight")!;
	copy_layer_weight(layer.conv_weight, sf, &name_buf, prefix_len, "mixer.conv1d.weight")!;
	copy_layer_weight(layer.conv_bias, sf, &name_buf, prefix_len, "mixer.conv1d.bias")!;
	copy_layer_weight(layer.a_log, sf, &name_buf, prefix_len, "mixer.A_log")!;
	copy_layer_weight(layer.d_vec, sf, &name_buf, prefix_len, "mixer.D")!;
	copy_layer_weight(layer.dt_bias, sf, &name_buf, prefix_len, "mixer.dt_bias")!;
	copy_layer_weight(layer.mixer_norm, sf, &name_buf, prefix_len, "mixer.norm.weight")!;
}

fn usz format_layer_prefix(char[128]* buf, usz layer_idx)
{
	char[] prefix = "backbone.layers.";
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
	assert(view.data.len == expected, "Weight byte size mismatch");

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
	assert(view.data.len == expected, "Weight byte size mismatch");

	dst.core().device.mem_transfer(view.data, dst_bytes[:expected],
		TransferDirection.HTOD);
}

fn void? load_layer_weights_sharded(Mamba2Layer* layer, safetensors::ShardedSafeTensors* st, usz layer_idx)
{
	char[128] name_buf;
	usz prefix_len = format_layer_prefix(&name_buf, layer_idx);

	copy_layer_weight_sharded(layer.norm_weight, st, &name_buf, prefix_len, "norm.weight")!;
	copy_layer_weight_sharded(layer.in_proj, st, &name_buf, prefix_len, "mixer.in_proj.weight")!;
	copy_layer_weight_sharded(layer.out_proj, st, &name_buf, prefix_len, "mixer.out_proj.weight")!;
	copy_layer_weight_sharded(layer.conv_weight, st, &name_buf, prefix_len, "mixer.conv1d.weight")!;
	copy_layer_weight_sharded(layer.conv_bias, st, &name_buf, prefix_len, "mixer.conv1d.bias")!;
	copy_layer_weight_sharded(layer.a_log, st, &name_buf, prefix_len, "mixer.A_log")!;
	copy_layer_weight_sharded(layer.d_vec, st, &name_buf, prefix_len, "mixer.D")!;
	copy_layer_weight_sharded(layer.dt_bias, st, &name_buf, prefix_len, "mixer.dt_bias")!;
	copy_layer_weight_sharded(layer.mixer_norm, st, &name_buf, prefix_len, "mixer.norm.weight")!;
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
fn void? generate_continuation(Mamba2Model* model, bpe::Bpe* tokenizer, GenerationState* state,
	char[] prompt, usz max_new_tokens, usz tokenizer_vocab_size, uint eos_token)
{
	// 1. Encode prompt
	uint[] prompt_ids = tokenizer.encode(prompt)!;

	// 2. Append prompt tokens to persistent history
	for (usz i = 0; i < prompt_ids.len; i++)
	{
		state.append(prompt_ids[i]);
	}

	// 3. Prefill: process all prompt tokens through model
	Clock prefill_clock = clock::now();
	for (usz i = 0; i < prompt_ids.len; i++)
	{
		model_step(model, (ulong)prompt_ids[i])!;
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
		// Cast logits to F32 on device for host sampling
		Tensor logits_f32 = model.logits.cast(F32)!;
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

		// Stop on EOS or out-of-vocab tokens
		if (next_token == eos_token || next_token >= (uint)tokenizer_vocab_size) { break; }

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
	float temperature = 0.8;
	uint top_k = 40;
	float top_p = 0.95;
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
		else if (arg == "--temperature" && i + 1 < args.len)
		{
			temperature = (float)(args[i + 1].to_double() ?? 0.8);
			i += 2;
		}
		else if (arg == "--top-k" && i + 1 < args.len)
		{
			top_k = (uint)(args[i + 1].to_int() ?? 40);
			i += 2;
		}
		else if (arg == "--top-p" && i + 1 < args.len)
		{
			top_p = (float)(args[i + 1].to_double() ?? 0.95);
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
			io::printfn("  --weights <path>      Codestral Mamba 7B weights dir");
			io::printfn("  --tokenizer <path>    HuggingFace tokenizer.json");
			io::printfn("");
			io::printfn("Optional:");
			io::printfn("  --prompt <text>       Initial prompt (generates, then enters REPL)");
			io::printfn("  --max-tokens <int>    Max tokens per generation, 0=unlimited (default: 0)");
			io::printfn("  --temperature <float> Sampling temperature (default: 0.8)");
			io::printfn("  --top-k <int>         Top-k sampling, 0=disabled (default: 40)");
			io::printfn("  --top-p <float>       Nucleus sampling, 0=disabled (default: 0.95)");
			io::printfn("  --repetition-penalty <float>  Repetition penalty (default: 1.1)");
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

	if (tokenizer_path == null)
	{
		io::printfn("Error: --tokenizer is required");
		return;
	}
	if (weights_path == null)
	{
		io::printfn("Error: --weights is required");
		return;
	}

	CudaDevice gpu = cuda_device::cuda_device_create({ .device_id = 0 });
	defer gpu.deinit();

	DeviceReference dev = gpu.reference();

	graph::init();

	io::printfn("\n=== Mamba-2 Interactive Generator (Codestral 7B) ===\n");

	// Use stable persistence so state tensors are preserved across
	// graph executions instead of being freed after their last consumer.
	graph::set_default_persistence(tensor_core::Persistence.STABLE);

	// 1. Init model (single config: Codestral 7B)
	Mamba2Config cfg = CODESTRAL_7B;

	io::printfn("Config: d_model=%d, d_inner=%d, nheads=%d, head_dim=%d, n_groups=%d, d_state=%d, n_layers=%d, vocab=%d",
		cfg.d_model, cfg.d_inner(), cfg.nheads, cfg.head_dim, cfg.n_groups, cfg.d_state, cfg.n_layers, cfg.vocab_size);

	io::printfn("\nBuilding model graph...");
	Mamba2Model model = model_init(&cfg, dev)!!;
	io::printfn("  Model graph built successfully");

	// 2. Load weights
	usz wlen = libc::strlen(weights_path);
	io::printfn("");
	load_weights(&model, weights_path[:wlen])!!;

	// 3. Load tokenizer
	io::printfn("\nLoading tokenizer...");
	bpe::Bpe tokenizer;
	tokenizer.init();
	usz tlen = libc::strlen(tokenizer_path);
	tokenizer.load_huggingface(tokenizer_path[:tlen])!!;
	io::printfn("  Tokenizer loaded");

	// 4. Setup sampling and generation state
	sampling::SamplingConfig sample_cfg = {
		.temperature = temperature,
		.top_k = top_k,
		.top_p = top_p,
		.repetition_penalty = repetition_penalty,
		.ngram_size = ngram_block,
		.frequency_penalty = frequency_penalty,
		.penalty_exempt_below = 0,
		.penalty_length_threshold = 4,
		.seed = seed,
	};
	io::printfn("\nSampling: temp=%.2f, top_k=%d, top_p=%.2f, rep_penalty=%.2f, ngram_block=%d, freq_penalty=%.2f, seed=%d",
		temperature, top_k, top_p, repetition_penalty, ngram_block, frequency_penalty, seed);

	usz tok_vocab = (usz)tokenizer.v.num_tokens;
	uint eos_token = tokenizer.v.special.has_eos ? tokenizer.v.special.eos : 0;

	// Build token length table from tokenizer vocab
	char* token_lengths = (char*)mem::calloc(tok_vocab);
	assert(token_lengths != null, "Failed to allocate token length table");
	for (usz ti = 0; ti < tok_vocab; ti++)
	{
		uint len = tokenizer.v.tokens[ti].len;
		token_lengths[ti] = len > 255 ? (char)255 : (char)len;
	}
	defer mem::free(token_lengths);

	// Init global sampler
	g_sampler.init((uint)tok_vocab, sample_cfg);
	g_sampler.set_token_lengths(token_lengths);
	defer g_sampler.deinit();

	// Host buffer for DtoH logits transfer
	g_host_logits = (float*)mem::malloc(float.sizeof * tok_vocab);
	assert(g_host_logits != null, "Failed to allocate host logits buffer");
	g_host_logits_size = tok_vocab;
	defer mem::free(g_host_logits);

	GenerationState state;
	state.init();
	defer state.deinit();

	// 5. If initial prompt provided, generate from it first
	if (initial_prompt != null)
	{
		usz plen = libc::strlen(initial_prompt);
		io::printfn("\n--- Initial prompt: \"%s\" ---\n", initial_prompt);
		generate_continuation(&model, &tokenizer, &state,
			initial_prompt[:plen], max_tokens, tok_vocab, eos_token)!!;
	}

	// 6. Interactive REPL loop
	io::printfn("\nReady. Type a prompt and press Enter. Commands: /tokenize, /reset, /quit, Ctrl+D to exit.\n");

	while (true)
	{
		libc::fflush(libc::stdout());

		if (try line = io::treadline())
		{
			if (line.len == 0) { continue; }

			if (line == "/reset")
			{
				model_reset(&model);
				state.reset();
				io::printfn("State reset.");
				continue;
			}
			if (line == "/quit" || line == "/exit")
			{
				break;
			}
			if (line == "/peak")
			{
				usz peak = model.dev.mem_peak();
				usz reserved = model.dev.mem_reserved();
				io::printfn("Peak GPU memory: %.1f MB", (float)peak / (1024.0 * 1024.0));
				io::printfn("Reserved:        %.1f MB", (float)reserved / (1024.0 * 1024.0));
				continue;
			}
			if (line.len > 10 && line[:10] == "/tokenize ")
			{
				char[] text = line[10..];
				if (try ids = tokenizer.encode(text))
				{
					io::printfn("Input:  \"%s\"", (ZString)text.ptr);
					io::printfn("Tokens: %d", ids.len);
					for (usz ti = 0; ti < ids.len; ti++)
					{
						uint[1] single = { ids[ti] };
						if (try decoded = tokenizer.decode(&single))
						{
							io::printfn("  [%d] %d  \"%s\"", ti, ids[ti], (ZString)decoded.ptr);
							mem::free(decoded.ptr);
						}
						else
						{
							io::printfn("  [%d] %d  <decode failed>", ti, ids[ti]);
						}
					}
					mem::free(ids.ptr);
				}
				else
				{
					io::printfn("Encode failed");
				}
				continue;
			}

			io::printn("");
			generate_continuation(&model, &tokenizer, &state,
				line, max_tokens, tok_vocab, eos_token)!!;
			io::printn("");
		}
		else
		{
			break;  // EOF (Ctrl+D) → exit
		}
	}

	// Cleanup
	tokenizer.deinit();

	io::printfn("\n=== Done ===");
}
