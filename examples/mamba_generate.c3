//! Mamba Text Generation Pipeline
//!
//! Step-based inference for pretrained Mamba models.
//! Loads weights from safetensors, tokenizes prompts via BPE,
//! and generates text one token at a time using greedy decoding.
//!
//! Architecture (step-based, single token per step):
//!   token (B,) u64
//!     → gather(embedding, dim=0)
//!   embedding (B, d_model)
//!     → N × MambaBlock (with residual + RMS norm)
//!   hidden (B, d_model)
//!     → rms_norm → linear(lm_head)
//!   logits (B, vocab_size)
//!
//! Run: c3c build mamba_generate && ./build/mamba_generate

module metaphor::examples::mamba_generate;
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
import metaphor::tensor::tensor_core;
import metaphor::utils::sampling;
import metaphor::utils::cjson;
import std::math;
import std::io;
import std::core::mem;
import libc;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HostDevice = host_device::HostDevice;
alias CudaDevice = cuda_device::CudaDevice;
alias TransferDirection = device_reference::TransferDirection;
alias Datatype = rtti::Datatype;
alias SliceRange = operations::SliceRange;

// Global sampler — persists PRNG state across all sample calls
sampling::Sampler g_sampler;
float* g_host_logits;
usz g_host_logits_size;

// ============================================================================
// Configuration
// ============================================================================

struct MambaConfig
{
	ulong d_model;
	ulong d_state;
	ulong d_conv;
	ulong expand;
	ulong dt_rank;    // 0 = auto (d_model / 16)
	ulong vocab_size;
	ulong n_layers;
	float eps;
	ulong batch_size;
	Datatype dtype;
}

fn ulong MambaConfig.d_inner(&self)
{
	return self.d_model * self.expand;
}

fn ulong MambaConfig.get_dt_rank(&self)
{
	if (self.dt_rank > 0) { return self.dt_rank; }
	ulong r = self.d_model / 16;
	return r > 0 ? r : 1;
}

// Mamba-130m defaults
const MambaConfig MAMBA_130M = {
	.d_model = 768,
	.d_state = 16,
	.d_conv = 4,
	.expand = 2,
	.dt_rank = 0,
	.vocab_size = 50280,
	.n_layers = 24,
	.eps = 1e-5,
	.batch_size = 1,
	.dtype = Datatype.F32,
};

const MambaConfig MAMBA_370M = {
	.d_model = 1024,
	.d_state = 16,
	.d_conv = 4,
	.expand = 2,
	.dt_rank = 64,
	.vocab_size = 50280,
	.n_layers = 48,
	.eps = 1e-5,
	.batch_size = 1,
	.dtype = Datatype.F32,
};

const MambaConfig MAMBA_790M = {
	.d_model = 1536,
	.d_state = 16,
	.d_conv = 4,
	.expand = 2,
	.dt_rank = 96,
	.vocab_size = 50280,
	.n_layers = 48,
	.eps = 1e-5,
	.batch_size = 1,
	.dtype = Datatype.F32,
};

const MambaConfig MAMBA_2_8B = {
	.d_model = 2560,
	.d_state = 16,
	.d_conv = 4,
	.expand = 2,
	.dt_rank = 160,
	.vocab_size = 50280,
	.n_layers = 64,
	.eps = 1e-5,
	.batch_size = 1,
	.dtype = Datatype.F32,
};

// ============================================================================
// Mamba Layer
// ============================================================================

struct MambaLayer
{
	// Weights (10 tensors)
	Tensor norm_weight;   // [d_model]
	Tensor in_proj;       // [2*d_inner, d_model]
	Tensor conv_weight;   // [d_inner, 1, d_conv]
	Tensor conv_bias;     // [d_inner]
	Tensor out_proj;      // [d_model, d_inner]
	Tensor a_log;         // [d_inner, d_state]
	Tensor d_vec;         // [d_inner]
	Tensor x_proj;        // [dt_rank + 2*d_state, d_inner]
	Tensor dt_proj;       // [d_inner, dt_rank]
	Tensor dt_proj_bias;  // [d_inner]

	// Cached state (input) — rank 3 with batch dim
	Tensor h;             // [B, d_inner, d_state]
	Tensor conv_state;    // [B, d_inner, d_conv-1]

	// Output state (computed by step, copied back)
	Tensor new_h;
	Tensor new_conv_state;

	DeviceReference dev;
}

fn MambaLayer? init_layer(MambaConfig* cfg, DeviceReference dev, usz layer_idx)
{
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong d_conv = cfg.d_conv;
	ulong dt_rank = cfg.get_dt_rank();
	Datatype dtype = cfg.dtype;

	MambaLayer layer;

	ulong[1] d_inner_shape = { d_inner };

	layer.norm_weight = tensor::empty(dtype, dev, { cfg.d_model })!;
	layer.in_proj = tensor::empty(dtype, dev, { d_inner * 2, cfg.d_model })!;
	layer.conv_weight = tensor::empty(dtype, dev, { d_inner, 1, d_conv })!;
	layer.conv_bias = tensor::empty(dtype, dev, &d_inner_shape)!;
	layer.out_proj = tensor::empty(dtype, dev, { cfg.d_model, d_inner })!;
	layer.a_log = tensor::empty(dtype, dev, { d_inner, d_state })!;
	layer.d_vec = tensor::empty(dtype, dev, &d_inner_shape)!;
	layer.x_proj = tensor::empty(dtype, dev, { dt_rank + d_state * 2, d_inner })!;
	layer.dt_proj = tensor::empty(dtype, dev, { d_inner, dt_rank })!;
	layer.dt_proj_bias = tensor::empty(dtype, dev, &d_inner_shape)!;

	// State tensors — zero-initialized (with batch dim)
	layer.h = tensor::zeros(dtype, dev, { cfg.batch_size, d_inner, d_state })!;
	layer.conv_state = tensor::zeros(dtype, dev, { cfg.batch_size, d_inner, d_conv - 1 })!;

	// Will be set during step()
	layer.new_h = layer.h;
	layer.new_conv_state = layer.conv_state;
	layer.dev = dev;

	return layer;
}

/// Single-token forward for inference (with batch dim for CUDA cuTENSOR compat).
/// Input x: (B, D)       — rank 2
/// Output:  (B, D)       — rank 2
///
/// State: h (B, E, d_state), conv_state (B, E, d_conv-1) — rank 3
fn Tensor? layer_step(MambaLayer* self, Tensor x, MambaConfig* cfg)
{
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong dt_rank = cfg.get_dt_rank();

	// 1. Input projection: (B, D) -> (B, 2*E)
	Tensor xz = functional::linear(x, self.in_proj)!;

	// 2. Split into x_branch and z_branch: both (B, E)
	SliceRange[2] x_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = d_inner, .step = 1 },
	};
	Tensor x_branch = xz.slice(&x_ranges)!;

	SliceRange[2] z_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = d_inner, .end = d_inner * 2, .step = 1 },
	};
	Tensor z_branch = xz.slice(&z_ranges)!;

	// 3. Conv step
	Tensor x_unsqueezed = x_branch.unsqueeze(2)!; // (B, E, 1)

	// Build conv_input = [conv_state, x]: (B, E, d_conv)
	Tensor[2] conv_cat_inputs = { self.conv_state, x_unsqueezed };
	Tensor conv_input = tensor::cat(&conv_cat_inputs, 2)!;

	// Update conv_state: shift left, append new x
	SliceRange[3] shift_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 1, .end = 0, .step = 1 },
	};
	Tensor shifted_state = self.conv_state.slice(&shift_ranges)!;
	Tensor[2] new_conv_cat = { shifted_state, x_unsqueezed };
	Tensor new_conv_state = tensor::cat(&new_conv_cat, 2)!;

	// Conv: squeeze weight, broadcast multiply, reduce, add bias, SiLU
	Tensor conv_weight_2d = self.conv_weight.reduce("eod->ed")!;
	Tensor conv_mul = conv_input.mul(conv_weight_2d)!;
	Tensor x_conv = conv_mul.reduce("bed->be")!;
	x_conv = x_conv.add(self.conv_bias)!;
	Tensor x_silu = x_conv.silu()!;

	// 4. SSM step
	// Project to get dt, B, C: (B, E) -> (B, dt_rank + 2*d_state)
	Tensor x_dbc = functional::linear(x_silu, self.x_proj)!;

	SliceRange[2] dt_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = dt_rank, .step = 1 },
	};
	Tensor dt_input = x_dbc.slice(&dt_ranges)!;

	SliceRange[2] b_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = dt_rank, .end = dt_rank + d_state, .step = 1 },
	};
	Tensor b_vec = x_dbc.slice(&b_ranges)!; // (B, d_state)

	SliceRange[2] c_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = dt_rank + d_state, .end = dt_rank + d_state * 2, .step = 1 },
	};
	Tensor c_vec = x_dbc.slice(&c_ranges)!; // (B, d_state)

	// dt = softplus(clamp(dt_proj(dt_input) + bias, -20, 20))
	Tensor dt_proj_out = functional::linear(dt_input, self.dt_proj)!; // (B, E)
	Tensor dt_biased = dt_proj_out.add(self.dt_proj_bias)!;
	Tensor lower = dt_biased.scalar_like(-20.0)!;
	Tensor upper = dt_biased.scalar_like(20.0)!;
	Tensor dt_clamped = dt_biased.clamp(&lower, &upper)!;
	Tensor dt = dt_clamped.softplus()!; // (B, E)

	// A = -exp(A_log): (E, d_state)
	Tensor a = self.a_log.exp()!.neg()!;

	// Discretize: A_bar = exp(dt * A)
	// dt (B, E) * A (E, d_state) -> (B, E, d_state)
	Tensor a_bar = dt.einsum(a, "be,ed->bed")!.exp()!;

	// B_bar = dt * B: outer product (B,E) x (B,d_state) -> (B,E,d_state)
	Tensor b_bar = dt.einsum(b_vec, "be,bd->bed")!;

	// SSM recurrence: h_new = A_bar * h + B_bar * x
	Tensor b_bar_x = b_bar.einsum(x_silu, "bed,be->bed")!;
	Tensor new_h = a_bar.mul(self.h)!.add(b_bar_x)!; // (B, E, d_state)

	// Output: y = C * h_new + D * x
	Tensor y_ssm = c_vec.einsum(new_h, "bd,bed->be")!;
	Tensor d_x = x_silu.mul(self.d_vec)!; // (B, E) * (E,) broadcast -> (B, E)
	Tensor y = y_ssm.add(d_x)!;

	// 5. Gate with z_branch
	Tensor z_silu = z_branch.silu()!;
	Tensor gated = y.mul(z_silu)!;

	// 6. Output projection: (B, E) -> (B, D)
	Tensor output = functional::linear(gated, self.out_proj)!;

	// Store output state — mark stable for collection
	self.new_h = new_h.stable();
	self.new_conv_state = new_conv_state.stable();

	b_bar_x.stable();

	return output;
}

/// Zero h and conv_state for new sequence.
fn void layer_reset(MambaLayer* self)
{
	self.h.fill(0.0);
	self.conv_state.fill(0.0);
}

/// Copy new_h/new_conv_state bytes back to h/conv_state.
fn void layer_copy_state(MambaLayer* self)
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
// Mamba Model
// ============================================================================

const usz MAX_LAYERS = 64;

struct MambaModel
{
	MambaConfig config;
	DeviceReference dev;

	// Weights
	Tensor embedding;           // [vocab_size, d_model]
	Tensor norm_f;              // [d_model]
	// lm_head is tied to embedding

	// Layers
	MambaLayer[MAX_LAYERS] layers;

	// Pipeline I/O
	Tensor tokens;              // [batch_size] u64
	Tensor logits;              // [batch_size, vocab_size]
}

fn MambaModel? model_init(MambaConfig* cfg, DeviceReference dev)
{
	assert(cfg.n_layers <= MAX_LAYERS, "Too many layers");

	MambaModel model;
	model.config = *cfg;
	model.dev = dev;

	// Embedding
	model.embedding = tensor::empty(cfg.dtype, dev, { cfg.vocab_size, cfg.d_model })!;

	// Final norm
	model.norm_f = tensor::empty(cfg.dtype, dev, { cfg.d_model })!;

	// Layers
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		model.layers[i] = init_layer(cfg, dev, i)!;
	}

	// Pipeline input: single token per batch
	model.tokens = tensor::empty(Datatype.U64, dev, { cfg.batch_size })!;

	// Build forward graph
	// 1. Embed: gather + unsqueeze for batch dim
	Tensor x = model.embedding.gather(0, model.tokens)!.unsqueeze(0)!;

	// 2. Mamba layers with residual
	for (usz i = 0; i < (usz)cfg.n_layers; i++)
	{
		Tensor normed = functional::rms_norm(x, model.layers[i].norm_weight, cfg.eps)!;
		Tensor layer_out = layer_step(&model.layers[i], normed, cfg)!;
		x = x.add(layer_out)!;
	}

	// 3. Final norm
	x = functional::rms_norm(x, model.norm_f, cfg.eps)!;

	// 4. Project to vocab (lm_head tied to embedding)
	model.logits = functional::linear(x, model.embedding)!.stable();

	return model;
}

/// Process a single token and return logits.
fn void? model_step(MambaModel* self, ulong token)
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
fn void model_reset(MambaModel* self)
{
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		layer_reset(&self.layers[i]);
	}
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Read vocab_size from config.json in the weights directory.
fn ulong? get_vocab_size_from_config(char[] weights_dir)
{
	String config_path = string::tformat("%s/config.json", (ZString)weights_dir.ptr);

	char[] data = io::file::load_temp(config_path)!;

	// Null-terminate for cJSON
	char* cstr = (char*)mem::malloc(data.len + 1);
	assert(cstr != null);
	mem::copy(cstr, data.ptr, data.len);
	cstr[data.len] = 0;

	cjson::CJson* root = cjson::cJSON_Parse(cstr);
	mem::free(cstr);
	if (root == null) return safetensors::ST_INVALID_HEADER~;

	cjson::CJson* vs = cjson::cJSON_GetObjectItemCaseSensitive(root, "vocab_size");
	if (vs == null || cjson::cJSON_IsNumber(vs) == 0)
	{
		cjson::cJSON_Delete(root);
		return safetensors::ST_INVALID_HEADER~;
	}

	ulong vocab = (ulong)vs.valueint;
	cjson::cJSON_Delete(root);
	return vocab;
}

fn void? load_weights(MambaModel* self, char[] path)
{
	io::printfn("Loading weights from: %s", (ZString)path.ptr);

	if (safetensors::is_sharded_model(path))
	{
		// Sharded model — multiple safetensors files
		load_weights_sharded(self, path)!;
	}
	else
	{
		// Single file
		load_weights_single(self, path)!;
	}
}

fn void? load_weights_single(MambaModel* self, char[] path)
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

	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		load_layer_weights(&self.layers[i], &sf, i)!;
		loaded += 10;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sf.free();
	mem::free(file_data.ptr);
}

fn void? load_weights_sharded(MambaModel* self, char[] path)
{
	safetensors::ShardedSafeTensors* sharded = (safetensors::ShardedSafeTensors*)mem::malloc(safetensors::ShardedSafeTensors.sizeof);
	assert(sharded != null, "Failed to allocate ShardedSafeTensors");
	*sharded = safetensors::sharded_open(path)!;

	usz loaded = 0;

	copy_weight_sharded(self.embedding, sharded, "backbone.embeddings.weight")!;
	loaded++;
	copy_weight_sharded(self.norm_f, sharded, "backbone.norm_f.weight")!;
	loaded++;

	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		load_layer_weights_sharded(&self.layers[i], sharded, i)!;
		loaded += 10;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sharded.free();
	mem::free(sharded);
}

fn void? load_layer_weights(MambaLayer* layer, safetensors::SafeTensorsFile* sf, usz layer_idx)
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
	copy_layer_weight(layer.x_proj, sf, &name_buf, prefix_len, "mixer.x_proj.weight")!;
	copy_layer_weight(layer.dt_proj, sf, &name_buf, prefix_len, "mixer.dt_proj.weight")!;
	copy_layer_weight(layer.dt_proj_bias, sf, &name_buf, prefix_len, "mixer.dt_proj.bias")!;
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

fn void? load_layer_weights_sharded(MambaLayer* layer, safetensors::ShardedSafeTensors* st, usz layer_idx)
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
	copy_layer_weight_sharded(layer.x_proj, st, &name_buf, prefix_len, "mixer.x_proj.weight")!;
	copy_layer_weight_sharded(layer.dt_proj, st, &name_buf, prefix_len, "mixer.dt_proj.weight")!;
	copy_layer_weight_sharded(layer.dt_proj_bias, st, &name_buf, prefix_len, "mixer.dt_proj.bias")!;
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
// LoRA Weight Merging
// ============================================================================

/// Merge LoRA adapter weights into base model weights.
/// W_merged = W_base + scale * (lora_b @ lora_a)
/// scale = alpha / rank (typically 1.0 for rank=alpha=16)
fn void? merge_lora_weights(MambaModel* self, char[] path, float scale)
{
	io::printfn("Merging LoRA weights from: %s (scale=%.2f)", (ZString)path.ptr, scale);

	char* c_path = (char*)mem::malloc(path.len + 1);
	assert(c_path != null, "Failed to allocate path buffer");
	mem::copy(c_path, path.ptr, path.len);
	c_path[path.len] = 0;

	CFile f = libc::fopen((ZString)c_path, "rb");
	mem::free(c_path);
	assert(f != null, "Could not open LoRA checkpoint");

	libc::fseek(f, 0, libc::SEEK_END);
	long file_size_l = libc::ftell(f);
	libc::fseek(f, 0, libc::SEEK_SET);
	assert(file_size_l > 0, "Empty LoRA file");
	usz file_size = (usz)file_size_l;

	char* data = (char*)libc::malloc(file_size);
	assert(data != null, "Failed to allocate LoRA file buffer");
	usz bytes_read = libc::fread(data, 1, file_size, f);
	libc::fclose(f);
	assert(bytes_read == file_size, "Incomplete LoRA file read");

	char[] file_data = data[:file_size];
	safetensors::SafeTensorsFile sf = safetensors::deserialize(file_data)!;

	usz merged = 0;
	for (usz i = 0; i < (usz)self.config.n_layers; i++)
	{
		// Try in_proj LoRA
		if (try_merge_layer_proj(&self.layers[i].in_proj, &sf, i, "in_proj", scale))
		{
			merged++;
		}
		// Try out_proj LoRA
		if (try_merge_layer_proj(&self.layers[i].out_proj, &sf, i, "out_proj", scale))
		{
			merged++;
		}
	}

	io::printfn("  Merged %d LoRA adapters into base weights", merged);

	// Load embed_ext from LoRA checkpoint if present — write ext rows into embedding
	if (try ext_view = sf.get("embed_ext"))
	{
		usz d_model = (usz)self.config.d_model;
		usz ext_rows = 6;  // EXT_VOCAB
		usz base_rows = 50277;  // BASE_VOCAB
		usz row_bytes = d_model * float.sizeof;
		usz ext_bytes = ext_rows * row_bytes;
		usz offset = base_rows * row_bytes;

		assert(ext_view.data.len == ext_bytes, "embed_ext size mismatch in LoRA checkpoint");

		char[] embed_bytes = self.embedding.value_bytes();
		assert(embed_bytes.len >= offset + ext_bytes, "Embedding too small for ext rows");

		char* ext_dst = embed_bytes.ptr + offset;
		self.dev.mem_transfer(ext_view.data[:ext_bytes],
			ext_dst[:ext_bytes], TransferDirection.HTOD);
		io::printfn("  Loaded embed_ext from LoRA checkpoint (%d rows into embedding[%d:%d])",
			ext_rows, base_rows, base_rows + ext_rows);
	}

	sf.free();
	libc::free(data);
}

/// Try to find and merge a LoRA adapter for a specific layer projection.
/// Returns true if the adapter was found and merged.
fn bool try_merge_layer_proj(Tensor* base_weight, safetensors::SafeTensorsFile* sf,
	usz layer_idx, char[] proj_name, float scale)
{
	// Build tensor names: "layer.N.{proj_name}_lora_a" / "_lora_b"
	char[128] name_a;
	char[128] name_b;
	usz pos = 0;

	char[] prefix = "layer.";
	for (usz i = 0; i < prefix.len; i++) { name_a[pos] = prefix[i]; name_b[pos] = prefix[i]; pos++; }
	if (layer_idx >= 10) { name_a[pos] = (char)('0' + layer_idx / 10); name_b[pos] = name_a[pos]; pos++; }
	name_a[pos] = (char)('0' + layer_idx % 10); name_b[pos] = name_a[pos]; pos++;
	name_a[pos] = '.'; name_b[pos] = '.'; pos++;
	for (usz i = 0; i < proj_name.len; i++) { name_a[pos] = proj_name[i]; name_b[pos] = proj_name[i]; pos++; }

	usz base_pos = pos;

	// "_lora_a"
	char[] suffix_a = "_lora_a";
	for (usz i = 0; i < suffix_a.len; i++) { name_a[base_pos + i] = suffix_a[i]; }
	usz len_a = base_pos + suffix_a.len;

	// "_lora_b"
	char[] suffix_b = "_lora_b";
	for (usz i = 0; i < suffix_b.len; i++) { name_b[base_pos + i] = suffix_b[i]; }
	usz len_b = base_pos + suffix_b.len;

	// Look up tensors
	safetensors::TensorView view_a;
	safetensors::TensorView view_b;
	if (catch sf.get(name_a[:len_a])) { return false; }
	view_a = sf.get(name_a[:len_a])!!;
	view_b = sf.get(name_b[:len_b])!!;

	// lora_a: [rank, in_features], lora_b: [out_features, rank]
	usz rank = (usz)view_a.info.shape[0];
	usz in_feat = (usz)view_a.info.shape[1];
	usz out_feat = (usz)view_b.info.shape[0];

	float* a_data = (float*)view_a.data.ptr;
	float* b_data = (float*)view_b.data.ptr;

	// Compute lora_b @ lora_a on CPU: [out_features, in_features]
	usz merged_size = out_feat * in_feat;
	float* merged = (float*)libc::calloc(merged_size, float.sizeof);
	assert(merged != null, "Failed to allocate merge buffer");

	for (usz o = 0; o < out_feat; o++)
	{
		for (usz r = 0; r < rank; r++)
		{
			float b_val = b_data[o * rank + r] * scale;
			for (usz j = 0; j < in_feat; j++)
			{
				merged[o * in_feat + j] += b_val * a_data[r * in_feat + j];
			}
		}
	}

	// Read base weight from GPU, add merged LoRA, write back
	usz weight_bytes = out_feat * in_feat * float.sizeof;
	float* host_w = (float*)libc::malloc(weight_bytes);
	assert(host_w != null, "Failed to allocate host weight buffer");

	char[] gpu_bytes = base_weight.value_bytes();
	base_weight.core().device.mem_transfer(gpu_bytes[:weight_bytes],
		((char*)host_w)[:weight_bytes], TransferDirection.DTOH);

	for (usz k = 0; k < merged_size; k++)
	{
		host_w[k] += merged[k];
	}

	base_weight.core().device.mem_transfer(((char*)host_w)[:weight_bytes],
		gpu_bytes[:weight_bytes], TransferDirection.HTOD);

	libc::free(merged);
	libc::free(host_w);

	return true;
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
/// SSM state and token history persist across calls.
fn void? generate_continuation(MambaModel* model, bpe::Bpe* tokenizer, GenerationState* state,
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
	for (usz i = 0; i < prompt_ids.len; i++)
	{
		model_step(model, (ulong)prompt_ids[i])!;
	}

	mem::free(prompt_ids.ptr);

	// 4. Generate new tokens, streaming each to stdout
	usz limit = max_new_tokens > 0 ? max_new_tokens : 200;
	for (usz i = 0; i < limit; i++)
	{
		char[] logits_bytes = model.logits.value_bytes();
		assert(logits_bytes.len > 0, "No logits data");

		// Transfer logits from GPU to host buffer
		// Use model's actual vocab size (may be smaller than tokenizer vocab)
		usz model_vocab = model.config.vocab_size;
		usz sample_vocab = tokenizer_vocab_size < model_vocab ? tokenizer_vocab_size : model_vocab;
		usz byte_size = sample_vocab * model.config.dtype.size_of();
		model.dev.mem_transfer(logits_bytes[:byte_size],
			((char*)g_host_logits)[:byte_size], TransferDirection.DTOH);

		// Sample using persistent global sampler (preserves PRNG state)
		float[] flogits = g_host_logits[:sample_vocab];
		uint[] ctx = state.all_tokens[:state.num_tokens];
		uint next_token = g_sampler.sample_with_context(flogits, ctx);

		// Stop on EOS or out-of-vocab tokens
		if (next_token == eos_token || next_token >= (uint)tokenizer_vocab_size) { break; }

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

	// Newline after generation
	io::printn("");
}

// ============================================================================
// Main
// ============================================================================

fn void main(String[] args)
{
	// --- Parse CLI args ---
	ZString weights_path = null;
	ZString lora_path = null;
	ZString tokenizer_path = null;
	ZString initial_prompt = null;
	ZString model_size = null;  // "130m", "370m", "790m"
	ZString dump_tokens_str = null;  // comma-separated token IDs for --dump-logits
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
		else if (arg == "--lora" && i + 1 < args.len)
		{
			lora_path = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--tokenizer" && i + 1 < args.len)
		{
			tokenizer_path = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--model" && i + 1 < args.len)
		{
			model_size = (ZString)args[i + 1].ptr;
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
		else if (arg == "--dump-logits" && i + 1 < args.len)
		{
			dump_tokens_str = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--help" || arg == "-h")
		{
			io::printfn("Usage: %s --weights <path> --tokenizer <path> [options]", args[0]);
			io::printfn("");
			io::printfn("Required:");
			io::printfn("  --weights <path>      Pretrained base weights (.safetensors)");
			io::printfn("  --tokenizer <path>    HuggingFace tokenizer.json");
			io::printfn("");
			io::printfn("Optional:");
			io::printfn("  --model <size>        Model config: 130m, 370m, 790m, 2.8b (default: 130m)");
			io::printfn("  --lora <path>         LoRA checkpoint to merge (.safetensors)");
			io::printfn("  --prompt <text>       Initial prompt (generates, then enters REPL)");
			io::printfn("  --max-tokens <int>    Max tokens per generation, 0=unlimited (default: 0)");
			io::printfn("  --temperature <float> Sampling temperature (default: 0.8)");
			io::printfn("  --top-k <int>         Top-k sampling, 0=disabled (default: 40)");
			io::printfn("  --top-p <float>       Nucleus sampling, 0=disabled (default: 0.95)");
			io::printfn("  --repetition-penalty <float>  Repetition penalty (default: 1.1)");
			io::printfn("  --ngram-block <int>   Block repeated n-grams, 0=disabled (default: 4)");
			io::printfn("  --frequency-penalty <float>  Per-occurrence penalty (default: 0.0)");
			io::printfn("  --seed <int>          Random seed, 0=random (default: 0)");
			io::printfn("  --dump-logits <ids>   Feed comma-separated token IDs, print logits, exit");
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

	if (tokenizer_path == null && dump_tokens_str == null)
	{
		io::printfn("Error: --tokenizer is required (unless using --dump-logits)");
		return;
	}
	if (weights_path == null)
	{
		io::printfn("Error: --weights is required");
		return;
	}

	CudaDevice gpu = cuda_device::create();
	defer gpu.deinit();

	DeviceReference dev = gpu.reference();

	graph::init();

	io::printfn("\n=== Mamba Interactive Generator ===\n");

	// Use stable persistence so state tensors are preserved across
	// graph executions instead of being freed after their last consumer.
	graph::set_default_persistence(STABLE);

	// 1. Select and init model
	MambaConfig cfg = MAMBA_130M;
	if (model_size != null)
	{
		usz mlen = libc::strlen(model_size);
		char[] ms = model_size[:mlen];
		if (ms == "370m" || ms == "370M")
		{
			cfg = MAMBA_370M;
		}
		else if (ms == "790m" || ms == "790M")
		{
			cfg = MAMBA_790M;
		}
		else if (ms == "2.8b" || ms == "2.8B" || ms == "2800m")
		{
			cfg = MAMBA_2_8B;
		}
		else if (ms != "130m" && ms != "130M")
		{
			io::printfn("Unknown model size: %s (use 130m, 370m, 790m, or 2.8b)", model_size);
			return;
		}
	}
	usz wlen = libc::strlen(weights_path);

	// Override vocab size from config.json if different (e.g. fused extended vocab)
	if (try vocab = get_vocab_size_from_config(weights_path[:wlen]))
	{
		if (vocab != cfg.vocab_size)
		{
			io::printfn("  Vocab override: %d -> %d (from weights)", cfg.vocab_size, vocab);
			cfg.vocab_size = vocab;
		}
	}

	io::printfn("Config: d_model=%d, d_inner=%d, n_layers=%d, vocab=%d",
		cfg.d_model, cfg.d_inner(), cfg.n_layers, cfg.vocab_size);

	io::printfn("\nBuilding model graph...");
	MambaModel model = model_init(&cfg, dev)!!;
	io::printfn("  Model graph built successfully");

	// 2. Load base weights
	io::printfn("");
	load_weights(&model, weights_path[:wlen])!!;

	// 3. Merge LoRA if provided
	if (lora_path != null)
	{
		usz llen = libc::strlen(lora_path);
		merge_lora_weights(&model, lora_path[:llen], 1.0)!!;
	}

	// 3b. Dump-logits mode: feed specific tokens, print logits, exit
	if (dump_tokens_str != null)
	{
		usz dlen = libc::strlen(dump_tokens_str);
		char[] dstr = dump_tokens_str[:dlen];

		// Allocate host buffer for logits
		usz vocab = (usz)cfg.vocab_size;
		float* host_logits = (float*)mem::malloc(float.sizeof * vocab);
		assert(host_logits != null);
		defer mem::free(host_logits);

		// Parse comma-separated token IDs and feed them
		usz pos = 0;
		usz token_idx = 0;
		while (pos <= dstr.len)
		{
			// Find next comma or end
			usz end = pos;
			while (end < dstr.len && dstr[end] != ',') { end++; }

			// Parse token ID manually
			ulong token = 0;
			for (usz ci = pos; ci < end; ci++)
			{
				assert(dstr[ci] >= '0' && dstr[ci] <= '9', "Invalid token ID in --dump-logits");
				token = token * 10 + (ulong)(dstr[ci] - '0');
			}

			io::printfn("\n--- Step %d: feeding token %d ---", token_idx, token);

			model_step(&model, token)!!;

			// Transfer logits from GPU to host
			char[] logits_bytes = model.logits.value_bytes();
			usz byte_size = vocab * cfg.dtype.size_of();
			dev.mem_transfer(logits_bytes[:byte_size],
				((char*)host_logits)[:byte_size], TransferDirection.DTOH);

			// Find argmax and top-5
			int argmax_idx = 0;
			float argmax_val = host_logits[0];
			for (usz vi = 1; vi < vocab; vi++)
			{
				if (host_logits[vi] > argmax_val)
				{
					argmax_val = host_logits[vi];
					argmax_idx = (int)vi;
				}
			}

			// Print first 10 logits
			io::printf("  logits[:10] = [");
			for (usz vi = 0; vi < 10 && vi < vocab; vi++)
			{
				if (vi > 0) io::printf(", ");
				io::printf("%.4f", host_logits[vi]);
			}
			io::printfn("]");
			io::printfn("  argmax = %d (logit=%.4f)", argmax_idx, argmax_val);

			// Top-5
			int[5] top5_idx;
			float[5] top5_val;
			for (usz k = 0; k < 5; k++) { top5_idx[k] = -1; top5_val[k] = -1e30; }
			for (usz vi = 0; vi < vocab; vi++)
			{
				float v = host_logits[vi];
				if (v > top5_val[4])
				{
					top5_val[4] = v;
					top5_idx[4] = (int)vi;
					// Bubble up
					for (int k = 3; k >= 0; k--)
					{
						if (top5_val[k + 1] > top5_val[k])
						{
							float tv = top5_val[k]; top5_val[k] = top5_val[k + 1]; top5_val[k + 1] = tv;
							int ti = top5_idx[k]; top5_idx[k] = top5_idx[k + 1]; top5_idx[k + 1] = ti;
						}
					}
				}
			}
			io::printf("  top5: [");
			for (usz k = 0; k < 5; k++)
			{
				if (k > 0) io::printf(", ");
				io::printf("(%d, %.4f)", top5_idx[k], top5_val[k]);
			}
			io::printfn("]");

			pos = end + 1;
			token_idx++;
		}

		io::printfn("\nDump complete.");
		return;
	}

	// 4. Load tokenizer
	io::printfn("\nLoading tokenizer...");
	bpe::Bpe tokenizer;
	tokenizer.init();
	usz tlen = libc::strlen(tokenizer_path);
	tokenizer.load_huggingface(tokenizer_path[:tlen])!!;
	io::printfn("  Tokenizer loaded");

	// 5. Setup sampling and generation state
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
	uint eos_token = tokenizer.v.special.has_eos ? tokenizer.v.special.eos : 50256;

	// Build token length table from tokenizer vocab
	char* token_lengths = (char*)mem::calloc(tok_vocab);
	assert(token_lengths != null, "Failed to allocate token length table");
	for (usz ti = 0; ti < tok_vocab; ti++)
	{
		uint len = tokenizer.v.tokens[ti].len;
		token_lengths[ti] = len > 255 ? (char)255 : (char)len;
	}
	defer mem::free(token_lengths);

	// Init global sampler (persistent PRNG state across all generations)
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

	// 6. If initial prompt provided, generate from it first
	if (initial_prompt != null)
	{
		usz plen = libc::strlen(initial_prompt);
		io::printfn("\n--- Initial prompt: \"%s\" ---\n", initial_prompt);
		generate_continuation(&model, &tokenizer, &state,
			initial_prompt[:plen], max_tokens, tok_vocab, eos_token)!!;
	}

	// 7. Interactive REPL loop
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
