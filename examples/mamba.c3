//! LoRA Mamba Training Example
//!
//! Full-scale training of a Mamba model with LoRA (Low-Rank Adaptation).
//! Only LoRA parameters are trained; base model weights remain frozen.
//!
//! Features:
//!   - Configurable training (steps, LR, scheduler)
//!   - Interval loss reporting
//!   - Early stopping with patience
//!   - Periodic safetensors checkpoints (LoRA weights only)
//!   - CUDA support (set USE_CUDA = true)
//!
//! Memory comparison for Mamba-2.8B:
//!   Full fine-tuning: ~45 GB overhead (gradients + optimizer)
//!   LoRA (r=16):      ~1 GB overhead
//!
//! Architecture:
//!   token → Embedding → N × LoRAMambaBlock → RMSNorm → LM Head → logits
//!
//! Each LoRAMambaBlock:
//!   x → RMSNorm → InProj(+LoRA) → split(x_branch, z_branch)
//!   x_branch → Conv1D → SiLU → x_proj(+LoRA) → SSM(parallel scan) → y
//!   y * SiLU(z_branch) → OutProj(+LoRA) → residual

module metaphor::examples::mamba;
import metaphor::device::host_device;
import metaphor::device::cuda_device;
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
import metaphor::graph::execution_sequence;
import libc;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias TransferDirection = device_reference::TransferDirection;
alias HostDevice = host_device::HostDevice;
alias CudaDevice = cuda_device::CudaDevice;
alias Datatype = rtti::Datatype;
alias SliceRange = operations::SliceRange;

// Vocabulary constants for extended embedding
const ulong PRETRAINED_VOCAB = 50280;  // rows in pretrained safetensors
const ulong BASE_VOCAB = 50277;        // frozen rows (real pretrained tokens)
const ulong EXT_VOCAB = 6;             // new special tokens (50277-50282)
const ulong TOTAL_VOCAB = 50283;       // BASE_VOCAB + EXT_VOCAB
alias Sr = SliceRange;
alias Optimizer = optimizer::Optimizer;

// ============================================================================
// Constants
// ============================================================================

const usz MAX_LORA_LAYERS = 64;

// ============================================================================
// Model Configuration
// ============================================================================

struct MambaConfig
{
	ulong d_model;
	ulong d_state;
	ulong d_conv;
	ulong expand;
	ulong dt_rank;
	ulong vocab_size;
	ulong n_layers;
	float eps;
	ulong batch_size;
	ulong seq_len;
	Datatype dtype;
}

fn ulong MambaConfig.d_inner(&self)
{
	return self.d_model * self.expand;
}

fn ulong MambaConfig.get_dt_rank(&self)
{
	if (self.dt_rank > 0) return self.dt_rank;
	ulong r = self.d_model / 16;
	return r > 0 ? r : 1;
}

// ============================================================================
// Training Configuration
// ============================================================================

struct TrainingConfig
{
	int num_steps;           // Total training steps
	float lr;                // Base learning rate
	int log_interval;        // Print loss every N steps
	int checkpoint_interval; // Save checkpoint every N steps (0 = disabled)
	int patience;            // Early stopping patience (0 = disabled)
	float min_delta;         // Minimum improvement for early stopping
}

// ============================================================================
// LoRA Configuration & Adapter
// ============================================================================

struct LoRAConfig
{
	ulong rank;
	float alpha;
	bool adapt_in_proj;
	bool adapt_out_proj;
	bool adapt_x_proj;
	bool adapt_dt_proj;
}

fn LoRAConfig default_lora_config()
{
	return {
		.rank = 16,
		.alpha = 16.0,
		.adapt_in_proj = true,
		.adapt_out_proj = true,
		.adapt_x_proj = false,
		.adapt_dt_proj = false,
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
	// A: (rank, in_features) — Kaiming init for stable gradients
	Tensor lora_a = tensor::random(dtype, dev, { rank, in_features },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)in_features })!!.enable_grad();

	// B: (out_features, rank) — zero init so LoRA starts as identity
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
	// Base path: x @ W^T (frozen, no gradients flow to W)
	Tensor base_out = functional::linear(x, base_weight)!!;

	// LoRA path: scale * (x @ A^T @ B^T)
	Tensor lora_down = functional::linear(x, self.lora_a)!!;          // -> (..., rank)
	Tensor lora_up = functional::linear(lora_down, self.lora_b)!!;    // -> (..., out_features)

	// Multiply by scale (alpha / rank)
	Tensor scale_t = lora_up.scalar_like((double)self.scale)!!;
	Tensor lora_out = lora_up.mul(scale_t)!!;

	return base_out.add(lora_out)!!;
}

// ============================================================================
// Parallel Scan (Blelloch-style prefix sum with decay)
// ============================================================================

/// Sequential SSM scan: computes h[t] = A_bar[t]*h[t-1] + B_bar_x[t]
/// Input shapes: [batch, seq_len, d_inner, d_state]
/// O(N) sequential steps using shift-and-accumulate on the full tensor.
/// Each step shifts h right by 1 (zero-padded), multiplies by a_bar,
/// and adds b_bar_x. After seq_len steps the recurrence has propagated
/// through all positions.
fn Tensor? sequential_ssm_scan(Tensor a_bar, Tensor b_bar_x, DeviceReference dev)
{
	ulong seq_len = a_bar.core().shape().get(1);

	if (seq_len <= 1)
	{
		return b_bar_x;
	}

	// Mark scan inputs as compilation boundaries — their gradients
	// accumulate across all seq_len steps, so we need boundaries
	// to prevent unbounded subgraph growth in backward.
	a_bar = a_bar.unstable();
	b_bar_x = b_bar_x.unstable();

	// h starts as b_bar_x (the "input" term at each position).
	// Each iteration propagates information one step to the right:
	//   h = a_bar * shift_right(h) + b_bar_x
	// After k iterations, position t has accumulated contributions
	// from positions [t-k, t]. After seq_len-1 iterations, all
	// positions have the full recurrence.
	Tensor h = b_bar_x;

	for (ulong step = 0; step < seq_len - 1; step++)
	{
		Tensor h_shifted = shift_and_pad_zeros(h, 1, seq_len, dev)!!;
		h = a_bar.mul(h_shifted)!!.add(b_bar_x)!!;
		// Force compilation boundary so backward can free intermediates
		h = h.unstable();
	}

	return h;
}

/// Parallel SSM scan: computes all hidden states h[t] = A_bar[t]*h[t-1] + B_bar_x[t]
/// Input shapes: [batch, seq_len, d_inner, d_state]
/// Uses O(log N) doubling steps.
fn Tensor? parallel_ssm_scan(Tensor a_bar, Tensor b_bar_x, DeviceReference dev)
{
	ulong seq_len = a_bar.core().shape().get(1);

	if (seq_len <= 1)
	{
		return b_bar_x;
	}

	// Mark inputs as compilation boundaries so backward doesn't
	// reach into scan internals — they get recomputed instead.
	a_bar = a_bar.unstable();
	b_bar_x = b_bar_x.unstable();
	return prefix_sum_with_decay(a_bar, b_bar_x, seq_len, dev);
}

/// Compute prefix sum with exponential decay using doubling.
/// h[0] = B_bar_x[0]
/// h[t] = A_bar[t] * h[t-1] + B_bar_x[t]
fn Tensor? prefix_sum_with_decay(Tensor a_bar, Tensor b_bar_x, ulong seq_len, DeviceReference dev)
{
	Tensor h = b_bar_x;
	ulong stride = 1;

	while (stride < seq_len)
	{
		Tensor a_stride = compute_stride_product(a_bar, stride, dev)!!;
		Tensor h_shifted = shift_and_pad_zeros(h, stride, seq_len, dev)!!;
		h = a_stride.mul(h_shifted)!!.add(h)!!;
		stride *= 2;
	}

	return h;
}

/// Compute product of A_bar over `stride` consecutive positions.
fn Tensor? compute_stride_product(Tensor a_bar, ulong stride, DeviceReference dev)
{
	if (stride == 1) return a_bar;

	Tensor result = a_bar;
	ulong s = 1;

	while (s < stride)
	{
		Tensor shifted = shift_and_pad_identity(result, s, dev)!!;
		result = result.mul(shifted)!!;
		s *= 2;
	}

	return result;
}

/// Shift tensor right along seq dim (dim 1), pad with 1s (identity).
fn Tensor? shift_and_pad_identity(Tensor a, ulong stride, DeviceReference dev)
{
	ulong seq_len = a.core().shape().get(1);

	if (stride >= seq_len)
	{
		ulong[1] one_shape = { 1 };
		Tensor zero_t = tensor::constant(Datatype.F32, dev, &one_shape, 0.0)!!;
		Tensor one_t = tensor::constant(Datatype.F32, dev, &one_shape, 1.0)!!;
		return a.mul(zero_t)!!.add(one_t);
	}

	Tensor left_part = a.slice(&&(Sr[4]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = seq_len - stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	})!!;

	ulong[1] one_shape = { 1 };
	Tensor zero_t = tensor::constant(Datatype.F32, dev, &one_shape, 0.0)!!;
	Tensor one_t = tensor::constant(Datatype.F32, dev, &one_shape, 1.0)!!;
	Tensor identity_pad = a.slice(&&(Sr[4]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	})!!.mul(zero_t)!!.add(one_t)!!;

	Tensor[2] cat_inputs = { identity_pad, left_part };
	return tensor::cat(&cat_inputs, 1);
}

/// Shift tensor right along seq dim (dim 1), pad with 0s.
fn Tensor? shift_and_pad_zeros(Tensor h, ulong stride, ulong seq_len, DeviceReference dev)
{
	if (stride >= seq_len)
	{
		ulong[1] z_shape = { 1 };
		Tensor zero_t = tensor::constant(Datatype.F32, dev, &z_shape, 0.0)!!;
		return h.mul(zero_t);
	}

	Tensor left_part = h.slice(&&(Sr[4]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = seq_len - stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	})!!;

	Tensor zero_t = tensor::constant(Datatype.F32, dev, { 1 }, 0.0)!!;
	Tensor zero_pad = h.slice(&&(Sr[4]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	})!!.mul(zero_t)!!;

	Tensor[2] cat_inputs = { zero_pad, left_part };
	return tensor::cat(&cat_inputs, 1);
}

// ============================================================================
// LoRA Mamba Layer
// ============================================================================

struct LoRAMambaLayer
{
	// Frozen base weights
	Tensor norm_weight;
	Tensor in_proj;
	Tensor conv_weight;
	Tensor conv_bias;
	Tensor out_proj;
	Tensor a_log;
	Tensor d_vec;
	Tensor x_proj;
	Tensor dt_proj;
	Tensor dt_proj_bias;

	// Trainable LoRA adapters (null if not adapted)
	LoRAAdapter in_proj_lora;
	bool has_in_proj_lora;
	LoRAAdapter out_proj_lora;
	bool has_out_proj_lora;
	LoRAAdapter x_proj_lora;
	bool has_x_proj_lora;
	LoRAAdapter dt_proj_lora;
	bool has_dt_proj_lora;

	MambaConfig config;
	LoRAConfig lora_config;

}

fn LoRAMambaLayer? layer_init(MambaConfig cfg, LoRAConfig lcfg, DeviceReference dev)
{
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong dt_rank = cfg.get_dt_rank();

	// Shape declarations
	ulong[1] d_inner_shape = { d_inner };

	// Base weights — random init (would be replaced by pretrained weights in production)
	Tensor norm_weight = tensor::ones(cfg.dtype, dev, { cfg.d_model })!!;
	Tensor in_proj = tensor::random(cfg.dtype, dev, { d_inner * 2, cfg.d_model },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)(cfg.d_model) })!!;
	Tensor conv_weight = tensor::random(cfg.dtype, dev, { d_inner, 1, cfg.d_conv },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)(d_inner) })!!;
	Tensor conv_bias = tensor::zeros(cfg.dtype, dev, &d_inner_shape)!!;
	Tensor out_proj = tensor::random(cfg.dtype, dev, { cfg.d_model, d_inner },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)(d_inner) })!!;
	Tensor a_log = tensor::random(cfg.dtype, dev, { d_inner, d_state },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)(d_inner) })!!;
	Tensor d_vec = tensor::ones(cfg.dtype, dev, &d_inner_shape)!!;
	Tensor x_proj = tensor::random(cfg.dtype, dev, { dt_rank + d_state * 2, d_inner },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)(d_inner) })!!;
	Tensor dt_proj = tensor::random(cfg.dtype, dev, { d_inner, dt_rank },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)(dt_rank) })!!;
	Tensor dt_proj_bias = tensor::zeros(cfg.dtype, dev, &d_inner_shape)!!;

	LoRAMambaLayer layer = {
		.norm_weight = norm_weight,
		.in_proj = in_proj,
		.conv_weight = conv_weight,
		.conv_bias = conv_bias,
		.out_proj = out_proj,
		.a_log = a_log,
		.d_vec = d_vec,
		.x_proj = x_proj,
		.dt_proj = dt_proj,
		.dt_proj_bias = dt_proj_bias,
		.config = cfg,
		.lora_config = lcfg,
	};

	// Create LoRA adapters
	if (lcfg.adapt_in_proj)
	{
		layer.in_proj_lora = lora_init(
			cfg.d_model, d_inner * 2, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!!;
		layer.has_in_proj_lora = true;
	}

	if (lcfg.adapt_out_proj)
	{
		layer.out_proj_lora = lora_init(
			d_inner, cfg.d_model, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!!;
		layer.has_out_proj_lora = true;
	}

	if (lcfg.adapt_x_proj)
	{
		layer.x_proj_lora = lora_init(
			d_inner, dt_rank + d_state * 2, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!!;
		layer.has_x_proj_lora = true;
	}

	if (lcfg.adapt_dt_proj)
	{
		layer.dt_proj_lora = lora_init(
			dt_rank, d_inner, lcfg.rank, lcfg.alpha, cfg.dtype, dev)!!;
		layer.has_dt_proj_lora = true;
	}

	return layer;
}

/// Forward pass using parallel scan (for training).
/// Input: [batch, seq_len, d_model]
/// Output: [batch, seq_len, d_model]
fn Tensor? layer_forward(LoRAMambaLayer* self, Tensor x, DeviceReference dev, int layer_idx = 0)
{
	MambaConfig cfg = self.config;
	ulong d_inner = cfg.d_inner();
	ulong d_state = cfg.d_state;
	ulong dt_rank = cfg.get_dt_rank();

	int li = layer_idx;

	// 1. Input projection with optional LoRA
	Tensor xz = self.has_in_proj_lora
		? lora_forward(&self.in_proj_lora, x, self.in_proj, dev)!!.stable()
		: functional::linear(x, self.in_proj)!!.stable();
	xz.fmt_label("L%d_xz", li);

	// 2. Split into x_branch [B,L,d_inner] and z_branch [B,L,d_inner]
	Tensor x_branch = xz.slice(&&(Sr[3]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = d_inner, .step = 1 },
	})!!;
	x_branch.fmt_label("L%d_x_branch", li);

	Tensor z_branch = xz.slice(&&(Sr[3]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = d_inner, .end = d_inner * 2, .step = 1 },
	})!!;
	z_branch.fmt_label("L%d_z_branch", li);

	// 3-5. Conv1d path: transpose → conv → transpose back → SiLU
	Tensor x_conv = causal_conv1d(
		x_branch.transpose(1, 2)!!, self.conv_weight, self.conv_bias)!!;
	x_conv.fmt_label("L%d_x_conv", li);

	Tensor x_silu = x_conv.transpose(1, 2)!!.silu()!!;
	x_silu.fmt_label("L%d_x_silu", li);

	// 6. Project to get dt, B, C (with optional LoRA on x_proj)
	Tensor x_dbc = self.has_x_proj_lora
		? lora_forward(&self.x_proj_lora, x_silu, self.x_proj, dev)!!
		: functional::linear(x_silu, self.x_proj)!!;
	x_dbc.fmt_label("L%d_x_dbc", li);

	Tensor dt_input = x_dbc.slice(&&(Sr[3]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = dt_rank, .step = 1 },
	})!!;
	dt_input.fmt_label("L%d_dt_input", li);

	Tensor b_proj = x_dbc.slice(&&(Sr[3]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = dt_rank, .end = dt_rank + d_state, .step = 1 },
	})!!;
	b_proj.fmt_label("L%d_b_proj", li);

	Tensor c_proj = x_dbc.slice(&&(Sr[3]){
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = dt_rank + d_state, .end = dt_rank + d_state * 2, .step = 1 },
	})!!;
	c_proj.fmt_label("L%d_c_proj", li);

	// 7. Compute dt = softplus(dt_proj(dt_input) + bias)
	Tensor dt_proj_out = self.has_dt_proj_lora
		? lora_forward(&self.dt_proj_lora, dt_input, self.dt_proj, dev)!!
			.einsum(self.dt_proj_bias, "ble,e->ble", inner: operations::BinaryOp.ADD)!!
		: functional::linear(dt_input, self.dt_proj, self.dt_proj_bias)!!;
	dt_proj_out.fmt_label("L%d_dt_proj_out", li);

	Tensor lower = dt_proj_out.scalar_like(-20.0)!!;
	Tensor upper = dt_proj_out.scalar_like(20.0)!!;
	Tensor dt = dt_proj_out.clamp(&lower, &upper)!!.softplus()!!;
	dt.fmt_label("L%d_dt", li);

	// 8. Discretize: A_bar = exp(dt * A), B_bar = dt * B
	Tensor a_bar = dt.einsum(self.a_log.exp()!!.neg()!!, "ble,ed->bled")!!.exp()!!;
	a_bar.fmt_label("L%d_a_bar", li);
	Tensor b_bar = dt.einsum(b_proj, "ble,bld->bled")!!;
	b_bar.fmt_label("L%d_b_bar", li);

	// 9. Sequential scan
	Tensor b_bar_x = b_bar.einsum(x_silu, "bled,ble->bled")!!;
	b_bar_x.fmt_label("L%d_b_bar_x", li);
	Tensor h = functional::selective_scan(a_bar, b_bar_x)!!;
	h.fmt_label("L%d_h", li);

	// 10. Output: y = C * h + D * x
	Tensor y = c_proj.einsum(h, "bld,bled->ble")!!
		.add(x_silu.einsum(self.d_vec, "ble,e->ble")!!)!!;
	y.fmt_label("L%d_y", li);

	// 11. Gate with z_branch
	Tensor gated = y.mul(z_branch.silu()!!)!!;
	gated.fmt_label("L%d_gated", li);

	// 12. Output projection with optional LoRA
	Tensor out = self.has_out_proj_lora
		? lora_forward(&self.out_proj_lora, gated, self.out_proj, dev)!!.stable()
		: functional::linear(gated, self.out_proj)!!.stable();
	out.fmt_label("L%d_out", li);
	return out;
}

/// Count LoRA parameters in this layer.
fn ulong layer_lora_param_count(LoRAMambaLayer* self)
{
	ulong count = 0;
	if (self.has_in_proj_lora)
	{
		count += self.in_proj_lora.lora_a.core().shape().size()
		       + self.in_proj_lora.lora_b.core().shape().size();
	}
	if (self.has_out_proj_lora)
	{
		count += self.out_proj_lora.lora_a.core().shape().size()
		       + self.out_proj_lora.lora_b.core().shape().size();
	}
	if (self.has_x_proj_lora)
	{
		count += self.x_proj_lora.lora_a.core().shape().size()
		       + self.x_proj_lora.lora_b.core().shape().size();
	}
	if (self.has_dt_proj_lora)
	{
		count += self.dt_proj_lora.lora_a.core().shape().size()
		       + self.dt_proj_lora.lora_b.core().shape().size();
	}
	return count;
}

// ============================================================================
// Causal Conv1D
// ============================================================================

/// Causal 1D convolution (depthwise).
/// x: [B, E, L], weight: [E, 1, K], bias: [E]
/// Output: [B, E, L]
/// Uses dedicated CUDA kernel for fused forward + backward.
fn Tensor? causal_conv1d(
	Tensor x,         // [B, E, L]
	Tensor weight,    // [E, 1, K]
	Tensor bias,      // [E]
)
{
	return functional::causal_conv1d(x, weight, bias);
}

// ============================================================================
// LoRA Mamba Model
// ============================================================================

struct LoRAMambaModel
{
	MambaConfig config;
	LoRAConfig lora_config;

	// Frozen base weights
	Tensor embedding;
	Tensor norm_f;
	Tensor lm_head;

	// LoRA-adapted layers
	LoRAMambaLayer[MAX_LORA_LAYERS] layers;
	usz num_layers;

	// Split embedding: frozen base [BASE_VOCAB, d] + trainable ext [EXT_VOCAB, d]
	Tensor embed_ext;       // [EXT_VOCAB, d_model] trainable extension
	Tensor embed_base;      // [BASE_VOCAB, d_model] frozen base

	// Pipeline tensors
	Tensor tokens;
	Tensor logits;

	// Layer residual outputs (for per-layer gradient collection)
	Tensor[MAX_LORA_LAYERS] residuals;
}

fn LoRAMambaModel? model_init(MambaConfig cfg, LoRAConfig lcfg, DeviceReference dev, int train_layers = 0)
{

	// Ensure vocab_size covers the full table
	MambaConfig ext_cfg = cfg;
	if (ext_cfg.vocab_size < TOTAL_VOCAB) { ext_cfg.vocab_size = TOTAL_VOCAB; }

	Tensor embedding;
	Tensor lm_head;
	Tensor embed_base = tensor::NULL_TENSOR;
	Tensor embed_ext = tensor::NULL_TENSOR;

	// Split embedding: frozen base [BASE_VOCAB, d] + trainable ext [EXT_VOCAB, d]
	// embed_ext is trained alongside LoRA adapters.
	{

		embed_base = tensor::zeros(cfg.dtype, dev, { BASE_VOCAB, cfg.d_model })!!;
		embed_ext = tensor::zeros(cfg.dtype, dev, { EXT_VOCAB, cfg.d_model })!!.enable_grad();
		embed_ext.set_label("embed_ext");

		embedding = tensor::cat(&&(Tensor[2]){ embed_base, embed_ext }, 0)!!;
		embedding.set_label("embedding");
		lm_head = tensor::cat(&&(Tensor[2]){ embed_base, embed_ext }, 0)!!;
		lm_head.set_label("lm_head");
	}

	LoRAMambaModel model = {
		.config = ext_cfg,
		.lora_config = lcfg,
		.num_layers = (usz)cfg.n_layers,
		.embedding = embedding,
		.norm_f = tensor::ones(cfg.dtype, dev, { cfg.d_model })!!.enable_grad(),
		.lm_head = lm_head,
		.embed_ext = embed_ext,
		.embed_base = embed_base,
	};
	cfg = ext_cfg;

	// Create layers — only last train_layers get LoRA (0 = all)
	usz lora_start = 0;
	if (train_layers > 0 && (usz)train_layers < model.num_layers)
	{
		lora_start = model.num_layers - (usz)train_layers;
	}

	LoRAConfig no_lora = {
		.rank = lcfg.rank,
		.alpha = lcfg.alpha,
		.adapt_in_proj = false,
		.adapt_out_proj = false,
		.adapt_x_proj = false,
		.adapt_dt_proj = false,
	};

	for (usz i = 0; i < model.num_layers; i++)
	{
		model.layers[i] = layer_init(cfg, i >= lora_start ? lcfg : no_lora, dev)!!;
		if (i >= lora_start)
		{
			LoRAMambaLayer* l = &model.layers[i];
			if (l.has_in_proj_lora)
			{
				l.in_proj_lora.lora_a.fmt_label("lora_L%d_in_a", (int)i);
				l.in_proj_lora.lora_b.fmt_label("lora_L%d_in_b", (int)i);
			}
			if (l.has_out_proj_lora)
			{
				l.out_proj_lora.lora_a.fmt_label("lora_L%d_out_a", (int)i);
				l.out_proj_lora.lora_b.fmt_label("lora_L%d_out_b", (int)i);
			}
		}
	}

	// Build forward graph — input tokens placeholder
	model.tokens = tensor::empty(Datatype.U64, dev, { cfg.batch_size, cfg.seq_len })!!;

	// Embedding lookup: [batch, seq_len] -> [batch, seq_len, d_model]
	Tensor x = model.embedding.gather(0, model.tokens)!!;
	x = x.stable();
	x.set_label("embed_out");

	for (usz i = 0; i < model.num_layers; i++)
	{
		Tensor normed = functional::rms_norm(x, model.layers[i].norm_weight, cfg.eps)!!;
		normed = normed.unstable();
		normed.fmt_label("L%d_normed", (int)i);
		Tensor layer_out = layer_forward(&model.layers[i], normed, dev, (int)i)!!;
		x = x.add(layer_out)!!.unstable();
		x.fmt_label("L%d_residual", (int)i);
		model.residuals[i] = x;
	}

	// Final norm
	x = functional::rms_norm(x, model.norm_f, cfg.eps)!!;
	x.set_label("final_norm");

	// Project to vocab
	model.logits = functional::linear(x, model.lm_head)!!.stable();
	model.logits.set_label("logits");

	return model;
}

/// Total count of frozen base parameters.
fn ulong total_base_params(MambaConfig cfg)
{
	ulong d_inner = cfg.d_inner();
	ulong dt_rank = cfg.get_dt_rank();

	ulong count = cfg.vocab_size * cfg.d_model;  // embedding
	count += cfg.d_model;                          // norm_f
	count += cfg.vocab_size * cfg.d_model;         // lm_head

	ulong per_layer = cfg.d_model                             // norm
		+ d_inner * 2 * cfg.d_model                           // in_proj
		+ d_inner * cfg.d_conv                                // conv_weight
		+ d_inner                                             // conv_bias
		+ cfg.d_model * d_inner                               // out_proj
		+ d_inner * cfg.d_state                               // A_log
		+ d_inner                                             // D
		+ (dt_rank + cfg.d_state * 2) * d_inner               // x_proj
		+ d_inner * dt_rank                                   // dt_proj
		+ d_inner;                                            // dt_proj_bias

	count += per_layer * cfg.n_layers;
	return count;
}

/// Total count of trainable LoRA parameters.
fn ulong total_lora_params(LoRAMambaModel* model)
{
	ulong count = 0;
	for (usz i = 0; i < model.num_layers; i++)
	{
		count += layer_lora_param_count(&model.layers[i]);
	}
	return count;
}

/// Register all trainable parameters with the optimizer.
/// Registers embed_ext + LoRA A/B matrices (base weights are frozen).
/// Returns the total number of parameter groups registered.
fn usz register_lora_params(LoRAMambaModel* model, Optimizer* opt)
{
	usz count = 0;

	// Always register embed_ext — trained alongside LoRA adapters
	opt.adam(&&(Tensor[1]){ model.embed_ext },
		{ ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
	count += 1;

	// Register LoRA parameters — base weights are frozen
	// Register layers in reverse order (last layer = closest to loss)
	for (usz ri = 0; ri < model.num_layers; ri++)
	{
		usz i = model.num_layers - 1 - ri;
		LoRAMambaLayer* l = &model.layers[i];
		if (l.has_in_proj_lora)
		{
			opt.adam(&&(Tensor[2]){ l.in_proj_lora.lora_a, l.in_proj_lora.lora_b },
				{ ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
			count += 2;
		}
		if (l.has_out_proj_lora)
		{
			opt.adam(&&(Tensor[2]){ l.out_proj_lora.lora_a, l.out_proj_lora.lora_b },
				{ ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
			count += 2;
		}
		if (l.has_x_proj_lora)
		{
			opt.adam(&&(Tensor[2]){ l.x_proj_lora.lora_a, l.x_proj_lora.lora_b },
				{ ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
			count += 2;
		}
		if (l.has_dt_proj_lora)
		{
			opt.adam(&&(Tensor[2]){ l.dt_proj_lora.lora_a, l.dt_proj_lora.lora_b },
				{ ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
			count += 2;
		}
	}
	return count;
}

// ============================================================================
// Checkpoint Saving (LoRA weights only, safetensors format)
// ============================================================================

struct NamedParam
{
	char[] name;
	Tensor tensor;
}

fn void save_lora_checkpoint(LoRAMambaModel* model, char[] path)
{
	// Build descriptors (max 64 layers * 4 projs * 2 + 1 embed_ext = 513)
	safetensors::TensorDescriptor[513] descs;
	char[64][513] name_bufs;
	usz d = 0;

	// Always save embed_ext — trained alongside LoRA adapters
	{
		char[] en = io::bprintf(&name_bufs[d], "embed_ext")!!;
		descs[d] = make_lora_desc(en, model.embed_ext);
		d++;
	}

	for (usz i = 0; i < model.num_layers; i++)
	{
		LoRAMambaLayer* l = &model.layers[i];

		if (l.has_in_proj_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.in_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.in_proj_lora.lora_a);
			d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.in_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.in_proj_lora.lora_b);
			d++;
		}
		if (l.has_out_proj_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.out_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.out_proj_lora.lora_a);
			d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.out_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.out_proj_lora.lora_b);
			d++;
		}
		if (l.has_x_proj_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.x_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.x_proj_lora.lora_a);
			d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.x_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.x_proj_lora.lora_b);
			d++;
		}
		if (l.has_dt_proj_lora)
		{
			char[] na = io::bprintf(&name_bufs[d], "layer.%d.dt_proj_lora_a", i)!!;
			descs[d] = make_lora_desc(na, l.dt_proj_lora.lora_a);
			d++;
			char[] nb = io::bprintf(&name_bufs[d], "layer.%d.dt_proj_lora_b", i)!!;
			descs[d] = make_lora_desc(nb, l.dt_proj_lora.lora_b);
			d++;
		}
	}

	// Serialize and write
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

fn safetensors::TensorDescriptor make_lora_desc(char[] name, Tensor t)
{
	assert(t.core().has_data(), "make_lora_desc: tensor has no data — cannot save");
	ulong[] shape_slice = t.core().shape().slice();
	char[] data_bytes = t.value_bytes();

	// If tensor lives on GPU, copy to host for serialization
	ZString dev_name = t.core().device.name();
	if (dev_name != "host")
	{
		char* host_buf = (char*)libc::malloc(data_bytes.len);
		assert(host_buf != null, "Failed to allocate host buffer for LoRA save");
		char[] host_bytes = host_buf[:data_bytes.len];
		t.core().device.mem_transfer(data_bytes, host_bytes, TransferDirection.DTOH);
		data_bytes = host_bytes;
	}

	return {
		.name = name,
		.dtype = safetensors::Dtype.F32,
		.shape = shape_slice,
		.data = data_bytes,
	};
}

// ============================================================================
// Pretrained Weight Loading (single-shard safetensors)
// ============================================================================

/// Load pretrained Mamba weights from a safetensors file into a LoRA model.
/// Weight names follow HuggingFace convention (backbone.layers.N.mixer.*).
/// The lm_head is tied to embedding (same tensor loaded into both).
fn void? load_pretrained_weights(LoRAMambaModel* model, char[] path, DeviceReference dev)
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

	// Try to load embed_ext from a separate file (overrides rows from base model)
	{
		String ext_path = string::tformat("%s/embed_ext.safetensors", (ZString)path.ptr);
		if (try ext_data = io::file::load(mem, ext_path))
		{
			safetensors::SafeTensorsFile sf = safetensors::deserialize(ext_data)!;
			copy_pretrained_weight(model.embed_ext, &sf, "embed_ext", dev)!;
			io::printfn("  Loaded embed_ext from %s (%d rows)", ext_path, EXT_VOCAB);
			sf.free();
			mem::free(ext_data.ptr);
		}
	}
}

fn void? load_pretrained_weights_single(LoRAMambaModel* model, char[] path, DeviceReference dev)
{
	// If path ends with .safetensors, use it directly; otherwise assume directory
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

	// Load base embedding weights (first BASE_VOCAB rows) into embed_base.
	// embed_ext is loaded separately below (from embed_ext.safetensors or seeded model).
	copy_pretrained_weight_partial(model.embed_base, &sf, "backbone.embeddings.weight", dev)!;

	// If the model file has 50283 rows (seeded/fused), also load ext rows into embed_ext
	if (try view = sf.get("backbone.embeddings.weight"))
	{
		usz total_rows = (usz)view.info.shape[0];
		if (total_rows >= TOTAL_VOCAB)
		{
			usz d_model = (usz)view.info.shape[1];
			usz row_bytes = d_model * model.embed_ext.dtype().size_of();
			usz ext_offset = BASE_VOCAB * row_bytes;
			usz ext_bytes = EXT_VOCAB * row_bytes;
			if (view.data.len >= ext_offset + ext_bytes)
			{
				char* ext_ptr = view.data.ptr + ext_offset;
				char[] ext_src = ext_ptr[:ext_bytes];
				char[] ext_dst = model.embed_ext.value_bytes();
				if (ext_dst.len == 0)
				{
					model.embed_ext.core().alloc_data();
					ext_dst = model.embed_ext.value_bytes();
				}
				dev.mem_transfer(ext_src, ext_dst, TransferDirection.HTOD);
				io::printfn("  Loaded ext embedding rows from base model (%d rows)", EXT_VOCAB);
			}
		}
	}
	loaded++;

	// Load final norm
	copy_pretrained_weight(model.norm_f, &sf, "backbone.norm_f.weight", dev)!;
	loaded++;

	// Load each layer
	for (usz i = 0; i < model.num_layers; i++)
	{
		load_lora_layer_base_weights(&model.layers[i], &sf, i, dev)!;
		loaded += 10;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sf.free();
	mem::free(file_data.ptr);
}

fn void? load_pretrained_weights_sharded(LoRAMambaModel* model, char[] path, DeviceReference dev)
{
	safetensors::ShardedSafeTensors* sharded = (safetensors::ShardedSafeTensors*)mem::malloc(safetensors::ShardedSafeTensors.sizeof);
	assert(sharded != null, "Failed to allocate ShardedSafeTensors");
	*sharded = safetensors::sharded_open(path)!;

	usz loaded = 0;

	// Load base embedding weights (first BASE_VOCAB rows) into embed_base
	copy_pretrained_weight_partial_sharded(model.embed_base, sharded, "backbone.embeddings.weight", dev)!;

	// If the model has 50283 rows (seeded/fused), also load ext rows into embed_ext
	if (try view = sharded.get("backbone.embeddings.weight"))
	{
		usz d_model = (usz)model.config.d_model;
		usz row_bytes = d_model * model.embed_ext.dtype().size_of();
		usz ext_offset = BASE_VOCAB * row_bytes;
		usz ext_bytes = EXT_VOCAB * row_bytes;
		if (view.data.len >= ext_offset + ext_bytes)
		{
			char* ext_ptr = view.data.ptr + ext_offset;
			char[] ext_src = ext_ptr[:ext_bytes];
			char[] ext_dst = model.embed_ext.value_bytes();
			if (ext_dst.len == 0)
			{
				model.embed_ext.core().alloc_data();
				ext_dst = model.embed_ext.value_bytes();
			}
			if (ext_dst.len == ext_bytes)
			{
				dev.mem_transfer(ext_src, ext_dst, TransferDirection.HTOD);
				io::printfn("  Loaded ext embedding rows from base model (%d rows)", EXT_VOCAB);
			}
		}
	}
	loaded++;

	copy_pretrained_weight_sharded(model.norm_f, sharded, "backbone.norm_f.weight", dev)!;
	loaded++;

	for (usz i = 0; i < model.num_layers; i++)
	{
		load_lora_layer_base_weights_sharded(&model.layers[i], sharded, i, dev)!;
		loaded += 10;
	}

	io::printfn("  Loaded %d weight tensors successfully", loaded);

	sharded.free();
	mem::free(sharded);
}

fn void? load_lora_layer_base_weights(
	LoRAMambaLayer* layer,
	safetensors::SafeTensorsFile* sf,
	usz layer_idx,
	DeviceReference dev,
)
{
	char[128] name_buf;
	usz prefix_len = format_pretrained_prefix(&name_buf, layer_idx);

	copy_layer_pretrained(layer.norm_weight, sf, &name_buf, prefix_len, "norm.weight", dev)!;
	copy_layer_pretrained(layer.in_proj, sf, &name_buf, prefix_len, "mixer.in_proj.weight", dev)!;
	copy_layer_pretrained(layer.out_proj, sf, &name_buf, prefix_len, "mixer.out_proj.weight", dev)!;
	copy_layer_pretrained(layer.conv_weight, sf, &name_buf, prefix_len, "mixer.conv1d.weight", dev)!;
	copy_layer_pretrained(layer.conv_bias, sf, &name_buf, prefix_len, "mixer.conv1d.bias", dev)!;
	copy_layer_pretrained(layer.a_log, sf, &name_buf, prefix_len, "mixer.A_log", dev)!;
	copy_layer_pretrained(layer.d_vec, sf, &name_buf, prefix_len, "mixer.D", dev)!;
	copy_layer_pretrained(layer.x_proj, sf, &name_buf, prefix_len, "mixer.x_proj.weight", dev)!;
	copy_layer_pretrained(layer.dt_proj, sf, &name_buf, prefix_len, "mixer.dt_proj.weight", dev)!;
	copy_layer_pretrained(layer.dt_proj_bias, sf, &name_buf, prefix_len, "mixer.dt_proj.bias", dev)!;
}

fn usz format_pretrained_prefix(char[128]* buf, usz layer_idx)
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

fn void? copy_layer_pretrained(
	Tensor dst,
	safetensors::SafeTensorsFile* sf,
	char[128]* buf,
	usz prefix_len,
	char[] suffix,
	DeviceReference dev,
)
{
	usz pos = prefix_len;
	for (usz i = 0; i < suffix.len; i++)
	{
		(*buf)[pos + i] = suffix[i];
	}
	char[] name = (*buf)[:pos + suffix.len];
	copy_pretrained_weight(dst, sf, name, dev)!;
}

fn void? copy_pretrained_weight(
	Tensor dst,
	safetensors::SafeTensorsFile* sf,
	char[] name,
	DeviceReference dev,
)
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

	dev.mem_transfer(view.data, dst_bytes[:expected], TransferDirection.HTOD);
}

/// Copy only the first `dst_rows` rows from a pretrained weight into dst.
/// The source tensor may have more rows (e.g. 50280) than dst (e.g. 50277).
fn void? copy_pretrained_weight_partial(
	Tensor dst,
	safetensors::SafeTensorsFile* sf,
	char[] name,
	DeviceReference dev,
)
{
	safetensors::TensorView view = sf.get(name)!;

	char[] dst_bytes = dst.value_bytes();
	if (dst_bytes.len == 0)
	{
		dst.core().alloc_data();
		dst_bytes = dst.value_bytes();
	}

	usz partial_bytes = dst.core().shape().size() * dst.core().dtype().size_of();
	assert(view.data.len >= partial_bytes, "Source weight too small for partial copy");

	dev.mem_transfer(view.data[:partial_bytes], dst_bytes[:partial_bytes], TransferDirection.HTOD);
}

fn void? copy_pretrained_weight_sharded(
	Tensor dst,
	safetensors::ShardedSafeTensors* st,
	char[] name,
	DeviceReference dev,
)
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

	dev.mem_transfer(view.data, dst_bytes[:expected], TransferDirection.HTOD);
}

/// Copy only the first `dst_rows` rows from a sharded pretrained weight into dst.
fn void? copy_pretrained_weight_partial_sharded(
	Tensor dst,
	safetensors::ShardedSafeTensors* st,
	char[] name,
	DeviceReference dev,
)
{
	safetensors::TensorView view = st.get(name)!;

	char[] dst_bytes = dst.value_bytes();
	if (dst_bytes.len == 0)
	{
		dst.core().alloc_data();
		dst_bytes = dst.value_bytes();
	}

	usz partial_bytes = dst.core().shape().size() * dst.core().dtype().size_of();
	assert(view.data.len >= partial_bytes, "Source weight too small for partial copy");

	dev.mem_transfer(view.data[:partial_bytes], dst_bytes[:partial_bytes], TransferDirection.HTOD);
}

fn void? load_lora_layer_base_weights_sharded(
	LoRAMambaLayer* layer,
	safetensors::ShardedSafeTensors* st,
	usz layer_idx,
	DeviceReference dev,
)
{
	char[128] name_buf;
	usz prefix_len = format_pretrained_prefix(&name_buf, layer_idx);

	copy_layer_pretrained_sharded(layer.norm_weight, st, &name_buf, prefix_len, "norm.weight", dev)!;
	copy_layer_pretrained_sharded(layer.in_proj, st, &name_buf, prefix_len, "mixer.in_proj.weight", dev)!;
	copy_layer_pretrained_sharded(layer.out_proj, st, &name_buf, prefix_len, "mixer.out_proj.weight", dev)!;
	copy_layer_pretrained_sharded(layer.conv_weight, st, &name_buf, prefix_len, "mixer.conv1d.weight", dev)!;
	copy_layer_pretrained_sharded(layer.conv_bias, st, &name_buf, prefix_len, "mixer.conv1d.bias", dev)!;
	copy_layer_pretrained_sharded(layer.a_log, st, &name_buf, prefix_len, "mixer.A_log", dev)!;
	copy_layer_pretrained_sharded(layer.d_vec, st, &name_buf, prefix_len, "mixer.D", dev)!;
	copy_layer_pretrained_sharded(layer.x_proj, st, &name_buf, prefix_len, "mixer.x_proj.weight", dev)!;
	copy_layer_pretrained_sharded(layer.dt_proj, st, &name_buf, prefix_len, "mixer.dt_proj.weight", dev)!;
	copy_layer_pretrained_sharded(layer.dt_proj_bias, st, &name_buf, prefix_len, "mixer.dt_proj.bias", dev)!;
}

fn void? copy_layer_pretrained_sharded(
	Tensor dst,
	safetensors::ShardedSafeTensors* st,
	char[128]* buf,
	usz prefix_len,
	char[] suffix,
	DeviceReference dev,
)
{
	usz pos = prefix_len;
	for (usz i = 0; i < suffix.len; i++)
	{
		(*buf)[pos + i] = suffix[i];
	}
	char[] name = (*buf)[:pos + suffix.len];
	copy_pretrained_weight_sharded(dst, st, name, dev)!;
}

// ============================================================================
// LoRA Checkpoint Loading
// ============================================================================

/// Load LoRA weights from a safetensors checkpoint into a LoRA model.
/// Expects tensor names matching save_lora_checkpoint format:
///   layer.N.{in_proj,out_proj,x_proj,dt_proj}_lora_{a,b}
fn void? load_lora_checkpoint(LoRAMambaModel* model, char[] path, DeviceReference dev)
{
	io::printfn("Loading checkpoint from: %s", (ZString)path.ptr);

	String p = string::tformat("%s", (ZString)path.ptr);
	char[] file_data = io::file::load(mem, p)!;
	safetensors::SafeTensorsFile sf = safetensors::deserialize(file_data)!;

	io::printfn("  Parsed %d tensors from checkpoint", sf.len());

	usz loaded = 0;

	// Always load embed_ext if present in checkpoint
	if (try view = sf.get("embed_ext"))
	{
		copy_pretrained_weight(model.embed_ext, &sf, "embed_ext", dev)!;
		io::printfn("  Loaded embed_ext (%d rows)", EXT_VOCAB);
		loaded += 1;
	}
	else if (try view = sf.get("embedding"))
	{
		copy_pretrained_weight(model.embed_ext, &sf, "embedding", dev)!;
		io::printfn("  Loaded embed_ext from legacy 'embedding' key (%d rows)", EXT_VOCAB);
		loaded += 1;
	}

	for (usz i = 0; i < model.num_layers; i++)
	{
		LoRAMambaLayer* l = &model.layers[i];
		char[128] name_buf;

		if (l.has_in_proj_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.in_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_pretrained_weight(l.in_proj_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.in_proj_lora_b", i)!!;
				copy_pretrained_weight(l.in_proj_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_out_proj_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.out_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_pretrained_weight(l.out_proj_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.out_proj_lora_b", i)!!;
				copy_pretrained_weight(l.out_proj_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_x_proj_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.x_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_pretrained_weight(l.x_proj_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.x_proj_lora_b", i)!!;
				copy_pretrained_weight(l.x_proj_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
		if (l.has_dt_proj_lora)
		{
			char[] na = io::bprintf(&name_buf, "layer.%d.dt_proj_lora_a", i)!!;
			if (try v = sf.get(na))
			{
				copy_pretrained_weight(l.dt_proj_lora.lora_a, &sf, na, dev)!;
				char[] nb = io::bprintf(&name_buf, "layer.%d.dt_proj_lora_b", i)!!;
				copy_pretrained_weight(l.dt_proj_lora.lora_b, &sf, nb, dev)!;
				loaded += 2;
			}
		}
	}

	io::printfn("  Loaded %d tensors", loaded);

	sf.free();
	mem::free(file_data.ptr);
}

// ============================================================================
// CUDA Loss Reading
// ============================================================================

/// Read scalar loss value from a tensor on GPU.
fn float read_loss_cuda(Tensor loss_t, DeviceReference dev)
{
	float[1] host_loss;
	char[] host_buf = ((char*)&host_loss)[:float.sizeof];
	char[] gpu_buf = loss_t.value_bytes()[:float.sizeof];
	dev.mem_transfer(gpu_buf, host_buf, TransferDirection.DTOH);
	return host_loss[0];
}

// ============================================================================
// Model Config Presets
// ============================================================================

fn MambaConfig mamba_130m()
{
	return {
		.d_model = 768,
		.d_state = 16,
		.d_conv = 4,
		.expand = 2,
		.dt_rank = 48,
		.vocab_size = 50280,
		.n_layers = 24,
		.eps = 1e-5,
		.batch_size = 1,
		.seq_len = 511,
		.dtype = Datatype.F32,
	};
}

fn MambaConfig mamba_370m()
{
	return {
		.d_model = 1024,
		.d_state = 16,
		.d_conv = 4,
		.expand = 2,
		.dt_rank = 64,
		.vocab_size = 50280,
		.n_layers = 48,
		.eps = 1e-5,
		.batch_size = 1,
		.seq_len = 511,
		.dtype = Datatype.F32,
	};
}

fn MambaConfig mamba_790m()
{
	return {
		.d_model = 1536,
		.d_state = 16,
		.d_conv = 4,
		.expand = 2,
		.dt_rank = 96,
		.vocab_size = 50280,
		.n_layers = 48,
		.eps = 1e-5,
		.batch_size = 1,
		.seq_len = 511,
		.dtype = Datatype.F32,
	};
}

fn MambaConfig mamba_2_8b()
{
	return {
		.d_model = 2560,
		.d_state = 16,
		.d_conv = 4,
		.expand = 2,
		.dt_rank = 160,
		.vocab_size = 50280,
		.n_layers = 64,
		.eps = 1e-5,
		.batch_size = 1,
		.seq_len = 511,
		.dtype = Datatype.F32,
	};
}
