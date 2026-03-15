//! Mamba Overfit Test — Confidence Building
//!
//! Trains a tiny Mamba model on a single fixed string to verify
//! the full pipeline: forward → loss → backward → optimizer step.
//!
//! Success: loss decreases from ~4.85 (random over 128-class vocab)
//! toward < 1.0 over 20 SGD steps.
//!
//! Architecture (1 layer, d_model=32):
//!   tokens → Embedding → MambaBlock → RMSNorm → LM Head → logits
//!   → softmax_cross_entropy → backward → SGD

module metaphor::examples::mamba_overfit;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::device::common::operations;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::rtti;
import std::math;
import std::io;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HostDevice = host_device::HostDevice;
alias Datatype = rtti::Datatype;
alias SliceRange = operations::SliceRange;
alias Optimizer = optimizer::Optimizer;

// ============================================================================
// Tiny Config — minimal for overfitting
// ============================================================================

const ulong D_MODEL = 32;
const ulong D_STATE = 4;
const ulong D_CONV  = 4;
const ulong EXPAND  = 2;
const ulong D_INNER = D_MODEL * EXPAND;  // 64
const ulong DT_RANK = 2;
const ulong VOCAB_SIZE = 128;  // ASCII
const ulong SEQ_LEN = 7;      // "hello!!" → predict "ello!!!\n"
const ulong BATCH_SIZE = 1;
const float EPS = 1e-5;

const int NUM_STEPS = 200;
const float LEARNING_RATE = 0.003;

// ============================================================================
// Mamba Layer (no LoRA — all weights trainable)
// ============================================================================

struct MambaLayer
{
	Tensor norm_weight;   // [D_MODEL]
	Tensor in_proj;       // [D_INNER*2, D_MODEL]
	Tensor conv_weight;   // [D_INNER, 1, D_CONV]
	Tensor conv_bias;     // [D_INNER]
	Tensor out_proj;      // [D_MODEL, D_INNER]
	Tensor a_log;         // [D_INNER, D_STATE]
	Tensor d_vec;         // [D_INNER]
	Tensor x_proj;        // [DT_RANK + D_STATE*2, D_INNER]
	Tensor dt_proj;       // [D_INNER, DT_RANK]
	Tensor dt_proj_bias;  // [D_INNER]
}

fn MambaLayer? init_layer(DeviceReference dev)
{
	MambaLayer layer;

	// Norm: ones
	ulong[1] d_model_shape = { D_MODEL };
	layer.norm_weight = tensor::ones(Datatype.F32, dev, &d_model_shape)!!.enable_grad();

	// InProj: Kaiming init
	layer.in_proj = tensor::random(Datatype.F32, dev, { D_INNER * 2, D_MODEL },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)D_MODEL })!!.enable_grad();

	// Conv: Kaiming init
	layer.conv_weight = tensor::random(Datatype.F32, dev, { D_INNER, 1, D_CONV },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)D_CONV })!!.enable_grad();

	// Conv bias: zeros
	ulong[1] d_inner_shape = { D_INNER };
	layer.conv_bias = tensor::zeros(Datatype.F32, dev, &d_inner_shape)!!.enable_grad();

	// OutProj: Kaiming
	layer.out_proj = tensor::random(Datatype.F32, dev, { D_MODEL, D_INNER },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)D_INNER })!!.enable_grad();

	// A_log: initialized to log(1..d_state) pattern, negative for stability
	layer.a_log = tensor::zeros(Datatype.F32, dev, { D_INNER, D_STATE })!!.enable_grad();
	// Fill with log(1), log(2), ..., log(d_state) per row
	{
		char[] bytes = layer.a_log.value_bytes();
		float* data = (float*)bytes.ptr;
		for (ulong e = 0; e < D_INNER; e++)
		{
			for (ulong s = 0; s < D_STATE; s++)
			{
				data[e * D_STATE + s] = (float)math::ln((double)(s + 1));
			}
		}
	}

	// D: ones (skip connection)
	layer.d_vec = tensor::ones(Datatype.F32, dev, &d_inner_shape)!!.enable_grad();

	// x_proj: Kaiming
	layer.x_proj = tensor::random(Datatype.F32, dev, { DT_RANK + D_STATE * 2, D_INNER },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)D_INNER })!!.enable_grad();

	// dt_proj: Kaiming
	layer.dt_proj = tensor::random(Datatype.F32, dev, { D_INNER, DT_RANK },
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)DT_RANK })!!.enable_grad();

	// dt_proj_bias: softplus_inverse(0.001) for proper timescale
	float dt_init = 0.001;
	float dt_bias_val = (float)math::ln(math::exp((double)dt_init) - 1.0);
	layer.dt_proj_bias = tensor::constant(Datatype.F32, dev, &d_inner_shape, dt_bias_val)!!.enable_grad();

	return layer;
}

/// Collect all trainable parameters from a layer.
fn void collect_layer_params(MambaLayer* layer, Tensor* params, usz* idx)
{
	params[*idx] = layer.norm_weight; (*idx)++;
	params[*idx] = layer.in_proj;    (*idx)++;
	params[*idx] = layer.conv_weight; (*idx)++;
	params[*idx] = layer.conv_bias;  (*idx)++;
	params[*idx] = layer.out_proj;   (*idx)++;
	params[*idx] = layer.a_log;      (*idx)++;
	params[*idx] = layer.d_vec;      (*idx)++;
	params[*idx] = layer.x_proj;     (*idx)++;
	params[*idx] = layer.dt_proj;    (*idx)++;
	params[*idx] = layer.dt_proj_bias; (*idx)++;
}

// ============================================================================
// Causal Conv1D (from mamba.c3)
// ============================================================================

fn Tensor? causal_conv1d(Tensor x, Tensor weight, Tensor bias)
{
	return functional::causal_conv1d(x, weight, bias);
}

// ============================================================================
// Parallel Scan (simplified from mamba.c3)
// ============================================================================

fn Tensor? parallel_ssm_scan(Tensor a_bar, Tensor b_bar_x, DeviceReference dev)
{
	ulong seq_len = a_bar.core().shape().get(1);
	if (seq_len <= 1) return b_bar_x;

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

	SliceRange[4] left_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = seq_len - stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	};
	Tensor left_part = a.slice(&left_ranges)!!;

	SliceRange[4] pad_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	};
	ulong[1] one_shape = { 1 };
	Tensor zero_t = tensor::constant(Datatype.F32, dev, &one_shape, 0.0)!!;
	Tensor one_t = tensor::constant(Datatype.F32, dev, &one_shape, 1.0)!!;
	Tensor identity_pad = a.slice(&pad_ranges)!!.mul(zero_t)!!.add(one_t)!!;

	Tensor[2] cat_inputs = { identity_pad, left_part };
	return tensor::cat(&cat_inputs, 1);
}

fn Tensor? shift_and_pad_zeros(Tensor h, ulong stride, ulong seq_len, DeviceReference dev)
{
	if (stride >= seq_len)
	{
		ulong[1] z_shape = { 1 };
		Tensor zero_t = tensor::constant(Datatype.F32, dev, &z_shape, 0.0)!!;
		return h.mul(zero_t);
	}

	SliceRange[4] left_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = seq_len - stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	};
	Tensor left_part = h.slice(&left_ranges)!!;

	SliceRange[4] pad_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = stride, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
	};
	Tensor zero_t = tensor::constant(Datatype.F32, dev, { 1 }, 0.0)!!;
	Tensor zero_pad = h.slice(&pad_ranges)!!.mul(zero_t)!!;

	Tensor[2] cat_inputs = { zero_pad, left_part };
	return tensor::cat(&cat_inputs, 1);
}

// ============================================================================
// Mamba Layer Forward
// ============================================================================

fn Tensor? layer_forward(MambaLayer* layer, Tensor x, DeviceReference dev)
{
	// 1. Input projection: [B,L,D] -> [B,L,2*E]
	Tensor xz = functional::linear(x, layer.in_proj)!!;

	// 2. Split into x_branch and z_branch
	SliceRange[3] x_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = D_INNER, .step = 1 },
	};
	Tensor x_branch = xz.slice(&x_ranges)!!;

	SliceRange[3] z_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = D_INNER, .end = D_INNER * 2, .step = 1 },
	};
	Tensor z_branch = xz.slice(&z_ranges)!!;

	// 3. Conv: transpose [B,L,E] -> [B,E,L]
	Tensor x_transposed = x_branch.transpose(1, 2)!!;

	// 4. Causal conv1d
	Tensor x_conv = causal_conv1d(x_transposed, layer.conv_weight, layer.conv_bias)!!;

	// 5. Transpose back + SiLU
	Tensor x_back = x_conv.transpose(1, 2)!!;
	Tensor x_silu = x_back.silu()!!;

	// 6. Project to dt, B, C
	Tensor x_dbc = functional::linear(x_silu, layer.x_proj)!!;

	SliceRange[3] dt_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = DT_RANK, .step = 1 },
	};
	Tensor dt_input = x_dbc.slice(&dt_ranges)!!;

	SliceRange[3] b_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = DT_RANK, .end = DT_RANK + D_STATE, .step = 1 },
	};
	Tensor b_proj = x_dbc.slice(&b_ranges)!!;

	SliceRange[3] c_ranges = {
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = 0, .end = 0, .step = 1 },
		{ .start = DT_RANK + D_STATE, .end = DT_RANK + D_STATE * 2, .step = 1 },
	};
	Tensor c_proj = x_dbc.slice(&c_ranges)!!;

	// 7. dt = softplus(clamp(dt_proj(dt_input) + bias, -20, 20))
	Tensor dt_proj_out = functional::linear(dt_input, layer.dt_proj)!!
		.einsum(layer.dt_proj_bias, "ble,e->ble")!!;
	Tensor lower = dt_proj_out.scalar_like(-20.0)!!;
	Tensor upper = dt_proj_out.scalar_like(20.0)!!;
	Tensor dt = dt_proj_out.clamp(&lower, &upper)!!.softplus()!!;

	// 8. Discretize
	Tensor a_continuous = layer.a_log.exp()!!.neg()!!;
	Tensor dt_a = dt.einsum(a_continuous, "ble,ed->bled")!!;
	Tensor a_bar = dt_a.exp()!!;
	Tensor b_bar = dt.einsum(b_proj, "ble,bld->bled")!!;

	// 9. Parallel scan
	Tensor b_bar_x = b_bar.einsum(x_silu, "bled,ble->bled")!!;
	Tensor h = parallel_ssm_scan(a_bar, b_bar_x, dev)!!;

	// 10. Output: y = C * h + D * x
	Tensor y_ssm = c_proj.einsum(h, "bld,bled->ble")!!;
	Tensor d_x = x_silu.einsum(layer.d_vec, "ble,e->ble")!!;
	Tensor y = y_ssm.add(d_x)!!;

	// 11. Gate with z_branch
	Tensor z_silu = z_branch.silu()!!;
	Tensor gated = y.mul(z_silu)!!;

	// 12. Output projection
	return functional::linear(gated, layer.out_proj);
}

// ============================================================================
// Main — Training Loop
// ============================================================================

fn void main()
{
	HostDevice cpu;
	cpu.init();

	DeviceReference dev = cpu.reference();

	graph::init();

	io::printfn("\n=== Mamba Overfit Confidence Test (CUDA) ===\n");
	io::printfn("Config: d_model=%d, d_inner=%d, vocab=%d, seq_len=%d",
		D_MODEL, D_INNER, VOCAB_SIZE, SEQ_LEN);
	io::printfn("Training: %d steps, lr=%.4f, Adam\n", NUM_STEPS, LEARNING_RATE);

	// --- Initialize model ---
	MambaLayer layer = init_layer(dev)!!;

	// Embedding: [VOCAB_SIZE, D_MODEL]
	ulong[2] embed_shape = { VOCAB_SIZE, D_MODEL };
	Tensor embedding = tensor::random(Datatype.F32, dev, &embed_shape,
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)VOCAB_SIZE })!!.enable_grad();

	// Final norm: [D_MODEL]
	Tensor norm_f = tensor::ones(Datatype.F32, dev, { D_MODEL })!!.enable_grad();

	// LM head: [VOCAB_SIZE, D_MODEL]
	Tensor lm_head = tensor::random(Datatype.F32, dev, &embed_shape,
		device_reference::RandType.NORMAL,
		{ .mode = device_reference::ScaleMode.KAIMING, .value = (float)D_MODEL })!!.enable_grad();

	// --- Prepare input/target data ---
	// Input:  "hello!!" (7 chars)
	// Target: "ello!!!" (next-char prediction, shifted by 1)
	char[] input_str  = "hello!!";
	char[] target_str = "ello!!!";

	// Create input token indices as U64 tensor [1, SEQ_LEN]
	Tensor input_tokens = tensor::empty(Datatype.U64, dev, { BATCH_SIZE, SEQ_LEN })!!;
	{
		char[] token_bytes = input_tokens.value_bytes();
		ulong* token_data = (ulong*)token_bytes.ptr;
		for (usz i = 0; i < SEQ_LEN; i++)
		{
			token_data[i] = (ulong)input_str[i];
		}
	}

	// Create one-hot target tensor [SEQ_LEN, VOCAB_SIZE] for loss
	// (2D because softmax_cross_entropy expects [batch, classes])
	Tensor target_onehot = tensor::zeros(Datatype.F32, dev, { SEQ_LEN, VOCAB_SIZE })!!;
	{
		char[] target_bytes = target_onehot.value_bytes();
		float* target_data = (float*)target_bytes.ptr;
		for (usz i = 0; i < SEQ_LEN; i++)
		{
			usz char_idx = (usz)target_str[i];
			target_data[i * VOCAB_SIZE + char_idx] = 1.0;
		}
	}

	// --- Build forward graph ---
	// Embedding lookup: [1, SEQ_LEN] -> [1, SEQ_LEN, D_MODEL]
	Tensor x = embedding.gather(0, input_tokens)!!;

	// Mamba layer + residual
	Tensor normed = functional::rms_norm(x, layer.norm_weight, EPS)!!;
	Tensor layer_out = layer_forward(&layer, normed, dev)!!;
	x = x.add(layer_out)!!;

	// Final norm
	x = functional::rms_norm(x, norm_f, EPS)!!;

	// LM head: [1, SEQ_LEN, D_MODEL] -> [1, SEQ_LEN, VOCAB_SIZE]
	Tensor logits_3d = functional::linear(x, lm_head)!!;

	// Squeeze to [SEQ_LEN, VOCAB_SIZE] for loss
	Tensor logits = logits_3d.reduce("bsv->sv")!!;

	// Loss
	Tensor loss_t = loss::softmax_cross_entropy(logits, target_onehot)!!;

	// Mark logits and loss as stable for collection
	logits = logits.stable();
	loss_t = loss_t.stable();

	// --- Build backward graph (once, before the loop) ---
	loss_t.backward();

	// --- Set up optimizer ---
	// Collect all trainable parameters
	Tensor[13] params;
	usz param_idx = 0;
	collect_layer_params(&layer, &params, &param_idx);
	params[param_idx] = embedding; param_idx++;
	params[param_idx] = norm_f;    param_idx++;
	params[param_idx] = lm_head;   param_idx++;

	Optimizer opt = optimizer::init({
		.lr = LEARNING_RATE,
		.scheduler = optimizer::scheduler_warmup_cosine(10, NUM_STEPS, 1e-5),
	});
	opt.adam(params[:param_idx], { ...optimizer::DEFAULT_ADAM_OPTIONS, .clip_grad = 1.0 });

	// --- Training loop ---
	io::printfn("Step  Loss      Used(KB)  Peak(KB)");
	io::printfn("----  --------  --------  --------");

	graph::Graph* g = graph::get();

	for (int step = 0; step < NUM_STEPS; step++)
	{
		// Invalidate cached computations so collect() re-executes
		g.advance_generation();

		// Forward: collect loss
		loss_t.collect()!!;

		// Read loss value
		char[] loss_bytes = loss_t.value_bytes();
		float loss_val = *((float*)loss_bytes.ptr);

		// Memory before optimizer step
		usz used_before = dev.mem_used() / 1024;
		usz peak_before = dev.mem_peak() / 1024;
		io::printfn("%4d  %.6f  %8d  %8d  (pre-opt)", step, loss_val, used_before, peak_before);

		// Optimizer step (collects gradients + updates params)
		opt.step();

		// Memory after optimizer step
		usz used_after = dev.mem_used() / 1024;
		usz peak_after = dev.mem_peak() / 1024;
		io::printfn("                %8d  %8d  (post-opt)", used_after, peak_after);
	}

	io::printfn("\n=== Complete ===");
}
