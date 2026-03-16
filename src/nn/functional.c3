//! Neural Network Functional Operations
//!
//! Provides softmax, normalization, and linear layer operations.
//! Uses runtime dimension patterns instead of compile-time enums.
//!

module metaphor::nn::functional;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::tensor_ops;
import metaphor::tensor::shape;
import metaphor::device::device_reference;
import metaphor::device::common::index_expr;
import metaphor::device::common::operations;
import metaphor::device::common::op_payload;
import metaphor::graph;
import metaphor::rtti;

// Index expression helpers

const char[8] AXES = { 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p' };

struct ExprBuf
{
	char[20] buf;
	usz len;
}

fn void ExprBuf.put(&self, char c) { self.buf[self.len++] = c; }
fn void ExprBuf.arrow(&self) { self.put('-'); self.put('>'); }
fn void ExprBuf.axes(&self, usz start, usz count) { for (usz i = start; i < start + count; i++) { self.put(AXES[i]); } }
fn void ExprBuf.axes_skip(&self, usz count, usz skip) { for (usz i = 0; i < count; i++) { if (i != skip) self.put(AXES[i]); } }
fn char[] ExprBuf.slice(&self) { return self.buf[:self.len]; }

// Reduce/Expand Pattern Generation

/// Pre-computed reduce and expand pattern strings for a given rank and target dimension.
/// Maximum supported rank is 8.
struct ReduceExpandPatterns
{
	char[19] reduce_buf;
	usz reduce_len;
	char[19] expand_buf;
	usz expand_len;
	usz target_dim;
}

fn char[] ReduceExpandPatterns.reduce_pat(&self)
{
	return self.reduce_buf[:self.reduce_len];
}

fn char[] ReduceExpandPatterns.expand_pat(&self)
{
	return self.expand_buf[:self.expand_len];
}

/// Generate patterns for reducing over a specific dimension.
/// e.g., for rank=3, dim=1: reduce="ijk->ik", expand="ik->ijk"
fn ReduceExpandPatterns? patterns_for_dim(usz rank, usz dim)
{
	if (rank == 0 || rank > 8 || dim >= rank) return tensor::INCOMPATIBLE_SHAPE~;

	ReduceExpandPatterns result;
	result.target_dim = dim;

	// Reduce: "ijk->ik" (all dims → skip reduced dim)
	ExprBuf r;
	r.axes(0, rank);
	r.arrow();
	r.axes_skip(rank, dim);
	result.reduce_len = r.len;
	for (usz i = 0; i < r.len; i++) { result.reduce_buf[i] = r.buf[i]; }

	// Expand: "ik->ijk" (skip reduced dim → all dims)
	ExprBuf e;
	e.axes_skip(rank, dim);
	e.arrow();
	e.axes(0, rank);
	result.expand_len = e.len;
	for (usz i = 0; i < e.len; i++) { result.expand_buf[i] = e.buf[i]; }

	return result;
}

// Internal helpers for reduce/expand with options

/// Reduce with an explicit op (e.g., MAX, MIN).
fn tensor::Tensor? reduce_with_op(tensor::Tensor input, char[] expr, operations::BinaryOp op)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	return tensor_ops::reduce(input, &parsed, { .op = op });
}

/// Reduce with default (ADD) op.
fn tensor::Tensor? reduce_sum(tensor::Tensor input, char[] expr)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	return tensor_ops::reduce(input, &parsed);
}

/// Expand with sizes.
fn tensor::Tensor? expand_dim(tensor::Tensor input, char[] expr, ulong expand_size)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	ulong[1] sizes = { expand_size };
	return tensor_ops::expand(input, &parsed, { .sizes = &sizes });
}

// Softmax / Log-Softmax

/// Numerically stable softmax along the specified dimension.
///   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
fn tensor::Tensor? softmax(tensor::Tensor input, usz dim)
{
	return softmax_op(input, dim, false);
}

/// Numerically stable log-softmax along the specified dimension.
///   log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
fn tensor::Tensor? log_softmax(tensor::Tensor input, usz dim)
{
	return softmax_op(input, dim, true);
}

/// Emit a dedicated SOFTMAX boundary op (forward or backward handled by
/// optimized host/CUDA kernels rather than JIT-compiled element-wise code).
fn tensor::Tensor? softmax_op(tensor::Tensor input, usz dim, bool is_log)
{
	graph::Graph* g = input.graph();
	device_reference::DeviceReference dev = input.device();

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		input.shape().slice(), g, input.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_softmax(
		is_log, false, (int)dim);

	g.register_op(payload, { input }, result);

	return result;
}

// Normalization

/// L2 normalization: y = x / (||x||_2 + eps)
fn tensor::Tensor? l2_normalize(tensor::Tensor input, usz dim, double epsilon)
{
	shape::Shape* s = input.shape();
	ReduceExpandPatterns patterns = patterns_for_dim(s.len, dim)!;
	ulong expand_size = s.get(dim);

	tensor::Tensor sqr = input.sqr()!;
	tensor::Tensor sum_sqr = reduce_sum(sqr, patterns.reduce_pat())!;
	tensor::Tensor norm = sum_sqr.sqrt()!;
	tensor::Tensor eps = input.scalar_like(epsilon)!;
	tensor::Tensor norm_safe = norm.add(eps)!;
	tensor::Tensor norm_exp = expand_dim(norm_safe, patterns.expand_pat(), expand_size)!;
	return input.div(norm_exp);
}

/// L1 normalization: y = x / (||x||_1 + eps)
fn tensor::Tensor? l1_normalize(tensor::Tensor input, usz dim, double epsilon)
{
	shape::Shape* s = input.shape();
	ReduceExpandPatterns patterns = patterns_for_dim(s.len, dim)!;
	ulong expand_size = s.get(dim);

	tensor::Tensor abs_x = input.abs()!;
	tensor::Tensor norm = reduce_sum(abs_x, patterns.reduce_pat())!;
	tensor::Tensor eps = input.scalar_like(epsilon)!;
	tensor::Tensor norm_safe = norm.add(eps)!;
	tensor::Tensor norm_exp = expand_dim(norm_safe, patterns.expand_pat(), expand_size)!;
	return input.div(norm_exp);
}

/// Z-score normalization: y = (x - mean) / (std + eps)
fn tensor::Tensor? zscore(tensor::Tensor input, usz dim, double epsilon)
{
	shape::Shape* s = input.shape();
	ReduceExpandPatterns patterns = patterns_for_dim(s.len, dim)!;
	ulong expand_size = s.get(dim);

	// mean = sum(x) / n
	tensor::Tensor sum_val = reduce_sum(input, patterns.reduce_pat())!;
	tensor::Tensor n = input.scalar_like((double)expand_size)!;
	tensor::Tensor mean_val = sum_val.div(n)!;
	tensor::Tensor mean_exp = expand_dim(mean_val, patterns.expand_pat(), expand_size)!;

	// centered = x - mean
	tensor::Tensor centered = input.sub(mean_exp)!;

	// variance = sum((x - mean)^2) / n
	tensor::Tensor sqr_centered = centered.sqr()!;
	tensor::Tensor sum_sqr = reduce_sum(sqr_centered, patterns.reduce_pat())!;
	tensor::Tensor variance = sum_sqr.div(n)!;

	// std = sqrt(variance + eps)
	tensor::Tensor eps = input.scalar_like(epsilon)!;
	tensor::Tensor var_safe = variance.add(eps)!;
	tensor::Tensor std_val = var_safe.sqrt()!;
	tensor::Tensor std_exp = expand_dim(std_val, patterns.expand_pat(), expand_size)!;

	return centered.div(std_exp);
}

/// RMS normalization: y = x / (rms(x) + eps)
/// where rms(x) = sqrt(mean(x^2))
fn tensor::Tensor? rms_normalize(tensor::Tensor input, usz dim, double epsilon)
{
	shape::Shape* s = input.shape();
	ReduceExpandPatterns patterns = patterns_for_dim(s.len, dim)!;
	ulong expand_size = s.get(dim);

	tensor::Tensor sqr = input.sqr()!;
	tensor::Tensor sum_sqr = reduce_sum(sqr, patterns.reduce_pat())!;
	tensor::Tensor n = input.scalar_like((double)expand_size)!;
	tensor::Tensor mean_sqr = sum_sqr.div(n)!;

	tensor::Tensor eps = input.scalar_like(epsilon)!;
	tensor::Tensor mean_sqr_safe = mean_sqr.add(eps)!;
	tensor::Tensor rms_val = mean_sqr_safe.sqrt()!;
	tensor::Tensor rms_exp = expand_dim(rms_val, patterns.expand_pat(), expand_size)!;

	return input.div(rms_exp);
}

/// Min-max scaling: y = (x - min) / (max - min + eps)
fn tensor::Tensor? min_max_scale(tensor::Tensor input, usz dim, double epsilon)
{
	shape::Shape* s = input.shape();
	ReduceExpandPatterns patterns = patterns_for_dim(s.len, dim)!;
	ulong expand_size = s.get(dim);

	tensor::Tensor min_val = reduce_with_op(input, patterns.reduce_pat(), operations::BinaryOp.MIN)!;
	tensor::Tensor max_val = reduce_with_op(input, patterns.reduce_pat(), operations::BinaryOp.MAX)!;

	tensor::Tensor range = max_val.sub(min_val)!;
	tensor::Tensor eps = input.scalar_like(epsilon)!;
	tensor::Tensor range_safe = range.add(eps)!;

	tensor::Tensor min_exp = expand_dim(min_val, patterns.expand_pat(), expand_size)!;
	tensor::Tensor range_exp = expand_dim(range_safe, patterns.expand_pat(), expand_size)!;

	tensor::Tensor shifted = input.sub(min_exp)!;
	return shifted.div(range_exp);
}

// RMS Norm (dedicated kernel path)

/// RMS normalization with learnable weight parameter.
/// Uses dedicated kernels with f32 internal accumulation for stability.
///   y = (x / rms) * weight, where rms = sqrt(mean(x^2) + eps)
fn tensor::Tensor? rms_norm(tensor::Tensor x, tensor::Tensor weight, double eps)
{
	graph::Graph* g = x.graph();
	device_reference::DeviceReference dev = x.device();

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		x.shape().slice(), g, x.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_rms_norm(
		eps, operations::RmsOpType.FORWARD);

	g.register_op(payload, { x, weight }, result);

	return result;
}

// Selective Scan (fused SSM recurrence)

/// Fused selective scan: h[t] = a_bar[t] * h[t-1] + b_bar_x[t]
/// Inputs must be [batch, seq_len, d_inner, d_state].
/// Returns all hidden states h with the same shape.
fn tensor::Tensor? selective_scan(tensor::Tensor a_bar, tensor::Tensor b_bar_x)
{
	graph::Graph* g = a_bar.graph();
	device_reference::DeviceReference dev = a_bar.device();

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		a_bar.shape().slice(), g, a_bar.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_selective_scan(
		operations::SelectiveScanOpType.FORWARD);

	g.register_op(payload, { a_bar, b_bar_x }, result);

	return result;
}

// Causal Conv1D (fused depthwise kernel)

/// Fused depthwise causal 1D convolution.
/// x: [B, E, L], weight: [E, 1, K], bias: [E]
/// Returns: [B, E, L]
fn tensor::Tensor? causal_conv1d(tensor::Tensor x, tensor::Tensor weight, tensor::Tensor bias)
{
	graph::Graph* g = x.graph();
	device_reference::DeviceReference dev = x.device();

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		x.shape().slice(), g, x.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_causal_conv1d(
		operations::CausalConv1dOpType.FORWARD);

	g.register_op(payload, { x, weight, bias }, result);

	return result;
}

// Linear Layer

/// Linear layer: y = x @ weight^T + bias
///   x:      [..., in_features]
///   weight: [out_features, in_features]
///   bias:   [out_features] (optional, omit or pass NULL_TENSOR if no bias)
fn tensor::Tensor? linear(tensor::Tensor input, tensor::Tensor weight, tensor::Tensor bias = tensor::NULL_TENSOR)
{
	// Linear: y = x @ W^T
	// For rank 2 input [batch, in]:  "ij,kj->ik" (GEMM: C = A @ B^T)
	// For rank 1 input [in]:         "i,ki->k"   (GEMV-like)
	usz rank = input.shape().len;
	ExprBuf eb;

	if (rank == 1)
	{
		// "i,ji->j"
		eb.put(AXES[0]); eb.put(',');
		eb.put(AXES[1]); eb.put(AXES[0]);
		eb.arrow();
		eb.put(AXES[1]);
	}
	else
	{
		// e.g. rank 2: "ij,kj->ik", rank 3: "ijk,lk->ijl"
		eb.axes(0, rank); eb.put(',');
		char out_axis = AXES[rank];
		eb.put(out_axis); eb.put(AXES[rank - 1]);
		eb.arrow();
		eb.axes(0, rank - 1); eb.put(out_axis);
	}

	index_expr::IndexExpr expr = index_expr::index_expr_parse(eb.slice())!;
	tensor::Tensor result = tensor_ops::einsum(input, weight, &expr)!;

	if (bias._core != null)
	{
		result = result.add(bias)!;
	}

	return result;
}

// Mean reduction

/// Mean reduction over a specified dimension.
///   mean_dim(x, 1) for [batch, features] -> mean over features
fn tensor::Tensor? mean_dim(tensor::Tensor input, usz dim)
{
	shape::Shape* s = input.shape();
	ReduceExpandPatterns patterns = patterns_for_dim(s.len, dim)!;
	ulong expand_size = s.get(dim);

	tensor::Tensor sum_val = reduce_sum(input, patterns.reduce_pat())!;
	tensor::Tensor n = input.scalar_like((double)expand_size)!;
	return sum_val.div(n);
}

/// Mean over all elements (scalar result).
fn tensor::Tensor? mean(tensor::Tensor input)
{
	usz rank = input.shape().len;
	if (rank == 0) return input;

	ExprBuf eb;
	eb.axes(0, rank);
	eb.arrow();

	tensor::Tensor sum_val = reduce_sum(input, eb.slice())!;

	ulong total = 1;
	for (usz i = 0; i < rank; i++)
	{
		total *= input.shape().get(i);
	}
	tensor::Tensor n = input.scalar_like((double)total)!;
	return sum_val.div(n);
}

/// Sum over all elements (scalar result).
fn tensor::Tensor? sum_all(tensor::Tensor input)
{
	usz rank = input.shape().len;
	if (rank == 0) return input;

	ExprBuf eb;
	eb.axes(0, rank);
	eb.arrow();

	return reduce_sum(input, eb.slice());
}

// Dropout (dedicated kernel path)

/// Inverted dropout: randomly zeros elements with probability `p` and
/// scales remaining elements by `1/(1-p)` to maintain expected values.
///
/// The forward op generates a random binary mask tensor (STABLE persistence)
/// and stores it as a second input.  Backward reads the same mask — no seed
/// threading required.
///
/// This operation is training-only — at inference time, callers should
/// skip this function entirely (pass input through unchanged).
fn tensor::Tensor? dropout(tensor::Tensor input, float p)
{
	graph::Graph* g = input.graph();
	device_reference::DeviceReference dev = input.device();

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		input.shape().slice(), g, input.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Mask tensor (U8, 0 or 1) — filled by the forward executor.
	// STABLE so it survives between forward and backward execution.
	tensor_core::TensorCore* mask_core = tensor_ops::setup_result_core(
		input.shape().slice(), g, U8, dev, null)!;
	mask_core.flags.persistence.promote(STABLE);

	tensor::Tensor mask = tensor::tensor_from_core(mask_core);

	op_payload::OpPayload payload = op_payload::op_payload_dropout(p);

	g.register_op(payload, { input, mask }, result);

	return result;
}
