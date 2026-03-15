//! Backward Pass Runtime Helpers
//!
//! Utility functions for building gradient computation graphs at runtime.
//! These create symbolic operations (not executing them) — used by
//! OpPayload.symbolic_backward to construct the backward graph.
//!

module metaphor::device::common::backward_helpers;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::tensor_ops;
import metaphor::tensor::shape;
import metaphor::graph;
import metaphor::device::device_reference;
import metaphor::device::common::op_payload;
import metaphor::device::common::index_expr;
import metaphor::device::common::operations;
import metaphor::rtti;
import metaphor::logging;

// Index Helpers

/// Copy indices from slice to fixed array.
fn void copy_indices(char[8]* dst, char* dst_len, char[] src)
{
	usz len = src.len;
	assert(len <= 8, "copy_indices: source exceeds max 8 indices");
	for (usz i = 0; i < len; i++)
	{
		(*dst)[i] = src[i];
	}
	*dst_len = (char)len;
}

// Runtime Map

/// Create a unary map operation at runtime.
fn tensor::Tensor? map_runtime(tensor::Tensor a, operations::UnaryOp op)
{
	graph::Graph* g = tensor_ops::resolve_graph({a}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a}, null)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		a.core().shape().slice(),
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_map(op);
	g.register_op(payload, {a}, result);

	return result;
}

// Element-wise Combine

/// Element-wise combine of two tensors (same shape, no index manipulation).
fn tensor::Tensor? combine_elemwise(tensor::Tensor a, tensor::Tensor b, operations::BinaryOp op)
{
	graph::Graph* g = tensor_ops::resolve_graph({a, b}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a, b}, null)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		a.core().shape().slice(),
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Build element-wise expression: "ij...,ij...->ij..." for rank N
	index_expr::IndexExpr expr;
	char rank = (char)a.core().shape().len;
	for (char i = 0; i < rank; i++)
	{
		char idx = (char)('i' + i);
		expr.input0[i] = idx;
		expr.input1[i] = idx;
		expr.output[i] = idx;
	}
	expr.input_lens[0] = rank;
	expr.input_lens[1] = rank;
	expr.output_len = rank;
	expr.num_inputs = 2;
	expr.has_ellipsis = false;

	op_payload::OpPayload payload = op_payload::op_payload_einsum(expr, op, operations::BinaryOp.ADD);
	g.register_op(payload, {a, b}, result);

	return result;
}

// Scalar Operations

/// Combine a tensor with a scalar using broadcasting.
fn tensor::Tensor? combine_with_scalar(
	tensor::Tensor a,
	tensor::Tensor scalar,
	operations::BinaryOp op)
{
	graph::Graph* g = tensor_ops::resolve_graph({a, scalar}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a, scalar}, null)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		a.core().shape().slice(),
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Build expression: input0 has full rank, input1 is scalar (no indices), output = same as input0
	index_expr::IndexExpr expr;
	char rank = (char)a.core().shape().len;
	for (char i = 0; i < rank; i++)
	{
		char idx = (char)('i' + i);
		expr.input0[i] = idx;
		expr.output[i] = idx;
	}
	expr.input_lens[0] = rank;
	expr.input_lens[1] = 0;  // scalar has no indices
	expr.output_len = rank;
	expr.num_inputs = 2;
	expr.has_ellipsis = false;

	op_payload::OpPayload payload = op_payload::op_payload_einsum(expr, op, operations::BinaryOp.ADD);
	g.register_op(payload, {a, scalar}, result);

	return result;
}

/// Scale tensor by scalar: result = a * scalar.
/// Requires a pre-created scalar tensor.
fn tensor::Tensor? scale_with_tensor(tensor::Tensor a, tensor::Tensor scalar_tensor)
{
	return combine_with_scalar(a, scalar_tensor, operations::BinaryOp.MUL);
}

/// Negate a tensor: result = -a (maps to NEG unary op).
fn tensor::Tensor? negate_runtime(tensor::Tensor a)
{
	return map_runtime(a, operations::UnaryOp.NEG);
}

// Zero-like, Select, Compare

/// Runtime select: result[i] = condition[i] ? positive[i] : negative[i].
fn tensor::Tensor? select_runtime(
	tensor::Tensor condition,
	tensor::Tensor positive,
	tensor::Tensor negative)
{
	graph::Graph* g = tensor_ops::resolve_graph({condition, positive, negative}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device(
		{condition, positive, negative}, null)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		positive.core().shape().slice(),
		g, positive.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_select();
	g.register_op(payload, {condition, positive, negative}, result);

	return result;
}

/// Runtime element-wise comparison (for max/min gradients).
fn tensor::Tensor? compare_runtime(
	tensor::Tensor a,
	tensor::Tensor b,
	operations::BinaryOp cmp_op)
{
	graph::Graph* g = tensor_ops::resolve_graph({a, b}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a, b}, null)!;

	// Build broadcast expression
	ulong[] a_shape = a.core().shape().slice();
	ulong[] b_shape = b.core().shape().slice();
	ulong[8] out_buf;
	ulong[] out_shape = compute_broadcast_shape_runtime(a_shape, b_shape, &out_buf)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		out_shape,
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	index_expr::IndexExpr expr = build_broadcast_expr_runtime(a_shape, b_shape, out_shape);

	op_payload::OpPayload payload = op_payload::op_payload_einsum(expr, cmp_op, operations::BinaryOp.ADD);
	g.register_op(payload, {a, b}, result);

	return result;
}

// Runtime Reduce / Expand

/// Runtime reduce (for gradients).
fn tensor::Tensor? reduce_runtime(
	tensor::Tensor a,
	index_expr::IndexExpr* expr,
	operations::BinaryOp op)
{
	logging::@debug(OP_PAYLOAD, "input_shape_len=%d in='%s' out='%s'",
		(int)a.core().shape().len,
		(ZString)expr.get_input(0).ptr, (ZString)expr.get_output().ptr);

	graph::Graph* g = tensor_ops::resolve_graph({a}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a}, null)!;

	shape::Shape output_shape = compute_reduce_shape_runtime(expr, a.core().shape())!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		output_shape.slice(),
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_reduce(*expr, op);
	g.register_op(payload, {a}, result);

	return result;
}

/// Runtime expand (for gradients).
fn tensor::Tensor? expand_runtime(
	tensor::Tensor a,
	index_expr::IndexExpr* expr,
	ulong[8]* new_sizes,
	char num_new)
{
	logging::@debug(OP_PAYLOAD, "input_shape_len=%d in='%s' out='%s' num_new=%d",
		(int)a.core().shape().len,
		(ZString)expr.get_input(0).ptr, (ZString)expr.get_output().ptr,
		(int)num_new);

	graph::Graph* g = tensor_ops::resolve_graph({a}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a}, null)!;

	shape::Shape output_shape = tensor_ops::compute_expand_shape(
		expr, a.core().shape(), (*new_sizes)[:num_new])!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		output_shape.slice(),
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	operations::ExpandPayload expand_payload;
	expand_payload.tag = operations::OpTag.EXPAND;
	expand_payload.expr = *expr;
	expand_payload.new_sizes = *new_sizes;
	expand_payload.num_new = num_new;

	op_payload::OpPayload payload = { .expand = expand_payload };
	g.register_op(payload, {a}, result);

	return result;
}

// Combine with Broadcast

/// Combine two tensors with broadcasting.
fn tensor::Tensor? combine_with_broadcast(
	tensor::Tensor a,
	tensor::Tensor b,
	operations::BinaryOp op)
{
	graph::Graph* g = tensor_ops::resolve_graph({a, b}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a, b}, null)!;

	ulong[] a_shape = a.core().shape().slice();
	ulong[] b_shape = b.core().shape().slice();
	ulong[8] out_buf;
	ulong[] out_shape = compute_broadcast_shape_runtime(a_shape, b_shape, &out_buf)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		out_shape,
		g, a.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);
	index_expr::IndexExpr expr = build_broadcast_expr_runtime(a_shape, b_shape, out_shape);

	op_payload::OpPayload payload = op_payload::op_payload_einsum(expr, op, operations::BinaryOp.ADD);
	g.register_op(payload, {a, b}, result);

	return result;
}

// Combine Runtime (with explicit indices)

/// Runtime combine with explicit index slices.
fn tensor::Tensor? combine_runtime(
	tensor::Tensor a,
	tensor::Tensor b,
	shape::Shape* target_shape,
	char[] a_indices,
	char[] b_indices,
	char[] out_indices)
{
	// Ensure both operands have matching dtypes for the contraction.
	// Mixed dtypes (e.g., F32 grad × BF16 weight in backward) cause the
	// device contraction to misinterpret one operand's bytes, producing NaN.
	tensor::Tensor ca = a;
	tensor::Tensor cb = b;
	if (a.dtype() != b.dtype())
	{
		// Cast the higher-precision operand to match the lower-precision one.
		// Backward contractions should run in the parameter dtype.
		ca = tensor_ops::cast(a, b.dtype())!;
	}

	graph::Graph* g = tensor_ops::resolve_graph({ca, cb}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({ca, cb}, null)!;

	index_expr::IndexExpr expr;
	expr.has_ellipsis = false;
	expr.num_inputs = 2;

	for (usz i = 0; i < a_indices.len && i < 8; i++)
	{
		expr.input0[i] = a_indices[i];
	}
	expr.input_lens[0] = (char)a_indices.len;

	for (usz i = 0; i < b_indices.len && i < 8; i++)
	{
		expr.input1[i] = b_indices[i];
	}
	expr.input_lens[1] = (char)b_indices.len;

	for (usz i = 0; i < out_indices.len && i < 8; i++)
	{
		expr.output[i] = out_indices[i];
	}
	expr.output_len = (char)out_indices.len;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		target_shape.slice(),
		g, ca.core().dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_einsum(
		expr, operations::BinaryOp.MUL, operations::BinaryOp.ADD);
	g.register_op(payload, {ca, cb}, result);

	return result;
}

// Build Combine Grad Expression

/// Check if a backward expression involves contraction (summation over some index).
fn bool has_contraction(char[] in0, char[] in1, char[] out)
{
	// Check if any output index is NOT in both inputs
	for (usz i = 0; i < out.len; i++)
	{
		bool found = false;
		for (usz j = 0; j < in0.len; j++)
		{
			if (in0[j] == out[i])
			{
				found = true;
				break;
			}
		}
		if (!found) { return true; }

		found = false;
		for (usz j = 0; j < in1.len; j++)
		{
			if (in1[j] == out[i])
			{
				found = true;
				break;
			}
		}
		if (!found) { return true; }
	}

	// Check if any input index is NOT in the output (summation/contraction)
	for (usz i = 0; i < in0.len; i++)
	{
		bool found = false;
		for (usz j = 0; j < out.len; j++) { if (in0[i] == out[j]) { found = true; break; } }
		if (!found) { return true; }
	}
	for (usz i = 0; i < in1.len; i++)
	{
		bool found = false;
		for (usz j = 0; j < out.len; j++) { if (in1[i] == out[j]) { found = true; break; } }
		if (!found) { return true; }
	}

	return false;
}

/// Build gradient expression for matmul-like ops.
/// Handles broadcasting reduction when backward result is larger than target.
fn tensor::Tensor? build_combine_grad_expr(
	tensor::Tensor input_a,
	tensor::Tensor input_b,
	shape::Shape* output_shape,
	char[] input_a_indices,
	char[] input_b_indices,
	char[] output_indices)
{
	// Ensure matching dtypes (same fix as combine_runtime)
	tensor::Tensor ca = input_a;
	tensor::Tensor cb = input_b;
	if (input_a.dtype() != input_b.dtype())
	{
		ca = tensor_ops::cast(input_a, input_b.dtype())!;
	}
	input_a = ca;

	graph::Graph* g = tensor_ops::resolve_graph({input_a, input_b}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({input_a, input_b}, null)!;

	// Compute broadcast shape
	ulong[] a_shape = input_a.core().shape().slice();
	ulong[] b_shape = input_b.core().shape().slice();
	ulong[8] broadcast_buf;
	ulong[] broadcast_shape = compute_broadcast_shape_runtime(a_shape, b_shape, &broadcast_buf)!;

	ulong[] target_shape = output_shape.slice();
	usz target_rank = target_shape.len;

	// Check if we need reduction
	bool needs_reduction = false;
	if (broadcast_shape.len != target_rank)
	{
		needs_reduction = true;
	}
	else
	{
		for (usz i = 0; i < broadcast_shape.len; i++)
		{
			if (broadcast_shape[i] > target_shape[i])
			{
				needs_reduction = true;
				break;
			}
		}
	}

	if (!needs_reduction)
	{
		// Simple case: no broadcast expansion needed
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			target_shape,
			g, input_a.core().dtype(), dev, null)!;

		tensor::Tensor result = tensor::tensor_from_core(result_core);

		index_expr::IndexExpr expr;
		expr.has_ellipsis = false;
		expr.num_inputs = 2;

		for (usz i = 0; i < input_a_indices.len && i < 8; i++)
		{
			expr.input0[i] = input_a_indices[i];
		}
		expr.input_lens[0] = (char)input_a_indices.len;

		for (usz i = 0; i < input_b_indices.len && i < 8; i++)
		{
			expr.input1[i] = input_b_indices[i];
		}
		expr.input_lens[1] = (char)input_b_indices.len;

		for (usz i = 0; i < output_indices.len && i < 8; i++)
		{
			expr.output[i] = output_indices[i];
		}
		expr.output_len = (char)output_indices.len;

		op_payload::OpPayload payload = op_payload::op_payload_einsum(
			expr, operations::BinaryOp.MUL, operations::BinaryOp.ADD);
		g.register_op(payload, {input_a, input_b}, result);

		return result;
	}

	// Complex case: combine at broadcast shape, then reduce to target
	tensor_core::TensorCore* intermediate_core = tensor_ops::setup_result_core(
		broadcast_shape,
		g, input_a.core().dtype(), dev, null)!;
	tensor::Tensor intermediate = tensor::tensor_from_core(intermediate_core);

	index_expr::IndexExpr broadcast_expr = build_broadcast_expr_runtime(a_shape, b_shape, broadcast_shape);

	op_payload::OpPayload combine_payload = op_payload::op_payload_einsum(
		broadcast_expr, operations::BinaryOp.MUL, operations::BinaryOp.ADD);
	g.register_op(combine_payload, {input_a, input_b}, intermediate);

	// Reduce intermediate to target shape
	usz broadcast_rank = broadcast_shape.len;
	usz offset_val = broadcast_rank - target_rank;

	index_expr::IndexExpr reduce_expr;
	reduce_expr.has_ellipsis = false;
	reduce_expr.num_inputs = 1;

	for (usz i = 0; i < broadcast_rank && i < 8; i++)
	{
		reduce_expr.input0[i] = (char)('i' + i);
	}
	reduce_expr.input_lens[0] = (char)broadcast_rank;

	// Output pattern - only non-reduced dims
	char out_idx = 0;
	for (usz i = 0; i < broadcast_rank; i++)
	{
		if (i < offset_val) { continue; }
		usz target_idx = i - offset_val;
		logging::@debug(COMPILER, "build_combine_grad reduce dim[%d]: broadcast=%d target=%d match=%d",
			(int)i, (int)broadcast_shape[i], (int)target_shape[target_idx],
			broadcast_shape[i] == target_shape[target_idx] ? 1 : 0);
		if (broadcast_shape[i] == target_shape[target_idx])
		{
			reduce_expr.output[out_idx] = (char)('i' + i);
			out_idx++;
		}
	}
	reduce_expr.output_len = out_idx;
	if (out_idx == 0 && target_rank > 0)
	{
		logging::@warn(COMPILER, "build_combine_grad: EMPTY output but target_rank=%d! broadcast=[%d,%d,%d,%d] target=[%d,%d,%d]",
			(int)target_rank,
			broadcast_rank > 0 ? (int)broadcast_shape[0] : 0,
			broadcast_rank > 1 ? (int)broadcast_shape[1] : 0,
			broadcast_rank > 2 ? (int)broadcast_shape[2] : 0,
			broadcast_rank > 3 ? (int)broadcast_shape[3] : 0,
			target_rank > 0 ? (int)target_shape[0] : 0,
			target_rank > 1 ? (int)target_shape[1] : 0,
			target_rank > 2 ? (int)target_shape[2] : 0);
	}

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		target_shape,
		g, input_a.core().dtype(), dev, null)!;
	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload reduce_payload = op_payload::op_payload_reduce(
		reduce_expr, operations::BinaryOp.ADD);
	g.register_op(reduce_payload, {intermediate}, result);

	return result;
}

// Internal Shape Helpers

/// Compute numpy-style broadcast output shape at runtime.
fn ulong[]? compute_broadcast_shape_runtime(ulong[] a, ulong[] b, ulong[8]* out)
{
	usz out_len = a.len;
	if (b.len > out_len) { out_len = b.len; }
	if (out_len > 8) { return NOT_FOUND~; }

	for (usz i = 0; i < out_len; i++)
	{
		usz out_idx = out_len - 1 - i;
		ulong a_dim = 1;
		ulong b_dim = 1;
		if (i < a.len) { a_dim = a[a.len - 1 - i]; }
		if (i < b.len) { b_dim = b[b.len - 1 - i]; }

		if (a_dim == b_dim)
		{
			(*out)[out_idx] = a_dim;
		}
		else if (a_dim == 1)
		{
			(*out)[out_idx] = b_dim;
		}
		else if (b_dim == 1)
		{
			(*out)[out_idx] = a_dim;
		}
		else
		{
			return SHAPE_ERROR~;
		}
	}

	return (*out)[:out_len];
}

/// Build index expression for numpy-style broadcasting at runtime.
fn index_expr::IndexExpr build_broadcast_expr_runtime(
	ulong[] a_shape,
	ulong[] b_shape,
	ulong[] out_shape)
{
	index_expr::IndexExpr expr;
	expr.has_ellipsis = false;
	expr.num_inputs = 2;

	usz out_len = out_shape.len;
	char r = (char)out_len;
	assert(r <= 8, "build_broadcast_expr_runtime: rank exceeds max 8");

	// Output indices: ijk...
	for (char i = 0; i < r; i++)
	{
		expr.output[i] = (char)('i' + i);
	}
	expr.output_len = r;

	// Input A indices (right-aligned)
	usz a_offset = out_len - a_shape.len;
	char a_idx = 0;
	for (usz i = 0; i < a_shape.len; i++)
	{
		expr.input0[a_idx] = (char)('i' + a_offset + i);
		a_idx++;
	}
	expr.input_lens[0] = a_idx;

	// Input B indices (right-aligned)
	usz b_offset = out_len - b_shape.len;
	char b_idx = 0;
	for (usz i = 0; i < b_shape.len; i++)
	{
		expr.input1[b_idx] = (char)('i' + b_offset + i);
		b_idx++;
	}
	expr.input_lens[1] = b_idx;

	return expr;
}

/// Compute output shape for reduce from runtime expression.
fn shape::Shape? compute_reduce_shape_runtime(
	index_expr::IndexExpr* expr,
	shape::Shape* a_shape)
{
	ulong[8] dims;
	char[] out_indices = expr.get_output();
	char[] in_indices = expr.get_input(0);

	for (usz i = 0; i < out_indices.len; i++)
	{
		if (try pos = index_expr::index_in_pattern(out_indices[i], in_indices))
		{
			dims[i] = a_shape.get(pos);
		}
		else
		{
			return NOT_FOUND~;
		}
	}

	if (out_indices.len == 0)
	{
		return shape::shape_init({});
	}

	return shape::shape_init(dims[:out_indices.len]);
}

faultdef NOT_FOUND;
faultdef SHAPE_ERROR;

// Scalar Helpers (for backward pass)

/// Scale a tensor by a scalar constant: result = a * scalar.
fn tensor::Tensor? scale_runtime(tensor::Tensor a, float scalar)
{
	graph::Graph* g = tensor_ops::resolve_graph({a}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a}, null)!;

	// Create a true scalar constant tensor (0-dimensional)
	ulong[] scalar_dims;
	tensor_core::TensorCore* scalar_core = tensor_ops::setup_result_core(
		scalar_dims,
		g, a.core().dtype(), dev, null)!;
	tensor::Tensor scalar_t = tensor::tensor_from_core(scalar_core);
	scalar_core.alloc_data();
	scalar_core.fill(scalar);

	return combine_with_scalar(a, scalar_t, operations::BinaryOp.MUL);
}

/// Add a scalar constant to all elements: result = a + scalar.
fn tensor::Tensor? add_scalar_runtime(tensor::Tensor a, float scalar)
{
	graph::Graph* g = tensor_ops::resolve_graph({a}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a}, null)!;

	// Create a true scalar constant tensor (0-dimensional)
	ulong[] scalar_dims;
	tensor_core::TensorCore* scalar_core = tensor_ops::setup_result_core(
		scalar_dims,
		g, a.core().dtype(), dev, null)!;
	tensor::Tensor scalar_t = tensor::tensor_from_core(scalar_core);
	scalar_core.alloc_data();
	scalar_core.fill(scalar);

	return combine_with_scalar(a, scalar_t, operations::BinaryOp.ADD);
}

/// Create a zero-filled tensor with the same shape and dtype as the input.
fn tensor::Tensor? create_zero_like(tensor::Tensor a)
{
	graph::Graph* g = tensor_ops::resolve_graph({a}, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device({a}, null)!;

	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		a.core().shape().slice(),
		g, a.core().dtype(), dev, null)!;
	tensor::Tensor result = tensor::tensor_from_core(result_core);
	result_core.alloc_data();
	result_core.fill_zero();

	return result;
}
