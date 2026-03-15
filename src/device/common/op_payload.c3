//! Op Payload
//!
//! Unified operation payload union. This is the single payload type stored
//! in SymbolicNode, replacing both UnifiedCallback and various bypass callbacks.
//!
//! Each payload struct has OpTag as its first field, so all union members
//! share the tag at offset 0 — forming a manually tagged union.
//!

module metaphor::device::common::op_payload;
import metaphor::device::common::operations;
import metaphor::device::common::index_expr;
import std::io;

// OpPayload — tagged union over all operation types

/// Unified operation payload. Stored in SymbolicNode.
/// Each variant carries all data needed for forward and backward passes.
/// All variants have OpTag as their first field (offset 0).
union OpPayload
{
	EinsumPayload einsum;
	ReducePayload reduce;
	ExpandPayload expand;
	MapPayload map_payload;
	SelectPayload select;
	SlicePayload slice;
	PermutePayload permute;
	ConcatPayload concat;
	StackPayload stack;
	SoftmaxPayload softmax;
	RmsNormPayload rms_norm;
	SelectiveScanPayload selective_scan;
	CausalConv1dPayload causal_conv1d;
	DropoutPayload dropout;
	ConvPayload conv;
	ConvTransposePayload conv_transpose;
	GatherPayload gather;
	ScatterAddPayload scatter_add;
	CastPayload cast;
	ReshapePayload reshape;
	ReduceMaxMinBackwardPayload reduce_maxmin_backward;
	TransferPayload transfer;
	NumelPayload numel;
	DimPayload dim;
}

/// Read the discriminant tag. Safe because all variants share OpTag at offset 0.
fn OpTag OpPayload.tag(&self)
{
	return self.einsum.tag;
}

// Convenience Constructors

/// Create an einsum payload.
fn OpPayload op_payload_einsum(IndexExpr expr, BinaryOp inner, BinaryOp outer)
{
	return { .einsum = {
		.tag = EINSUM,
		.expr = expr,
		.inner = inner,
		.outer = outer,
		.alpha = 1.0,
		.beta = 0.0,
		.hash = 0,
	}};
}

/// Create a reduce payload.
fn OpPayload op_payload_reduce(IndexExpr expr, BinaryOp op)
{
	return { .reduce = {
		.tag = REDUCE,
		.expr = expr,
		.op = op,
		.hash = 0,
	}};
}

/// Create a map payload.
fn OpPayload op_payload_map(UnaryOp unary)
{
	return { .map_payload = {
		.tag = MAP,
		.unary = unary,
		.beta = 0.0,
		.hash = 0,
	}};
}

/// Create a select payload.
fn OpPayload op_payload_select()
{
	return { .select = { .tag = SELECT, .hash = 0 }};
}

/// Create a permute payload.
fn OpPayload op_payload_permute(IndexExpr expr)
{
	return { .permute = {
		.tag = PERMUTE,
		.expr = expr,
		.hash = 0,
	}};
}

/// Create a slice payload.
fn OpPayload op_payload_slice(SlicePayload p)
{
	p.tag = SLICE;
	return { .slice = p };
}

/// Create a concat payload.
fn OpPayload op_payload_concat(ConcatPayload p)
{
	p.tag = CONCAT;
	return { .concat = p };
}

/// Create a stack payload.
fn OpPayload op_payload_stack(StackPayload p)
{
	p.tag = STACK;
	return { .stack = p };
}

/// Create a conv payload.
fn OpPayload op_payload_conv(ConvParams params)
{
	return { .conv = { .tag = CONV, .params = params }};
}

/// Create a gather payload.
fn OpPayload op_payload_gather(GatherParams params)
{
	return { .gather = { .tag = GATHER, .params = params, .hash = 0 }};
}

/// Create a cast payload.
fn OpPayload op_payload_cast(char source_dtype, char target_dtype)
{
	return { .cast = { .tag = CAST, .source_dtype = source_dtype, .target_dtype = target_dtype }};
}

/// Create a reshape payload.
fn OpPayload op_payload_reshape()
{
	return { .reshape = { .tag = RESHAPE }};
}

fn OpPayload op_payload_transfer(device_reference::TransferDirection direction)
{
	return { .transfer = { .tag = TRANSFER, .direction = direction }};
}

/// Create a softmax payload.
fn OpPayload op_payload_softmax(bool is_log, bool is_backward, int axis = -1)
{
	return { .softmax = { .tag = SOFTMAX, .axis = axis, .is_log = is_log, .is_backward = is_backward, .hash = 0 }};
}

/// Create an rms_norm payload.
fn OpPayload op_payload_rms_norm(double eps, RmsOpType op_type)
{
	return { .rms_norm = { .tag = RMS_NORM, .eps = eps, .op_type = op_type, .hash = 0 }};
}

/// Create a selective_scan payload.
fn OpPayload op_payload_selective_scan(SelectiveScanOpType op_type)
{
	return { .selective_scan = { .tag = SELECTIVE_SCAN, .op_type = op_type, .hash = 0 }};
}

/// Create a causal_conv1d payload.
fn OpPayload op_payload_causal_conv1d(CausalConv1dOpType op_type)
{
	return { .causal_conv1d = { .tag = CAUSAL_CONV1D, .op_type = op_type, .hash = 0 }};
}

/// Create a dropout payload.
fn OpPayload op_payload_dropout(float p, ulong seed, bool is_backward = false)
{
	return { .dropout = { .tag = DROPOUT, .p = p, .seed = seed, .is_backward = is_backward, .hash = 0 }};
}

/// Create a scatter_add payload.
fn OpPayload op_payload_scatter_add(GatherParams params)
{
	return { .scatter_add = { .tag = SCATTER_ADD, .params = params, .hash = 0 }};
}

/// Create a reduce_maxmin_backward payload.
fn OpPayload op_payload_reduce_maxmin_backward(ulong row, ulong col)
{
	return { .reduce_maxmin_backward = { .tag = REDUCE_MAXMIN_BACKWARD, .row = row, .col = col, .hash = 0 }};
}

/// Create a conv_transpose payload.
fn OpPayload op_payload_conv_transpose(ConvParams params, ulong[4] output_shape)
{
	return { .conv_transpose = { .tag = CONV_TRANSPOSE, .params = params, .output_shape = output_shape, .hash = 0 }};
}

/// Create a numel payload.
fn OpPayload op_payload_numel()
{
	return { .numel = { .tag = NUMEL, .hash = 0 }};
}

/// Create a dim payload.
fn OpPayload op_payload_dim(usz axis)
{
	return { .dim = { .tag = DIM, .axis = axis, .hash = 0 }};
}

// Hash Accessors

/// Get the kernel hash (set by compile()).
fn ulong OpPayload.get_hash(&self)
{
	switch (self.tag())
	{
		case EINSUM:   return self.einsum.hash;
		case REDUCE:   return self.reduce.hash;
		case EXPAND:   return self.expand.hash;
		case MAP:      return self.map_payload.hash;
		case SELECT:   return self.select.hash;
		case SLICE:    return self.slice.hash;
		case PERMUTE:  return self.permute.hash;
		case CONCAT:   return self.concat.hash;
		case STACK:    return self.stack.hash;
		case SOFTMAX:  return self.softmax.hash;
		case RMS_NORM: return self.rms_norm.hash;
		case SELECTIVE_SCAN: return self.selective_scan.hash;
		case CAUSAL_CONV1D: return self.causal_conv1d.hash;
		case DROPOUT: return self.dropout.hash;
		case GATHER:   return self.gather.hash;
		case SCATTER_ADD: return self.scatter_add.hash;
		case CONV_TRANSPOSE: return self.conv_transpose.hash;
		case REDUCE_MAXMIN_BACKWARD: return self.reduce_maxmin_backward.hash;
		default: unreachable("OpPayload.get_hash: unhandled tag");
	}
}

/// Set the kernel hash (called by compile()).
fn void OpPayload.set_hash(&self, ulong hash)
{
	switch (self.tag())
	{
		case EINSUM:   self.einsum.hash = hash;
		case REDUCE:   self.reduce.hash = hash;
		case EXPAND:   self.expand.hash = hash;
		case MAP:      self.map_payload.hash = hash;
		case SELECT:   self.select.hash = hash;
		case SLICE:    self.slice.hash = hash;
		case PERMUTE:  self.permute.hash = hash;
		case CONCAT:   self.concat.hash = hash;
		case STACK:    self.stack.hash = hash;
		case SOFTMAX:  self.softmax.hash = hash;
		case RMS_NORM: self.rms_norm.hash = hash;
		case SELECTIVE_SCAN: self.selective_scan.hash = hash;
		case CAUSAL_CONV1D: self.causal_conv1d.hash = hash;
		case DROPOUT: self.dropout.hash = hash;
		case GATHER:   self.gather.hash = hash;
		case SCATTER_ADD: self.scatter_add.hash = hash;
		case CONV_TRANSPOSE: self.conv_transpose.hash = hash;
		case REDUCE_MAXMIN_BACKWARD: self.reduce_maxmin_backward.hash = hash;
		default: unreachable("set_hash: unhandled tag");
	}
}

// Symbolic Backward

import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::tensor_ops;
import metaphor::graph;
import metaphor::device::common::backward_helpers;
import metaphor::device::device_reference;
import metaphor::logging;

/// Compute symbolic gradient for a specific child of this operation.
/// Returns null if no gradient can be computed (non-differentiable op).
fn tensor::Tensor? OpPayload.symbolic_backward(&self, usz child_idx, tensor::Tensor[] children, tensor::Tensor upstream)
{
	// Get upstream gradient — if it doesn't exist, can't backprop
	graph::Graph* g = upstream.graph();
	tensor_core::TensorCore* grad_core = g.get_gradient_core(upstream.core());
	if (grad_core == null)
	{
		return NO_GRADIENT~;
	}
	tensor::Tensor grad = tensor::tensor_from_core(grad_core);

	logging::@debug(OP_PAYLOAD, "tag=%d child_idx=%d num_children=%d upstream=%p grad=%p",
		(int)self.tag(), (int)child_idx, (int)children.len, upstream.core(), grad_core);

	switch (self.tag())
	{
		case EINSUM:  return self.bwd_einsum(child_idx, children, grad);
		case REDUCE:  return self.bwd_reduce(children, grad);
		case EXPAND:  return self.bwd_expand(children, grad);
		case MAP:     return self.bwd_map(children, upstream, grad);
		case SELECT:  return self.bwd_select(child_idx, children, grad);
		case SLICE:   return self.bwd_slice(children, grad);
		case PERMUTE: return self.bwd_permute(grad);
		case CONCAT:  return self.bwd_concat(child_idx, children, grad);
		case STACK:   return self.bwd_stack(child_idx, children, grad);
		case SOFTMAX: return self.bwd_softmax(children, upstream, grad);
		case RMS_NORM: return self.bwd_rms_norm(child_idx, children, upstream, grad);
		case SELECTIVE_SCAN: return self.bwd_selective_scan(child_idx, children, upstream, grad);
		case CAUSAL_CONV1D: return self.bwd_causal_conv1d(child_idx, children, upstream, grad);
		case DROPOUT: return self.bwd_dropout(children, upstream, grad);
		case GATHER:  return self.bwd_gather(child_idx, children, upstream);
		case CAST:    return self.bwd_cast(children, upstream);
		case RESHAPE: return self.bwd_reshape(children, grad);
		case TRANSFER: return grad;  // gradient passes through unchanged
		// Shape queries — genuinely not differentiable
		case NUMEL: return NO_GRADIENT~;
		case DIM:   return NO_GRADIENT~;
		default: unreachable("backward not implemented for op tag %d", (int)self.tag());
	}
}

/// Backward for einsum: handle add/sub/mul/div.
fn tensor::Tensor? OpPayload.bwd_einsum(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor grad)
{
	char[] out_indices = self.einsum.expr.get_output();
	char[] a_indices = self.einsum.expr.get_input(0);
	char[] b_indices = self.einsum.expr.get_input(1);

	// Element-wise ADD: gradient is just the upstream gradient
	if (self.einsum.inner == operations::BinaryOp.ADD)
	{
		char[] target_indices = child_idx == 0 ? a_indices : b_indices;

		if (indices_equal(out_indices, target_indices))
		{
			// Same indices — identity copy
			return backward_helpers::map_runtime(grad, operations::UnaryOp.IDENTITY);
		}
		else
		{
			// Need to reduce due to broadcasting
			index_expr::IndexExpr reduce_expr;
			reduce_expr.has_ellipsis = false;
			reduce_expr.num_inputs = 1;
			backward_helpers::copy_indices(&reduce_expr.input0, &reduce_expr.input_lens[0], out_indices);
			backward_helpers::copy_indices(&reduce_expr.output, &reduce_expr.output_len, target_indices);
			return backward_helpers::reduce_runtime(grad, &reduce_expr, operations::BinaryOp.ADD);
		}
	}

	// SUB: dc/da = grad, dc/db = -grad
	if (self.einsum.inner == operations::BinaryOp.SUB)
	{
		char[] target_indices = child_idx == 0 ? a_indices : b_indices;
		tensor::Tensor? base_grad;
		if (child_idx == 0)
		{
			base_grad = backward_helpers::map_runtime(grad, operations::UnaryOp.IDENTITY);
		}
		else
		{
			base_grad = backward_helpers::negate_runtime(grad);
		}

		if (try bg = base_grad)
		{
			if (indices_equal(out_indices, target_indices))
			{
				return bg;
			}
			else
			{
				index_expr::IndexExpr reduce_expr;
				reduce_expr.has_ellipsis = false;
				reduce_expr.num_inputs = 1;
				backward_helpers::copy_indices(&reduce_expr.input0, &reduce_expr.input_lens[0], out_indices);
				backward_helpers::copy_indices(&reduce_expr.output, &reduce_expr.output_len, target_indices);
				return backward_helpers::reduce_runtime(bg, &reduce_expr, operations::BinaryOp.ADD);
			}
		}
		return NO_GRADIENT~;
	}

	// DIV: dc/da = grad / b, dc/db = -(grad * a / b²)
	if (self.einsum.inner == operations::BinaryOp.DIV)
	{
		char[] target_indices = child_idx == 0 ? a_indices : b_indices;

		if (child_idx == 0)
		{
			// grad_a = grad / b
			tensor::Tensor grad_a = backward_helpers::combine_with_broadcast(
				grad, args[1], operations::BinaryOp.DIV)!;

			if (indices_equal(out_indices, target_indices))
			{
				return grad_a;
			}
			else
			{
				index_expr::IndexExpr reduce_expr;
				reduce_expr.has_ellipsis = false;
				reduce_expr.num_inputs = 1;
				backward_helpers::copy_indices(&reduce_expr.input0, &reduce_expr.input_lens[0], out_indices);
				backward_helpers::copy_indices(&reduce_expr.output, &reduce_expr.output_len, target_indices);
				return backward_helpers::reduce_runtime(grad_a, &reduce_expr, operations::BinaryOp.ADD);
			}
		}
		else
		{
			// grad_b = -(grad * a / b²)
			tensor::Tensor grad_times_a = backward_helpers::combine_with_broadcast(
				grad, args[0], operations::BinaryOp.MUL)!;
			tensor::Tensor b_squared = backward_helpers::combine_with_broadcast(
				args[1], args[1], operations::BinaryOp.MUL)!;
			tensor::Tensor ratio = backward_helpers::combine_with_broadcast(
				grad_times_a, b_squared, operations::BinaryOp.DIV)!;
			tensor::Tensor grad_b = backward_helpers::negate_runtime(ratio)!;

			if (indices_equal(out_indices, target_indices))
			{
				return grad_b;
			}
			else
			{
				index_expr::IndexExpr reduce_expr;
				reduce_expr.has_ellipsis = false;
				reduce_expr.num_inputs = 1;
				backward_helpers::copy_indices(&reduce_expr.input0, &reduce_expr.input_lens[0], out_indices);
				backward_helpers::copy_indices(&reduce_expr.output, &reduce_expr.output_len, target_indices);
				return backward_helpers::reduce_runtime(grad_b, &reduce_expr, operations::BinaryOp.ADD);
			}
		}
	}

	// MUL and other ops: dA = einsum(dC, B), dB = einsum(A, dC)
	// Build expanded index arrays (handling ellipsis if needed)
	char[8] exp_a;
	char[8] exp_b;
	char[8] exp_out;
	usz exp_a_len = 0;
	usz exp_b_len = 0;
	usz exp_out_len = 0;

	for (usz i = 0; i < a_indices.len; i++)
	{
		exp_a[exp_a_len] = a_indices[i];
		exp_a_len++;
	}
	for (usz i = 0; i < b_indices.len; i++)
	{
		exp_b[exp_b_len] = b_indices[i];
		exp_b_len++;
	}
	for (usz i = 0; i < out_indices.len; i++)
	{
		exp_out[exp_out_len] = out_indices[i];
		exp_out_len++;
	}

	char[] final_a = exp_a[:exp_a_len];
	char[] final_b = exp_b[:exp_b_len];
	char[] final_out = exp_out[:exp_out_len];

	if (child_idx == 0)
	{
		if (backward_helpers::has_contraction(final_out, final_b, final_a))
		{
			return backward_helpers::combine_runtime(
				grad, args[1], args[0].core().shape(),
				final_out, final_b, final_a);
		}
		else
		{
			return backward_helpers::build_combine_grad_expr(
				grad, args[1], args[0].core().shape(),
				final_out, final_b, final_a);
		}
	}
	else
	{
		if (backward_helpers::has_contraction(final_a, final_out, final_b))
		{
			return backward_helpers::combine_runtime(
				args[0], grad, args[1].core().shape(),
				final_a, final_out, final_b);
		}
		else
		{
			return backward_helpers::build_combine_grad_expr(
				args[0], grad, args[1].core().shape(),
				final_a, final_out, final_b);
		}
	}
}

/// Backward for reduce: expand gradient back to input shape.
fn tensor::Tensor? OpPayload.bwd_reduce(&self, tensor::Tensor[] args, tensor::Tensor grad)
{
	tensor::Tensor input = args[0];

	index_expr::IndexExpr expand_expr;
	expand_expr.has_ellipsis = false;
	expand_expr.num_inputs = 1;

	char[] explicit_in = self.reduce.expr.get_input(0);
	char[] explicit_out = self.reduce.expr.get_output();

	// Expand ellipsis into explicit indices using the input tensor's rank
	char[8] full_in;
	usz full_in_len = 0;
	char[8] full_out;
	usz full_out_len = 0;

	if (self.reduce.expr.has_ellipsis)
	{
		usz ellipsis_len = input.core().shape().len - explicit_in.len;

		// Batch dims come first
		for (usz d = 0; d < ellipsis_len; d++)
		{
			char batch_char = (char)('a' + d);
			full_in[full_in_len++] = batch_char;
			full_out[full_out_len++] = batch_char;
		}
		for (usz i = 0; i < explicit_in.len; i++) { full_in[full_in_len++] = explicit_in[i]; }
		for (usz i = 0; i < explicit_out.len; i++) { full_out[full_out_len++] = explicit_out[i]; }
	}
	else
	{
		for (usz i = 0; i < explicit_in.len; i++) { full_in[full_in_len++] = explicit_in[i]; }
		for (usz i = 0; i < explicit_out.len; i++) { full_out[full_out_len++] = explicit_out[i]; }
	}

	char[] in_indices = full_in[:full_in_len];
	char[] out_indices = full_out[:full_out_len];

	logging::@debug(OP_PAYLOAD, "in='%s' out='%s' input_shape_len=%d grad_shape_len=%d",
		(ZString)in_indices.ptr, (ZString)out_indices.ptr,
		(int)input.core().shape().len, (int)grad.core().shape().len);

	// Swap: the reduce's input becomes the expand's output and vice versa
	backward_helpers::copy_indices(&expand_expr.input0, &expand_expr.input_lens[0], out_indices);
	backward_helpers::copy_indices(&expand_expr.output, &expand_expr.output_len, in_indices);

	// Compute new_sizes for dimensions being expanded
	ulong[8] new_sizes;
	char num_new = 0;

	for (usz pos = 0; pos < in_indices.len; pos++)
	{
		// Check if this index is NOT in the output (was reduced away)
		bool found = false;
		for (usz j = 0; j < out_indices.len; j++)
		{
			if (in_indices[pos] == out_indices[j])
			{
				found = true;
				break;
			}
		}
		if (!found)
		{
			new_sizes[num_new] = input.core().shape().get(pos);
			num_new++;
		}
	}

	// For sum/add reduction: just expand gradient back
	if (self.reduce.op != operations::BinaryOp.MAX && self.reduce.op != operations::BinaryOp.MIN)
	{
		return backward_helpers::expand_runtime(grad, &expand_expr, &new_sizes, num_new);
	}

	// Max/min backward not yet implemented
	return NO_GRADIENT~;
}

/// Backward for expand: reduce gradient back to input shape.
fn tensor::Tensor? OpPayload.bwd_expand(&self, tensor::Tensor[] args, tensor::Tensor grad)
{
	index_expr::IndexExpr reduce_expr;
	reduce_expr.has_ellipsis = false;
	reduce_expr.num_inputs = 1;

	char[] explicit_in = self.expand.expr.get_input(0);
	char[] explicit_out = self.expand.expr.get_output();

	logging::@debug(OP_PAYLOAD, "in='%s' out='%s' grad_shape_len=%d",
		(ZString)explicit_in.ptr, (ZString)explicit_out.ptr,
		(int)grad.core().shape().len);

	// Swap: the expand's output becomes the reduce's input and vice versa
	backward_helpers::copy_indices(&reduce_expr.input0, &reduce_expr.input_lens[0], explicit_out);
	backward_helpers::copy_indices(&reduce_expr.output, &reduce_expr.output_len, explicit_in);

	return backward_helpers::reduce_runtime(grad, &reduce_expr, operations::BinaryOp.ADD);
}

/// Backward for map: chain rule with derivative.
fn tensor::Tensor? OpPayload.bwd_map(&self, tensor::Tensor[] args, tensor::Tensor upstream, tensor::Tensor grad)
{
	tensor::Tensor x = args[0];
	tensor::Tensor y = upstream;

	switch (self.map_payload.unary)
	{
		case IDENTITY: return grad;
		case NEG: return backward_helpers::map_runtime(grad, operations::UnaryOp.NEG);

		case EXP: return backward_helpers::combine_elemwise(grad, y, operations::BinaryOp.MUL);

		case RELU:
		{
			tensor::Tensor h = backward_helpers::map_runtime(x, operations::UnaryOp.HEAVISIDE)!;
			return backward_helpers::combine_elemwise(grad, h, operations::BinaryOp.MUL);
		}

		case SIN:
		{
			tensor::Tensor cos_x = backward_helpers::map_runtime(x, operations::UnaryOp.COS)!;
			return backward_helpers::combine_elemwise(grad, cos_x, operations::BinaryOp.MUL);
		}

		case COS:
		{
			tensor::Tensor sin_x = backward_helpers::map_runtime(x, operations::UnaryOp.SIN)!;
			tensor::Tensor neg_sin = backward_helpers::map_runtime(sin_x, operations::UnaryOp.NEG)!;
			return backward_helpers::combine_elemwise(grad, neg_sin, operations::BinaryOp.MUL);
		}

		case ABS:
		{
			tensor::Tensor sign_x = backward_helpers::map_runtime(x, operations::UnaryOp.SIGN)!;
			return backward_helpers::combine_elemwise(grad, sign_x, operations::BinaryOp.MUL);
		}

		case RECIP:
		{
			// d/dx(1/x) = -1/x²
			tensor::Tensor x_sqr = backward_helpers::map_runtime(x, operations::UnaryOp.SQR)!;
			tensor::Tensor neg_recip = backward_helpers::map_runtime(x_sqr, operations::UnaryOp.RECIP)!;
			tensor::Tensor deriv = backward_helpers::map_runtime(neg_recip, operations::UnaryOp.NEG)!;
			return backward_helpers::combine_elemwise(grad, deriv, operations::BinaryOp.MUL);
		}

		case TANH:
		{
			// d/dx(tanh(x)) = 1 - tanh²(x) = 1 - y²
			tensor::Tensor y_sqr = backward_helpers::map_runtime(y, operations::UnaryOp.SQR)!;
			tensor::Tensor neg_ys = backward_helpers::map_runtime(y_sqr, operations::UnaryOp.NEG)!;
			tensor::Tensor one_minus = backward_helpers::add_scalar_runtime(neg_ys, 1.0)!;
			return backward_helpers::combine_elemwise(grad, one_minus, operations::BinaryOp.MUL);
		}

		case SIGMOID:
		{
			// d/dx(σ(x)) = σ(x)(1 - σ(x)) = y(1 - y)
			tensor::Tensor neg_y = backward_helpers::map_runtime(y, operations::UnaryOp.NEG)!;
			tensor::Tensor one_minus_y = backward_helpers::add_scalar_runtime(neg_y, 1.0)!;
			tensor::Tensor y_times = backward_helpers::combine_elemwise(y, one_minus_y, operations::BinaryOp.MUL)!;
			return backward_helpers::combine_elemwise(grad, y_times, operations::BinaryOp.MUL);
		}

		case LOG:
		{
			tensor::Tensor recip_x = backward_helpers::map_runtime(x, operations::UnaryOp.RECIP)!;
			return backward_helpers::combine_elemwise(grad, recip_x, operations::BinaryOp.MUL);
		}

		case SQR:
		{
			tensor::Tensor two_x = backward_helpers::scale_runtime(x, 2.0)!;
			return backward_helpers::combine_elemwise(grad, two_x, operations::BinaryOp.MUL);
		}

		case SQRT:
		{
			// d/dx(√x) = 1/(2√x) = 0.5/y
			tensor::Tensor recip_y = backward_helpers::map_runtime(y, operations::UnaryOp.RECIP)!;
			tensor::Tensor half_recip = backward_helpers::scale_runtime(recip_y, 0.5)!;
			return backward_helpers::combine_elemwise(grad, half_recip, operations::BinaryOp.MUL);
		}

		case GELU:
		{
			// GELU'(x) = 0.5 * (1 + tanh(k)) + 0.5 * x * sech²(k) * dk/dx
			// where k = 0.7978845608 * (x + 0.044715 * x³)
			//       dk/dx = 0.7978845608 * (1 + 0.134145 * x²)
			tensor::Tensor xs = backward_helpers::map_runtime(x, operations::UnaryOp.SQR)!;
			tensor::Tensor xc = backward_helpers::combine_elemwise(x, xs, operations::BinaryOp.MUL)!;
			tensor::Tensor cxc = backward_helpers::scale_runtime(xc, 0.044715)!;
			tensor::Tensor xpc = backward_helpers::combine_elemwise(x, cxc, operations::BinaryOp.ADD)!;
			tensor::Tensor kk = backward_helpers::scale_runtime(xpc, 0.7978845608)!;
			tensor::Tensor tk = backward_helpers::map_runtime(kk, operations::UnaryOp.TANH)!;
			tensor::Tensor tks = backward_helpers::map_runtime(tk, operations::UnaryOp.SQR)!;
			tensor::Tensor ntks = backward_helpers::map_runtime(tks, operations::UnaryOp.NEG)!;
			tensor::Tensor s2k = backward_helpers::add_scalar_runtime(ntks, 1.0)!;
			tensor::Tensor sxs = backward_helpers::scale_runtime(xs, 0.134145)!;
			tensor::Tensor ops2 = backward_helpers::add_scalar_runtime(sxs, 1.0)!;
			tensor::Tensor dkdx = backward_helpers::scale_runtime(ops2, 0.7978845608)!;
			tensor::Tensor xs2 = backward_helpers::combine_elemwise(x, s2k, operations::BinaryOp.MUL)!;
			tensor::Tensor xsd = backward_helpers::combine_elemwise(xs2, dkdx, operations::BinaryOp.MUL)!;
			tensor::Tensor opt = backward_helpers::add_scalar_runtime(tk, 1.0)!;
			tensor::Tensor s = backward_helpers::combine_elemwise(opt, xsd, operations::BinaryOp.ADD)!;
			tensor::Tensor deriv = backward_helpers::scale_runtime(s, 0.5)!;
			return backward_helpers::combine_elemwise(grad, deriv, operations::BinaryOp.MUL);
		}

		case SILU:
		{
			// SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
			tensor::Tensor sig = backward_helpers::map_runtime(x, operations::UnaryOp.SIGMOID)!;
			tensor::Tensor neg_sig = backward_helpers::map_runtime(sig, operations::UnaryOp.NEG)!;
			tensor::Tensor one_minus_sig = backward_helpers::add_scalar_runtime(neg_sig, 1.0)!;
			tensor::Tensor x_times_oms = backward_helpers::combine_elemwise(x, one_minus_sig, operations::BinaryOp.MUL)!;
			tensor::Tensor one_plus_x_oms = backward_helpers::add_scalar_runtime(x_times_oms, 1.0)!;
			tensor::Tensor deriv = backward_helpers::combine_elemwise(sig, one_plus_x_oms, operations::BinaryOp.MUL)!;
			return backward_helpers::combine_elemwise(grad, deriv, operations::BinaryOp.MUL);
		}

		case SOFTPLUS:
		{
			// softplus'(x) = sigmoid(x)
			tensor::Tensor sig = backward_helpers::map_runtime(x, operations::UnaryOp.SIGMOID)!;
			return backward_helpers::combine_elemwise(grad, sig, operations::BinaryOp.MUL);
		}

		// Non-differentiable ops
		case SIGN: return NO_GRADIENT~;
		case HEAVISIDE: return NO_GRADIENT~;
		case NOT: return NO_GRADIENT~;
	}
}

/// Backward for slice: scatter gradient back to original positions.
fn tensor::Tensor? OpPayload.bwd_slice(&self, tensor::Tensor[] args, tensor::Tensor grad)
{
	return tensor_ops::slice_scatter(args[0], grad, self.slice);
}

/// Backward for permute: apply inverse permutation.
/// Swaps input and output of the forward expression to get the inverse.
fn tensor::Tensor? OpPayload.bwd_permute(&self, tensor::Tensor grad)
{
	index_expr::IndexExpr inv_expr;
	inv_expr.has_ellipsis = self.permute.expr.has_ellipsis;
	inv_expr.num_inputs = 1;

	// Inverse: input becomes output and vice versa
	char[] fwd_in = self.permute.expr.get_input(0);
	char[] fwd_out = self.permute.expr.get_output();

	backward_helpers::copy_indices(&inv_expr.input0, &inv_expr.input_lens[0], fwd_out);
	backward_helpers::copy_indices(&inv_expr.output, &inv_expr.output_len, fwd_in);

	return tensor_ops::permute(grad, &inv_expr, {});
}

/// Backward for concat: slice gradient back to each input's portion.
fn tensor::Tensor? OpPayload.bwd_concat(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor grad)
{
	shape::Shape* grad_shape = grad.shape();
	usz ndim = grad_shape.len;

	// Build slice ranges for this child's portion of the gradient
	operations::SliceRange[8] ranges;

	for (usz d = 0; d < ndim; d++)
	{
		if (d == (usz)self.concat.dim)
		{
			// Concat dimension — slice from offsets[child_idx] to offsets[child_idx+1]
			ranges[d] = {
				.start = self.concat.offsets[child_idx],
				.end = self.concat.offsets[child_idx + 1],
				.step = 1,
			};
		}
		else
		{
			// Other dimensions — take the full extent
			ranges[d] = {
				.start = 0,
				.end = 0, // 0 means full extent in C3 slice
				.step = 1,
			};
		}
	}

	return tensor_ops::slice(grad, ranges[:ndim]);
}

/// Backward for stack: slice gradient and remove the stack dimension.
fn tensor::Tensor? OpPayload.bwd_stack(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor grad)
{
	shape::Shape* grad_shape = grad.shape();
	usz ndim = grad_shape.len;

	// Build slice ranges that select index child_idx along the stack dimension
	operations::SliceRange[8] ranges;

	for (usz d = 0; d < ndim; d++)
	{
		if (d == (usz)self.stack.dim)
		{
			// Stack dimension — select just this index (size=1)
			ranges[d] = {
				.start = (ulong)child_idx,
				.end = (ulong)(child_idx + 1),
				.step = 1,
			};
		}
		else
		{
			// Other dimensions — take the full extent
			ranges[d] = {
				.start = 0,
				.end = 0, // 0 means full extent
				.step = 1,
			};
		}
	}

	// Slice the gradient, then squeeze out the stack dimension
	tensor::Tensor? sliced = tensor_ops::slice(grad, ranges[:ndim]);
	if (try s = sliced)
	{
		return tensor_ops::squeeze(s, (usz)self.stack.dim);
	}
	return NO_GRADIENT~;
}

/// Backward for select: gradient flows only to the selected branch.
fn tensor::Tensor? OpPayload.bwd_select(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor grad)
{
	tensor::Tensor condition = args[0];

	switch (child_idx)
	{
		case 0: return NO_GRADIENT~; // Condition is boolean — not differentiable
		case 1:
		{
			tensor::Tensor? zero = backward_helpers::create_zero_like(grad);
			if (try z = zero)
			{
				return backward_helpers::select_runtime(condition, grad, z);
			}
			return NO_GRADIENT~;
		}
		case 2:
		{
			tensor::Tensor? zero = backward_helpers::create_zero_like(grad);
			if (try z = zero)
			{
				return backward_helpers::select_runtime(condition, z, grad);
			}
			return NO_GRADIENT~;
		}
		default: return NO_GRADIENT~;
	}
}

// Boundary Op Backward Methods

/// Backward for softmax/log_softmax.
/// Registers a new softmax op with is_backward=true.
/// Inputs to backward kernel: [upstream_output, grad]
fn tensor::Tensor? OpPayload.bwd_softmax(&self, tensor::Tensor[] args, tensor::Tensor upstream, tensor::Tensor grad)
{
	tensor::Tensor[] resolve_args = { upstream, grad };
	graph::Graph* g = tensor_ops::resolve_graph(resolve_args, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device(resolve_args, null)!;

	ulong[] shape_slice = grad.shape().slice();
	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		shape_slice, g, grad.dtype(), dev, null)!;
	tensor::Tensor result = tensor::tensor_from_core(result_core);

	OpPayload bwd_payload = op_payload_softmax(
		self.softmax.is_log, true, self.softmax.axis);

	// sources[0] = y (forward output), sources[1] = grad_y (upstream gradient)
	tensor::Tensor[2] op_args = { upstream, grad };
	g.register_op(bwd_payload, &op_args, result);

	return result;
}

/// Backward for dropout.
/// Registers a new dropout op with the forward's seed, is_backward=true.
/// Only passes the upstream gradient — the mask is determined by seed alone.
fn tensor::Tensor? OpPayload.bwd_dropout(&self, tensor::Tensor[] args, tensor::Tensor upstream, tensor::Tensor grad)
{
	tensor::Tensor[] resolve_args = { grad };
	graph::Graph* g = tensor_ops::resolve_graph(resolve_args, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device(resolve_args, null)!;

	ulong[] shape_slice = grad.shape().slice();
	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		shape_slice, g, grad.dtype(), dev, null)!;
	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Backward payload: same seed/p, is_backward=true
	OpPayload bwd_payload = op_payload_dropout(self.dropout.p, self.dropout.seed, true);

	// Only input is the upstream gradient
	g.register_op(bwd_payload, { grad }, result);

	return result;
}

/// Backward for RMS normalization.
/// child 0 (input x): registers rms_norm with BACKWARD_X, inputs=[grad, x, weight]
/// child 1 (weight):  registers rms_norm with BACKWARD_WEIGHT, inputs=[grad, x]
fn tensor::Tensor? OpPayload.bwd_rms_norm(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor upstream, tensor::Tensor grad)
{
	tensor::Tensor x = args[0];
	tensor::Tensor weight = args[1];

	if (child_idx == 0)
	{
		// Gradient w.r.t. input x
		tensor::Tensor[3] resolve_args = { x, weight, grad };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = x.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, x.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_rms_norm(self.rms_norm.eps, operations::RmsOpType.BACKWARD_X);

		tensor::Tensor[3] op_args = { grad, x, weight };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
	else
	{
		// Gradient w.r.t. weight
		tensor::Tensor[2] resolve_args = { x, grad };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = weight.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, weight.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_rms_norm(self.rms_norm.eps, operations::RmsOpType.BACKWARD_WEIGHT);

		tensor::Tensor[2] op_args = { grad, x };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
}

/// Backward for selective scan.
/// child 0 (a_bar):   BACKWARD_A_BAR kernel, inputs=[grad, a_bar, h]
/// child 1 (b_bar_x): BACKWARD_B_BAR_X kernel, inputs=[grad, a_bar]
fn tensor::Tensor? OpPayload.bwd_selective_scan(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor upstream, tensor::Tensor grad)
{
	tensor::Tensor a_bar = args[0];
	tensor::Tensor b_bar_x = args[1];

	if (child_idx == 0)
	{
		// grad_a_bar: needs [grad, a_bar, h(=upstream)]
		tensor::Tensor[3] resolve_args = { a_bar, grad, upstream };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = a_bar.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, a_bar.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_selective_scan(operations::SelectiveScanOpType.BACKWARD_A_BAR);
		tensor::Tensor[3] op_args = { grad, a_bar, upstream };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
	else
	{
		// grad_b_bar_x: needs [grad, a_bar] (no h needed)
		tensor::Tensor[2] resolve_args = { b_bar_x, grad };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = b_bar_x.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, b_bar_x.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_selective_scan(operations::SelectiveScanOpType.BACKWARD_B_BAR_X);
		tensor::Tensor[2] op_args = { grad, a_bar };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
}

/// Backward for causal conv1d.
/// child 0 (x):      BACKWARD_X kernel, inputs=[grad, weight]
/// child 1 (weight):  BACKWARD_WEIGHT kernel, inputs=[grad, x]
/// child 2 (bias):    BACKWARD_BIAS kernel, inputs=[grad]
fn tensor::Tensor? OpPayload.bwd_causal_conv1d(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor upstream, tensor::Tensor grad)
{
	tensor::Tensor x = args[0];
	tensor::Tensor weight = args[1];
	tensor::Tensor bias = args[2];

	if (child_idx == 0)
	{
		// Gradient w.r.t. input x: needs [grad, weight]
		tensor::Tensor[2] resolve_args = { x, grad };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = x.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, x.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_causal_conv1d(operations::CausalConv1dOpType.BACKWARD_X);

		tensor::Tensor[2] op_args = { grad, weight };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
	else if (child_idx == 1)
	{
		// Gradient w.r.t. weight: needs [grad, x]
		tensor::Tensor[2] resolve_args = { weight, grad };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = weight.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, weight.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_causal_conv1d(operations::CausalConv1dOpType.BACKWARD_WEIGHT);

		tensor::Tensor[2] op_args = { grad, x };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
	else
	{
		// Gradient w.r.t. bias: needs [grad]
		tensor::Tensor[1] resolve_args = { grad };
		graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
		device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

		ulong[] shape_slice = bias.shape().slice();
		tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
			shape_slice, g, bias.dtype(), dev, null)!;
		tensor::Tensor result = tensor::tensor_from_core(result_core);

		OpPayload bwd_payload = op_payload_causal_conv1d(operations::CausalConv1dOpType.BACKWARD_BIAS);

		tensor::Tensor[1] op_args = { grad };
		g.register_op(bwd_payload, &op_args, result);

		return result;
	}
}

/// Backward for gather: scatter gradient back to input shape.
/// Only gradient for input (child 0), not indices (child 1).
fn tensor::Tensor? OpPayload.bwd_gather(&self, usz child_idx, tensor::Tensor[] args, tensor::Tensor upstream)
{
	if (child_idx != 0) return NO_GRADIENT~;

	tensor::Tensor input = args[0];
	tensor::Tensor indices = args[1];
	graph::Graph* g_upstream = upstream.graph();
	tensor_core::TensorCore* grad_core = g_upstream.get_gradient_core(upstream.core());
	if (grad_core == null) return NO_GRADIENT~;
	tensor::Tensor grad = tensor::tensor_from_core(grad_core);

	// Create scatter_add operation to accumulate gradients
	tensor::Tensor[3] resolve_args = { input, indices, grad };
	graph::Graph* g = tensor_ops::resolve_graph(&resolve_args, null);
	device_reference::DeviceReference dev = tensor_ops::resolve_device(&resolve_args, null)!;

	ulong[] shape_slice = input.shape().slice();
	tensor_core::TensorCore* result_core = tensor_ops::setup_result_core(
		shape_slice, g, grad.dtype(), dev, null)!;
	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Build output_shape array
	ulong[8] output_shape;
	for (usz d = 0; d < input.shape().len; d++)
	{
		output_shape[d] = input.shape().get(d);
	}

	OpPayload payload = { .scatter_add = {
		.tag = operations::OpTag.SCATTER_ADD,
		.params = { .dim = self.gather.params.dim },
		.output_shape = output_shape,
		.output_ndim = (char)input.shape().len,
		.hash = 0,
	}};

	tensor::Tensor[2] op_args = { grad, indices };
	g.register_op(payload, &op_args, result);

	return result;
}

/// Backward for cast: cast gradient back to source dtype.
fn tensor::Tensor? OpPayload.bwd_cast(&self, tensor::Tensor[] children, tensor::Tensor upstream)
{
	graph::Graph* g = upstream.graph();
	tensor_core::TensorCore* grad_core = g.get_gradient_core(upstream.core());
	if (grad_core == null) return NO_GRADIENT~;
	tensor::Tensor grad = tensor::tensor_from_core(grad_core);
	// Cast gradient back to the input's dtype (e.g., F32 grad → BF16 for mixed-precision training)
	if (grad.dtype() != children[0].dtype())
	{
		return tensor_ops::cast(grad, children[0].dtype());
	}
	return grad;
}

/// Backward for reshape: reshape gradient back to input shape.
fn tensor::Tensor? OpPayload.bwd_reshape(&self, tensor::Tensor[] children, tensor::Tensor grad)
{
	// children[0] is the input — reshape grad to match its shape
	return tensor_ops::reshape(grad, children[0].shape().slice());
}

// Index Helpers

/// Compare two index slices for equality.
fn bool indices_equal(char[] a, char[] b)
{
	if (a.len != b.len) return false;
	for (usz i = 0; i < a.len; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

faultdef NO_GRADIENT;
