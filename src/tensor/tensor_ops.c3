//! Tensor Operations
//!
//! Fundamental tensor operations: element-wise binary/unary, in-place,
//! unified einsum/reduce/expand/permute/map, shape ops, and select.
//!
//!
//! All operations follow the pattern:
//!   1. resolve_graph — determine which graph owns this computation
//!   2. resolve_device — all inputs must be on the same device
//!   3. setup_result_core — create or reinitialize result tensor
//!   4. build payload — create OpPayload for the operation
//!   5. register_op — link into the symbolic graph

module metaphor::tensor::tensor_ops;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::dense_core;
import metaphor::tensor::shape;
import metaphor::graph;
import metaphor::device::device_reference;
import metaphor::device::common::op_payload;
import metaphor::device::common::index_expr;
import metaphor::device::common::operations;
import metaphor::graph::op_cache;
import metaphor::rtti;
import std::io;

// Resolution Utilities

/// Resolves the graph for an operation.
///
/// Rules:
/// 1. Sink (output) tensor always wins — if provided, use its graph
/// 2. Otherwise, use the first tensor's graph
///
/// Note: ambiguous-graph is NOT checked per design decision.
fn graph::Graph* resolve_graph(tensor::Tensor[] args, tensor::Tensor* sink)
{
	if (sink != null)
	{
		return sink.graph();
	}
	assert(args.len > 0, "resolve_graph: no arguments and no sink");
	return args[0].graph();
}

/// Resolves device from input tensors.
/// All tensors must be on the same device.
fn device_reference::DeviceReference? resolve_device(
	tensor::Tensor[] args, tensor::Tensor* sink)
{
	device_reference::DeviceReference dev;
	bool have_dev = false;

	if (sink != null)
	{
		dev = sink.device();
		have_dev = true;
	}

	for (usz i = 0; i < args.len; i++)
	{
		device_reference::DeviceReference arg_dev = args[i].device();
		if (have_dev)
		{
			if (!dev.is_compatible(arg_dev))
			{
				return tensor::AMBIGUOUS_DEVICE~;
			}
		}
		else
		{
			dev = arg_dev;
			have_dev = true;
		}
	}

	return dev;
}

/// CSE lookup — returns cached tensor if found, null core otherwise.
fn tensor_core::TensorCore* cse_lookup(graph::Graph* g, tensor::Tensor* sink,
	tensor_core::TensorCore*[] inputs, operations::OpTag tag, ulong hash)
{
	if (sink != null || !g.flags.op_caching) return null;
	return g.cse_cache.lookup(inputs, tag, hash);
}

/// CSE register — stores result in cache.
fn void cse_register(graph::Graph* g, tensor::Tensor* sink,
	tensor_core::TensorCore*[] inputs, operations::OpTag tag, ulong hash,
	tensor_core::TensorCore* result)
{
	if (sink != null || !g.flags.op_caching) return;
	g.cse_cache.register(inputs, tag, hash, result);
}

/// Sets up a result TensorCore for an operation.
/// If sink is null, creates a new TensorCore.
/// If sink is provided, reinitializes it with the new shape.
fn tensor_core::TensorCore*? setup_result_core(
	ulong[] shape_slice,
	graph::Graph* g,
	rtti::Datatype dtype,
	device_reference::DeviceReference dev,
	tensor::Tensor* sink)
{
	if (sink != null)
	{
		tensor_core::TensorCore* core = sink.core();
		core.flags.alive = true;
		shape::Shape new_shape = shape::shape_init(shape_slice);
		*core.shape() = new_shape;
		core.state.dense.num_elements = new_shape.size();
		core.state.dense.strides = new_shape.strides().buffer;
		return core;
	}
	else
	{
		return g.create_tensor_core(shape_slice, dtype, dev);
	}
}

/// Determine result dtype based on operation.
/// TODO: Proper mixed-type support requires changes across cuTENSOR, hipTENSOR,
/// JIT codegen, and custom kernels. For now we special-case int+float promotion
/// (e.g. numel U64 ÷ F32 sum) and let the backend handle the cast. If the cast
/// is unsafe (e.g. large count into F16), it fails at the backend level.
fn rtti::Datatype result_dtype(
	rtti::Datatype a_dtype, rtti::Datatype b_dtype, operations::BinaryOp op)
{
	if (op.produces_bool()) return rtti::Datatype.BOOL;

	// Same type — fast path
	if (a_dtype == b_dtype) return a_dtype;

	// Mixed int+float: promote to the float type
	if (a_dtype.is_float() && b_dtype.is_int()) return a_dtype;
	if (a_dtype.is_int() && b_dtype.is_float()) return b_dtype;

	unreachable("Mixed-dtype operation detected — cast explicitly before combining");
}

// Binary Element-wise Operations (via combine_elemwise)

/// Element-wise addition: c = a + b.
fn tensor::Tensor? add(tensor::Tensor a, tensor::Tensor b, tensor::Tensor* sink = null)
{
	return combine_elemwise(a, b, operations::BinaryOp.ADD, sink);
}

/// Element-wise subtraction: c = a - b.
fn tensor::Tensor? sub(tensor::Tensor a, tensor::Tensor b, tensor::Tensor* sink = null)
{
	return combine_elemwise(a, b, operations::BinaryOp.SUB, sink);
}

/// Element-wise multiplication: c = a * b.
fn tensor::Tensor? mul(tensor::Tensor a, tensor::Tensor b, tensor::Tensor* sink = null)
{
	return combine_elemwise(a, b, operations::BinaryOp.MUL, sink);
}

/// Element-wise division: c = a / b.
fn tensor::Tensor? div(tensor::Tensor a, tensor::Tensor b, tensor::Tensor* sink = null)
{
	return combine_elemwise(a, b, operations::BinaryOp.DIV, sink);
}

/// Element-wise maximum: c = max(a, b).
fn tensor::Tensor? max(tensor::Tensor a, tensor::Tensor b, tensor::Tensor* sink = null)
{
	return combine_elemwise(a, b, operations::BinaryOp.MAX, sink);
}

/// Element-wise minimum: c = min(a, b).
fn tensor::Tensor? min(tensor::Tensor a, tensor::Tensor b, tensor::Tensor* sink = null)
{
	return combine_elemwise(a, b, operations::BinaryOp.MIN, sink);
}

// Comparison Operations (produce bool tensors)

fn tensor::Tensor? cmp_eq(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.EQL, null); }
fn tensor::Tensor? cmp_ne(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.NEQ, null); }
fn tensor::Tensor? cmp_lt(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.LT, null); }
fn tensor::Tensor? cmp_le(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.LTE, null); }
fn tensor::Tensor? cmp_gt(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.GT, null); }
fn tensor::Tensor? cmp_ge(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.GTE, null); }

// Binary Logical Operations (bool, bool -> bool)

fn tensor::Tensor? logical_and(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.AND, null); }
fn tensor::Tensor? logical_or(tensor::Tensor a, tensor::Tensor b)  { return combine_elemwise(a, b, operations::BinaryOp.OR, null); }
fn tensor::Tensor? logical_xor(tensor::Tensor a, tensor::Tensor b) { return combine_elemwise(a, b, operations::BinaryOp.XOR, null); }

// Internal: combine_elemwise — element-wise with broadcasting

fn tensor::Tensor? combine_elemwise(
	tensor::Tensor a, tensor::Tensor b,
	operations::BinaryOp op, tensor::Tensor* sink)
{
	graph::Graph* g = resolve_graph({ a, b }, sink);
	device_reference::DeviceReference dev = resolve_device({ a, b }, sink)!;

	tensor_core::TensorCore* cached = cse_lookup(g, sink, { a.core(), b.core() }, operations::OpTag.EINSUM, (ulong)op);
	if (cached != null) return tensor::tensor_from_core(cached);

	ulong[] a_shape = a.shape().slice();
	ulong[] b_shape = b.shape().slice();

	// Compute broadcast output shape
	ulong[8] out_buf = { [0..7] = 0 };
	ulong[] out_shape = compute_broadcast_shape(a_shape, b_shape, &out_buf)!;

	tensor_core::TensorCore* result_core = setup_result_core(out_shape,
		g, result_dtype(a.dtype(), b.dtype(), op), dev, sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Build broadcast expression
	index_expr::IndexExpr expr = build_elemwise_broadcast_expr(a_shape, b_shape, out_shape);

	op_payload::OpPayload payload = op_payload::op_payload_einsum(
		expr, op, operations::BinaryOp.ADD);

	g.register_op(payload, { a, b }, result);

	cse_register(g, sink, { a.core(), b.core() }, operations::OpTag.EINSUM, (ulong)op, result_core);

	return result;
}

// Broadcasting Helpers

/// Compute numpy-style broadcast output shape.
fn ulong[]? compute_broadcast_shape(ulong[] a, ulong[] b, ulong[8]* out)
{
	usz out_len = a.len > b.len ? a.len : b.len;
	if (out_len > 8) return tensor::INCOMPATIBLE_SHAPE~;

	for (usz i = 0; i < out_len; i++)
	{
		usz out_idx = out_len - 1 - i;
		ulong a_dim = i < a.len ? a[a.len - 1 - i] : 1;
		ulong b_dim = i < b.len ? b[b.len - 1 - i] : 1;

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
			return tensor::INCOMPATIBLE_SHAPE~;
		}
	}

	return (*out)[:out_len];
}

/// Build index expression for element-wise broadcasting.
fn index_expr::IndexExpr build_elemwise_broadcast_expr(
	ulong[] a_shape, ulong[] b_shape, ulong[] out_shape)
{
	index_expr::IndexExpr expr = {
		.input0 = { [0..7] = 0 },
		.input1 = { [0..7] = 0 },
		.input_lens = { 0, 0 },
		.num_inputs = 2,
		.output = { [0..7] = 0 },
		.output_len = 0,
		.has_ellipsis = false,
	};

	usz out_len = out_shape.len;
	char r = out_len < 8 ? (char)out_len : 8;

	for (usz i = 0; i < (usz)r; i++)
	{
		expr.output[i] = (char)('i' + i);
	}
	expr.output_len = r;

	// Input A indices (right-aligned)
	usz a_offset = out_len - a_shape.len;
	char a_idx = 0;
	for (usz j = 0; j < a_shape.len; j++)
	{
		expr.input0[a_idx] = (char)('i' + a_offset + j);
		a_idx += 1;
	}
	expr.input_lens[0] = a_idx;

	// Input B indices (right-aligned)
	usz b_offset = out_len - b_shape.len;
	char b_idx = 0;
	for (usz j = 0; j < b_shape.len; j++)
	{
		expr.input1[b_idx] = (char)('i' + b_offset + j);
		b_idx += 1;
	}
	expr.input_lens[1] = b_idx;

	return expr;
}

/// Build index expression for scalar broadcasting.
fn index_expr::IndexExpr build_broadcast_expr(usz a_rank, usz b_rank, usz out_rank)
{
	index_expr::IndexExpr expr = {
		.input0 = { [0..7] = 0 },
		.input1 = { [0..7] = 0 },
		.input_lens = { 0, 0 },
		.num_inputs = 2,
		.output = { [0..7] = 0 },
		.output_len = 0,
		.has_ellipsis = false,
	};

	char r = out_rank < 8 ? (char)out_rank : 8;
	for (usz i = 0; i < (usz)r; i++)
	{
		expr.output[i] = (char)('i' + i);
	}
	expr.output_len = r;

	char a_r = a_rank < 8 ? (char)a_rank : 8;
	char b_r = b_rank < 8 ? (char)b_rank : 8;

	for (usz i = 0; i < (usz)a_r; i++)
	{
		expr.input0[i] = (char)('i' + i);
	}
	expr.input_lens[0] = a_r;

	for (usz i = 0; i < (usz)b_r; i++)
	{
		expr.input1[i] = (char)('i' + i);
	}
	expr.input_lens[1] = b_r;

	return expr;
}

// Unary Element-wise Operations (via map_unary)

fn tensor::Tensor? cos(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.COS, sink); }
fn tensor::Tensor? sin(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.SIN, sink); }
fn tensor::Tensor? neg(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.NEG, sink); }
fn tensor::Tensor? abs(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.ABS, sink); }
fn tensor::Tensor? sqr(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.SQR, sink); }
fn tensor::Tensor? sqrt(tensor::Tensor a, tensor::Tensor* sink = null)      { return map_unary(a, operations::UnaryOp.SQRT, sink); }
fn tensor::Tensor? recip(tensor::Tensor a, tensor::Tensor* sink = null)     { return map_unary(a, operations::UnaryOp.RECIP, sink); }
fn tensor::Tensor? exp(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.EXP, sink); }
fn tensor::Tensor? log(tensor::Tensor a, tensor::Tensor* sink = null)       { return map_unary(a, operations::UnaryOp.LOG, sink); }
fn tensor::Tensor? tanh(tensor::Tensor a, tensor::Tensor* sink = null)      { return map_unary(a, operations::UnaryOp.TANH, sink); }
fn tensor::Tensor? relu(tensor::Tensor a, tensor::Tensor* sink = null)      { return map_unary(a, operations::UnaryOp.RELU, sink); }
fn tensor::Tensor? sigmoid(tensor::Tensor a, tensor::Tensor* sink = null)   { return map_unary(a, operations::UnaryOp.SIGMOID, sink); }
fn tensor::Tensor? heaviside(tensor::Tensor a, tensor::Tensor* sink = null) { return map_unary(a, operations::UnaryOp.HEAVISIDE, sink); }
fn tensor::Tensor? gelu(tensor::Tensor a, tensor::Tensor* sink = null)      { return map_unary(a, operations::UnaryOp.GELU, sink); }
fn tensor::Tensor? silu(tensor::Tensor a, tensor::Tensor* sink = null)      { return map_unary(a, operations::UnaryOp.SILU, sink); }
fn tensor::Tensor? softplus(tensor::Tensor a, tensor::Tensor* sink = null)  { return map_unary(a, operations::UnaryOp.SOFTPLUS, sink); }

/// Element-wise logical NOT.
fn tensor::Tensor? logical_not(tensor::Tensor a) { return map_unary(a, operations::UnaryOp.NOT, null); }

/// Internal: unary map operation.
fn tensor::Tensor? map_unary(
	tensor::Tensor a, operations::UnaryOp op, tensor::Tensor* sink)
{
	graph::Graph* g = resolve_graph({ a }, sink);
	device_reference::DeviceReference dev = resolve_device({ a }, sink)!;

	tensor_core::TensorCore* cached = cse_lookup(g, sink, { a.core() }, operations::OpTag.MAP, (ulong)op);
	if (cached != null) return tensor::tensor_from_core(cached);

	tensor_core::TensorCore* rc = setup_result_core(a.shape().slice(), g, a.dtype(), dev, sink)!;
	tensor::Tensor result = tensor::tensor_from_core(rc);
	g.register_op(op_payload::op_payload_map(op), { a }, result);

	cse_register(g, sink, { a.core() }, operations::OpTag.MAP, (ulong)op, rc);
	return result;
}


// Unified Operations

/// Options for einsum operation.
struct EinsumOpts
{
	operations::BinaryOp inner;
	operations::BinaryOp outer;
	double alpha;
	double beta;
	tensor::Tensor* sink;
}

const EinsumOpts DEFAULT_EINSUM_OPTS = {
	.inner = operations::BinaryOp.MUL,
	.outer = operations::BinaryOp.ADD,
	.alpha = 1.0,
	.beta = 0.0,
	.sink = null,
};

/// Options for reduce operation.
struct ReduceOpts
{
	operations::BinaryOp op;
	tensor::Tensor* sink;
}

const ReduceOpts DEFAULT_REDUCE_OPTS = {
	.op = operations::BinaryOp.ADD,
	.sink = null,
};

/// Options for expand operation.
struct ExpandOpts
{
	ulong[] sizes;
	tensor::Tensor* sink;
}

/// Options for map operation (unified).
struct MapOpts
{
	operations::UnaryOp unary;
	double beta;
	tensor::Tensor* sink;
}

/// Options for permute operation.
struct PermuteOpts
{
	tensor::Tensor* sink;
}

/// Binary operation with index pattern (einsum).
fn tensor::Tensor? einsum(
	tensor::Tensor a, tensor::Tensor b,
	index_expr::IndexExpr* expr,
	EinsumOpts opts = DEFAULT_EINSUM_OPTS)
{
	graph::Graph* g = resolve_graph({ a, b }, opts.sink);
	device_reference::DeviceReference dev = resolve_device({ a, b }, opts.sink)!;

	// CSE lookup — hash expr + inner + outer
	ulong ph = op_cache::hash_bytes(((char*)expr)[:$sizeof(index_expr::IndexExpr)]);
	ph = op_cache::mix(ph, (ulong)opts.inner);
	ph = op_cache::mix(ph, (ulong)opts.outer);
	tensor_core::TensorCore* cached = cse_lookup(g, opts.sink, { a.core(), b.core() }, operations::OpTag.EINSUM, ph);
	if (cached != null) return tensor::tensor_from_core(cached);

	shape::Shape output_shape = compute_einsum_shape(expr, a.shape(), b.shape())!;

	tensor_core::TensorCore* result_core = setup_result_core(output_shape.slice(),
		g, result_dtype(a.dtype(), b.dtype(), opts.inner), dev, opts.sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_einsum(
		*expr, opts.inner, opts.outer);
	payload.einsum.alpha = opts.alpha;
	payload.einsum.beta = opts.beta;

	g.register_op(payload, { a, b }, result);

	cse_register(g, opts.sink, { a.core(), b.core() }, operations::OpTag.EINSUM, ph, result_core);

	return result;
}

/// Unary contraction over specified indices (reduce).
fn tensor::Tensor? reduce(
	tensor::Tensor a, index_expr::IndexExpr* expr,
	ReduceOpts opts = DEFAULT_REDUCE_OPTS)
{
	graph::Graph* g = resolve_graph({ a }, opts.sink);
	device_reference::DeviceReference dev = resolve_device({ a }, opts.sink)!;

	// CSE lookup — hash expr + reduce op
	ulong ph = op_cache::hash_bytes(((char*)expr)[:$sizeof(index_expr::IndexExpr)]);
	ph = op_cache::mix(ph, (ulong)opts.op);
	tensor_core::TensorCore* cached = cse_lookup(g, opts.sink, { a.core() }, operations::OpTag.REDUCE, ph);
	if (cached != null) return tensor::tensor_from_core(cached);

	shape::Shape output_shape = compute_reduce_shape(expr, a.shape())!;

	tensor_core::TensorCore* result_core = setup_result_core(output_shape.slice(),
		g, a.dtype(), dev, opts.sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_reduce(*expr, opts.op);

	g.register_op(payload, { a }, result);

	cse_register(g, opts.sink, { a.core() }, operations::OpTag.REDUCE, ph, result_core);

	return result;
}

/// Unary expansion (inverse of reduce).
fn tensor::Tensor? expand(
	tensor::Tensor a, index_expr::IndexExpr* expr, ExpandOpts opts)
{
	graph::Graph* g = resolve_graph({ a }, opts.sink);
	device_reference::DeviceReference dev = resolve_device({ a }, opts.sink)!;

	// CSE lookup — hash expr + sizes
	ulong ph = op_cache::hash_bytes(((char*)expr)[:$sizeof(index_expr::IndexExpr)]);
	ulong[8] sizes_buf;
	char num = opts.sizes.len < 8 ? (char)opts.sizes.len : 8;
	for (usz i = 0; i < (usz)num; i++)
	{
		sizes_buf[i] = opts.sizes[i];
	}
	ph = op_cache::mix(ph,
		op_cache::hash_bytes(((char*)&sizes_buf)[:num * ulong.sizeof]));
	tensor_core::TensorCore* cached = cse_lookup(g, opts.sink, { a.core() }, operations::OpTag.EXPAND, ph);
	if (cached != null) return tensor::tensor_from_core(cached);

	shape::Shape output_shape = compute_expand_shape(expr, a.shape(), opts.sizes)!;

	tensor_core::TensorCore* result_core = setup_result_core(output_shape.slice(),
		g, a.dtype(), dev, opts.sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	operations::ExpandPayload ep = {
		.tag = operations::OpTag.EXPAND,
		.expr = *expr,
		.num_new = 0,
	};
	char num_new = opts.sizes.len < 8 ? (char)opts.sizes.len : 8;
	for (usz i = 0; i < (usz)num_new; i++)
	{
		ep.new_sizes[i] = opts.sizes[i];
	}
	ep.num_new = num_new;

	op_payload::OpPayload payload = { .expand = ep };
	g.register_op(payload, { a }, result);

	cse_register(g, opts.sink, { a.core() }, operations::OpTag.EXPAND, ph, result_core);

	return result;
}

/// Dimension reordering (transpose/permute).
fn tensor::Tensor? permute(
	tensor::Tensor a, index_expr::IndexExpr* expr, PermuteOpts opts = {})
{
	graph::Graph* g = resolve_graph({ a }, opts.sink);
	device_reference::DeviceReference dev = resolve_device({ a }, opts.sink)!;

	// CSE lookup — hash expr
	ulong ph = op_cache::hash_bytes(((char*)expr)[:$sizeof(index_expr::IndexExpr)]);
	tensor_core::TensorCore* cached = cse_lookup(g, opts.sink, { a.core() }, operations::OpTag.PERMUTE, ph);
	if (cached != null) return tensor::tensor_from_core(cached);

	shape::Shape output_shape = compute_permute_shape(expr, a.shape())!;

	tensor_core::TensorCore* result_core = setup_result_core(output_shape.slice(),
		g, a.dtype(), dev, opts.sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_permute(*expr);
	g.register_op(payload, { a }, result);

	// Permute is a compilation boundary — the changed traversal pattern
	// cannot be fused with adjacent ops, and the backward pass needs its own
	// execution unit to correctly propagate gradients through the permutation.
	result_core.flags.persistence.promote(tensor_core::Persistence.UNSTABLE);

	cse_register(g, opts.sink, { a.core() }, operations::OpTag.PERMUTE, ph, result_core);

	return result;
}

/// Unary transform preserving shape (map with options).
fn tensor::Tensor? map_with_opts(tensor::Tensor a, MapOpts opts)
{
	graph::Graph* g = resolve_graph({ a }, opts.sink);
	device_reference::DeviceReference dev = resolve_device({ a }, opts.sink)!;

	// CSE lookup — hash unary op + beta
	ulong ph = op_cache::mix((ulong)opts.unary,
		op_cache::hash_bytes(((char*)&opts.beta)[:double.sizeof]));
	tensor_core::TensorCore* cached = cse_lookup(g, opts.sink, { a.core() }, operations::OpTag.MAP, ph);
	if (cached != null) return tensor::tensor_from_core(cached);

	tensor_core::TensorCore* result_core = setup_result_core(
		a.shape().slice(), g, a.dtype(), dev, opts.sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_map(opts.unary);
	payload.map_payload.beta = opts.beta;

	g.register_op(payload, { a }, result);

	cse_register(g, opts.sink, { a.core() }, operations::OpTag.MAP, ph, result_core);

	return result;
}

// Shape Operations

/// Extract dimension size as a u64 scalar tensor.
fn tensor::Tensor? dim_op(tensor::Tensor a, usz axis)
{
	tensor_core::TensorCore* a_core = a.core();
	graph::Graph* g = (graph::Graph*)a_core.graph;
	ulong[1] scalar_shape = { 1 };

	tensor_core::TensorCore* result_core = g.create_tensor_core(
		&scalar_shape, U64, a_core.device);

	tensor::Tensor result = tensor::tensor_from_core(result_core);
	result_core.flags._requires_grad = false;

	op_payload::OpPayload payload = op_payload::op_payload_dim(axis);
	g.register_op(payload, { a }, result);

	return result;
}

/// Returns the total number of elements as a u64 scalar tensor.
fn tensor::Tensor? numel_op(tensor::Tensor a)
{
	tensor_core::TensorCore* a_core = a.core();
	graph::Graph* g = (graph::Graph*)a_core.graph;
	ulong[1] scalar_shape = { 1 };

	tensor_core::TensorCore* result_core = g.create_tensor_core(
		&scalar_shape, U64, a_core.device);

	tensor::Tensor result = tensor::tensor_from_core(result_core);
	result_core.flags._requires_grad = false;

	op_payload::OpPayload payload = op_payload::op_payload_numel();
	g.register_op(payload, { a }, result);

	return result;
}

// Select (ternary)

/// Ternary selection: result[i] = condition[i] ? positive[i] : negative[i]
fn tensor::Tensor? select(
	tensor::Tensor condition, tensor::Tensor positive,
	tensor::Tensor negative, tensor::Tensor* sink = null)
{
	if (condition.dtype() != rtti::Datatype.BOOL)
	{
		return tensor::INCOMPATIBLE_DTYPE~;
	}

	graph::Graph* g = resolve_graph({ condition, positive, negative }, sink);
	device_reference::DeviceReference dev = resolve_device(
		{ condition, positive, negative }, sink)!;

	tensor_core::TensorCore* result_core = setup_result_core(positive.shape().slice(),
		g, positive.dtype(), dev, sink)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_select();
	g.register_op(payload, { condition, positive, negative }, result);

	return result;
}

// Shape Computation Helpers

/// Compute output shape for einsum operation.
fn shape::Shape? compute_einsum_shape(
	index_expr::IndexExpr* expr, shape::Shape* a_shape, shape::Shape* b_shape)
{
	ulong[8] dims;
	usz dim_count = 0;

	if (expr.has_ellipsis)
	{
		usz a_explicit = expr.get_input(0).len;
		usz b_explicit = expr.get_input(1).len;

		usz a_ellipsis_len = a_shape.len - a_explicit;
		usz b_ellipsis_len = b_shape.len - b_explicit;

		usz max_ellipsis = a_ellipsis_len > b_ellipsis_len
			? a_ellipsis_len : b_ellipsis_len;

		for (usz i = 0; i < max_ellipsis; i++)
		{
			long a_idx = (long)i - (long)(max_ellipsis - a_ellipsis_len);
			long b_idx = (long)i - (long)(max_ellipsis - b_ellipsis_len);

			ulong a_dim = a_idx >= 0 ? a_shape.get((usz)a_idx) : 1;
			ulong b_dim = b_idx >= 0 ? b_shape.get((usz)b_idx) : 1;

			if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
			dims[dim_count] = a_dim > b_dim ? a_dim : b_dim;
			dim_count += 1;
		}

		char[] out_indices = expr.get_output();
		for (usz i = 0; i < out_indices.len; i++)
		{
			if (try pos = index_expr::index_in_pattern(
					out_indices[i], expr.get_input(0)))
			{
				dims[dim_count] = a_shape.get(a_ellipsis_len + pos);
			}
			else if (try pos2 = index_expr::index_in_pattern(
					out_indices[i], expr.get_input(1)))
			{
				dims[dim_count] = b_shape.get(b_ellipsis_len + pos2);
			}
			else
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
			dim_count += 1;
		}
	}
	else
	{
		// Validate: without ellipsis, pattern length must match tensor rank
		char[] in0_pat = expr.get_input(0);
		char[] in1_pat = expr.get_input(1);
		if (in0_pat.len != a_shape.len || in1_pat.len != b_shape.len)
		{
			io::eprintfn("[EINSUM MISMATCH] in0: pat_len=%d rank=%d, in1: pat_len=%d rank=%d",
				in0_pat.len, a_shape.len, in1_pat.len, b_shape.len);
			io::eprintf("[EINSUM MISMATCH] in0 shape=[");
			for (usz si = 0; si < a_shape.len; si++)
			{
				if (si > 0) io::eprintf(",");
				io::eprintf("%d", a_shape.get(si));
			}
			io::eprintf("] in1 shape=[");
			for (usz si = 0; si < b_shape.len; si++)
			{
				if (si > 0) io::eprintf(",");
				io::eprintf("%d", b_shape.get(si));
			}
			io::eprintfn("]");
		}
		assert(in0_pat.len == a_shape.len,
			"einsum: input0 pattern length (%d) != tensor rank (%d) -- missing ellipsis?",
			in0_pat.len, a_shape.len);
		assert(in1_pat.len == b_shape.len,
			"einsum: input1 pattern length (%d) != tensor rank (%d) -- missing ellipsis?",
			in1_pat.len, b_shape.len);

		char[] out_indices = expr.get_output();
		for (usz i = 0; i < out_indices.len; i++)
		{
			usz? a_pos = index_expr::index_in_pattern(
					out_indices[i], expr.get_input(0));
			usz? b_pos = index_expr::index_in_pattern(
					out_indices[i], expr.get_input(1));

			if (try ap = a_pos)
			{
				ulong a_dim = a_shape.get(ap);
				if (try bp = b_pos)
				{
					// Index in both inputs — take max for broadcasting
					ulong b_dim = b_shape.get(bp);
					dims[i] = a_dim > b_dim ? a_dim : b_dim;
				}
				else
				{
					dims[i] = a_dim;
				}
			}
			else if (try bp = b_pos)
			{
				dims[i] = b_shape.get(bp);
			}
			else
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
		}
		dim_count = out_indices.len;
	}

	return shape::shape_init(dims[:dim_count]);
}

/// Compute output shape for reduce operation.
fn shape::Shape? compute_reduce_shape(
	index_expr::IndexExpr* expr, shape::Shape* a_shape)
{
	ulong[8] dims;
	char[] out_indices = expr.get_output();
	char[] in_indices = expr.get_input(0);
	usz dim_count = 0;

	if (expr.has_ellipsis)
	{
		usz ellipsis_len = a_shape.len - in_indices.len;

		// Ellipsis dims pass through to output
		for (usz i = 0; i < ellipsis_len; i++)
		{
			dims[dim_count] = a_shape.get(i);
			dim_count++;
		}

		// Explicit output indices — look up in input pattern, offset by ellipsis
		for (usz i = 0; i < out_indices.len; i++)
		{
			if (try pos = index_expr::index_in_pattern(out_indices[i], in_indices))
			{
				dims[dim_count] = a_shape.get(ellipsis_len + pos);
			}
			else
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
			dim_count++;
		}
	}
	else
	{
		for (usz i = 0; i < out_indices.len; i++)
		{
			if (try pos = index_expr::index_in_pattern(out_indices[i], in_indices))
			{
				dims[dim_count] = a_shape.get(pos);
			}
			else
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
			dim_count++;
		}
	}

	if (dim_count == 0)
	{
		ulong[] empty;
		return shape::shape_init(empty);
	}

	return shape::shape_init(dims[:dim_count]);
}

/// Compute output shape for expand operation.
fn shape::Shape? compute_expand_shape(
	index_expr::IndexExpr* expr, shape::Shape* a_shape, ulong[] new_sizes)
{
	ulong[8] dims;
	char[] out_indices = expr.get_output();
	char[] in_indices = expr.get_input(0);
	usz new_idx = 0;
	usz dim_count = 0;

	if (expr.has_ellipsis)
	{
		usz ellipsis_len = a_shape.len - in_indices.len;

		// Ellipsis dims pass through
		for (usz i = 0; i < ellipsis_len; i++)
		{
			dims[dim_count] = a_shape.get(i);
			dim_count++;
		}

		// Explicit output indices
		for (usz i = 0; i < out_indices.len; i++)
		{
			if (try pos = index_expr::index_in_pattern(out_indices[i], in_indices))
			{
				dims[dim_count] = a_shape.get(ellipsis_len + pos);
			}
			else
			{
				if (new_idx >= new_sizes.len) return tensor::INCOMPATIBLE_SHAPE~;
				dims[dim_count] = new_sizes[new_idx];
				new_idx++;
			}
			dim_count++;
		}
	}
	else
	{
		for (usz i = 0; i < out_indices.len; i++)
		{
			if (try pos = index_expr::index_in_pattern(out_indices[i], in_indices))
			{
				dims[dim_count] = a_shape.get(pos);
			}
			else
			{
				if (new_idx >= new_sizes.len) return tensor::INCOMPATIBLE_SHAPE~;
				dims[dim_count] = new_sizes[new_idx];
				new_idx++;
			}
			dim_count++;
		}
	}

	return shape::shape_init(dims[:dim_count]);
}

/// Compute output shape for permute from expression and input shape.
fn shape::Shape? compute_permute_shape(
	index_expr::IndexExpr* expr, shape::Shape* input_shape)
{
	char[] input_indices = expr.get_input(0);
	char[] output_indices = expr.get_output();

	if (input_indices.len != output_indices.len
		|| input_indices.len != input_shape.len)
	{
		return tensor::INCOMPATIBLE_SHAPE~;
	}

	ulong[8] dims;
	for (usz out_pos = 0; out_pos < output_indices.len; out_pos++)
	{
		bool found = false;
		for (usz in_pos = 0; in_pos < input_indices.len; in_pos++)
		{
			if (input_indices[in_pos] == output_indices[out_pos])
			{
				dims[out_pos] = input_shape.get(in_pos);
				found = true;
				break;
			}
		}
		if (!found)
		{
			return tensor::INCOMPATIBLE_SHAPE~;
		}
	}

	return shape::shape_init(dims[:output_indices.len]);
}

// Gather (index select / embedding lookup)

/// Gather elements from input along dim using indices.
///
/// For input [A, B, C] and indices [X, Y] at dim=1:
///   output shape is [A, X, Y, C]
///   output[a, x, y, c] = input[a, indices[x, y], c]
fn tensor::Tensor? gather(tensor::Tensor input, usz gather_dim, tensor::Tensor indices)
{
	graph::Graph* g = resolve_graph({ input, indices }, null);
	device_reference::DeviceReference dev = resolve_device({ input, indices }, null)!;

	shape::Shape* input_shape = input.shape();
	shape::Shape* indices_shape = indices.shape();

	// Validate dimension
	if (gather_dim >= input_shape.len)
	{
		return tensor::INCOMPATIBLE_SHAPE~;
	}

	// Compute output shape: input shape with gather_dim replaced by indices shape
	ulong[8] output_dims;
	usz out_idx = 0;

	// Dimensions before gather_dim
	for (usz d = 0; d < gather_dim; d++)
	{
		output_dims[out_idx] = input_shape.get(d);
		out_idx += 1;
	}

	// Insert indices shape
	for (usz d = 0; d < indices_shape.len; d++)
	{
		output_dims[out_idx] = indices_shape.get(d);
		out_idx += 1;
	}

	// Dimensions after gather_dim
	for (usz d = gather_dim + 1; d < input_shape.len; d++)
	{
		output_dims[out_idx] = input_shape.get(d);
		out_idx += 1;
	}

	tensor_core::TensorCore* result_core = setup_result_core(
		output_dims[:out_idx], g, input.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_gather(
		{ .dim = (char)gather_dim });
	g.register_op(payload, { input, indices }, result);

	return result;
}

// Scatter-add (backward for gather)

/// Scatter-add operation: accumulates values from src into output at positions
/// specified by indices. This is the backward pass for gather.
fn tensor::Tensor? scatter_add(
	tensor::Tensor input, tensor::Tensor indices,
	tensor::Tensor grad, usz scatter_dim)
{
	graph::Graph* g = resolve_graph({ input, indices, grad }, null);
	device_reference::DeviceReference dev = resolve_device({ input, indices, grad }, null)!;

	shape::Shape* input_shape = input.shape();

	ulong[8] output_shape;
	for (usz d = 0; d < input_shape.len; d++)
	{
		output_shape[d] = input_shape.get(d);
	}

	tensor_core::TensorCore* result_core = setup_result_core(
		input_shape.slice(), g, grad.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_scatter_add(
		{ .dim = (char)scatter_dim });
	payload.scatter_add.output_shape = output_shape;
	payload.scatter_add.output_ndim = (char)input_shape.len;

	g.register_op(payload, { grad, indices }, result);

	return result;
}

// Slice Operations

/// Create a slice view of a tensor.
/// ranges: slice specification for each dimension.
fn tensor::Tensor? slice(tensor::Tensor source, operations::SliceRange[] ranges, usz offset = 0)
{
	graph::Graph* g = resolve_graph({ source }, null);
	device_reference::DeviceReference dev = resolve_device({ source }, null)!;

	shape::Shape* source_shape = source.shape();

	if (offset + ranges.len > source_shape.len)
	{
		return tensor::INCOMPATIBLE_SHAPE~;
	}

	ulong[8] output_shape;
	operations::SlicePayload sp = {
		.tag = operations::OpTag.SLICE,
		.ndim = (char)source_shape.len,
		.is_scatter = false,
		.hash = 0,
	};

	for (usz i = 0; i < source_shape.len; i++)
	{
		ulong dim_size = source_shape.get(i);
		usz range_idx = i - offset;

		if (i >= offset && range_idx < ranges.len)
		{
			ulong start = ranges[range_idx].start;
			ulong end = ranges[range_idx].end == 0 ? dim_size : ranges[range_idx].end;
			long step = ranges[range_idx].step == 0 ? 1 : ranges[range_idx].step;

			if (start >= dim_size || end > dim_size || start >= end)
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}

			sp.offsets[i] = start;
			sp.steps[i] = step;

			// Compute output size for this dimension
			ulong range = end - start;
			ulong abs_step = step > 0 ? (ulong)step : (ulong)(-step);
			output_shape[i] = (range + abs_step - 1) / abs_step;
		}
		else
		{
			// Dimension not specified — take full extent
			sp.offsets[i] = 0;
			sp.steps[i] = 1;
			output_shape[i] = dim_size;
		}
	}

	tensor_core::TensorCore* result_core = setup_result_core(
		output_shape[:source_shape.len], g, source.dtype(), dev, null)!;

	// Set view strides: slice_stride[d] = source_stride[d] * step[d]
	shape::Strides source_strides = source.core().shape().strides();
	for (usz d = 0; d < source_shape.len; d++)
	{
		long step = sp.steps[d];
		ulong abs_step = step > 0 ? (ulong)step : (ulong)(-step);
		result_core.state.dense.strides[d] = source_strides.get(d) * abs_step;
	}

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_slice(sp);
	g.register_op(payload, { source }, result);

	return result;
}

/// Scatter a gradient back into a zero tensor at the slice positions.
/// Used for backward pass of slice operations.
fn tensor::Tensor? slice_scatter(
	tensor::Tensor source, tensor::Tensor grad, operations::SlicePayload sp)
{
	graph::Graph* g = resolve_graph({ grad }, null);
	device_reference::DeviceReference dev = resolve_device({ grad }, null)!;

	// Create result with source's shape (zeros will be filled in by executor)
	tensor_core::TensorCore* result_core = setup_result_core(
		source.shape().slice(), g, grad.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	// Build scatter payload — same as forward slice but is_scatter = true
	operations::SlicePayload scatter_payload = sp;
	scatter_payload.hash = 0;
	scatter_payload.is_scatter = true;

	// Register as scatter op — only grad is a child (source is just for shape)
	op_payload::OpPayload payload = op_payload::op_payload_slice(scatter_payload);
	g.register_op(payload, { grad }, result);

	return result;
}

// Concatenation and Stacking

/// Concatenate tensors along an existing dimension.
///
/// All input tensors must have the same shape except along the concat dimension.
/// Output shape equals input shape, with concat dimension summed.
fn tensor::Tensor? cat(tensor::Tensor[] inputs, char concat_dim)
{
	if (inputs.len < 2) return tensor::INCOMPATIBLE_SHAPE~;
	if (inputs.len > 8) return tensor::INCOMPATIBLE_SHAPE~;

	graph::Graph* g = resolve_graph(inputs, null);
	device_reference::DeviceReference dev = resolve_device(inputs, null)!;

	shape::Shape* first_shape = inputs[0].shape();
	usz ndim = first_shape.len;

	if ((usz)concat_dim >= ndim) return tensor::INCOMPATIBLE_SHAPE~;

	// Validate shapes and compute cumulative offsets
	operations::ConcatPayload cp = {
		.tag = operations::OpTag.CONCAT,
		.dim = concat_dim,
		.num_inputs = (char)inputs.len,
		.hash = 0,
	};
	cp.offsets[0] = 0;

	ulong total_concat_size = 0;

	for (usz i = 0; i < inputs.len; i++)
	{
		shape::Shape* s = inputs[i].shape();
		if (s.len != ndim) return tensor::INCOMPATIBLE_SHAPE~;

		// Check all dimensions except concat dim match
		for (usz d = 0; d < ndim; d++)
		{
			if (d != (usz)concat_dim && s.get(d) != first_shape.get(d))
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
		}

		total_concat_size += s.get((usz)concat_dim);
		cp.offsets[i + 1] = total_concat_size;
	}

	// Compute output shape
	ulong[8] output_shape;
	for (usz d = 0; d < ndim; d++)
	{
		if (d == (usz)concat_dim)
		{
			output_shape[d] = total_concat_size;
		}
		else
		{
			output_shape[d] = first_shape.get(d);
		}
	}

	tensor_core::TensorCore* result_core = setup_result_core(
		output_shape[:ndim], g, inputs[0].dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_concat(cp);
	g.register_op(payload, inputs, result);

	return result;
}

/// Stack tensors along a new dimension.
///
/// All input tensors must have identical shapes.
/// Output has one more dimension at position `dim` with size = num_inputs.
fn tensor::Tensor? stack(tensor::Tensor[] inputs, char stack_dim)
{
	if (inputs.len < 2) return tensor::INCOMPATIBLE_SHAPE~;
	if (inputs.len > 8) return tensor::INCOMPATIBLE_SHAPE~;

	graph::Graph* g = resolve_graph(inputs, null);
	device_reference::DeviceReference dev = resolve_device(inputs, null)!;

	shape::Shape* first_shape = inputs[0].shape();
	usz ndim = first_shape.len;

	if ((usz)stack_dim > ndim) return tensor::INCOMPATIBLE_SHAPE~;

	// Validate all shapes are identical
	for (usz i = 1; i < inputs.len; i++)
	{
		shape::Shape* s = inputs[i].shape();
		if (s.len != ndim) return tensor::INCOMPATIBLE_SHAPE~;

		for (usz d = 0; d < ndim; d++)
		{
			if (s.get(d) != first_shape.get(d))
			{
				return tensor::INCOMPATIBLE_SHAPE~;
			}
		}
	}

	// Compute output shape: insert new dimension at stack_dim with size = num_inputs
	ulong[8] output_shape;
	for (usz d = 0; d < (usz)stack_dim; d++)
	{
		output_shape[d] = first_shape.get(d);
	}
	output_shape[(usz)stack_dim] = inputs.len;
	for (usz d = (usz)stack_dim; d < ndim; d++)
	{
		output_shape[d + 1] = first_shape.get(d);
	}

	operations::StackPayload sp = {
		.tag = operations::OpTag.STACK,
		.dim = stack_dim,
		.num_inputs = (char)inputs.len,
		.hash = 0,
	};

	tensor_core::TensorCore* result_core = setup_result_core(
		output_shape[:ndim + 1], g, inputs[0].dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_stack(sp);
	g.register_op(payload, inputs, result);

	return result;
}

// Squeeze

/// Remove a dimension of size 1 from a tensor.
/// Zero-copy: delegates to reshape since the dimension must be size 1.
fn tensor::Tensor? squeeze(tensor::Tensor input, usz squeeze_dim)
{
	usz ndim = input.shape().len;
	assert(squeeze_dim < ndim, "squeeze: dim out of range");
	assert(input.shape().get(squeeze_dim) == 1, "squeeze: dim size must be 1");

	ulong[8] new_shape;
	usz out_idx = 0;
	for (usz d = 0; d < ndim; d++)
	{
		if (d != squeeze_dim)
		{
			new_shape[out_idx] = input.shape().get(d);
			out_idx++;
		}
	}

	return reshape(input, new_shape[:out_idx]);
}

// Cast Operations

/// Cast tensor to a different dtype.
///
/// This is a boundary operation — forces materialization.
fn tensor::Tensor? cast(tensor::Tensor input, rtti::Datatype target_dtype)
{
	// If already the target dtype, return identity
	if (input.dtype() == target_dtype) return input;

	graph::Graph* g = resolve_graph({ input }, null);
	device_reference::DeviceReference dev = resolve_device({ input }, null)!;

	tensor_core::TensorCore* result_core = setup_result_core(
		input.shape().slice(), g, target_dtype, dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_cast(
		(char)input.dtype().ordinal, (char)target_dtype.ordinal);
	g.register_op(payload, { input }, result);

	return result;
}

// Transfer — cross-device tensor movement

/// Move a tensor to a different device, returning a new tensor on target_dev.
/// If already on the target device, returns the input unchanged (no-op).
fn tensor::Tensor? to(tensor::Tensor input, device_reference::DeviceReference target_dev)
{
	// No-op if already on compatible device
	device_reference::DeviceReference src_dev = input.device();
	if (src_dev.is_compatible(target_dev)) return input;

	device_reference::TransferDirection direction;
	if (src_dev.is_host()) { direction = HTOD; }
	else if (target_dev.is_host()) { direction = DTOH; }
	else { direction = DTOD; }

	graph::Graph* g = resolve_graph({ input }, null);

	// Create result on the target device (not setup_result_core — cross-device)
	tensor_core::TensorCore* result_core = g.create_tensor_core(
		input.shape().slice(), input.dtype(), target_dev);
	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_transfer(direction);
	g.register_op(payload, { input }, result);

	return result;
}

// Reshape — zero-copy shape reinterpretation

/// Reshape a tensor to a new shape with the same total number of elements.
/// Zero-copy: the result shares the source's data buffer via refcounting.
fn tensor::Tensor? reshape(tensor::Tensor input, ulong[] new_shape)
{
	// Validate element count matches
	usz old_size = input.shape().size();
	usz new_size = 1;
	for (usz i = 0; i < new_shape.len; i++)
	{
		new_size *= (usz)new_shape[i];
	}
	assert(old_size == new_size,
		"reshape: element count mismatch (old=%d, new=%d)", old_size, new_size);

	graph::Graph* g = resolve_graph({ input }, null);
	device_reference::DeviceReference dev = resolve_device({ input }, null)!;

	tensor_core::TensorCore* result_core = setup_result_core(
		new_shape, g, input.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_reshape();
	g.register_op(payload, { input }, result);

	return result;
}

// Convolution Operations

/// 2D convolution.
///
/// Input:  [N, C_in, H, W]
/// Weight: [C_out, C_in/groups, kH, kW]
/// Output: [N, C_out, H_out, W_out]
fn tensor::Tensor? conv2d(
	tensor::Tensor input, tensor::Tensor weight,
	operations::ConvParams params = operations::DEFAULT_CONV_PARAMS)
{
	graph::Graph* g = resolve_graph({ input, weight }, null);
	device_reference::DeviceReference dev = resolve_device({ input, weight }, null)!;

	shape::Shape* input_shape = input.shape();
	shape::Shape* weight_shape = weight.shape();

	// Validate shapes
	if (input_shape.len != 4 || weight_shape.len != 4)
	{
		return tensor::INCOMPATIBLE_SHAPE~;
	}

	// Compute output shape
	ulong n_batch = input_shape.get(0);
	ulong c_out = weight_shape.get(0);
	ulong h_in = input_shape.get(2);
	ulong w_in = input_shape.get(3);
	ulong kh = weight_shape.get(2);
	ulong kw = weight_shape.get(3);

	ulong h_out = (h_in + 2 * params.padding[0]
		- params.dilation[0] * (kh - 1) - 1) / params.stride[0] + 1;
	ulong w_out = (w_in + 2 * params.padding[1]
		- params.dilation[1] * (kw - 1) - 1) / params.stride[1] + 1;

	ulong[4] output_shape = { n_batch, c_out, h_out, w_out };

	tensor_core::TensorCore* result_core = setup_result_core(
		&output_shape, g, input.dtype(), dev, null)!;

	tensor::Tensor result = tensor::tensor_from_core(result_core);

	op_payload::OpPayload payload = op_payload::op_payload_conv(params);
	g.register_op(payload, { input, weight }, result);

	return result;
}
