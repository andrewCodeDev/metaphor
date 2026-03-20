//! CAS Mapper — symbolic graph → IREntry translation
//!
//! Walks the symbolic graph from a target tensor backward, building CAS
//! expressions that the device compiler consumes. Returns IREntries in
//! dependency order (deepest first).
//!

module metaphor::graph;
import metaphor::cas;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::device::device_reference;
import metaphor::device::common::operations;
import metaphor::device::common::index_expr;
import metaphor::utils::hash_map;
import metaphor::logging;
import std::core::mem::allocator;
import std::collections::list;

alias IREntryList = list::List{cas::IREntry};
alias TensorCoreList = list::List{tensor_core::TensorCore*};

// Constants

const usz MAX_CHILD_EXPRS = 8;

// Compiled Set — tracks which tensors have been compiled this session

alias CompiledSet = hash_map::HashMap {uptr, bool};

// map_graph — main entry point

/// Map a graph starting from root tensor.
/// Returns IR entries in dependency order (deepest first).
/// Uses temp allocator for arena-style allocation.
fn cas::IREntry[] map_graph(
	SymbolicSequence* seq,
	tensor_core::TensorCore* root,
	Graph* g,
)
{
	IREntryList ir_list;
	ir_list.tinit();

	CompiledSet compiled;
	compiled.tinit();

	cas::ExprBuilder builder = cas::expr_builder_init(tmem);

	build_subgraph_ir(seq, root, &ir_list, &compiled, &builder, g);

	return ir_list.array_view();
}

// build_subgraph_ir — recursive subgraph builder

fn void build_subgraph_ir(
	SymbolicSequence* seq,
	tensor_core::TensorCore* tc,
	IREntryList* ir_list,
	CompiledSet* compiled,
	cas::ExprBuilder* builder,
	Graph* g,
)
{
	// Already compiled this session
	if (compiled.has_key((uptr)tc)) return;

	// Already has execution unit from prior collect
	if (g.has_execution_unit(tc)) return;

	// Look up the single node for this tensor (no node = leaf)
	SymbolicNode*? node = seq.get(tc);
	if (catch node) return;

	device_reference::DeviceReference subgraph_device = tc.device;

	TensorCoreList inputs_list;
	inputs_list.tinit();

	cas::Expr* base_expr = walk(seq, tc, node, subgraph_device,
		&inputs_list, ir_list, compiled, builder, g);

	if (base_expr == null) return;

	// If walk_reduce or walk_einsum_cutensor already emitted
	if (compiled.has_key((uptr)tc)) return;

	// Fix empty indices for element-wise operations
	FixResult fix_result = fix_empty_indices(base_expr, tc, builder);

	logging::@debug(COMPILER, "build_subgraph_ir: output_ndim=%d output_indices=%d expr_kind=%d",
		(int)tc.shape().len, (int)fix_result.output_indices.len, (int)fix_result.expr.kind);

	compiled.set((uptr)tc, true);

	// Copy inputs to temp-allocated slice
	usz inputs_count = inputs_list.len();
	tensor_core::TensorCore*[] inputs_slice = allocator::alloc_array_try(tmem, tensor_core::TensorCore*, inputs_count)!!;
	for (usz i = 0; i < inputs_count; i++)
	{
		inputs_slice[i] = inputs_list[i];
	}

	ir_list.push({
		.output = tc,
		.expr = fix_result.expr,
		.inputs = inputs_slice,
		.output_indices = fix_result.output_indices,
		.payload = null,
	});
}

// walk_child — process a child tensor to get its expression

fn cas::Expr* walk_child(
	SymbolicSequence* seq,
	tensor_core::TensorCore* child,
	device_reference::DeviceReference current_device,
	TensorCoreList* inputs_list,
	IREntryList* ir_list,
	CompiledSet* compiled,
	cas::ExprBuilder* builder,
	Graph* g,
)
{
	// Different device? Compile separately, becomes input
	if (!child.device.is_compatible(current_device))
	{
		build_subgraph_ir(seq, child, ir_list, compiled, builder, g);
		add_input_if_new(inputs_list, child);
		return builder.access(child, (cas::IndexId[]){}, false)!!;
	}

	// Already compiled? Becomes input
	if (compiled.has_key((uptr)child) || g.has_execution_unit(child))
	{
		add_input_if_new(inputs_list, child);
		return builder.access(child, (cas::IndexId[]){}, false)!!;
	}

	// Compilation boundary? Compile separately
	if (child.is_compilation_boundary())
	{
		build_subgraph_ir(seq, child, ir_list, compiled, builder, g);
		add_input_if_new(inputs_list, child);
		return builder.access(child, (cas::IndexId[]){}, false)!!;
	}

	// Get node for child — if none, it's a leaf
	SymbolicNode*? child_node = seq.get(child);
	if (catch child_node)
	{
		add_input_if_new(inputs_list, child);
		return builder.access(child, (cas::IndexId[]){}, false)!!;
	}

	// Recurse into child
	return walk(seq, child, child_node, current_device,
		inputs_list, ir_list, compiled, builder, g);
}

// walk — main recursive traversal

fn cas::Expr* walk(
	SymbolicSequence* seq,
	tensor_core::TensorCore* tc,
	SymbolicNode* node,
	device_reference::DeviceReference current_device,
	TensorCoreList* inputs_list,
	IREntryList* ir_list,
	CompiledSet* compiled,
	cas::ExprBuilder* builder,
	Graph* g,
)
{
	operations::OpTag tag = node.payload.tag();

	switch (tag)
	{
		case REDUCE:
			return walk_reduce(seq, tc, node,
				inputs_list, ir_list, compiled, builder, g);

		case EXPAND:
			return walk_expand(seq, tc, node,
				inputs_list, ir_list, compiled, builder, g);

		case EINSUM:
			operations::EinsumPayload* cp = &node.payload.einsum;
			if (einsum_has_contractions(cp) || einsum_needs_index_aware_path(cp))
			{
				return walk_einsum_cutensor(seq, tc, node, current_device,
					inputs_list, ir_list, compiled, builder, g, cp);
			}
			// Element-wise einsum (same indices) — fall through to generic path for chain fusion

		// Boundary ops — can't be fused, emit as opaque IR entries
		case SLICE:
		case CONCAT:
		case STACK:
		case SOFTMAX:
		case RMS_NORM:
		case CONV:
		case CONV_TRANSPOSE:
		case GATHER:
		case SCATTER_ADD:
		case CAST:
		case RESHAPE:
		case TRANSFER:
		case REDUCE_MAXMIN_BACKWARD:
		case SELECTIVE_SCAN:
		case CAUSAL_CONV1D:
		case DROPOUT:
		case DIM:
		case NUMEL:
			tensor::Tensor[] boundary_children = node.children.slice();
			for (usz i = 0; i < boundary_children.len; i++)
			{
				build_subgraph_ir(seq, boundary_children[i].core(), ir_list, compiled, builder, g);
			}
			tensor_core::TensorCore*[] boundary_inputs = allocator::alloc_array_try(tmem, tensor_core::TensorCore*, boundary_children.len)!!;
			for (usz i = 0; i < boundary_children.len; i++)
			{
				boundary_inputs[i] = boundary_children[i].core();
			}
			compiled.set((uptr)tc, true);
			logging::@debug(COMPILER, "boundary_spill: tag=%d output_ndim=%d",
				(int)node.payload.tag(), (int)tc.shape().len);
			ir_list.push({
				.output = tc,
				.expr = builder.access(tc, (cas::IndexId[]){}, true)!!,
				.inputs = boundary_inputs,
				.output_indices = (cas::IndexId[]){},
				.payload = &node.payload,
			});
			add_input_if_new(inputs_list, tc);
			return builder.access(tc, (cas::IndexId[]){}, false)!!;

		default:
			// Generic element-wise ops (MAP, SELECT, PERMUTE, etc.)
	}

	// Process children via walk_child
	tensor::Tensor[] children = node.children.slice();
	cas::Expr*[MAX_CHILD_EXPRS] child_exprs_buf;
	usz child_exprs_count = 0;

	for (usz ci = 0; ci < children.len; ci++)
	{
		cas::Expr* child_expr = walk_child(seq, children[ci].core(), current_device,
			inputs_list, ir_list, compiled, builder, g);
		child_exprs_buf[child_exprs_count] = child_expr;
		child_exprs_count++;
	}

	cas::Expr*[] child_exprs = child_exprs_buf[:child_exprs_count];
	return build_expr_from_node(builder, tc, node, child_exprs);
}

// build_expr_from_node — convert a SymbolicNode + child exprs to CAS Expr

fn cas::Expr* build_expr_from_node(
	cas::ExprBuilder* builder,
	tensor_core::TensorCore* tc,
	SymbolicNode* node,
	cas::Expr*[] child_exprs,
)
{
	operations::OpTag tag = node.payload.tag();

	switch (tag)
	{
		case MAP:
			if (child_exprs.len == 0) return builder.access(tc, (cas::IndexId[]){}, false)!!;
			cas::Expr* child = child_exprs[0];
			return map_unary_to_expr(builder, node.payload.map_payload.unary, child);

		case EINSUM:
			if (child_exprs.len < 2)
			{
				if (child_exprs.len == 1) return child_exprs[0];
				return builder.access(tc, (cas::IndexId[]){}, false)!!;
			}
			return einsum_binary_to_expr(builder, node.payload.einsum.inner,
				child_exprs[0], child_exprs[1]);

		case SELECT:
			if (child_exprs.len < 3) return builder.access(tc, (cas::IndexId[]){}, false)!!;
			return builder.create_select(child_exprs[0], child_exprs[1], child_exprs[2])!!;

		case EXPAND:
			if (child_exprs.len == 0) unreachable("expand requires a child expression");
			return build_expand_expr(builder, &node.payload.expand, child_exprs[0]);

		case PERMUTE:
			if (child_exprs.len == 0) unreachable("permute requires a child expression");
			return build_permute_expr(builder, node, child_exprs[0]);

		default:
			// Boundary ops — opaque to CAS
			return builder.access(tc, (cas::IndexId[]){}, true)!!;
	}
}

// Expression builders for specific op types

fn cas::Expr* map_unary_to_expr(cas::ExprBuilder* builder, operations::UnaryOp op, cas::Expr* child)
{
	switch (op)
	{
		case IDENTITY: return child;
		case NEG: return builder.create_negate(child)!!;
		case ABS: return builder.create_unary(cas::UnaryOp.ABS, child)!!;
		case SQR: return builder.create_unary(cas::UnaryOp.SQR, child)!!;
		case SQRT: return builder.create_unary(cas::UnaryOp.SQRT, child)!!;
		case RECIP: return builder.create_unary(cas::UnaryOp.RECIP, child)!!;
		case EXP: return builder.create_unary(cas::UnaryOp.EXP, child)!!;
		case LOG: return builder.create_unary(cas::UnaryOp.LOG, child)!!;
		case SIN: return builder.create_unary(cas::UnaryOp.SIN, child)!!;
		case COS: return builder.create_unary(cas::UnaryOp.COS, child)!!;
		case TANH: return builder.create_unary(cas::UnaryOp.TANH, child)!!;
		case RELU: return builder.create_unary(cas::UnaryOp.RELU, child)!!;
		case HEAVISIDE: return builder.create_unary(cas::UnaryOp.HEAVISIDE, child)!!;
		case SIGMOID: return builder.create_unary(cas::UnaryOp.SIGMOID, child)!!;
		case GELU: return builder.create_unary(cas::UnaryOp.GELU, child)!!;
		case SILU: return builder.create_unary(cas::UnaryOp.SILU, child)!!;
		case SOFTPLUS: return builder.create_unary(cas::UnaryOp.SOFTPLUS, child)!!;
		case NOT: return builder.create_unary(cas::UnaryOp.LOGIC_NOT, child)!!;
		case SIGN: return builder.create_unary(cas::UnaryOp.SIGN, child)!!;
		default: unreachable("map_unary_to_expr: unhandled UnaryOp");
	}
}

fn cas::Expr* einsum_binary_to_expr(cas::ExprBuilder* builder, operations::BinaryOp op, cas::Expr* lhs, cas::Expr* rhs)
{
	switch (op)
	{
		case ADD:
			cas::Expr*[2] sum_args = { lhs, rhs };
			return builder.sum(&sum_args)!!;
		case MUL:
			cas::Expr*[2] prod_args = { lhs, rhs };
			return builder.product(&prod_args)!!;
		case SUB: return builder.create_binary(cas::BinaryOp.SUB, lhs, rhs)!!;
		case DIV: return builder.create_binary(cas::BinaryOp.DIV, lhs, rhs)!!;
		case MAX: return builder.create_binary(cas::BinaryOp.MAX, lhs, rhs)!!;
		case MIN: return builder.create_binary(cas::BinaryOp.MIN, lhs, rhs)!!;
		case EQL: return builder.create_compare(cas::CompareOp.EQ, lhs, rhs)!!;
		case NEQ: return builder.create_compare(cas::CompareOp.NE, lhs, rhs)!!;
		case GT:  return builder.create_compare(cas::CompareOp.GT, lhs, rhs)!!;
		case LT:  return builder.create_compare(cas::CompareOp.LT, lhs, rhs)!!;
		case GTE: return builder.create_compare(cas::CompareOp.GTE, lhs, rhs)!!;
		case LTE: return builder.create_compare(cas::CompareOp.LTE, lhs, rhs)!!;
		case AND: return builder.create_binary(cas::BinaryOp.LOGIC_AND, lhs, rhs)!!;
		case OR:  return builder.create_binary(cas::BinaryOp.LOGIC_OR, lhs, rhs)!!;
		case XOR: return builder.create_binary(cas::BinaryOp.LOGIC_XOR, lhs, rhs)!!;
		default:
			unreachable("einsum_binary_to_expr: unhandled BinaryOp");
	}
}

fn cas::Expr* build_expand_expr(cas::ExprBuilder* builder, operations::ExpandPayload* payload, cas::Expr* child_expr)
{
	index_expr::IndexExpr* expr = &payload.expr;
	char[] in_indices = expr.get_input(0);
	char[] out_indices = expr.get_output();

	// Build input_ids using OUTPUT POSITIONS as IndexIds
	cas::IndexId[] input_ids = allocator::alloc_array_try(builder.arena, cas::IndexId, in_indices.len)!!;
	for (usz i = 0; i < in_indices.len; i++)
	{
		// Find position of this character in output indices
		bool found = false;
		for (usz j = 0; j < out_indices.len; j++)
		{
			if (in_indices[i] == out_indices[j])
			{
				input_ids[i] = (cas::IndexId)j;
				found = true;
				break;
			}
		}
		assert(found, "expand input char not in output indices");
	}

	// Remap child access indices
	return remap_access_indices(builder.arena, child_expr, input_ids);
}

fn cas::Expr* build_permute_expr(cas::ExprBuilder* builder, SymbolicNode* node, cas::Expr* child_expr)
{
	operations::PermutePayload* payload = &node.payload.permute;
	index_expr::IndexExpr* expr = &payload.expr;
	char[] in_indices = expr.get_input(0);
	char[] out_indices = expr.get_output();

	// Build permutation mapping
	char[8] mapping_buf;
	usz mapping_len = 0;
	for (usz oi = 0; oi < out_indices.len; oi++)
	{
		for (usz ii = 0; ii < in_indices.len; ii++)
		{
			if (out_indices[oi] == in_indices[ii])
			{
				mapping_buf[mapping_len] = (char)ii;
				mapping_len++;
				break;
			}
		}
	}

	return builder.permute(child_expr, mapping_buf[:mapping_len])!!;
}

// walk_einsum_cutensor — einsums routed to cuTENSOR (contractions, outer products, non-trivial broadcasts)

fn cas::Expr* walk_einsum_cutensor(
	SymbolicSequence* seq,
	tensor_core::TensorCore* tc,
	SymbolicNode* node,
	device_reference::DeviceReference current_device,
	TensorCoreList* inputs_list,
	IREntryList* ir_list,
	CompiledSet* compiled,
	cas::ExprBuilder* builder,
	Graph* g,
	operations::EinsumPayload* combine_payload,
)
{
	tensor::Tensor[] children = node.children.slice();

	// Compile all children first
	for (usz i = 0; i < children.len; i++)
	{
		build_subgraph_ir(seq, children[i].core(), ir_list, compiled, builder, g);
	}

	// Build a fresh builder for this contraction's IR entry
	cas::ExprBuilder contraction_builder = cas::expr_builder_init(tmem);

	tensor_core::TensorCore*[] contraction_inputs = allocator::alloc_array_try(tmem, tensor_core::TensorCore*, children.len)!!;
	for (usz i = 0; i < children.len; i++)
	{
		contraction_inputs[i] = children[i].core();
	}

	// Use index expressions to build proper einsum contraction
	index_expr::IndexExpr* idx_expr = &combine_payload.expr;
	char[] in0_explicit = idx_expr.get_input(0);
	char[] in1_explicit = idx_expr.get_input(1);
	char[] out_explicit = idx_expr.get_output();

	// Create Index objects for each character
	cas::Index[256] index_map;
	bool[256] index_created = {};

	tensor_core::TensorCore* child0 = children[0].core();
	tensor_core::TensorCore* child1 = children.len > 1 ? children[1].core() : child0;
	ulong[] shape0 = child0.shape().slice();
	ulong[] shape1 = child1.shape().slice();

	// Expand ellipsis: prepend batch dimension chars before explicit indices.
	// Use chars starting from 0x01 to avoid colliding with user-specified letters.
	char[8] in0_buf;
	char[8] in1_buf;
	char[8] out_buf;
	usz in0_len = 0;
	usz in1_len = 0;
	usz out_len = 0;
	usz ellipsis_dims = 0;

	if (idx_expr.has_ellipsis)
	{
		usz a_ellipsis = shape0.len > in0_explicit.len ? shape0.len - in0_explicit.len : 0;
		usz b_ellipsis = shape1.len > in1_explicit.len ? shape1.len - in1_explicit.len : 0;
		ellipsis_dims = a_ellipsis > b_ellipsis ? a_ellipsis : b_ellipsis;

		// Create indices for batch dims and prepend to all patterns
		for (usz d = 0; d < ellipsis_dims; d++)
		{
			char batch_char = (char)(1 + d);  // 0x01, 0x02, ...

			// Size from whichever input has this batch dim (broadcast: take max)
			long a_idx = (long)d - (long)(ellipsis_dims - a_ellipsis);
			long b_idx = (long)d - (long)(ellipsis_dims - b_ellipsis);
			ulong a_dim = a_idx >= 0 ? shape0[(usz)a_idx] : 1;
			ulong b_dim = b_idx >= 0 ? shape1[(usz)b_idx] : 1;
			ulong size = a_dim > b_dim ? a_dim : b_dim;

			index_map[batch_char] = contraction_builder.index(size);
			index_created[batch_char] = true;

			in0_buf[in0_len++] = batch_char;
			in1_buf[in1_len++] = batch_char;
			out_buf[out_len++] = batch_char;
		}
	}

	// Append explicit indices
	for (usz i = 0; i < in0_explicit.len; i++) { in0_buf[in0_len++] = in0_explicit[i]; }
	for (usz i = 0; i < in1_explicit.len; i++) { in1_buf[in1_len++] = in1_explicit[i]; }
	for (usz i = 0; i < out_explicit.len; i++) { out_buf[out_len++] = out_explicit[i]; }

	char[] in0 = in0_buf[:in0_len];
	char[] in1 = in1_buf[:in1_len];
	char[] out = out_buf[:out_len];

	// Create indices for explicit chars (offset into shape by ellipsis_dims)
	for (usz i = 0; i < in0_explicit.len; i++)
	{
		char c = in0_explicit[i];
		if (!index_created[c])
		{
			usz shape_idx = ellipsis_dims + i;
			ulong size = shape_idx < shape0.len ? shape0[shape_idx] : 1;
			index_map[c] = contraction_builder.index(size);
			index_created[c] = true;
		}
	}
	for (usz i = 0; i < in1_explicit.len; i++)
	{
		char c = in1_explicit[i];
		if (!index_created[c])
		{
			usz shape_idx = ellipsis_dims + i;
			ulong size = shape_idx < shape1.len ? shape1[shape_idx] : 1;
			index_map[c] = contraction_builder.index(size);
			index_created[c] = true;
		}
	}

	// Build access expressions with proper indices
	cas::IndexId[] ids0 = allocator::alloc_array_try(tmem, cas::IndexId, in0.len)!!;
	for (usz i = 0; i < in0.len; i++) { ids0[i] = index_map[in0[i]].id; }

	cas::IndexId[] ids1 = allocator::alloc_array_try(tmem, cas::IndexId, in1.len)!!;
	for (usz i = 0; i < in1.len; i++) { ids1[i] = index_map[in1[i]].id; }

	cas::Expr* access0 = contraction_builder.access(child0, ids0, false)!!;
	cas::Expr* access1 = contraction_builder.access(child1, ids1, false)!!;

	// Combine the two accesses using the inner op (MUL → product, ADD → sum)
	cas::Expr*[2] combine_args = { access0, access1 };
	cas::Expr* body;
	if (combine_payload.inner == operations::BinaryOp.ADD)
	{
		body = contraction_builder.sum(&combine_args)!!;
	}
	else
	{
		body = contraction_builder.product(&combine_args)!!;
	}

	// Contract over indices not in output
	cas::ReduceOp reduce_op = binary_op_to_reduce_op(combine_payload.outer);
	for (usz i = 0; i < in0.len; i++)
	{
		char c = in0[i];
		if (!char_in_slice(c, out))
		{
			body = contraction_builder.contract(index_map[c], body, reduce_op)!!;
		}
	}
	for (usz i = 0; i < in1.len; i++)
	{
		char c = in1[i];
		if (!char_in_slice(c, out) && !char_in_slice(c, in0))
		{
			body = contraction_builder.contract(index_map[c], body, reduce_op)!!;
		}
	}

	// Build output indices
	cas::IndexId[] output_ids = allocator::alloc_array_try(tmem, cas::IndexId, out.len)!!;
	for (usz i = 0; i < out.len; i++)
	{
		output_ids[i] = index_map[out[i]].id;
	}

	// Emit IR entry
	compiled.set((uptr)tc, true);
	logging::@debug(COMPILER, "walk_einsum_cutensor: output_ndim=%d output_indices=%d expr_kind=%d",
		(int)tc.shape().len, (int)output_ids.len, (int)body.kind);
	ir_list.push({
		.output = tc,
		.expr = body,
		.inputs = contraction_inputs,
		.output_indices = output_ids,
		.payload = &node.payload,
	});

	// Return access to result
	add_input_if_new(inputs_list, tc);
	return builder.access(tc, (cas::IndexId[]){}, false)!!;
}

// walk_reduce — reduction operations

fn cas::Expr* walk_reduce(
	SymbolicSequence* seq,
	tensor_core::TensorCore* tc,
	SymbolicNode* node,
	TensorCoreList* inputs_list,
	IREntryList* ir_list,
	CompiledSet* compiled,
	cas::ExprBuilder* builder,
	Graph* g,
)
{
	tensor::Tensor[] children = node.children.slice();
	assert(children.len > 0, "reduce node has no children");

	tensor_core::TensorCore* child = children[0].core();
	build_subgraph_ir(seq, child, ir_list, compiled, builder, g);

	// Build a fresh builder for this reduce's IR entry
	cas::ExprBuilder reduce_builder = cas::expr_builder_init(tmem);

	tensor_core::TensorCore*[] reduce_inputs = allocator::alloc_array_try(tmem, tensor_core::TensorCore*, 1)!!;
	reduce_inputs[0] = child;

	operations::ReducePayload* reduce_payload = &node.payload.reduce;
	index_expr::IndexExpr* expr = &reduce_payload.expr;
	ulong[] input_shape = child.shape().slice();

	logging::@debug(COMPILER, "walk_reduce: input_ndim=%d output_ndim=%d has_ellipsis=%d in_len=%d out_len=%d in='%s' out='%s'",
		(int)input_shape.len, (int)tc.shape().len, (int)expr.has_ellipsis,
		(int)expr.input_lens[0], (int)expr.output_len,
		(ZString)expr.get_input(0).ptr, (ZString)expr.get_output().ptr);

	cas::ReduceOp reduce_op = binary_op_to_reduce_op(reduce_payload.op);

	// Handle ellipsis expansion
	char[8] expanded_in;
	usz expanded_in_len = 0;
	char[8] expanded_out;
	usz expanded_out_len = 0;
	char[] out_indices;

	if (expr.has_ellipsis)
	{
		char[] explicit_in = expr.get_input(0);
		char[] explicit_out = expr.get_output();
		usz ellipsis_dims = input_shape.len > explicit_in.len ? input_shape.len - explicit_in.len : 0;

		// Batch dims (ellipsis) come first, then explicit indices
		for (usz d = 0; d < ellipsis_dims; d++)
		{
			expanded_in[expanded_in_len] = (char)('a' + d);
			expanded_in_len++;
		}
		for (usz i = 0; i < explicit_in.len; i++)
		{
			expanded_in[expanded_in_len] = explicit_in[i];
			expanded_in_len++;
		}

		// Expand ellipsis in output: batch dims + explicit output indices
		for (usz d = 0; d < ellipsis_dims; d++)
		{
			expanded_out[expanded_out_len] = (char)('a' + d);
			expanded_out_len++;
		}
		for (usz i = 0; i < explicit_out.len; i++)
		{
			expanded_out[expanded_out_len] = explicit_out[i];
			expanded_out_len++;
		}
		out_indices = expanded_out[:expanded_out_len];
	}
	else
	{
		char[] explicit_in = expr.get_input(0);
		for (usz i = 0; i < explicit_in.len; i++)
		{
			expanded_in[i] = explicit_in[i];
		}
		expanded_in_len = explicit_in.len;
		out_indices = expr.get_output();
	}
	char[] in_indices = expanded_in[:expanded_in_len];

	// Create Index objects for each input index character
	cas::Index[256] index_map;
	bool[256] index_created = {};
	for (usz i = 0; i < in_indices.len; i++)
	{
		char c = in_indices[i];
		if (!index_created[c])
		{
			ulong size = i < input_shape.len ? input_shape[i] : 1;
			index_map[c] = reduce_builder.index(size);
			index_created[c] = true;
		}
	}

	// Create access to input with proper indices
	cas::IndexId[] ids = allocator::alloc_array_try(tmem, cas::IndexId, in_indices.len)!!;
	for (usz i = 0; i < in_indices.len; i++)
	{
		ids[i] = index_map[in_indices[i]].id;
	}
	cas::Expr* body = reduce_builder.access(child, ids, false)!!;

	// Contract over indices not in output
	for (usz i = 0; i < in_indices.len; i++)
	{
		char c = in_indices[i];
		if (!char_in_slice(c, out_indices))
		{
			body = reduce_builder.contract(index_map[c], body, reduce_op)!!;
		}
	}

	// Build output_indices
	cas::IndexId[] output_ids = allocator::alloc_array_try(tmem, cas::IndexId, out_indices.len)!!;
	for (usz i = 0; i < out_indices.len; i++)
	{
		output_ids[i] = index_map[out_indices[i]].id;
	}

	// Emit IR entry
	compiled.set((uptr)tc, true);
	logging::@debug(COMPILER, "walk_reduce: output_ndim=%d output_indices=%d expr_kind=%d",
		(int)tc.shape().len, (int)output_ids.len, (int)body.kind);
	ir_list.push({
		.output = tc,
		.expr = body,
		.inputs = reduce_inputs,
		.output_indices = output_ids,
	});

	// Return access to result
	add_input_if_new(inputs_list, tc);
	return builder.access(tc, (cas::IndexId[]){}, false)!!;
}

// walk_expand — expand/broadcast operations (consistent IndexId handling)

fn cas::Expr* walk_expand(
	SymbolicSequence* seq,
	tensor_core::TensorCore* tc,
	SymbolicNode* node,
	TensorCoreList* inputs_list,
	IREntryList* ir_list,
	CompiledSet* compiled,
	cas::ExprBuilder* builder,
	Graph* g,
)
{
	tensor::Tensor[] children = node.children.slice();
	assert(children.len > 0, "expand node has no children");

	tensor_core::TensorCore* child = children[0].core();
	build_subgraph_ir(seq, child, ir_list, compiled, builder, g);

	// Build a fresh builder for this expand's IR entry
	cas::ExprBuilder expand_builder = cas::expr_builder_init(tmem);

	tensor_core::TensorCore*[] expand_inputs = allocator::alloc_array_try(tmem, tensor_core::TensorCore*, 1)!!;
	expand_inputs[0] = child;

	operations::ExpandPayload* expand_payload = &node.payload.expand;
	index_expr::IndexExpr* expr = &expand_payload.expr;
	char[] in_indices = expr.get_input(0);
	char[] out_indices = expr.get_output();
	ulong[] input_shape = child.shape().slice();
	ulong[] output_shape = tc.shape().slice();

	// Create Index objects for each output index character
	cas::Index[256] index_map;
	bool[256] index_created = {};
	for (usz i = 0; i < out_indices.len; i++)
	{
		char c = out_indices[i];
		if (!index_created[c])
		{
			ulong size = i < output_shape.len ? output_shape[i] : 1;
			index_map[c] = expand_builder.index(size);
			index_created[c] = true;
		}
	}

	// Build input access indices — only the dims that exist in input
	cas::IndexId[] input_ids = allocator::alloc_array_try(tmem, cas::IndexId, in_indices.len)!!;
	for (usz i = 0; i < in_indices.len; i++)
	{
		input_ids[i] = index_map[in_indices[i]].id;
	}
	cas::Expr* body = expand_builder.access(child, input_ids, false)!!;

	// Build output_indices
	cas::IndexId[] output_ids = allocator::alloc_array_try(tmem, cas::IndexId, out_indices.len)!!;
	for (usz i = 0; i < out_indices.len; i++)
	{
		output_ids[i] = index_map[out_indices[i]].id;
	}

	// Emit IR entry with payload so the CUDA compiler knows this is an expand
	compiled.set((uptr)tc, true);
	logging::@debug(COMPILER, "walk_expand: output_ndim=%d output_indices=%d expr_kind=%d",
		(int)tc.shape().len, (int)output_ids.len, (int)body.kind);
	ir_list.push({
		.output = tc,
		.expr = body,
		.inputs = expand_inputs,
		.output_indices = output_ids,
		.payload = &node.payload,
	});

	// Return access to result
	add_input_if_new(inputs_list, tc);
	return builder.access(tc, (cas::IndexId[]){}, false)!!;
}

// fix_empty_indices — assign shared indices to element-wise accesses

struct FixResult
{
	cas::Expr* expr;
	cas::IndexId[] output_indices;
}

fn FixResult fix_empty_indices(cas::Expr* expr, tensor_core::TensorCore* output, cas::ExprBuilder* builder)
{
	ulong[] output_shape = output.shape().slice();
	cas::IndexId[] shared_indices = allocator::alloc_array_try(builder.arena, cas::IndexId, output_shape.len)!!;
	for (usz i = 0; i < output_shape.len; i++)
	{
		shared_indices[i] = builder.index(output_shape[i]).id;
	}

	cas::Expr* fixed = fix_empty_indices_impl(expr, shared_indices, builder);
	return { .expr = fixed, .output_indices = shared_indices };
}

fn cas::Expr* fix_empty_indices_impl(cas::Expr* expr, cas::IndexId[] shared_indices, cas::ExprBuilder* builder)
{
	switch (expr.kind)
	{
		case ACCESS:
			if (expr.access.indices.len == 0)
			{
				ulong[] tensor_shape = expr.access.tensor.shape().slice();
				usz offset = shared_indices.len >= tensor_shape.len ?
					shared_indices.len - tensor_shape.len : 0;
				usz num_indices = tensor_shape.len < shared_indices.len ? tensor_shape.len : shared_indices.len;
				return builder.access(expr.access.tensor,
					shared_indices[offset:num_indices],
					expr.access.is_stable)!!;
			}
			return expr;

		case SCALAR: return expr;

		case PRODUCT:
			cas::Expr*[] fixed_p = allocator::alloc_array_try(builder.arena, cas::Expr*, expr.operands.len)!!;
			for (usz i = 0; i < expr.operands.len; i++)
			{
				fixed_p[i] = fix_empty_indices_impl(expr.operands[i], shared_indices, builder);
			}
			return builder.product(fixed_p)!!;

		case SUM:
			cas::Expr*[] fixed_s = allocator::alloc_array_try(builder.arena, cas::Expr*, expr.operands.len)!!;
			for (usz i = 0; i < expr.operands.len; i++)
			{
				fixed_s[i] = fix_empty_indices_impl(expr.operands[i], shared_indices, builder);
			}
			return builder.sum(fixed_s)!!;

		case CONTRACT:
			// The contracted index is consumed by the contract — the body needs
			// shared_indices + the contracted index so inner accesses get the
			// right number of indices.
			usz ext_len = shared_indices.len + 1;
			cas::IndexId[] extended = allocator::alloc_array_try(builder.arena, cas::IndexId, ext_len)!!;
			for (usz i = 0; i < shared_indices.len; i++) { extended[i] = shared_indices[i]; }
			extended[shared_indices.len] = expr.contract.index.id;
			cas::Expr* fixed_body = fix_empty_indices_impl(expr.contract.body, extended, builder);
			return builder.contract(expr.contract.index, fixed_body, expr.contract.op)!!;

		case NEGATE:
			return builder.create_negate(fix_empty_indices_impl(expr.inner, shared_indices, builder))!!;

		case UNARY:
			return builder.create_unary(expr.unary.op,
				fix_empty_indices_impl(expr.unary.child, shared_indices, builder))!!;

		case BINARY:
			return builder.create_binary(expr.binary.op,
				fix_empty_indices_impl(expr.binary.lhs, shared_indices, builder),
				fix_empty_indices_impl(expr.binary.rhs, shared_indices, builder))!!;

		case COMPARE:
			return builder.create_compare(expr.compare.op,
				fix_empty_indices_impl(expr.compare.lhs, shared_indices, builder),
				fix_empty_indices_impl(expr.compare.rhs, shared_indices, builder))!!;

		case SELECT:
			return builder.create_select(
				fix_empty_indices_impl(expr.select_.cond, shared_indices, builder),
				fix_empty_indices_impl(expr.select_.then_expr, shared_indices, builder),
				fix_empty_indices_impl(expr.select_.else_expr, shared_indices, builder))!!;

		case PERMUTE:
			// Un-permute shared_indices for the child
			usz num_dims = expr.permute.mapping.len;
			cas::IndexId[] child_indices = allocator::alloc_array_try(builder.arena, cas::IndexId, num_dims)!!;
			for (usz out_dim = 0; out_dim < num_dims; out_dim++)
			{
				usz in_dim = (usz)expr.permute.mapping[out_dim];
				child_indices[in_dim] = shared_indices[out_dim];
			}
			cas::Expr* fixed_child;
			if (expr.permute.child.kind == ACCESS)
			{
				fixed_child = builder.access(
					expr.permute.child.access.tensor,
					child_indices,
					expr.permute.child.access.is_stable)!!;
			}
			else
			{
				fixed_child = fix_empty_indices_impl(expr.permute.child, child_indices, builder);
			}
			return builder.permute(fixed_child, expr.permute.mapping)!!;

		case BUILTIN:
			cas::Expr*[] fixed_args = allocator::alloc_array_try(builder.arena, cas::Expr*, expr.builtin.args.len)!!;
			for (usz i = 0; i < expr.builtin.args.len; i++)
			{
				fixed_args[i] = fix_empty_indices_impl(expr.builtin.args[i], shared_indices, builder);
			}
			return builder.builtin(expr.builtin.func, fixed_args, expr.builtin.output_indices)!!;
	}
}

// remap_access_indices — used by expand

fn cas::Expr* remap_access_indices(Allocator alloc, cas::Expr* expr, cas::IndexId[] new_indices)
{
	switch (expr.kind)
	{
		case ACCESS:
			cas::Expr* result = allocator::alloc_try(alloc, cas::Expr)!!;
			*result = { .kind = ACCESS, .access = {
				.tensor = expr.access.tensor,
				.indices = new_indices,
				.is_stable = expr.access.is_stable,
			}};
			return result;

		case SCALAR: return expr;

		case PRODUCT:
			cas::Expr*[] new_factors = allocator::alloc_array_try(alloc, cas::Expr*, expr.operands.len)!!;
			for (usz i = 0; i < expr.operands.len; i++)
			{
				new_factors[i] = remap_access_indices(alloc, expr.operands[i], new_indices);
			}
			cas::Expr* prod = allocator::alloc_try(alloc, cas::Expr)!!;
			*prod = { .kind = PRODUCT, .operands = new_factors };
			return prod;

		case SUM:
			cas::Expr*[] new_terms = allocator::alloc_array_try(alloc, cas::Expr*, expr.operands.len)!!;
			for (usz i = 0; i < expr.operands.len; i++)
			{
				new_terms[i] = remap_access_indices(alloc, expr.operands[i], new_indices);
			}
			cas::Expr* s = allocator::alloc_try(alloc, cas::Expr)!!;
			*s = { .kind = SUM, .operands = new_terms };
			return s;

		case NEGATE:
			cas::Expr* neg = allocator::alloc_try(alloc, cas::Expr)!!;
			*neg = { .kind = NEGATE, .inner = remap_access_indices(alloc, expr.inner, new_indices) };
			return neg;

		case UNARY:
			cas::Expr* u = allocator::alloc_try(alloc, cas::Expr)!!;
			*u = { .kind = UNARY, .unary = {
				.op = expr.unary.op,
				.child = remap_access_indices(alloc, expr.unary.child, new_indices),
			}};
			return u;

		case BINARY:
			cas::Expr* b = allocator::alloc_try(alloc, cas::Expr)!!;
			*b = { .kind = BINARY, .binary = {
				.op = expr.binary.op,
				.lhs = remap_access_indices(alloc, expr.binary.lhs, new_indices),
				.rhs = remap_access_indices(alloc, expr.binary.rhs, new_indices),
			}};
			return b;

		default: return expr;
	}
}

// Helpers

fn bool einsum_has_contractions(operations::EinsumPayload* payload)
{
	index_expr::IndexExpr* expr = &payload.expr;
	char[] out = expr.get_output();

	for (usz input_idx = 0; input_idx < (usz)expr.num_inputs; input_idx++)
	{
		char[] inp = expr.get_input(input_idx);
		for (usz i = 0; i < inp.len; i++)
		{
			if (!char_in_slice(inp[i], out)) return true;
		}
	}
	return false;
}

/// Returns true if any input's indices are NOT a right-aligned suffix of the output indices.
/// This means the einsum needs the index-aware walk_einsum_cutensor path
/// because the generic element-wise path (which assumes right-aligned broadcasting) would
/// assign incorrect index mappings.
///
/// Examples:
///   "ijk,ijk->ijk" → false (both inputs match output, element-wise)
///   "ij,j->ij"     → false (input1 is suffix of output, standard broadcast)
///   "be,bd->bed"   → true  (input0 "be" != suffix "ed" of output "bed")
///   "bed,be->bed"  → true  (input1 "be" != suffix "ed" of output "bed")
///   "be,ed->bed"   → true  (input0 "be" != suffix "ed", outer product)
fn bool einsum_needs_index_aware_path(operations::EinsumPayload* payload)
{
	index_expr::IndexExpr* expr = &payload.expr;
	char[] out = expr.get_output();

	for (usz input_idx = 0; input_idx < (usz)expr.num_inputs; input_idx++)
	{
		char[] inp = expr.get_input(input_idx);
		if (inp.len > out.len) return true;

		// Check if input indices are a right-aligned suffix of output
		usz offset = out.len - inp.len;
		for (usz i = 0; i < inp.len; i++)
		{
			if (inp[i] != out[offset + i]) return true;
		}
	}
	return false;
}

fn cas::ReduceOp binary_op_to_reduce_op(operations::BinaryOp op)
{
	switch (op)
	{
		case ADD: return cas::ReduceOp.ADD;
		case MUL: return cas::ReduceOp.MUL;
		case MAX: return cas::ReduceOp.MAX;
		case MIN: return cas::ReduceOp.MIN;
		default: unreachable("BinaryOp cannot be used as reduce operation");
	}
}

fn void add_input_if_new(TensorCoreList* inputs_list, tensor_core::TensorCore* tc)
{
	for (usz i = 0; i < inputs_list.len(); i++)
	{
		if ((*inputs_list)[i] == tc) return;
	}
	inputs_list.push(tc);
}

fn bool char_in_slice(char c, char[] slice)
{
	for (usz i = 0; i < slice.len; i++)
	{
		if (slice[i] == c) return true;
	}
	return false;
}
