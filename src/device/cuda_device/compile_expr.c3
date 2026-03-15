//! CUDA Graph Compiler — CAS Expression to CudaOps
//!
//! Translates CAS expression trees into CudaOp sequences.
//! Handles contractions (cuTENSOR), reductions, permutations,
//! and linearization of elementwise chains.
//!
//! Part of the compile module — functions here are visible to compile.c3.

module metaphor::device::cuda_device::compile;

import metaphor::cas;
import metaphor::device::cuda_device;
import metaphor::device::cuda_device::types;
import metaphor::device::cuda_device::utils;
import metaphor::device::common::codegen_context;
import metaphor::device::common::operations;
import metaphor::tensor::tensor_core;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import std::core::mem::allocator;
import metaphor::logging;
import std::io;

// Access helper — extracted from CAS expression find_first_access

faultdef SEARCH_MISSING;

struct AccessInfo
{
	void* tensor;
	IndexId[] indices;
}

/// Find the first access node in a CAS expression tree (depth-first).
fn AccessInfo? find_first_access(Expr* expr)
{
	switch (expr.kind)
	{
		case ACCESS:
			return { .tensor = (void*)expr.access.tensor, .indices = expr.access.indices };
		case UNARY:
			return find_first_access(expr.unary.child);
		case NEGATE:
			return find_first_access(expr.inner);
		case BINARY:
			if (try a = find_first_access(expr.binary.lhs))
			{
				return a;
			}
			return find_first_access(expr.binary.rhs);
		case CONTRACT:
			return find_first_access(expr.contract.body);
		case PRODUCT:
			foreach (f : expr.operands)
			{
				if (try a = find_first_access(f))
				{
					return a;
				}
			}
			return SEARCH_MISSING~;
		case SUM:
			foreach (t : expr.operands)
			{
				if (try a = find_first_access(t))
				{
					return a;
				}
			}
			return SEARCH_MISSING~;
		case SELECT:
			if (try a = find_first_access(expr.select_.cond))
			{
				return a;
			}
			if (try a = find_first_access(expr.select_.then_expr))
			{
				return a;
			}
			return find_first_access(expr.select_.else_expr);
		case COMPARE:
			if (try a = find_first_access(expr.compare.lhs))
			{
				return a;
			}
			return find_first_access(expr.compare.rhs);
		case PERMUTE:
			return find_first_access(expr.permute.child);
		default:
			return SEARCH_MISSING~;
	}
}

/// Unwrap nested contract expressions to find the innermost non-contract body.
fn Expr* unwrap_nested_contracts(Expr* expr)
{
	Expr* current = expr;
	while (current.kind == ExprKind.CONTRACT)
	{
		current = current.contract.body;
	}
	return current;
}

/// Check if two index ID slices are identical (same length and same IDs).
fn bool indices_match(cas::IndexId[] a, cas::IndexId[] b)
{
	if (a.len != b.len) return false;
	for (usz i = 0; i < a.len; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

// Index Mapping — fixed-size arrays instead of hashmaps

const usz MAX_INDEX_MAP = 32;

struct IndexCharMap
{
	IndexId[MAX_INDEX_MAP] ids;
	char[MAX_INDEX_MAP] chars;
	usz len;
	char next_char;
}

fn void IndexCharMap.init(&self)
{
	self.len = 0;
	self.next_char = 'a';
}

fn char IndexCharMap.get(&self, IndexId id)
{
	for (usz i = 0; i < self.len; i++)
	{
		if (self.ids[i] == id) return self.chars[i];
	}
	unreachable("IndexCharMap.get: index ID not found in map");
}

fn bool IndexCharMap.contains(&self, IndexId id)
{
	for (usz i = 0; i < self.len; i++)
	{
		if (self.ids[i] == id) return true;
	}
	return false;
}

fn void IndexCharMap.put(&self, IndexId id)
{
	if (self.contains(id)) return;
	assert(self.len < MAX_INDEX_MAP, "IndexCharMap.put: index map overflow — too many unique indices");
	self.ids[self.len] = id;
	self.chars[self.len] = self.next_char;
	self.len++;
	self.next_char++;
}

// Pattern building

fn void build_pattern(IndexId[] indices, IndexCharMap* map, char[] out_buf, char* out_len)
{
	usz len = indices.len < types::MAX_PATTERN_LEN ? indices.len : types::MAX_PATTERN_LEN;
	for (usz i = 0; i < len; i++)
	{
		out_buf[i] = map.get(indices[i]);
	}
	*out_len = (char)len;
}

/// Set input on a CudaOp from an access expression.
fn void set_input_from_access(
	void* tensor,
	IndexId[] indices,
	CudaOp* cuda_op,
	usz slot,
	IndexCharMap* map,
)
{
	if (slot < 3)
	{
		cuda_op.inputs[slot] = tensor;
		if (slot >= (usz)cuda_op.input_count)
		{
			cuda_op.input_count = (char)(slot + 1);
		}
	}

	char[types::MAX_PATTERN_LEN] pattern;
	char pattern_len;
	build_pattern(indices, map, &pattern, &pattern_len);

	if (slot < 2)
	{
		cuda_op.set_input_pattern(slot, pattern[:pattern_len]);
	}
}

/// Set sequential identity pattern for a tensor.
fn void set_sequential_pattern(CudaOp* cuda_op, usz slot, void* tensor, usz ndim)
{
	usz len = ndim < 16 ? ndim : 16;
	char[16] pattern;
	for (usz i = 0; i < len; i++)
	{
		pattern[i] = (char)('a' + i);
	}
	if (slot < 2)
	{
		cuda_op.set_input_pattern(slot, pattern[:len]);
	}
	cuda_op.inputs[slot] = tensor;
	if (slot >= (usz)cuda_op.input_count)
	{
		cuda_op.input_count = (char)(slot + 1);
	}
}

// Collect indices from expression tree

fn void collect_contracted_indices(Expr* expr, IndexCharMap* map)
{
	switch (expr.kind)
	{
		case CONTRACT:
			map.put(expr.contract.index.id);
			collect_contracted_indices(expr.contract.body, map);
		case PRODUCT:
			foreach (f : expr.operands) collect_contracted_indices(f, map);
		case SUM:
			foreach (t : expr.operands) collect_contracted_indices(t, map);
		case NEGATE:
			collect_contracted_indices(expr.inner, map);
		case UNARY:
			collect_contracted_indices(expr.unary.child, map);
		case BINARY:
			collect_contracted_indices(expr.binary.lhs, map);
			collect_contracted_indices(expr.binary.rhs, map);
		case COMPARE:
			collect_contracted_indices(expr.compare.lhs, map);
			collect_contracted_indices(expr.compare.rhs, map);
		case SELECT:
			collect_contracted_indices(expr.select_.cond, map);
			collect_contracted_indices(expr.select_.then_expr, map);
			collect_contracted_indices(expr.select_.else_expr, map);
		case PERMUTE:
			collect_contracted_indices(expr.permute.child, map);
		default: {}
	}
}

fn void collect_all_access_indices(Expr* expr, IndexCharMap* map)
{
	switch (expr.kind)
	{
		case ACCESS:
			foreach (idx_id : expr.access.indices) map.put(idx_id);
		case CONTRACT:
			collect_all_access_indices(expr.contract.body, map);
		case PRODUCT:
			foreach (f : expr.operands) collect_all_access_indices(f, map);
		case SUM:
			foreach (t : expr.operands) collect_all_access_indices(t, map);
		case NEGATE:
			collect_all_access_indices(expr.inner, map);
		case UNARY:
			collect_all_access_indices(expr.unary.child, map);
		case BINARY:
			collect_all_access_indices(expr.binary.lhs, map);
			collect_all_access_indices(expr.binary.rhs, map);
		case COMPARE:
			collect_all_access_indices(expr.compare.lhs, map);
			collect_all_access_indices(expr.compare.rhs, map);
		case SELECT:
			collect_all_access_indices(expr.select_.cond, map);
			collect_all_access_indices(expr.select_.then_expr, map);
			collect_all_access_indices(expr.select_.else_expr, map);
		case PERMUTE:
			collect_all_access_indices(expr.permute.child, map);
		default: {}
	}
}

// compile_cas_expr — main CAS expression compiler

/// Maximum ops we collect before registering.
const usz MAX_COMPILE_OPS = 64;

<*
 @require device != null : "compile_cas_expr: device must not be null"
 @require g != null : "compile_cas_expr: graph must not be null"
 @require entry != null : "compile_cas_expr: entry must not be null"
 @require entry.expr != null : "compile_cas_expr: entry must have an expression"
 @require entry.output != null : "compile_cas_expr: entry must have an output"
*>
fn void compile_cas_expr(
	CudaDevice* device,
	graph::Graph* g,
	IREntry* entry,
)
{
	// Only decompose elementwise expressions (SUM, PRODUCT, NEGATE, UNARY,
	// BINARY, SELECT, COMPARE). CONTRACT, ACCESS, and PERMUTE are handled
	// by cuTENSOR/permutation backends that have no input count limitation.
	ExprKind kind = entry.expr.kind;
	bool is_elementwise = kind == ExprKind.SUM || kind == ExprKind.PRODUCT
		|| kind == ExprKind.NEGATE || kind == ExprKind.UNARY
		|| kind == ExprKind.BINARY || kind == ExprKind.SELECT
		|| kind == ExprKind.COMPARE;

	if (is_elementwise)
	{
		usz num_unique = count_unique_inputs(entry.expr);
		if (num_unique > MAX_UNIQUE_INPUTS)
		{
			decompose_expression(device, g, entry);
			return;
		}
	}

	void* output = (void*)entry.output;

	// Build index mapping: IndexId → character for einsum patterns
	IndexCharMap map;
	map.init();

	// Assign characters to output indices first
	foreach (idx_id : entry.output_indices)
	{
		map.put(idx_id);
	}

	// Collect contracted and access indices
	collect_contracted_indices(entry.expr, &map);
	collect_all_access_indices(entry.expr, &map);

	// Build output pattern
	char[types::MAX_PATTERN_LEN] output_pattern;
	char output_pattern_len;
	build_pattern(entry.output_indices, &map, &output_pattern, &output_pattern_len);

	// Log compilation entry
	logging::@debug(COMPILER, "compile_cas_expr: kind=%d output_ndim=%d output_indices=%d map_len=%d out_pat_len=%d",
		(int)entry.expr.kind,
		(int)((tensor_core::TensorCore*)entry.output).shape().len,
		(int)entry.output_indices.len,
		(int)map.len,
		(int)output_pattern_len);
	if (logging::is_enabled(logging::Scope.COMPILER, logging::Level.DEBUG))
	{
		// Log index map
		for (usz mi = 0; mi < map.len; mi++)
		{
			logging::@debug(COMPILER, "  map[%d]: IndexId=%d -> '%c'",
				(int)mi, (int)map.ids[mi], map.chars[mi]);
		}
		// Log output pattern chars
		for (usz pi = 0; pi < (usz)output_pattern_len; pi++)
		{
			logging::@debug(COMPILER, "  out_pat[%d]: '%c' (0x%02x)",
				(int)pi, output_pattern[pi], (int)output_pattern[pi]);
		}
	}

	// Analyze expression and build CudaOp(s)
	CudaOp[MAX_COMPILE_OPS] ops_buf;
	usz ops_count = 0;

	expr_to_cuda_ops(
		device,
		entry.expr,
		output,
		entry,
		&map,
		output_pattern[:output_pattern_len],
		&ops_buf,
		&ops_count,
	);

	assert(ops_count <= MAX_COMPILE_OPS, "compile_cas_expr: ops_count exceeded buffer capacity");

	if (ops_count == 0) return;

	// Log compiled ops before fusion
	if (logging::is_enabled(logging::Scope.COMPILER, logging::Level.DEBUG))
	{
		for (usz oi = 0; oi < ops_count; oi++)
		{
			CudaOp* cop = &ops_buf[oi];
			char[] opat = cop.get_output_pattern();
			logging::@debug(COMPILER, "  op[%d]: type=%d reduce_op=%d input_count=%d out_pat_len=%d",
				(int)oi, (int)cop.op_type, (int)cop.contraction.reduce_op,
				(int)cop.input_count, (int)opat.len);
			for (usz pi = 0; pi < opat.len; pi++)
			{
				logging::@debug(COMPILER, "    out_pat[%d]: '%c' (0x%02x)", (int)pi, opat[pi], (int)opat[pi]);
			}
			for (usz ii = 0; ii < (usz)cop.input_count; ii++)
			{
				char[] ipat = cop.get_input_pattern(ii);
				tensor_core::TensorCore* inp_tc = (tensor_core::TensorCore*)cop.inputs[ii];
				logging::@debug(COMPILER, "    input[%d]: ndim=%d pat_len=%d",
					(int)ii, inp_tc != null ? (int)inp_tc.shape().len : -1, (int)ipat.len);
				for (usz pi = 0; pi < ipat.len; pi++)
				{
					logging::@debug(COMPILER, "      in_pat[%d]: '%c' (0x%02x)", (int)pi, ipat[pi], (int)ipat[pi]);
				}
			}
		}
	}

	// Apply fusion pass
	apply_fusion_pass(&ops_buf, &ops_count);

	// Register each op as an execution unit
	for (usz i = 0; i < ops_count; i++)
	{
		register_cuda_op_with_inputs(device, g, ops_buf[i]);
	}
}

// expr_to_cuda_ops — recursive expression walker

<*
 @require expr != null : "expr_to_cuda_ops: expression must not be null"
 @require output != null : "expr_to_cuda_ops: output must not be null"
 @require ops_buf != null : "expr_to_cuda_ops: ops buffer must not be null"
 @require ops_count != null : "expr_to_cuda_ops: ops count must not be null"
*>
fn void expr_to_cuda_ops(
	CudaDevice* device,
	Expr* expr,
	void* output,
	IREntry* entry,
	IndexCharMap* map,
	char[] output_pattern,
	CudaOp* ops_buf,    // pointer to fixed-size buffer
	usz* ops_count,
)
{
	if (*ops_count >= MAX_COMPILE_OPS) return;

	switch (expr.kind)
	{
		case CONTRACT:
			compile_contract(expr, output, entry, map, output_pattern, ops_buf, ops_count);

		case ACCESS:
			compile_access(expr, output, map, output_pattern, ops_buf, ops_count);

		case COMPARE:
			compile_compare(expr, output, entry, ops_buf, ops_count);

		case SELECT:
			compile_select(expr, output, entry, ops_buf, ops_count);

		case PERMUTE:
			compile_permute(expr, output, map, output_pattern, ops_buf, ops_count);

		case SCALAR:
			{} // Scalar at top level — nothing to emit

		case BUILTIN:
			{} // Should have been handled as boundary op

		// All elementwise: try linearization, fallback to JIT
		case SUM:
			compile_elementwise(device, expr, output, entry, map, output_pattern, ops_buf, ops_count);
		case PRODUCT:
			// Check if this is a product of two indexed accesses with DIFFERENT index
			// patterns (outer product or broadcast from einsum). Route to cuTENSOR
			// contraction which handles these natively. Element-wise products (same
			// indices on both inputs) stay on the linear chain path.
			if (expr.operands.len >= 2 &&
				expr.operands[0].kind == ExprKind.ACCESS &&
				expr.operands[0].access.indices.len > 0 &&
				expr.operands[1].kind == ExprKind.ACCESS &&
				expr.operands[1].access.indices.len > 0 &&
				!indices_match(expr.operands[0].access.indices, expr.operands[1].access.indices))
			{
				CudaOp cuda_op;
				cuda_op.op_type = CONTRACTION;
				cuda_op.result = output;
				cuda_op.set_output_pattern(output_pattern);
				cuda_op.contraction.reduce_op = operations::BinaryOp.ADD;
				set_input_from_factor(expr.operands[0], &cuda_op, 0, map);
				set_input_from_factor(expr.operands[1], &cuda_op, 1, map);
				ops_buf[*ops_count] = cuda_op;
				(*ops_count)++;
			}
			else
			{
				compile_elementwise(device, expr, output, entry, map, output_pattern, ops_buf, ops_count);
			}
		case UNARY:
			compile_elementwise(device, expr, output, entry, map, output_pattern, ops_buf, ops_count);
		case NEGATE:
			compile_elementwise(device, expr, output, entry, map, output_pattern, ops_buf, ops_count);
		case BINARY:
			compile_elementwise(device, expr, output, entry, map, output_pattern, ops_buf, ops_count);
	}
}

fn void compile_contract(
	Expr* expr,
	void* output,
	IREntry* entry,
	IndexCharMap* map,
	char[] output_pattern,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	Expr* inner_body = unwrap_nested_contracts(expr);

	if (inner_body.kind == ExprKind.PRODUCT && inner_body.operands.len >= 2)
	{
		// contract(..., product[A, B]) → cuTENSOR contraction
		CudaOp cuda_op;
		cuda_op.op_type = CONTRACTION;
		cuda_op.result = output;
		cuda_op.set_output_pattern(output_pattern);
		cuda_op.contraction.reduce_op = cas_reduce_to_binary(expr.contract.op);

		// Extract inputs from factors
		set_input_from_factor(inner_body.operands[0], &cuda_op, 0, map);
		set_input_from_factor(inner_body.operands[1], &cuda_op, 1, map);

		ops_buf[*ops_count] = cuda_op;
		(*ops_count)++;
	}
	else if (inner_body.kind == ExprKind.ACCESS)
	{
		// contract(..., access(A)) → cuTENSOR reduction
		CudaOp cuda_op;
		cuda_op.op_type = REDUCTION;
		cuda_op.result = output;
		cuda_op.set_output_pattern(output_pattern);
		cuda_op.reduction.reduce_op = cas_reduce_to_binary(expr.contract.op);
		set_input_from_access(
			(void*)inner_body.access.tensor,
			inner_body.access.indices,
			&cuda_op, 0, map,
		);

		ops_buf[*ops_count] = cuda_op;
		(*ops_count)++;
	}
	else
	{
		// contract over complex expression → reduction
		CudaOp cuda_op;
		cuda_op.op_type = REDUCTION;
		cuda_op.result = output;
		cuda_op.set_output_pattern(output_pattern);
		cuda_op.reduction.reduce_op = cas_reduce_to_binary(expr.contract.op);

		if (try acc = find_first_access(inner_body))
		{
			set_input_from_access(acc.tensor, acc.indices, &cuda_op, 0, map);
		}
		else if (entry.inputs.len > 0)
		{
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.input_count = 1;
		}

		ops_buf[*ops_count] = cuda_op;
		(*ops_count)++;
	}
}

fn void compile_access(
	Expr* expr,
	void* output,
	IndexCharMap* map,
	char[] output_pattern,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	CudaOp cuda_op;
	cuda_op.op_type = PERMUTATION;
	cuda_op.result = output;
	cuda_op.set_output_pattern(output_pattern);

	char[types::MAX_PATTERN_LEN] in_pattern;
	char in_len;
	build_pattern(expr.access.indices, map, &in_pattern, &in_len);
	cuda_op.set_input_pattern(0, in_pattern[:in_len]);
	cuda_op.inputs[0] = (void*)expr.access.tensor;
	cuda_op.input_count = 1;

	ops_buf[*ops_count] = cuda_op;
	(*ops_count)++;
}

fn void compile_compare(
	Expr* expr,
	void* output,
	IREntry* entry,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	operations::BinaryOp binary_op;
	switch (expr.compare.op)
	{
		case EQ:  binary_op = EQL;
		case NE:  binary_op = NEQ;
		case GT:  binary_op = GT;
		case LT:  binary_op = LT;
		case GTE: binary_op = GTE;
		case LTE: binary_op = LTE;
	}

	CudaOp cuda_op;
	cuda_op.op_type = ELEMENTWISE_BINARY;
	cuda_op.result = output;
	cuda_op.elem_binary.binary_op = binary_op;

	if (try la = find_first_access(expr.compare.lhs))
	{
		cuda_op.inputs[0] = la.tensor;
	}
	else if (entry.inputs.len > 0)
	{
		cuda_op.inputs[0] = (void*)entry.inputs[0];
	}

	if (try ra = find_first_access(expr.compare.rhs))
	{
		cuda_op.inputs[1] = ra.tensor;
	}
	else if (entry.inputs.len > 1)
	{
		cuda_op.inputs[1] = (void*)entry.inputs[1];
	}
	cuda_op.input_count = 2;

	ops_buf[*ops_count] = cuda_op;
	(*ops_count)++;
}

fn void compile_select(
	Expr* expr,
	void* output,
	IREntry* entry,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	CudaOp cuda_op;
	cuda_op.op_type = ELEMENTWISE_SELECT;
	cuda_op.result = output;

	if (try a = find_first_access(expr.select_.cond)) {
		cuda_op.inputs[0] = a.tensor;
	} else if (entry.inputs.len > 0) {
		cuda_op.inputs[0] = (void*)entry.inputs[0];
	}

	if (try a = find_first_access(expr.select_.then_expr)) {
		cuda_op.inputs[1] = a.tensor;
	} else if (entry.inputs.len > 1) {
		cuda_op.inputs[1] = (void*)entry.inputs[1];
	}

	if (try a = find_first_access(expr.select_.else_expr)) {
		cuda_op.inputs[2] = a.tensor;
	} else if (entry.inputs.len > 2) {
		cuda_op.inputs[2] = (void*)entry.inputs[2];
	}

	cuda_op.input_count = 3;

	ops_buf[*ops_count] = cuda_op;
	(*ops_count)++;
}

fn void compile_permute(
	Expr* expr,
	void* output,
	IndexCharMap* map,
	char[] output_pattern,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	if (try acc = find_first_access(expr.permute.child))
	{
		CudaOp cuda_op;
		cuda_op.op_type = PERMUTATION;
		cuda_op.result = output;
		cuda_op.set_output_pattern(output_pattern);

		char[types::MAX_PATTERN_LEN] in_pattern;
		char in_len;
		build_pattern(acc.indices, map, &in_pattern, &in_len);
		cuda_op.set_input_pattern(0, in_pattern[:in_len]);
		cuda_op.inputs[0] = acc.tensor;
		cuda_op.input_count = 1;

		ops_buf[*ops_count] = cuda_op;
		(*ops_count)++;
	}
}

fn void set_input_from_factor(
	Expr* factor,
	CudaOp* cuda_op,
	usz slot,
	IndexCharMap* map,
)
{
	if (factor.kind == ExprKind.ACCESS)
	{
		set_input_from_access(
			(void*)factor.access.tensor,
			factor.access.indices,
			cuda_op, slot, map,
		);
	}
	else if (try acc = find_first_access(factor))
	{
		set_input_from_access(acc.tensor, acc.indices, cuda_op, slot, map);
	}
}

// Expression decomposition — split large expressions into smaller JIT kernels

/// Maximum number of groups when partitioning n-ary operands.
const usz MAX_DECOMPOSE_GROUPS = 64;

/// Decompose an expression that exceeds MAX_UNIQUE_INPUTS.
///
/// Strategy:
/// - SUM/PRODUCT (n-ary): partition operands into groups, compile each group
///   into an intermediate tensor, then combine intermediates in a final kernel.
/// - All other kinds: find the child subtree with the most unique inputs,
///   materialize it into an intermediate tensor, rebuild the parent with an
///   ACCESS to the intermediate, and re-enter compilation (which may recurse).
///
/// IMPORTANT: all recursive compilation goes through compile_jit_entry (not
/// compile_cas_expr) so sub-expressions stay on the JIT path and are never
/// re-routed to cuTENSOR / permutation / reduction backends.
fn void decompose_expression(
	CudaDevice* device,
	graph::Graph* g,
	IREntry* entry,
)
{
	Expr* expr = entry.expr;

	if (expr.kind == ExprKind.SUM || expr.kind == ExprKind.PRODUCT)
	{
		decompose_nary(device, g, entry);
	}
	else
	{
		materialize_heaviest_child(device, g, entry);
	}
}

/// Compile an IREntry as a JIT kernel, decomposing if needed.
/// Unlike compile_cas_expr, this ALWAYS stays on the JIT path — it never
/// routes sub-expressions to cuTENSOR or other backends.
fn void compile_jit_entry(
	CudaDevice* device,
	graph::Graph* g,
	IREntry* entry,
)
{
	usz num_unique = count_unique_inputs(entry.expr);
	if (num_unique > MAX_UNIQUE_INPUTS)
	{
		// Still too large — decompose further (stays on JIT path)
		decompose_expression(device, g, entry);
		return;
	}

	// Small enough — emit a single JIT kernel directly
	CudaOp cuda_op;
	cuda_op.op_type = ELEMENTWISE_CAS_EXPR;
	cuda_op.result = (void*)entry.output;
	generate_cuda_kernel_from_cas(device, entry.expr, entry, (void*)entry.output, &cuda_op);
	register_cuda_op_with_inputs(device, g, cuda_op);
}

/// Partition a SUM or PRODUCT's operands into groups with ≤MAX_UNIQUE_INPUTS
/// each. Each group is compiled into an intermediate tensor, then a final
/// kernel combines the intermediates with the same n-ary operation.
fn void decompose_nary(
	CudaDevice* device,
	graph::Graph* g,
	IREntry* entry,
)
{
	Expr* expr = entry.expr;
	ExprKind kind = expr.kind;
	assert(kind == ExprKind.SUM || kind == ExprKind.PRODUCT);

	Expr*[] operands = expr.operands;
	usz num_operands = operands.len;

	// Greedy partition: walk operands left-to-right, accumulating unique inputs.
	// When adding the next operand would exceed MAX_UNIQUE_INPUTS, close the group.
	usz[MAX_DECOMPOSE_GROUPS + 1] group_starts;
	usz num_groups = 0;
	group_starts[0] = 0;

	CodegenContext current_inputs;
	current_inputs.tinit();

	for (usz i = 0; i < num_operands; i++)
	{
		// Save current state so we can undo if this operand overflows the group.
		usz saved_len = current_inputs.count();
		codegen_context::collect_unique_inputs(operands[i], &current_inputs);

		// Close the group if adding this operand overflows AND the group
		// already has at least one operand (i > group start).
		if (current_inputs.count() > MAX_UNIQUE_INPUTS && i > group_starts[num_groups])
		{
			// Undo: truncate back to saved state
			current_inputs.tensors.size = saved_len;

			num_groups++;
			assert(num_groups < MAX_DECOMPOSE_GROUPS, "decompose_nary: too many groups");
			group_starts[num_groups] = i;

			// Start fresh group with just this operand
			current_inputs.tensors.size = 0;
			codegen_context::collect_unique_inputs(operands[i], &current_inputs);
		}
	}
	num_groups++;
	group_starts[num_groups] = num_operands;

	assert(num_groups >= 2, "decompose_nary: partition produced fewer than 2 groups");

	// Get output tensor core for creating intermediates
	tensor_core::TensorCore* out_tc = (tensor_core::TensorCore*)entry.output;

	// Compile each group into an intermediate tensor
	tensor_core::TensorCore*[MAX_DECOMPOSE_GROUPS] intermediates;

	for (usz gi = 0; gi < num_groups; gi++)
	{
		usz start = group_starts[gi];
		usz end = group_starts[gi + 1];
		usz group_len = end - start;

		tensor_core::TensorCore* intermediate = g.create_tensor_core(
			out_tc.shape().slice(),
			out_tc.dtype(),
			out_tc.device,
		);
		intermediates[gi] = intermediate;

		// Build group expression (single operand or n-ary of group)
		Expr* group_expr;
		if (group_len == 1)
		{
			group_expr = operands[start];
		}
		else
		{
			Expr*[] group_ops = allocator::alloc_array_try(tmem, Expr*, group_len)!!;
			for (usz j = 0; j < group_len; j++)
			{
				group_ops[j] = operands[start + j];
			}
			Expr* node = allocator::alloc_try(tmem, Expr)!!;
			*node = { .kind = kind, .operands = group_ops };
			group_expr = node;
		}

		IREntry group_entry = {
			.output = intermediate,
			.expr = group_expr,
			.inputs = entry.inputs,
			.output_indices = entry.output_indices,
			.payload = null,
		};
		compile_jit_entry(device, g, &group_entry);
	}

	// Build final n-ary of ACCESS nodes pointing to the intermediates
	Expr*[] final_ops = allocator::alloc_array_try(tmem, Expr*, num_groups)!!;
	for (usz gi = 0; gi < num_groups; gi++)
	{
		IndexId[] idx_copy = allocator::alloc_array_try(tmem, IndexId, entry.output_indices.len)!!;
		idx_copy[..] = entry.output_indices[..];

		Expr* acc = allocator::alloc_try(tmem, Expr)!!;
		*acc = {
			.kind = ACCESS,
			.access = {
				.tensor = intermediates[gi],
				.indices = idx_copy,
				.is_stable = false,
			},
		};
		final_ops[gi] = acc;
	}

	Expr* final_expr = allocator::alloc_try(tmem, Expr)!!;
	*final_expr = { .kind = kind, .operands = final_ops };

	tensor_core::TensorCore*[] final_inputs = allocator::alloc_array_try(tmem, tensor_core::TensorCore*, num_groups)!!;
	for (usz gi = 0; gi < num_groups; gi++)
	{
		final_inputs[gi] = intermediates[gi];
	}

	IREntry final_entry = {
		.output = entry.output,
		.expr = final_expr,
		.inputs = final_inputs,
		.output_indices = entry.output_indices,
		.payload = null,
	};
	compile_jit_entry(device, g, &final_entry);
}

/// For non-n-ary expressions (BINARY, UNARY, NEGATE, SELECT, etc.):
/// find the child subtree with the most unique inputs, compile it into
/// an intermediate tensor, replace it with ACCESS(intermediate), and
/// re-enter compilation with the simplified expression.
fn void materialize_heaviest_child(
	CudaDevice* device,
	graph::Graph* g,
	IREntry* entry,
)
{
	Expr* expr = entry.expr;
	tensor_core::TensorCore* out_tc = (tensor_core::TensorCore*)entry.output;

	// Find the heaviest child and materialize it
	Expr* heaviest;
	usz heaviest_count = 0;

	switch (expr.kind)
	{
		case NEGATE:
			heaviest = expr.inner;
			heaviest_count = count_unique_inputs(heaviest);
		case UNARY:
			heaviest = expr.unary.child;
			heaviest_count = count_unique_inputs(heaviest);
		case BINARY:
			usz lc = count_unique_inputs(expr.binary.lhs);
			usz rc = count_unique_inputs(expr.binary.rhs);
			if (lc >= rc) { heaviest = expr.binary.lhs; heaviest_count = lc; }
			else { heaviest = expr.binary.rhs; heaviest_count = rc; }
		case SELECT:
			Expr*[3] sel_children = { expr.select_.cond, expr.select_.then_expr, expr.select_.else_expr };
			heaviest = sel_children[0];
			heaviest_count = count_unique_inputs(sel_children[0]);
			for (usz i = 1; i < 3; i++)
			{
				usz c = count_unique_inputs(sel_children[i]);
				if (c > heaviest_count) { heaviest = sel_children[i]; heaviest_count = c; }
			}
		case COMPARE:
			usz lc2 = count_unique_inputs(expr.compare.lhs);
			usz rc2 = count_unique_inputs(expr.compare.rhs);
			if (lc2 >= rc2) { heaviest = expr.compare.lhs; heaviest_count = lc2; }
			else { heaviest = expr.compare.rhs; heaviest_count = rc2; }
		case PERMUTE:
			heaviest = expr.permute.child;
			heaviest_count = count_unique_inputs(heaviest);
		default:
			unreachable("materialize_heaviest_child: unhandled expression kind");
	}

	assert(heaviest_count > 0, "materialize_heaviest_child: heaviest child has no inputs");

	// Compile the heaviest child into an intermediate tensor
	tensor_core::TensorCore* intermediate = g.create_tensor_core(
		out_tc.shape().slice(),
		out_tc.dtype(),
		out_tc.device,
	);

	IREntry child_entry = {
		.output = intermediate,
		.expr = heaviest,
		.inputs = entry.inputs,
		.output_indices = entry.output_indices,
		.payload = null,
	};
	compile_jit_entry(device, g, &child_entry);

	// Build ACCESS node to the intermediate
	IndexId[] idx_copy = allocator::alloc_array_try(tmem, IndexId, entry.output_indices.len)!!;
	idx_copy[..] = entry.output_indices[..];

	Expr* access_node = allocator::alloc_try(tmem, Expr)!!;
	*access_node = {
		.kind = ACCESS,
		.access = {
			.tensor = intermediate,
			.indices = idx_copy,
			.is_stable = false,
		},
	};

	// Rebuild parent expression with the heaviest child replaced
	Expr* new_expr = rebuild_with_replacement(expr, heaviest, access_node);

	IREntry new_entry = {
		.output = entry.output,
		.expr = new_expr,
		.inputs = entry.inputs,
		.output_indices = entry.output_indices,
		.payload = entry.payload,
	};
	compile_jit_entry(device, g, &new_entry);
}

/// Rebuild an expression, replacing one child pointer with a new expression.
/// Only rebuilds the immediate node — does not recurse into children.
fn Expr* rebuild_with_replacement(Expr* expr, Expr* old_child, Expr* new_child)
{
	Expr* result = allocator::alloc_try(tmem, Expr)!!;

	switch (expr.kind)
	{
		case NEGATE:
			*result = { .kind = NEGATE, .inner = new_child };
		case UNARY:
			*result = { .kind = UNARY, .unary = { .op = expr.unary.op, .child = new_child } };
		case BINARY:
			Expr* lhs = (expr.binary.lhs == old_child) ? new_child : expr.binary.lhs;
			Expr* rhs = (expr.binary.rhs == old_child) ? new_child : expr.binary.rhs;
			*result = { .kind = BINARY, .binary = { .op = expr.binary.op, .lhs = lhs, .rhs = rhs } };
		case SELECT:
			Expr* cond = (expr.select_.cond == old_child) ? new_child : expr.select_.cond;
			Expr* then_ = (expr.select_.then_expr == old_child) ? new_child : expr.select_.then_expr;
			Expr* else_ = (expr.select_.else_expr == old_child) ? new_child : expr.select_.else_expr;
			*result = { .kind = SELECT, .select_ = { .cond = cond, .then_expr = then_, .else_expr = else_ } };
		case COMPARE:
			Expr* cl = (expr.compare.lhs == old_child) ? new_child : expr.compare.lhs;
			Expr* cr = (expr.compare.rhs == old_child) ? new_child : expr.compare.rhs;
			*result = { .kind = COMPARE, .compare = { .op = expr.compare.op, .lhs = cl, .rhs = cr } };
		case PERMUTE:
			*result = { .kind = PERMUTE, .permute = { .child = new_child, .mapping = expr.permute.mapping } };
		default:
			unreachable("rebuild_with_replacement: unhandled expression kind");
	}
	return result;
}

// Elementwise compilation — JIT kernel emission

/// Count the number of unique inputs in an expression. Uses the CodegenContext
/// from compile_jit.c3.
fn usz count_unique_inputs(Expr* expr)
{
	CodegenContext imap;
	imap.tinit();
	codegen_context::collect_unique_inputs(expr, &imap);
	return imap.count();
}

fn void compile_elementwise(
	CudaDevice* device,
	Expr* expr,
	void* output,
	IREntry* entry,
	IndexCharMap* map,
	char[] output_pattern,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	usz num_inputs = count_unique_inputs(expr);

	// Depth is not a concern for JIT kernels — arbitrary depth just means
	// more lines of CUDA code per thread. Only input count matters because
	// each unique input becomes a kernel parameter.
	assert(num_inputs <= MAX_UNIQUE_INPUTS,
		"Expression has too many unique inputs for a single JIT kernel");

	emit_jit_kernel(device, expr, output, entry, ops_buf, ops_count);
}

/// Emit a single JIT kernel for an expression.
fn void emit_jit_kernel(
	CudaDevice* device,
	Expr* expr,
	void* output,
	IREntry* entry,
	CudaOp* ops_buf,
	usz* ops_count,
)
{
	if (*ops_count >= MAX_COMPILE_OPS) return;

	CudaOp cuda_op;
	cuda_op.op_type = ELEMENTWISE_CAS_EXPR;
	cuda_op.result = output;
	generate_cuda_kernel_from_cas(device, expr, entry, output, &cuda_op);

	ops_buf[*ops_count] = cuda_op;
	(*ops_count)++;
}


