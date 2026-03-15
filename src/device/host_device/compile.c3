//! Compile Entry Point
//!
//! Main compilation entry point that orchestrates the compilation pipeline:
//! 1. Analyze CAS expression to extract index info
//! 2. Hash the IR entry for cache lookup
//! 3. Build FusedKernel from IR
//! 4. Generate C3 source via kernel_codegen
//! 5. Compile and cache via kernel_cache
//! 6. Create ExecutableCallback for execution
//!

module metaphor::device::host_device::compile;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::rtti;
import metaphor::cas;
import metaphor::codegen::kernel_ir;
import metaphor::codegen::kernel_codegen;
import metaphor::device::host_device;
import metaphor::device::host_device::kernel_cache;
import metaphor::device::host_device::executable_callback;
import metaphor::device::common::jit_cache;
import metaphor::device::common::op_payload;
import metaphor::device::host_device::boundary_callbacks;
import metaphor::device::host_device::blas::vendor_kernels;
import metaphor::device::host_device::blas::blas_callbacks;
import metaphor::device::common::index_expr;
import metaphor::device::common::operations;
import metaphor::logging;
import std::hash::fnv64a;
import std::core::mem::allocator;
import std::io;

// Specialize JitCache types for host's CachedKernel.
alias JitKey = jit_cache::JitKey{kernel_cache::CachedKernel};

// Constants

const usz MAX_INPUTS = 32;
const usz MAX_INDICES = 16;
const usz MAX_SOURCE_SIZE = 64 * 1024; // 64KB source buffer

// Main Entry Point

/// Compile a single IR entry to an execution unit on the graph.
/// Called after the CAS mapper has built the IR.
fn void compile(host_device::HostDevice* device, graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;

	logging::@debug(COMPILER, "output=%p inputs=%d", output, (int)entry.inputs.len);

	// Skip if already compiled
	if (g.has_execution_unit(output))
	{
		logging::@debug(COMPILER, "already compiled, skipping");
		return;
	}

	kernel_cache::initialize_kernel_cache();

	// Boundary op dispatch: conv, gather, scatter, slice, concat, etc.
	// If the payload tag identifies a boundary op, handle it and return.
	if (boundary_callbacks::dispatch_boundary_op(g, entry))
	{
		return;
	}

	// BLAS vendor kernel dispatch: try to match EINSUM payloads
	// to optimized CBLAS operations before falling through to JIT.
	if (entry.payload != null && entry.payload.tag() == operations::OpTag.EINSUM)
	{
		EinsumPayload* cp = &entry.payload.einsum;
		if (entry.inputs.len == 2)
		{
			// Reconstruct pattern string from IndexExpr
			char[64] pat_buf;
			usz pat_len = 0;

			IndexExpr* expr = &cp.expr;
			// Input 0
			for (usz pi = 0; pi < (usz)expr.input_lens[0]; pi++)
			{
				pat_buf[pat_len] = expr.input0[pi];
				pat_len++;
			}
			pat_buf[pat_len] = ',';
			pat_len++;
			// Input 1
			for (usz pi = 0; pi < (usz)expr.input_lens[1]; pi++)
			{
				pat_buf[pat_len] = expr.input1[pi];
				pat_len++;
			}
			pat_buf[pat_len] = '-';
			pat_len++;
			pat_buf[pat_len] = '>';
			pat_len++;
			// Output
			for (usz pi = 0; pi < (usz)expr.output_len; pi++)
			{
				pat_buf[pat_len] = expr.output[pi];
				pat_len++;
			}

			TensorCore*[2] children = { entry.inputs[0], entry.inputs[1] };
			rtti::Datatype dtype = output.dtype();

			if (try vkr = vendor_kernels::lookup(
				pat_buf[:pat_len],
				children,
				output,
				cp.inner,
				cp.outer,
				dtype))
			{
				TensorCore*[2] source_cores;
				for (usz si = 0; si < (usz)vkr.source_count; si++)
				{
					source_cores[si] = vkr.sources[si];
				}
				g.add_execution_unit(
					vkr.operation,
					source_cores[:(usz)vkr.source_count],
					vkr.target,
				);
				return;
			}
		}
	}

	// Analyze CAS expression to extract index info
	CASAnalysis analysis = analyze_cas_expr(entry);

	assert(!analysis.is_inplace, "Inplace operations from CAS not yet supported");

	usz children_count = entry.inputs.len;
	assert(children_count <= MAX_INPUTS, "IR entry has too many inputs (max 32)");

	// ---- Build JitKey ----
	// Host kernels handle all sizes at runtime (loop bounds from dims[]),
	// so we encode rank only (all dims → DYNAMIC) rather than sentinelized shapes.
	JitKey key = {};
	defer key.free();

	key.write_cas_preamble((void**)entry.inputs.ptr, children_count, output.dtype());

	// Expression tree structure
	key.write_expr(entry.expr, (void**)entry.inputs.ptr, children_count);

	// Input ranks (all dims DYNAMIC — host kernels are size-independent)
	for (usz i = 0; i < children_count; i++)
	{
		key.write_rank(entry.inputs[i].shape().len);
	}
	// Output rank
	key.write_rank(output.shape().len);

	ulong hash = key.hash();

	// JIT compilation path
	logging::@debug(COMPILER, "hash=%lx free=%d bound=%d", hash,
		(int)analysis.num_free, (int)analysis.num_bound);

	// ---- Fast path: JitCache (in-memory) ----
	if (catch device.jit_kernel_cache.get(&key))
	{
		// JitCache miss — check disk cache (kernel_cache)
		if (catch kernel_cache::lookup_cached_kernel(hash))
		{
			// Disk miss — build kernel, generate source, compile
			FusedKernel kernel = build_fused_kernel_from_ir_entry(entry, &analysis, hash);
			char[MAX_SOURCE_SIZE] source_buf;
			char[] source = kernel_codegen::generate_kernel_source(&kernel, &source_buf);

			// Log kernel details
			if (logging::is_enabled(logging::Scope.DISPLAY_KERNEL, logging::Level.DEBUG))
			{
				for (usz di = 0; di < kernel.num_inputs; di++)
				{
					TensorBinding* b = &kernel.inputs[di];
					logging::@debug(DISPLAY_KERNEL, "input[%d]: indices_len=%d shape_len=%d",
						(int)di, (int)b.indices_len, (int)b.tensor.shape().len);
				}
			}

			// The kernel name is a ZString for dlsym
			char[kernel_ir::MAX_NAME_LEN + 1] name_z_buf;
			char[] name = kernel.name();
			for (usz i = 0; i < name.len; i++)
			{
				name_z_buf[i] = name[i];
			}
			name_z_buf[name.len] = 0;
			ZString func_name = (ZString)&name_z_buf;
			kernel_cache::compile_and_cache(hash, func_name, source)!!;
		}

		// Promote from kernel_cache into JitCache for fast subsequent lookups
		if (try cached = kernel_cache::lookup_cached_kernel(hash))
		{
			device.jit_kernel_cache.put(&key, cached);
		}
	}

	// Build bindings for the ExecutableCallback
	// Each input gets an INPUT binding, plus one OUTPUT binding

	// Build ID-to-name mapping for index patterns
	IdNameEntry[MAX_INDICES] id_to_name;
	usz id_to_name_count = 0;

	// Free indices from output_indices
	for (usz i = 0; i < entry.output_indices.len; i++)
	{
		id_to_name[id_to_name_count] = {
			.id = entry.output_indices[i],
			.name = (char)('i' + i),
		};
		id_to_name_count++;
	}

	// Bound indices from contract nodes
	IdNameEntry[MAX_INDICES] bound_ids;
	usz bound_ids_count = 0;
	collect_bound_ids(entry.expr, &bound_ids, &bound_ids_count, &analysis);
	for (usz i = 0; i < bound_ids_count; i++)
	{
		id_to_name[id_to_name_count] = bound_ids[i];
		id_to_name_count++;
	}

	// Collect CAS accesses (tensor → index pattern)
	AccessEntry[MAX_INPUTS] access_map;
	usz access_map_count = 0;
	collect_cas_accesses(entry.expr, &access_map, &access_map_count);

	// Create the callback (allocated on the graph's allocator so it persists)
	ExecutableCallback* callback = allocator::new(g.allocator, ExecutableCallback);
	*callback = {};
	// Set bindings directly on the callback (embedded fixed array)
	for (usz i = 0; i < children_count; i++)
	{
		callback.bindings_buf[i] = {
			.slot = (char)i,
			.source = INPUT,
			.source_idx = (char)i,
		};
	}
	callback.bindings_buf[children_count] = {
		.slot = 0,
		.source = OUTPUT,
		.source_idx = 0,
	};
	callback.bindings_count = children_count + 1;
	callback.input_count = (char)children_count;
	callback.hash = hash;
	callback.is_inplace = analysis.is_inplace;

	// Set index patterns on callback for each input
	for (usz i = 0; i < children_count; i++)
	{
		TensorCore* input = entry.inputs[i];
		usz actual_rank = input.shape().len;

		// Look up CAS indices for this input
		IndexId[8] cas_indices;
		usz cas_len = 0;
		for (usz a = 0; a < access_map_count; a++)
		{
			if (access_map[a].tensor == input)
			{
				cas_len = access_map[a].count;
				for (usz j = 0; j < cas_len; j++)
				{
					cas_indices[j] = access_map[a].indices[j];
				}
				break;
			}
		}

		if (cas_len > 0)
		{
			usz pat_len = cas_len < actual_rank ? cas_len : actual_rank;
			char[8] pattern_buf;
			for (usz j = 0; j < pat_len; j++)
			{
				pattern_buf[j] = lookup_id_name(&id_to_name, id_to_name_count, cas_indices[j]);
			}
			callback.set_index_pattern((char)i, pattern_buf[:pat_len]);
		}
		else
		{
			// Element-wise: sequential indices based on shape
			char[8] pattern_buf;
			for (usz j = 0; j < actual_rank; j++)
			{
				pattern_buf[j] = (char)('i' + j);
			}
			callback.set_index_pattern((char)i, pattern_buf[:actual_rank]);
		}
	}

	// Set dim names and add dims
	char dim_idx = 0;
	for (usz i = 0; i < analysis.num_free; i++)
	{
		callback.set_dim_name(dim_idx, analysis.free_indices[i].name_buf[0]);
		callback.add_dim(analysis.free_indices[i].size);
		dim_idx++;
	}
	callback.num_free_dims = (char)analysis.num_free;
	for (usz i = 0; i < analysis.num_bound; i++)
	{
		callback.set_dim_name(dim_idx, analysis.bound_indices[i].name_buf[0]);
		callback.add_dim(analysis.bound_indices[i].size);
		dim_idx++;
	}

	// Build source cores array
	TensorCore*[MAX_INPUTS] source_cores;
	for (usz i = 0; i < children_count; i++)
	{
		source_cores[i] = entry.inputs[i];
	}

	// Add execution unit to the graph
	g.add_execution_unit(
		(any)callback,
		source_cores[:children_count],
		output,
	);
}

// CAS Expression Analysis

struct CASAnalysis
{
	KernelIndex[MAX_INDICES] free_indices;
	usz num_free;
	KernelIndex[MAX_INDICES] bound_indices;
	usz num_bound;
	cas::ReduceOp reduce_op;
	bool has_reduce_op;
	bool is_inplace;
}

/// Analyze a CAS expression to extract free/bound indices and reduce op.
fn CASAnalysis analyze_cas_expr(cas::IREntry* entry)
{
	CASAnalysis analysis;

	// Free indices: from output_indices, named 'i', 'j', 'k', ...
	if (entry.output_indices.len > 0)
	{
		ulong[] output_shape = entry.output.shape().slice();
		for (usz i = 0; i < entry.output_indices.len; i++)
		{
			ulong size = i < output_shape.len ? output_shape[i] : 1;
			analysis.free_indices[analysis.num_free] =
				kernel_ir::kernel_index_from_char((char)('i' + i), size);
			analysis.num_free++;
		}
	}
	else
	{
		// Fallback: derive from output shape
		ulong[] output_shape = entry.output.shape().slice();
		for (usz i = 0; i < output_shape.len; i++)
		{
			analysis.free_indices[analysis.num_free] =
				kernel_ir::kernel_index_from_char((char)('i' + i), output_shape[i]);
			analysis.num_free++;
		}
	}

	// Walk CAS expression to find contract nodes (bound indices)
	extract_bound_indices(entry.expr, &analysis);

	return analysis;
}

/// Walk CAS expression to find contract nodes and extract bound indices.
fn void extract_bound_indices(cas::Expr* expr, CASAnalysis* analysis)
{
	switch (expr.kind)
	{
		case CONTRACT:
			// Use 'a', 'b', 'c', ... for bound indices to avoid collision with free 'i', 'j', 'k', ...
			// Previous 'x' + idx scheme overflowed past 'z' with 4+ bound indices.
			usz bound_idx = analysis.num_bound;
			analysis.bound_indices[analysis.num_bound] =
				kernel_ir::kernel_index_from_char((char)('a' + bound_idx), expr.contract.index.size);
			analysis.num_bound++;
			analysis.reduce_op = expr.contract.op;
			analysis.has_reduce_op = true;
			extract_bound_indices(expr.contract.body, analysis);

		case ACCESS: {}
		case SCALAR: {}

		case PRODUCT:
			foreach (f : expr.operands)
			{
				extract_bound_indices(f, analysis);
			}

		case SUM:
			foreach (t : expr.operands)
			{
				extract_bound_indices(t, analysis);
			}

		case NEGATE:
			extract_bound_indices(expr.inner, analysis);

		case UNARY:
			extract_bound_indices(expr.unary.child, analysis);

		case BINARY:
			extract_bound_indices(expr.binary.lhs, analysis);
			extract_bound_indices(expr.binary.rhs, analysis);

		case COMPARE:
			extract_bound_indices(expr.compare.lhs, analysis);
			extract_bound_indices(expr.compare.rhs, analysis);

		case SELECT:
			extract_bound_indices(expr.select_.cond, analysis);
			extract_bound_indices(expr.select_.then_expr, analysis);
			extract_bound_indices(expr.select_.else_expr, analysis);

		case PERMUTE:
			extract_bound_indices(expr.permute.child, analysis);

		case BUILTIN:
			foreach (arg : expr.builtin.args)
			{
				extract_bound_indices(arg, analysis);
			}
	}
}

/// Hash a CAS expression tree into the hasher.
fn void hash_cas_expr(Fnv64a* h, cas::Expr* expr)
{
	// Hash the expression kind
	h.update_char((char)expr.kind);

	switch (expr.kind)
	{
		case ACCESS:
			// Hash tensor dtype and rank (NOT pointer — same structure = same kernel)
			h.update_char((char)expr.access.tensor.dtype_raw());
			uint ndim = (uint)expr.access.tensor.rank();
			for (usz i = 0; i < 4; i++)
			{
				h.update_char((char)((ndim >> (i * 8)) & 0xFF));
			}
			// Hash index count and IDs
			uint idx_count = (uint)expr.access.indices.len;
			for (usz i = 0; i < 4; i++)
			{
				h.update_char((char)((idx_count >> (i * 8)) & 0xFF));
			}
			foreach (id : expr.access.indices)
			{
				h.update_char((char)(id & 0xFF));
				h.update_char((char)((id >> 8) & 0xFF));
			}
			// Hash is_stable flag
			h.update_char(expr.access.is_stable ? 1 : 0);

		case SCALAR:
			// Hash scalar value bytes
			ulong bits = *(ulong*)&expr.scalar;
			for (usz i = 0; i < 8; i++)
			{
				h.update_char((char)((bits >> (i * 8)) & 0xFF));
			}

		case PRODUCT:
			foreach (f : expr.operands) { hash_cas_expr(h, f); }

		case SUM:
			foreach (t : expr.operands) { hash_cas_expr(h, t); }

		case CONTRACT:
			h.update_char((char)expr.contract.op);
			hash_cas_expr(h, expr.contract.body);

		case NEGATE:
			hash_cas_expr(h, expr.inner);

		case UNARY:
			h.update_char((char)expr.unary.op);
			hash_cas_expr(h, expr.unary.child);

		case BINARY:
			h.update_char((char)expr.binary.op);
			hash_cas_expr(h, expr.binary.lhs);
			hash_cas_expr(h, expr.binary.rhs);

		case COMPARE:
			h.update_char((char)expr.compare.op);
			hash_cas_expr(h, expr.compare.lhs);
			hash_cas_expr(h, expr.compare.rhs);

		case SELECT:
			hash_cas_expr(h, expr.select_.cond);
			hash_cas_expr(h, expr.select_.then_expr);
			hash_cas_expr(h, expr.select_.else_expr);

		case PERMUTE:
			foreach (m : expr.permute.mapping)
			{
				h.update_char(m);
			}
			hash_cas_expr(h, expr.permute.child);

		case BUILTIN:
			h.update_char((char)expr.builtin.func);
			foreach (arg : expr.builtin.args) { hash_cas_expr(h, arg); }
	}
}

// FusedKernel Builder

/// Build a FusedKernel from an IREntry and analysis.
fn FusedKernel build_fused_kernel_from_ir_entry(
	cas::IREntry* entry,
	CASAnalysis* analysis,
	ulong hash,
)
{
	FusedKernel kernel;

	// Name
	kernel.set_name_from_hash(hash);

	// Copy indices
	for (usz i = 0; i < analysis.num_free; i++)
	{
		kernel.add_free_index(analysis.free_indices[i]);
	}
	for (usz i = 0; i < analysis.num_bound; i++)
	{
		kernel.add_bound_index(analysis.bound_indices[i]);
	}

	// Prologue expression: strip top-level contract wrappers
	kernel.prologue_expr = strip_contracts(entry.expr);

	// Reduce op
	kernel.has_reduce_op = analysis.has_reduce_op;
	if (analysis.has_reduce_op)
	{
		kernel.reduce_op = analysis.reduce_op;
	}

	// Collect accesses and build ID-to-name mapping
	AccessEntry[MAX_INPUTS] access_map;
	usz access_map_count = 0;
	collect_cas_accesses(entry.expr, &access_map, &access_map_count);

	IdNameEntry[MAX_INDICES] id_to_name;
	usz id_to_name_count = 0;

	// Free indices
	for (usz i = 0; i < entry.output_indices.len; i++)
	{
		id_to_name[id_to_name_count] = {
			.id = entry.output_indices[i],
			.name = (char)('i' + i),
		};
		id_to_name_count++;
	}

	// Bound indices
	IdNameEntry[MAX_INDICES] bound_ids;
	usz bound_ids_count = 0;
	collect_bound_ids(entry.expr, &bound_ids, &bound_ids_count, analysis);
	for (usz i = 0; i < bound_ids_count; i++)
	{
		id_to_name[id_to_name_count] = bound_ids[i];
		id_to_name_count++;
	}

	// Input bindings
	foreach (input : entry.inputs)
	{
		// Find CAS indices for this input
		char[8] indices_buf;
		usz indices_len = 0;

		for (usz a = 0; a < access_map_count; a++)
		{
			if (access_map[a].tensor == input)
			{
				for (usz j = 0; j < access_map[a].count; j++)
				{
					indices_buf[j] = lookup_id_name(&id_to_name, id_to_name_count, access_map[a].indices[j]);
				}
				indices_len = access_map[a].count;
				break;
			}
		}

		if (indices_len == 0)
		{
			// Element-wise: sequential indices
			usz rank = input.shape().len;
			for (usz j = 0; j < rank; j++)
			{
				indices_buf[j] = (char)('i' + j);
			}
			indices_len = rank;
		}

		kernel.add_input(input, indices_buf[:indices_len]);
	}

	// Output binding
	char[8] out_indices;
	for (usz i = 0; i < analysis.num_free; i++)
	{
		out_indices[i] = analysis.free_indices[i].name_buf[0];
	}
	kernel.set_output(entry.output, out_indices[:analysis.num_free]);

	kernel.dtype = entry.output.dtype();
	kernel.is_inplace = analysis.is_inplace;

	return kernel;
}

/// Strip top-level contract expressions to get the inner body.
/// Contracts are handled by the kernel loop structure (bound_indices).
fn cas::Expr* strip_contracts(cas::Expr* expr)
{
	cas::Expr* current = expr;
	while (current.kind == cas::ExprKind.CONTRACT)
	{
		current = current.contract.body;
	}
	return current;
}

// CAS Access Collection

/// An entry in the access map: tensor → index pattern.
struct AccessEntry
{
	TensorCore* tensor;
	cas::IndexId[8] indices;
	usz count;
}

/// Collect all tensor accesses from a CAS expression.
fn void collect_cas_accesses(cas::Expr* expr, AccessEntry* map, usz* count)
{
	switch (expr.kind)
	{
		case ACCESS:
			// Check if we already have this tensor
			for (usz i = 0; i < *count; i++)
			{
				if (map[i].tensor == expr.access.tensor) return;
			}
			assert(*count < MAX_INPUTS, "too many unique tensor accesses");
			map[*count].tensor = expr.access.tensor;
			usz len = expr.access.indices.len < 8 ? expr.access.indices.len : 8;
			for (usz i = 0; i < len; i++)
			{
				map[*count].indices[i] = expr.access.indices[i];
			}
			map[*count].count = len;
			(*count)++;

		case SCALAR: {}

		case PRODUCT:
			foreach (f : expr.operands) { collect_cas_accesses(f, map, count); }

		case SUM:
			foreach (t : expr.operands) { collect_cas_accesses(t, map, count); }

		case CONTRACT:
			collect_cas_accesses(expr.contract.body, map, count);

		case NEGATE:
			collect_cas_accesses(expr.inner, map, count);

		case UNARY:
			collect_cas_accesses(expr.unary.child, map, count);

		case BINARY:
			collect_cas_accesses(expr.binary.lhs, map, count);
			collect_cas_accesses(expr.binary.rhs, map, count);

		case COMPARE:
			collect_cas_accesses(expr.compare.lhs, map, count);
			collect_cas_accesses(expr.compare.rhs, map, count);

		case SELECT:
			collect_cas_accesses(expr.select_.cond, map, count);
			collect_cas_accesses(expr.select_.then_expr, map, count);
			collect_cas_accesses(expr.select_.else_expr, map, count);

		case PERMUTE:
			collect_cas_accesses(expr.permute.child, map, count);

		case BUILTIN:
			foreach (arg : expr.builtin.args) { collect_cas_accesses(arg, map, count); }
	}
}

// Bound Index Collection

/// Maps CAS IndexId to kernel index name character.
struct IdNameEntry
{
	cas::IndexId id;
	char name;
}

/// Collect bound (contracted) index IDs from a CAS expression.
fn void collect_bound_ids(
	cas::Expr* expr,
	IdNameEntry* map,
	usz* count,
	CASAnalysis* analysis,
)
{
	switch (expr.kind)
	{
		case CONTRACT:
			if (*count < analysis.num_bound)
			{
				map[*count] = {
					.id = expr.contract.index.id,
					.name = analysis.bound_indices[*count].name_buf[0],
				};
				(*count)++;
			}
			collect_bound_ids(expr.contract.body, map, count, analysis);

		case ACCESS: {}
		case SCALAR: {}

		case PRODUCT:
			foreach (f : expr.operands) { collect_bound_ids(f, map, count, analysis); }

		case SUM:
			foreach (t : expr.operands) { collect_bound_ids(t, map, count, analysis); }

		case NEGATE:
			collect_bound_ids(expr.inner, map, count, analysis);

		case UNARY:
			collect_bound_ids(expr.unary.child, map, count, analysis);

		case BINARY:
			collect_bound_ids(expr.binary.lhs, map, count, analysis);
			collect_bound_ids(expr.binary.rhs, map, count, analysis);

		case COMPARE:
			collect_bound_ids(expr.compare.lhs, map, count, analysis);
			collect_bound_ids(expr.compare.rhs, map, count, analysis);

		case SELECT:
			collect_bound_ids(expr.select_.cond, map, count, analysis);
			collect_bound_ids(expr.select_.then_expr, map, count, analysis);
			collect_bound_ids(expr.select_.else_expr, map, count, analysis);

		case PERMUTE:
			collect_bound_ids(expr.permute.child, map, count, analysis);

		case BUILTIN:
			foreach (arg : expr.builtin.args) { collect_bound_ids(arg, map, count, analysis); }
	}
}

/// Look up name for an IndexId. Falls back to 'i' + id if not found.
fn char lookup_id_name(IdNameEntry* map, usz count, cas::IndexId id)
{
	for (usz i = 0; i < count; i++)
	{
		if (map[i].id == id) return map[i].name;
	}
	return (char)('i' + id);
}
