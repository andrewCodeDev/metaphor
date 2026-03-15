//! JIT Kernel Cache
//!
//! Device-agnostic cache for compiled JIT kernels. Shared by all device
//! backends (host, HIP, CUDA). Each backend specializes Value with its
//! own handle type (CachedKernel, KernelHandle, etc.).
//!
//! ## Key Format
//!
//! Keys are canonical byte strings built by serializing operation structure.
//! The format uses marker bytes, counted fields, and pre-order traversal
//! for expression trees. Two operations produce the same key if and only if
//! they require the same compiled kernel.
//!
//! Dimensions are sentinelized: power-of-2 values are kept verbatim
//! (enabling shift/mask codegen), non-power-of-2 values become DYNAMIC.
//!
//! ## Canonical Key Grammar
//!
//! key          := header body
//! header       := MARKER_HEADER num_dtypes:u8 (dtype:u8 * num_dtypes)
//! body         := flat_op | expr_op
//!
//! flat_op      := unary_op | binary_op | comparison_op | select_op
//!              | permute_op | reduce_op | contraction_op
//! unary_op     := MARKER_UNARY  unary_code:u8  shape_block
//! binary_op    := MARKER_BINARY binary_code:u8  shape_block shape_block
//! comparison_op:= MARKER_COMPARE compare_code:u8 shape_block shape_block
//! select_op    := MARKER_SELECT  shape_block shape_block shape_block
//! permute_op   := MARKER_PERMUTE ndim:u8 symbols:(u8 * ndim) shape_block
//! reduce_op    := MARKER_REDUCE  reduce_code:u8 symbols_src symbols_dst shape_block
//! contraction_op:= MARKER_CONTRACT symbols_x symbols_y symbols_z shape_block shape_block shape_block
//!
//! expr_op      := fused_chain | cas_expr
//! fused_chain  := MARKER_FUSED num_ops:u8 (chain_step * num_ops)
//!                 num_inputs:u8 (shape_block * num_inputs) shape_block
//! chain_step   := is_unary:u8 op_code:u8
//! cas_expr     := MARKER_CAS num_inputs:u8 expr_tree
//!                 (shape_block * num_inputs) shape_block
//!
//! expr_tree    := access_node | scalar_node | unary_node | binary_node
//!              | compare_node | select_node | negate_node
//!              | sum_node | product_node
//! access_node  := NODE_ACCESS input_index:u8
//! scalar_node  := NODE_SCALAR float_bits:u64
//! unary_node   := NODE_UNARY unary_code:u8 expr_tree
//! binary_node  := NODE_BINARY binary_code:u8 expr_tree expr_tree
//! compare_node := NODE_COMPARE compare_code:u8 expr_tree expr_tree
//! select_node  := NODE_SELECT expr_tree expr_tree expr_tree
//! negate_node  := NODE_NEGATE expr_tree
//! sum_node     := NODE_SUM count:u8 (expr_tree * count)
//! product_node := NODE_PRODUCT count:u8 (expr_tree * count)
//!
//! shape_block  := ndim:u8 (dim:u64 * ndim)
//!
//! All dims in shape_block are sentinelized (pow2 or DYNAMIC).
//!
//! Not thread-safe — owned by a single graph/device context.

module metaphor::device::common::jit_cache <Value>;
import metaphor::utils::hash;
import metaphor::utils::hash_map;
import metaphor::cas;
import metaphor::rtti;
import metaphor::tensor::tensor_core;
import std::core::mem::allocator;
import std::core::mem;
import std::core::dstring;

// Faults

faultdef CACHE_MISS;

// Constants

/// Sentinel for non-power-of-2 dimensions. Kernel uses runtime params.
const ulong DYNAMIC = 0xFFFFFFFFFFFFFFFF;

// ---- Marker bytes (top-level operation type) ----
const char MARKER_HEADER   = 0x00;
const char MARKER_UNARY    = 0x01;
const char MARKER_BINARY   = 0x02;
const char MARKER_COMPARE  = 0x03;
const char MARKER_SELECT   = 0x04;
const char MARKER_PERMUTE  = 0x05;
const char MARKER_REDUCE   = 0x06;
const char MARKER_CONTRACT = 0x07;
const char MARKER_FUSED    = 0x08;
const char MARKER_CAS      = 0x09;

// ---- Node bytes (CAS expression tree nodes) ----
const char NODE_ACCESS     = 0x10;
const char NODE_SCALAR     = 0x11;
const char NODE_UNARY      = 0x12;
const char NODE_BINARY     = 0x13;
const char NODE_COMPARE    = 0x14;
const char NODE_SELECT     = 0x15;
const char NODE_NEGATE     = 0x16;
const char NODE_SUM        = 0x17;
const char NODE_PRODUCT    = 0x18;
const char NODE_CONTRACT   = 0x19;
const char NODE_PERMUTE    = 0x1A;
const char NODE_BUILTIN    = 0x1B;

// JitKey — canonical byte-string key

struct JitKey
{
	DString data;
}

fn void JitKey.free(&self)
{
	self.data.free();
}

fn char[] JitKey.bytes(&self)
{
	return self.data.str_view();
}

// ---- Primitive writers ----

fn void JitKey.write_u8(&self, char val)
{
	self.data.append_char(val);
}

fn void JitKey.write_u64(&self, ulong val)
{
	// Little-endian write
	for (usz i = 0; i < 8; i++)
	{
		self.data.append_char((char)(val >> (i * 8)));
	}
}

// ---- Structured writers ----

/// Write header: marker + dtype count + dtypes.
/// Every operation must declare all participating dtypes upfront.
/// Single-dtype ops: write_header({dtype})
/// Comparison: write_header({input_dtype, output_dtype})
/// Cast: write_header({src_dtype, dst_dtype})
fn void JitKey.write_header(&self, rtti::Datatype[] dtypes)
{
	self.write_u8(MARKER_HEADER);
	self.write_u8((char)dtypes.len);
	for (usz i = 0; i < dtypes.len; i++)
	{
		self.write_u8((char)dtypes[i]);
	}
}

/// Write the standard CAS preamble: header (all input + output dtypes),
/// CAS marker byte, and input count. All backends must use this to ensure
/// identical key prefixes — input dtypes distinguish kernels with different
/// type combinations (e.g., f32/f32 div vs f32/u64 div).
///
/// input_tensors: void** pointing to TensorCore* entries (cast from CodegenContext or IREntry.inputs)
/// num_inputs:    number of input tensors
/// output_dtype:  dtype of the output tensor
fn void JitKey.write_cas_preamble(&self, void** input_tensors, usz num_inputs, rtti::Datatype output_dtype)
{
	const usz MAX_CAS_DTYPES = 33; // 32 inputs + 1 output
	assert(num_inputs < MAX_CAS_DTYPES, "too many CAS inputs for key header");
	rtti::Datatype[MAX_CAS_DTYPES] dtypes;
	for (usz i = 0; i < num_inputs; i++)
	{
		tensor_core::TensorCore* tc = (tensor_core::TensorCore*)input_tensors[i];
		dtypes[i] = tc.dtype();
	}
	dtypes[num_inputs] = output_dtype;
	self.write_header(dtypes[:(num_inputs + 1)]);
	self.write_u8(MARKER_CAS);
	self.write_u8((char)num_inputs);
}

/// Write a shape block: ndim followed by sentinelized dimensions.
/// Used by GPU backends where pow2 dims → shift/mask codegen.
fn void JitKey.write_shape(&self, ulong[] dims)
{
	self.write_u8((char)dims.len);
	for (usz i = 0; i < dims.len; i++)
	{
		self.write_u64(sentinelize(dims[i]));
	}
}

/// Write a rank-only shape block: ndim followed by DYNAMIC for every dim.
/// Used by the host backend where ALL sizes are runtime loop bounds —
/// the compiled kernel is identical regardless of actual dimension values.
fn void JitKey.write_rank(&self, usz ndim)
{
	self.write_u8((char)ndim);
	for (usz i = 0; i < ndim; i++)
	{
		self.write_u64(DYNAMIC);
	}
}

/// Write a CAS expression tree in pre-order traversal.
/// input_ptrs/num_inputs map tensor pointers to input indices
/// (same ordering as compile_jit's CodegenContext).
fn void JitKey.write_expr(&self, cas::Expr* expr, void** input_ptrs, usz num_inputs)
{
	switch (expr.kind)
	{
		case ACCESS:
			self.write_u8(NODE_ACCESS);
			self.write_u8((char)find_input_index(input_ptrs, num_inputs, (void*)expr.access.tensor));
			// Encode index IDs so different access patterns
			// (e.g., A[i,j] vs A[j,i]) produce different keys.
			self.write_u8((char)expr.access.indices.len);
			foreach (idx_id : expr.access.indices)
			{
				// IndexId is ushort — write as 2 bytes
				self.write_u8((char)(idx_id & 0xFF));
				self.write_u8((char)((idx_id >> 8) & 0xFF));
			}

		case SCALAR:
			self.write_u8(NODE_SCALAR);
			self.write_u64(bitcast(expr.scalar, ulong));

		case NEGATE:
			self.write_u8(NODE_NEGATE);
			self.write_expr(expr.inner, input_ptrs, num_inputs);

		case UNARY:
			self.write_u8(NODE_UNARY);
			self.write_u8((char)expr.unary.op);
			self.write_expr(expr.unary.child, input_ptrs, num_inputs);

		case BINARY:
			self.write_u8(NODE_BINARY);
			self.write_u8((char)expr.binary.op);
			self.write_expr(expr.binary.lhs, input_ptrs, num_inputs);
			self.write_expr(expr.binary.rhs, input_ptrs, num_inputs);

		case COMPARE:
			self.write_u8(NODE_COMPARE);
			self.write_u8((char)expr.compare.op);
			self.write_expr(expr.compare.lhs, input_ptrs, num_inputs);
			self.write_expr(expr.compare.rhs, input_ptrs, num_inputs);

		case SELECT:
			self.write_u8(NODE_SELECT);
			self.write_expr(expr.select_.cond, input_ptrs, num_inputs);
			self.write_expr(expr.select_.then_expr, input_ptrs, num_inputs);
			self.write_expr(expr.select_.else_expr, input_ptrs, num_inputs);

		case SUM:
			self.write_u8(NODE_SUM);
			self.write_u8((char)expr.operands.len);
			foreach (child : expr.operands)
			{
				self.write_expr(child, input_ptrs, num_inputs);
			}

		case PRODUCT:
			self.write_u8(NODE_PRODUCT);
			self.write_u8((char)expr.operands.len);
			foreach (child : expr.operands)
			{
				self.write_expr(child, input_ptrs, num_inputs);
			}

		case CONTRACT:
			self.write_u8(NODE_CONTRACT);
			self.write_u8((char)expr.contract.op);
			// Encode contracted index ID so contracting over different
			// dimensions (e.g., axis=0 vs axis=1) produces different keys.
			cas::IndexId cid = expr.contract.index.id;
			self.write_u8((char)(cid & 0xFF));
			self.write_u8((char)((cid >> 8) & 0xFF));
			self.write_expr(expr.contract.body, input_ptrs, num_inputs);

		case PERMUTE:
			self.write_u8(NODE_PERMUTE);
			self.write_u8((char)expr.permute.mapping.len);
			foreach (m : expr.permute.mapping)
			{
				self.write_u8(m);
			}
			self.write_expr(expr.permute.child, input_ptrs, num_inputs);

		case BUILTIN:
			self.write_u8(NODE_BUILTIN);
			self.write_u8((char)expr.builtin.func);
			self.write_u8((char)expr.builtin.args.len);
			foreach (arg : expr.builtin.args)
			{
				self.write_expr(arg, input_ptrs, num_inputs);
			}

		default:
			unreachable("unexpected CAS node in JitKey serialization: %s", expr.kind);
	}
}

/// Write a fused chain step.
fn void JitKey.write_chain_step(&self, bool is_unary, char op_code)
{
	self.write_u8(is_unary ? 1 : 0);
	self.write_u8(op_code);
}

// ---- Input index lookup ----

fn usz find_input_index(void** ptrs, usz count, void* target)
{
	for (usz i = 0; i < count; i++)
	{
		if (ptrs[i] == target) return i;
	}
	unreachable("JitKey: tensor not in input list");
}

// Dimension sentinelization

/// Power-of-2 dimensions keep their value (shift/mask codegen).
/// Non-power-of-2 → DYNAMIC (runtime params in kernel).
fn ulong sentinelize(ulong dim_size) @inline
{
	if (dim_size > 0 && (dim_size & (dim_size - 1)) == 0)
	{
		return dim_size;
	}
	return DYNAMIC;
}

// Key hashing

fn ulong JitKey.hash(&self)
{
	return hash::fnv1a(self.bytes());
}

// JitCache

struct CacheEntry
{
	JitKey key;
	Value value;
}

alias CacheMap = hash_map::HashMap{ulong, CacheEntry};

struct JitCache
{
	CacheMap map;
	allocator::LibcAllocator alloc;
	ulong hits;
	ulong misses;
	bool initialized;
}

// Lifecycle

fn void JitCache.init(&self)
{
	if (!self.initialized)
	{
		self.map.init(&self.alloc);
		self.hits = 0;
		self.misses = 0;
		self.initialized = true;
	}
}

fn void JitCache.free(&self)
{
	if (!self.initialized) return;
	self.map.free();
	self.initialized = false;
}

// Core operations

/// Look up a cached kernel by its key.
fn Value? JitCache.get(&self, JitKey* key)
{
	ulong h = key.hash();
	if (try entry = self.map.get(h))
	{
		if (key.data == entry.key.data)
		{
			self.hits++;
			return entry.value;
		}
		// Hash collision — different key, same hash. Treat as miss.
		self.misses++;
		return CACHE_MISS~;
	}
	self.misses++;
	return CACHE_MISS~;
}

/// Insert a kernel handle for the given key. Copies key bytes into owned storage.
fn void JitCache.put(&self, JitKey* key, Value handle)
{
	DString owned_data;
	owned_data.append_bytes(key.bytes());
	JitKey owned = { .data = owned_data };
	CacheEntry entry = { .key = owned, .value = handle };
	self.map.set(key.hash(), entry);
}

// Statistics

struct CacheStats
{
	ulong hits;
	ulong misses;
	usz entries;
}

fn CacheStats JitCache.stats(&self)
{
	return {
		.hits = self.hits,
		.misses = self.misses,
		.entries = self.map.len(),
	};
}

fn void JitCache.reset_stats(&self)
{
	self.hits = 0;
	self.misses = 0;
}
