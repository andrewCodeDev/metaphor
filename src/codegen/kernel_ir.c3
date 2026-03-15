//! Unified Kernel IR
//!
//! Intermediate representation for fused kernels covering:
//! - Element-wise fusion (no bound indices)
//! - Reductions (bound indices with reduce op)
//! - Einsum patterns (bound indices, map is product)
//!
//! The IR captures:
//! - Loop structure via free (output) and bound (contracted) indices
//! - CAS expression tree (prologue) inside innermost loop
//! - Optional reduce operator (add, mul, max, min)
//! - Optional epilogue expression (after reduction)
//!

module metaphor::codegen::kernel_ir;
import metaphor::cas;
import metaphor::tensor::tensor_core;
import metaphor::rtti;
import std::io;

// Constants

const usz MAX_FREE_INDICES = 8;
const usz MAX_BOUND_INDICES = 8;
const usz MAX_INPUTS = 64;
const usz MAX_INDEX_NAME = 4;
const usz MAX_INDICES_PER_BINDING = 8;
const usz MAX_SHAPE_EXPRS = 4;
const usz MAX_NAME_LEN = 32;

// KernelIndex — a named dimension with known size

struct KernelIndex
{
	/// Index name (single char like "i", "j", "k") stored as bytes.
	char[MAX_INDEX_NAME] name_buf;
	usz name_len;
	/// Size of this dimension.
	ulong size;
}

fn char[] KernelIndex.name(&self)
{
	return self.name_buf[:self.name_len];
}

fn KernelIndex kernel_index_from_name(char[] name, ulong size)
{
	KernelIndex idx;
	usz len = name.len < MAX_INDEX_NAME ? name.len : MAX_INDEX_NAME;
	for (usz i = 0; i < len; i++)
	{
		idx.name_buf[i] = name[i];
	}
	idx.name_len = len;
	idx.size = size;
	return idx;
}

fn KernelIndex kernel_index_from_char(char c, ulong size)
{
	KernelIndex idx;
	idx.name_buf[0] = c;
	idx.name_len = 1;
	idx.size = size;
	return idx;
}

// TensorBinding — binds a tensor to an index pattern

struct TensorBinding
{
	/// The tensor.
	TensorCore* tensor;
	/// Index pattern, e.g., "ij", "jk", "i".
	/// Order matches tensor dimensions.
	char[MAX_INDICES_PER_BINDING] indices_buf;
	usz indices_len;
}

fn char[] TensorBinding.indices(&self)
{
	return self.indices_buf[:self.indices_len];
}

fn void TensorBinding.set_indices(&self, char[] indices)
{
	usz len = indices.len < MAX_INDICES_PER_BINDING ? indices.len : MAX_INDICES_PER_BINDING;
	for (usz i = 0; i < len; i++)
	{
		self.indices_buf[i] = indices[i];
	}
	self.indices_len = len;
}

// ReduceOp — imported from CAS

// Re-use the CAS ReduceOp directly. It has exactly the same variants.
alias ReduceOp = cas::ReduceOp;

// ShapeExpr — shape-derived expressions computed at kernel runtime

enum ShapeExprKind : char
{
	NUMEL,
	DIM,
}

struct ShapeExpr
{
	ShapeExprKind kind;
	/// Input index (for both NUMEL and DIM).
	usz input;
	/// Axis character (for DIM only).
	char axis;
}

fn ShapeExpr shape_expr_numel(usz input_idx)
{
	ShapeExpr se;
	se.kind = NUMEL;
	se.input = input_idx;
	return se;
}

fn ShapeExpr shape_expr_dim(usz input_idx, char axis)
{
	ShapeExpr se;
	se.kind = DIM;
	se.input = input_idx;
	se.axis = axis;
	return se;
}

// FusableOp — operations that can be fused into compiled kernels

enum FusableOp : char
{
	// Binary arithmetic
	ADD,
	SUB,
	MUL,
	DIV,

	// Unary arithmetic
	NEG,
	ABS,
	SQRT,
	EXP,
	LOG,
	SIN,
	COS,
	TANH,
	RELU,
	SIGMOID,
	HEAVISIDE,

	// Comparison (produce bool)
	CMP_EQ,
	CMP_NE,
	CMP_GT,
	CMP_LT,
	CMP_GTE,
	CMP_LTE,

	// Logical (bool -> bool)
	LOGIC_AND,
	LOGIC_OR,
	LOGIC_XOR,
	LOGIC_NOT,
}

faultdef NOT_FUSABLE;

/// Maps from OpTag to FusableOp. Returns fault if the operation cannot be fused.
fn FusableOp? fusable_op_from_tag(rtti::OpTag tag)
{
	switch (tag)
	{
		case ADD: return ADD;
		case SUB: return SUB;
		case MUL: return MUL;
		case DIV: return DIV;
		case NEG: return NEG;
		case SIN: return SIN;
		case COS: return COS;
		case RELU: return RELU;
		case HEAVISIDE: return HEAVISIDE;
		case CMP_EQ: return CMP_EQ;
		case CMP_GT: return CMP_GT;
		case CMP_LT: return CMP_LT;
		case CMP_GTE: return CMP_GTE;
		case CMP_LTE: return CMP_LTE;
		case LOGIC_AND: return LOGIC_AND;
		case LOGIC_OR: return LOGIC_OR;
		case LOGIC_XOR: return LOGIC_XOR;
		case LOGIC_NOT: return LOGIC_NOT;
		default: return NOT_FUSABLE~;
	}
}

// Granularity — fusion boundary classification

enum Granularity : char
{
	/// Element-wise: scalar -> scalar.
	RANK_0,
	/// Vector operations: vector -> scalar (BLAS).
	RANK_1,
	/// Full reduction: entire tensor -> scalar.
	RANK_N,
}

fn bool Granularity.is_elementwise(&self)
{
	return *self == RANK_0;
}

fn Granularity FusableOp.granularity(&self)
{
	// All currently fusable ops are element-wise.
	return Granularity.RANK_0;
}

// FusedKernel — the unified kernel IR

struct FusedKernel
{
	/// Kernel name (e.g., "fk_abc123").
	char[MAX_NAME_LEN] name_buf;
	usz name_len;

	/// Free indices — appear in output, define output shape.
	KernelIndex[MAX_FREE_INDICES] free_indices;
	usz num_free;

	/// Bound indices — contracted over, don't appear in output.
	KernelIndex[MAX_BOUND_INDICES] bound_indices;
	usz num_bound;

	/// CAS expression tree for the prologue phase.
	/// Evaluated inside the innermost (bound) loop.
	cas::Expr* prologue_expr;

	/// Reduction operator applied to prologue_expr results.
	/// Null = pure element-wise (no bound indices).
	bool has_reduce_op;
	ReduceOp reduce_op;

	/// CAS expression tree for epilogue.
	/// Applied after reduction completes, before store.
	cas::Expr* epilogue_expr;

	/// Input tensor bindings.
	TensorBinding[MAX_INPUTS] inputs;
	usz num_inputs;

	/// Output tensor binding.
	TensorBinding output;

	/// Output data type.
	rtti::Datatype dtype;

	/// Number of runtime parameters.
	usz num_params;

	/// True if this kernel modifies input[0] in-place.
	bool is_inplace;

	/// Shape expressions — computed at kernel runtime from input shapes.
	ShapeExpr[MAX_SHAPE_EXPRS] shape_exprs;
	usz num_shape_exprs;
}

fn char[] FusedKernel.name(&self)
{
	return self.name_buf[:self.name_len];
}

fn void FusedKernel.set_name(&self, char[] name)
{
	usz len = name.len < MAX_NAME_LEN ? name.len : MAX_NAME_LEN;
	for (usz i = 0; i < len; i++)
	{
		self.name_buf[i] = name[i];
	}
	self.name_len = len;
}

/// Format a kernel name from a hash: "fk_{hash:x}".
fn void FusedKernel.set_name_from_hash(&self, ulong hash)
{
	// "fk_" prefix + up to 16 hex digits
	self.name_buf[0] = 'f';
	self.name_buf[1] = 'k';
	self.name_buf[2] = '_';
	usz pos = 3;

	// Write hex digits
	char[16] hex_buf;
	usz hex_len = 0;
	ulong h = hash;
	if (h == 0)
	{
		hex_buf[0] = '0';
		hex_len = 1;
	}
	else
	{
		while (h > 0 && hex_len < 16)
		{
			char digit = (char)(h & 0xF);
			hex_buf[hex_len] = digit < 10 ? ('0' + digit) : ('a' + digit - 10);
			hex_len++;
			h >>= 4;
		}
	}
	// hex_buf is reversed, write it forwards
	for (usz i = 0; i < hex_len && pos < MAX_NAME_LEN; i++)
	{
		self.name_buf[pos] = hex_buf[hex_len - 1 - i];
		pos++;
	}
	self.name_len = pos;
}

fn void FusedKernel.add_free_index(&self, KernelIndex idx)
{
	assert(self.num_free < MAX_FREE_INDICES, "too many free indices");
	self.free_indices[self.num_free] = idx;
	self.num_free++;
}

fn void FusedKernel.add_bound_index(&self, KernelIndex idx)
{
	assert(self.num_bound < MAX_BOUND_INDICES, "too many bound indices");
	self.bound_indices[self.num_bound] = idx;
	self.num_bound++;
}

fn void FusedKernel.add_input(&self, TensorCore* tensor, char[] indices)
{
	assert(self.num_inputs < MAX_INPUTS, "too many inputs");
	self.inputs[self.num_inputs].tensor = tensor;
	self.inputs[self.num_inputs].set_indices(indices);
	self.num_inputs++;
}

fn void FusedKernel.set_output(&self, TensorCore* tensor, char[] indices)
{
	self.output.tensor = tensor;
	self.output.set_indices(indices);
}

fn void FusedKernel.add_shape_expr(&self, ShapeExpr se)
{
	assert(self.num_shape_exprs < MAX_SHAPE_EXPRS, "too many shape exprs");
	self.shape_exprs[self.num_shape_exprs] = se;
	self.num_shape_exprs++;
}

/// Find full index name from a single character.
/// Panics if index not found (indicates a bug in kernel construction).
fn char[] FusedKernel.find_index_name(&self, char c)
{
	for (usz i = 0; i < self.num_free; i++)
	{
		char[] n = self.free_indices[i].name();
		if (n.len > 0 && n[0] == c) return n;
	}
	for (usz i = 0; i < self.num_bound; i++)
	{
		char[] n = self.bound_indices[i].name();
		if (n.len > 0 && n[0] == c) return n;
	}
	unreachable("find_index_name: index not found in kernel");
}

/// Find the size of an index by its character.
fn ulong FusedKernel.find_index_size(&self, char c)
{
	for (usz i = 0; i < self.num_free; i++)
	{
		char[] n = self.free_indices[i].name();
		if (n.len > 0 && n[0] == c) return self.free_indices[i].size;
	}
	for (usz i = 0; i < self.num_bound; i++)
	{
		char[] n = self.bound_indices[i].name();
		if (n.len > 0 && n[0] == c) return self.bound_indices[i].size;
	}
	unreachable("find_index_size: index not found in kernel");
}

/// Check if a bound index character is used by any input's index pattern.
fn bool FusedKernel.bound_index_is_used(&self, char idx_char)
{
	for (usz i = 0; i < self.num_inputs; i++)
	{
		char[] indices = self.inputs[i].indices();
		for (usz j = 0; j < indices.len; j++)
		{
			if (indices[j] == idx_char) return true;
		}
	}
	return false;
}
