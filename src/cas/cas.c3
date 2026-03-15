//! Einsum CAS — Computer Algebra System for Tensor Expressions
//!
//! Goals:
//! 1. Detect distribution: (xy + xz) -> x(y + z)
//! 2. Detect fusable permutations: fold permutes into index patterns
//! 3. Respect stable memory nodes (atoms that can't be simplified away)
//!
//! Core insight: Everything is einsum, multiplication is commutative.
//! We can apply algebraic rewrite rules to simplify expressions.
//!

module metaphor::cas;
import metaphor::tensor::tensor_core;
import std::core::mem::allocator;

// Index — a symbolic dimension identifier

alias IndexId = ushort;

struct Index
{
	IndexId id;
	ulong size;
}

// Built-in Functions (NN ops, special functions)

// Complex NN operations that are fusion boundaries.
// These are "opaque" to the CAS — we don't look inside for algebraic simplification.
// Simple element-wise ops (sqrt, exp, relu) use the Expr.unary/binary variants instead.
enum BuiltinFn : char
{
	// Activation functions (complex ones that may have special implementations)
	GELU,
	SILU,
	SIGMOID,
	TANH,
	LEAKY_RELU,
	ELU,
	SOFTPLUS,
	MISH,
	HEAVISIDE,
	CLAMP,

	// Softmax variants (fused implementations)
	SOFTMAX,
	LOG_SOFTMAX,

	// Normalization layers (fused implementations)
	LAYER_NORM,
	RMS_NORM,
	BATCH_NORM,
	GROUP_NORM,
	INSTANCE_NORM,

	// Scaling / normalization functions (shape-preserving)
	L2_NORMALIZE,
	L1_NORMALIZE,
	MAX_ABS_SCALE,
	ZSCORE_NORMALIZE,
	MIN_MAX_SCALE,
	RMS_NORMALIZE,

	// Norm computations (reduce dimensions)
	L2_NORM,
	L1_NORM,
	RMS,

	// Statistical reductions
	MEAN,
	VARIANCE,
	STD_DEV,
	SUM,
	PROD,
	MAX,
	MIN,
	ARGMAX,
	ARGMIN,

	// Loss functions
	CROSS_ENTROPY,
	NLL_LOSS,
	MSE_LOSS,
	HUBER_LOSS,
	KL_DIV,
	BINARY_CROSS_ENTROPY,
	FOCAL_LOSS,
	SMOOTH_L1_LOSS,
	COSINE_EMBEDDING_LOSS,
	TRIPLET_MARGIN_LOSS,
	CONTRASTIVE_LOSS,

	// Attention mechanisms
	SCALED_DOT_PRODUCT_ATTENTION,
	FLASH_ATTENTION,
	MULTI_HEAD_ATTENTION,

	// Pooling operations
	MAX_POOL_1D,
	MAX_POOL_2D,
	MAX_POOL_3D,
	AVG_POOL_1D,
	AVG_POOL_2D,
	AVG_POOL_3D,
	ADAPTIVE_MAX_POOL_1D,
	ADAPTIVE_MAX_POOL_2D,
	ADAPTIVE_AVG_POOL_1D,
	ADAPTIVE_AVG_POOL_2D,
	GLOBAL_MAX_POOL,
	GLOBAL_AVG_POOL,
	LP_POOL_1D,
	LP_POOL_2D,

	// Convolution operations
	CONV1D,
	CONV2D,
	CONV3D,
	DEPTHWISE_CONV2D,
	CONV_TRANSPOSE1D,
	CONV_TRANSPOSE2D,
	CONV_TRANSPOSE3D,
	SEPARABLE_CONV2D,

	// RNN cells and layers
	LSTM_CELL,
	GRU_CELL,
	RNN_CELL,
	LSTM,
	GRU,
	RNN,

	// Embedding operations
	EMBEDDING,
	ROTARY_EMBEDDING,
	SINUSOIDAL_EMBEDDING,
	LEARNED_POSITIONAL_EMBEDDING,

	// Regularization
	DROPOUT,
	DROPOUT2D,
	DROPOUT3D,
	ALPHA_DROPOUT,

	// Spatial transformations
	INTERPOLATE,
	UPSAMPLE,
	UPSAMPLE_NEAREST,
	UPSAMPLE_BILINEAR,
	GRID_SAMPLE,
	AFFINE_GRID,
	ROI_ALIGN,
	ROI_POOL,

	// Padding operations
	PAD,
	REFLECTION_PAD_1D,
	REFLECTION_PAD_2D,
	REPLICATION_PAD_1D,
	REPLICATION_PAD_2D,
	CONSTANT_PAD,
	ZERO_PAD,

	// Other common operations
	ONE_HOT,
	GATHER,
	SCATTER,
	SCATTER_ADD,
	INDEX_SELECT,
	MASKED_FILL,
	WHERE,
	TOPK,
	SORT,
	UNIQUE,
}

// Element-wise Operations

// Unary element-wise operations (fusable).
enum UnaryOp : char
{
	// Math functions
	ABS,
	SQRT,
	SQR,
	RECIP,
	EXP,
	LOG,
	SIN,
	COS,
	SIGN,
	FLOOR,
	CEIL,
	ROUND,

	// Activation functions
	RELU,
	SIGMOID,
	TANH,
	GELU,
	SILU,
	SOFTPLUS,
	HEAVISIDE,

	// Logical
	LOGIC_NOT,
}

// Binary element-wise operations (fusable).
// For operations that can't be expressed as algebraic product/sum.
enum BinaryOp : char
{
	DIV,
	SUB,
	POW,
	MAX,
	MIN,
	LOGIC_AND,
	LOGIC_OR,
	LOGIC_XOR,
}

// Comparison operations (produce 0.0 or 1.0).
enum CompareOp : char
{
	EQ,
	NE,
	GT,
	LT,
	GTE,
	LTE,
}

// Reduction operations for contractions.
// Determines how values are accumulated across contracted indices.
enum ReduceOp : char
{
	ADD,
	MUL,
	MAX,
	MIN,
}

// Expr — the core expression type

// An algebraic expression over tensors.
// Expressions are immutable — simplification produces new expressions.

enum ExprKind : char
{
	ACCESS,
	SCALAR,
	PRODUCT,
	SUM,
	CONTRACT,
	NEGATE,
	UNARY,
	BINARY,
	COMPARE,
	SELECT,
	PERMUTE,
	BUILTIN,
}

struct AccessData
{
	TensorCore* tensor;
	IndexId[] indices;
	bool is_stable;
}

struct ContractData
{
	Index index;
	Expr* body;
	ReduceOp op;
}

struct UnaryData
{
	UnaryOp op;
	Expr* child;
}

struct BinaryData
{
	BinaryOp op;
	Expr* lhs;
	Expr* rhs;
}

struct CompareData
{
	CompareOp op;
	Expr* lhs;
	Expr* rhs;
}

struct SelectData
{
	Expr* cond;
	Expr* then_expr;
	Expr* else_expr;
}

struct PermuteData
{
	Expr* child;
	char[] mapping;
}

struct BuiltinData
{
	BuiltinFn func;
	Expr*[] args;
	IndexId[] output_indices;
}

struct Expr
{
	ExprKind kind;
	union
	{
		AccessData access;
		double scalar;
		Expr*[] operands; // product or sum children
		ContractData contract;
		Expr* inner;      // negate child
		UnaryData unary;
		BinaryData binary;
		CompareData compare;
		SelectData select_;
		PermuteData permute;
		BuiltinData builtin;
	}
}

// Expression queries

// Check if expression depends on a given index.
fn bool Expr.depends_on(&self, IndexId idx)
{
	switch (self.kind)
	{
		case ACCESS:
			foreach (id : self.access.indices)
			{
				if (id == idx) return true;
			}
			return false;

		case SCALAR:
			return false;

		case PRODUCT:
			foreach (f : self.operands)
			{
				if (f.depends_on(idx)) return true;
			}
			return false;

		case SUM:
			foreach (t : self.operands)
			{
				if (t.depends_on(idx)) return true;
			}
			return false;

		case CONTRACT:
			// The contracted index is bound, not free
			if (self.contract.index.id == idx) return false;
			return self.contract.body.depends_on(idx);

		case NEGATE:
			return self.inner.depends_on(idx);

		case UNARY:
			return self.unary.child.depends_on(idx);

		case BINARY:
			return self.binary.lhs.depends_on(idx) || self.binary.rhs.depends_on(idx);

		case COMPARE:
			return self.compare.lhs.depends_on(idx) || self.compare.rhs.depends_on(idx);

		case SELECT:
			return self.select_.cond.depends_on(idx)
				|| self.select_.then_expr.depends_on(idx)
				|| self.select_.else_expr.depends_on(idx);

		case PERMUTE:
			return self.permute.child.depends_on(idx);

		case BUILTIN:
			foreach (id : self.builtin.output_indices)
			{
				if (id == idx) return true;
			}
			foreach (arg : self.builtin.args)
			{
				if (arg.depends_on(idx)) return true;
			}
			return false;
	}
}

// Check if this expression contains a stable tensor.
fn bool Expr.has_stable(&self)
{
	switch (self.kind)
	{
		case ACCESS:
			return self.access.is_stable;

		case SCALAR:
			return false;

		case PRODUCT:
			foreach (f : self.operands)
			{
				if (f.has_stable()) return true;
			}
			return false;

		case SUM:
			foreach (t : self.operands)
			{
				if (t.has_stable()) return true;
			}
			return false;

		case CONTRACT:
			return self.contract.body.has_stable();

		case NEGATE:
			return self.inner.has_stable();

		case UNARY:
			return self.unary.child.has_stable();

		case BINARY:
			return self.binary.lhs.has_stable() || self.binary.rhs.has_stable();

		case COMPARE:
			return self.compare.lhs.has_stable() || self.compare.rhs.has_stable();

		case SELECT:
			return self.select_.cond.has_stable()
				|| self.select_.then_expr.has_stable()
				|| self.select_.else_expr.has_stable();

		case PERMUTE:
			return self.permute.child.has_stable();

		case BUILTIN:
			foreach (arg : self.builtin.args)
			{
				if (arg.has_stable()) return true;
			}
			return false;
	}
}

// Get the inner expression (for wrapper types).
// Returns null for leaf types or multi-child expressions.
fn Expr* Expr.prev(&self)
{
	switch (self.kind)
	{
		case CONTRACT: return self.contract.body;
		case NEGATE:   return self.inner;
		case UNARY:    return self.unary.child;
		case PERMUTE:  return self.permute.child;
		default:       return null;
	}
}

// Check structural equality of expressions.
fn bool expr_equal(Expr* a, Expr* b)
{
	if (a == b) return true;

	if (a.kind != b.kind) return false;

	switch (a.kind)
	{
		case ACCESS:
			if (a.access.tensor != b.access.tensor) return false;
			if (a.access.indices.len != b.access.indices.len) return false;
			foreach (i, id : a.access.indices)
			{
				if (id != b.access.indices[i]) return false;
			}
			return true;

		case SCALAR:
			return a.scalar == b.scalar;

		case PRODUCT:
			if (a.operands.len != b.operands.len) return false;
			foreach (i, x : a.operands)
			{
				if (!expr_equal(x, b.operands[i])) return false;
			}
			return true;

		case SUM:
			if (a.operands.len != b.operands.len) return false;
			foreach (i, x : a.operands)
			{
				if (!expr_equal(x, b.operands[i])) return false;
			}
			return true;

		case CONTRACT:
			return a.contract.index.id == b.contract.index.id
				&& a.contract.op == b.contract.op
				&& expr_equal(a.contract.body, b.contract.body);

		case NEGATE:
			return expr_equal(a.inner, b.inner);

		case UNARY:
			return a.unary.op == b.unary.op
				&& expr_equal(a.unary.child, b.unary.child);

		case BINARY:
			return a.binary.op == b.binary.op
				&& expr_equal(a.binary.lhs, b.binary.lhs)
				&& expr_equal(a.binary.rhs, b.binary.rhs);

		case COMPARE:
			return a.compare.op == b.compare.op
				&& expr_equal(a.compare.lhs, b.compare.lhs)
				&& expr_equal(a.compare.rhs, b.compare.rhs);

		case SELECT:
			return expr_equal(a.select_.cond, b.select_.cond)
				&& expr_equal(a.select_.then_expr, b.select_.then_expr)
				&& expr_equal(a.select_.else_expr, b.select_.else_expr);

		case PERMUTE:
			if (a.permute.mapping.len != b.permute.mapping.len) return false;
			foreach (i, m : a.permute.mapping)
			{
				if (m != b.permute.mapping[i]) return false;
			}
			return expr_equal(a.permute.child, b.permute.child);

		case BUILTIN:
			if (a.builtin.func != b.builtin.func) return false;
			if (a.builtin.args.len != b.builtin.args.len) return false;
			foreach (i, x : a.builtin.args)
			{
				if (!expr_equal(x, b.builtin.args[i])) return false;
			}
			if (a.builtin.output_indices.len != b.builtin.output_indices.len) return false;
			foreach (i, id : a.builtin.output_indices)
			{
				if (id != b.builtin.output_indices[i]) return false;
			}
			return true;
	}
}

// Check if candidate is a common factor in all terms.
// Each term must be a product containing the candidate.
fn bool is_common_factor(Expr* candidate, Expr*[] terms)
{
	foreach (term : terms)
	{
		if (term.kind != PRODUCT) return false;

		bool found = false;
		foreach (f : term.operands)
		{
			if (expr_equal(f, candidate))
			{
				found = true;
				break;
			}
		}
		if (!found) return false;
	}
	return true;
}
