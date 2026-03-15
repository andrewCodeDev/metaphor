//! Operation Types
//!
//! Fundamental operation types for the unified JIT compiler:
//! BinaryOp, UnaryOp, and all operation payload structs.
//!

module metaphor::device::common::operations;
import metaphor::device::common::index_expr;
import metaphor::device::device_reference;

// Binary Operations

/// Binary operations for einsum (inner/outer) and reduce.
enum BinaryOp : char
{
	// Arithmetic (numeric, numeric) -> numeric
	ADD,
	SUB,
	MUL,
	DIV,
	MAX,
	MIN,

	// Comparison (numeric, numeric) -> bool
	EQL,
	NEQ,
	GT,
	LT,
	GTE,
	LTE,

	// Logical (bool, bool) -> bool
	AND,
	OR,
	XOR,
}

fn bool BinaryOp.is_comparison(&self)
{
	switch (*self)
	{
		case EQL: return true;
		case NEQ: return true;
		case GT:  return true;
		case LT:  return true;
		case GTE: return true;
		case LTE: return true;
		default:  return false;
	}
}

fn bool BinaryOp.is_logical(&self)
{
	switch (*self)
	{
		case AND: return true;
		case OR:  return true;
		case XOR: return true;
		default:  return false;
	}
}

/// Returns true if this operation produces a bool result.
fn bool BinaryOp.produces_bool(&self)
{
	return self.is_comparison() || self.is_logical();
}

// Unary Operations

/// Unary operations for map (shape-preserving transforms).
/// Order matches MapOp in CUDA interop.h exactly — cast directly, no conversion needed.
enum UnaryOp : char
{
	IDENTITY,   // 0  MAP_IDENTITY
	NEG,        // 1  MAP_NEG
	ABS,        // 2  MAP_ABS
	SQR,        // 3  MAP_SQR
	SQRT,       // 4  MAP_SQRT
	RECIP,      // 5  MAP_RECIP
	EXP,        // 6  MAP_EXP
	LOG,        // 7  MAP_LOG
	SIN,        // 8  MAP_SIN
	COS,        // 9  MAP_COS
	TANH,       // 10 MAP_TANH
	RELU,       // 11 MAP_RELU
	HEAVISIDE,  // 12 MAP_HEAVISIDE
	SIGMOID,    // 13 MAP_SIGMOID
	GELU,       // 14 MAP_GELU
	SILU,       // 15 MAP_SILU
	SOFTPLUS,   // 16 MAP_SOFTPLUS
	NOT,        // 17 MAP_NOT
	SIGN,       // 18 MAP_SIGN
}

/// Returns true if backward pass only needs input (args[0]).
/// If false, backward needs collected output.
fn bool UnaryOp.backward_needs_input_only(&self)
{
	switch (*self)
	{
		// Can compute gradient from input alone
		case IDENTITY: return true;
		case NEG:      return true;
		case ABS:      return true;
		case SQR:      return true;
		case RECIP:    return true;
		case EXP:      return true;
		case LOG:      return true;
		case COS:      return true;
		case SIN:      return true;
		case RELU:     return true;
		case GELU:     return true;
		case SILU:     return true;
		case SOFTPLUS: return true;
		// Need output for efficient gradient
		case SQRT:     return false;
		case TANH:     return false;
		case SIGMOID:  return false;
		// Not differentiable
		case SIGN:      return false;
		case HEAVISIDE: return false;
		case NOT:       return false;
	}
}

// Op Tag — discriminant for the OpPayload union

enum OpTag : char
{
	// Unified operations (fusable)
	EINSUM,
	REDUCE,
	EXPAND,
	MAP,
	SELECT,
	SLICE,
	PERMUTE,

	// Multi-input operations
	CONCAT,
	STACK,

	// Special operations (boundary, not fusable)
	SOFTMAX,
	RMS_NORM,
	CONV,
	CONV_TRANSPOSE,
	CONV_WEIGHT_GRAD,
	GATHER,
	SCATTER_ADD,
	CAST,
	REDUCE_MAXMIN_BACKWARD,
	SELECTIVE_SCAN,
	CAUSAL_CONV1D,
	DROPOUT,

	// Zero-copy operations
	RESHAPE,

	// Shape query operations (compile-time resolved)
	DIM,
	NUMEL,

	// Device transfer
	TRANSFER,
}

// Operation Payloads

/// Payload for einsum operation.
/// Binary contraction: einsum(a, b, expr, inner_op, outer_op).
struct EinsumPayload
{
	OpTag tag;
	IndexExpr expr;
	BinaryOp inner;
	BinaryOp outer;
	double alpha;
	double beta;
	ulong hash;
}

/// Payload for reduce operation.
/// Unary contraction: reduce(a, expr, op).
struct ReducePayload
{
	OpTag tag;
	IndexExpr expr;
	BinaryOp op;
	ulong hash;
}

/// Payload for expand operation.
/// Unary expansion (inverse of reduce).
struct ExpandPayload
{
	OpTag tag;
	IndexExpr expr;
	ulong[8] new_sizes;
	char num_new;
	ulong hash;
}

/// Payload for map operation.
/// Element-wise unary transform.
struct MapPayload
{
	OpTag tag;
	UnaryOp unary;
	double beta;
	ulong hash;
}

/// Payload for select (ternary conditional) operation.
struct SelectPayload
{
	OpTag tag;
	ulong hash;
}

/// Payload for numel (total element count) operation.
struct NumelPayload
{
	OpTag tag;
	ulong hash;
}

/// Payload for dim (dimension size query) operation.
struct DimPayload
{
	OpTag tag;
	usz axis;
	ulong hash;
}

/// Payload for slice operation.
struct SlicePayload
{
	OpTag tag;
	ulong[8] offsets;
	long[8] steps;
	char ndim;
	bool is_scatter;
	ulong hash;
}

/// Payload for permute (dimension reorder) operation.
struct PermutePayload
{
	OpTag tag;
	IndexExpr expr;
	ulong hash;
}

/// Payload for concat operation.
struct ConcatPayload
{
	OpTag tag;
	char dim;
	char num_inputs;
	ulong[8] offsets;
	ulong hash;
}

/// Payload for stack operation.
struct StackPayload
{
	OpTag tag;
	char dim;
	char num_inputs;
	ulong hash;
}

/// Payload for softmax operation.
/// Uses torch-style axis indexing: -1 = last dim, -2 = second-to-last, etc.
struct SoftmaxPayload
{
	OpTag tag;
	/// Dimension to reduce over. Negative values index from end (-1 = last).
	int axis;
	bool is_log;
	bool is_backward;
	ulong hash;
}

/// Payload for RMS normalization operation.
struct RmsNormPayload
{
	OpTag tag;
	double eps;
	RmsOpType op_type;
	ulong hash;
}

enum RmsOpType : char
{
	FORWARD,
	BACKWARD_X,
	BACKWARD_WEIGHT,
}

/// Payload for selective scan (SSM recurrence) operation.
struct SelectiveScanPayload
{
	OpTag tag;
	SelectiveScanOpType op_type;
	ulong hash;
}

enum SelectiveScanOpType : char
{
	FORWARD,
	BACKWARD_A_BAR,
	BACKWARD_B_BAR_X,
}

/// Payload for causal conv1d operation.
struct CausalConv1dPayload
{
	OpTag tag;
	CausalConv1dOpType op_type;
	ulong hash;
}

enum CausalConv1dOpType : char
{
	FORWARD,
	BACKWARD_X,
	BACKWARD_WEIGHT,
	BACKWARD_BIAS,
}

/// Payload for dropout operation.
/// Stores probability and seed for mask generation.
struct DropoutPayload
{
	OpTag tag;
	float p;          // drop probability (0..1)
	ulong seed;       // RNG seed for mask generation
	bool is_backward; // false=forward, true=backward
	ulong hash;
}

// Convolution Parameters

/// Parameters for convolution operations.
struct ConvParams
{
	ulong[2] padding;
	ulong[2] stride;
	ulong[2] dilation;
	ulong groups;
}

const ConvParams DEFAULT_CONV_PARAMS = {
	.padding = { 0, 0 },
	.stride = { 1, 1 },
	.dilation = { 1, 1 },
	.groups = 1,
};

/// Payload for convolution operation.
struct ConvPayload
{
	OpTag tag;
	ConvParams params;
}

/// Payload for transposed convolution.
struct ConvTransposePayload
{
	OpTag tag;
	ConvParams params;
	ulong[4] output_shape;
	ulong hash;
}

/// Parameters for gather/scatter operations.
struct GatherParams
{
	char dim;
}

/// Payload for gather operation.
struct GatherPayload
{
	OpTag tag;
	GatherParams params;
	ulong hash;
}

/// Payload for scatter-add operation.
struct ScatterAddPayload
{
	OpTag tag;
	GatherParams params;
	ulong[8] output_shape;
	char output_ndim;
	ulong hash;
}

/// Payload for dtype cast operation.
struct CastPayload
{
	OpTag tag;
	char source_dtype;
	char target_dtype;
}

/// Payload for reshape (zero-copy shape reinterpretation).
/// No extra data needed — the target TensorCore already has the new shape.
struct ReshapePayload
{
	OpTag tag;
}

struct TransferPayload
{
	OpTag tag;
	device_reference::TransferDirection direction;
}

/// Payload for reduce max/min backward operation.
struct ReduceMaxMinBackwardPayload
{
	OpTag tag;
	ulong row;
	ulong col;
	ulong hash;
}

// Slice Range (public API type)

/// Slice range for a single dimension.
struct SliceRange
{
	ulong start;
	ulong end;
	long step;
}
