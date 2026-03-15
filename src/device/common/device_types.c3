//! Common Device Compiler Types
//!
//! Operation type enum and internal DeviceOp representation used during
//! graph compilation. A DeviceOp describes a single device operation to dispatch.
//! Shared between HIP and CUDA backends.

module metaphor::device::common::device_types;
import metaphor::device::common::operations;
import metaphor::device::device_reference;
import std::collections::elastic_array;

alias Pattern = ElasticArray{char, MAX_PATTERN_LEN};

// Device Operation Types

/// Operation type for device dispatch.
enum OpType : char
{
	CONTRACTION,                  // Tensor contraction (einsum with bound indices)
	REDUCTION,                    // Tensor reduction
	PERMUTATION,                  // Tensor permutation (transpose)
	ELEMENTWISE_BINARY_TENSOR,    // Tensor library elementwise binary (einsum broadcasting)
	ELEMENTWISE_UNARY,            // JIT element-wise unary (map)
	ELEMENTWISE_BINARY,           // JIT element-wise binary (numpy-style broadcast only)
	ELEMENTWISE_SELECT,           // JIT ternary select: mask ? true_vals : false_vals
	ELEMENTWISE_FUSED_CHAIN,      // JIT fused mixed unary/binary chain
	ELEMENTWISE_CAS_EXPR,         // JIT kernel from CAS expression tree
	SLICE,                        // Strided copy (slice forward or backward scatter)
	CONCAT,                       // Concatenate tensors along existing dimension
	STACK,                        // Stack tensors along new dimension
	GATHER,                       // Index-based gather operation
	SCATTER_ADD,                  // Scatter-add (backward for gather)
	SOFTMAX,                      // 2D row-wise softmax
	RMS_NORM,                     // RMS normalization forward
	RMS_NORM_BWD_X,               // RMS norm backward w.r.t. input
	RMS_NORM_BWD_WEIGHT,          // RMS norm backward w.r.t. weight
	SELECTIVE_SCAN,               // Fused selective scan forward
	SELECTIVE_SCAN_BWD_A,         // Selective scan backward w.r.t. a_bar
	SELECTIVE_SCAN_BWD_BX,        // Selective scan backward w.r.t. b_bar_x
	CAUSAL_CONV1D,                // Fused causal conv1d forward
	CAUSAL_CONV1D_BWD_X,          // Causal conv1d backward w.r.t. input
	CAUSAL_CONV1D_BWD_WEIGHT,     // Causal conv1d backward w.r.t. weight
	CAUSAL_CONV1D_BWD_BIAS,       // Causal conv1d backward w.r.t. bias
	CAST,                         // Dtype cast operation
	TRANSFER,                     // Device transfer (DtoH, HtoD, DtoD)
	REDUCE_MAXMIN_BACKWARD,       // Backward for reduce max/min
	DROPOUT,                      // Inverted dropout with mask
	VENDOR_OVERRIDE,              // Dispatched to registered override
}

enum ReduceDir : char
{
	ROWS,
	COLS,
	FLAT,
}

// Chain Operation (for fused chains)

/// Operation in a fused chain. Matches interop ChainOp.
/// Layout must match C: bool (+ 3 pad) + int-sized enum union = 8 bytes.
struct ChainOp
{
	bool is_unary;
	// C enums are int-sized; C3 UnaryOp/BinaryOp are char-sized.
	// Store as CInt to match the C ChainOp layout.
	union
	{
		CInt unary;
		CInt binary;
	}
}

// Internal Operation Representation

/// Maximum number of regular inputs per op.
const usz MAX_INPUTS = 3;
/// Maximum number of multi-inputs (concat/stack).
const usz MAX_MULTI_INPUTS = 8;
/// Maximum number of fused chain ops/inputs.
const usz MAX_FUSED_CHAIN = 8;
/// Maximum number of JIT inputs.
const usz MAX_JIT_INPUTS = 32;
/// Maximum einsum pattern length.
const usz MAX_PATTERN_LEN = 16;
/// Maximum number of dimensions for slice.
const usz MAX_SLICE_DIMS = 8;

/// Distinct type for cached kernel handles.
/// Prevents accidental misuse as a generic void*.
typedef KernelHandle = void*;

struct ContractionOp {

	BinaryOp reduce_op;
}

struct ReductionOp {

	BinaryOp reduce_op;
}


struct ElemBinaryTensorOp {

	BinaryOp binary_op;
}

struct ElemUnaryOp {

	UnaryOp map_op;
}

struct ElemBinaryOp {

	BinaryOp binary_op;
}


struct FusedChainOp {

	ChainOp[MAX_FUSED_CHAIN] ops;
	void*[MAX_FUSED_CHAIN] chain_inputs;
	char ops_len;
	char inputs_len;
}

struct CasExprOp {
	char* jit_source;
	usz jit_source_len;
	char jit_num_inputs;
	bool jit_needs_broadcast;
	ulong jit_hash;
	void*[MAX_JIT_INPUTS] jit_inputs;
	ulong[16] jit_dynamic_params;
	char jit_num_dynamic_params;
}

struct SliceOp {
	ulong[MAX_SLICE_DIMS] offsets;
	ulong[MAX_SLICE_DIMS] steps;
	char ndim;
	bool is_scatter;
}

struct ConcatOp {
	ulong[9] offsets;
	char dim;
	char num_inputs;
	void*[MAX_MULTI_INPUTS] multi_inputs;
}

struct StackOp {
	char dim;
	char num_inputs;
	void*[MAX_MULTI_INPUTS] multi_inputs;
}

struct GatherOp {
	char dim;
	void* indices_tensor;
}

struct ScatterAddOp {
	char dim;
	void* indices_tensor;
}

struct SoftmaxOp {
	int axis;
	bool is_log;
}

struct RmsNormOp {
	float eps;
}

struct TransferOp {
	device_reference::TransferDirection direction;
}

struct ReduceMaxMinBwdOp {
	ulong row;
	ulong col;
	ReduceDir reduce_type;
	bool is_min;
}

struct DropoutOp {
	float p;
	ulong seed;
	bool is_bwd;
}

/// Internal representation of a compiled device operation.
/// Tagged union — each OpType has its own data variant.
struct DeviceOp
{
	OpType op_type;
	void* result;
	void*[MAX_INPUTS] inputs;
	char input_count;
	uint branch_id;
	KernelHandle cached_handle;

	Pattern[2] in_pat;
	Pattern out_pat;

	union {
		ContractionOp contraction;
		ReductionOp reduction;
		ElemBinaryTensorOp elem_binary_tensor;
		ElemUnaryOp elem_unary;
		ElemBinaryOp elem_binary;
		FusedChainOp fused_chain;
		CasExprOp cas_expr;
		SliceOp slice;
		ConcatOp concat;
		StackOp stack;
		GatherOp gather;
		ScatterAddOp scatter_add;
		SoftmaxOp softmax;
		RmsNormOp rms_norm;
		TransferOp transfer;
		ReduceMaxMinBwdOp reduce_maxmin_bwd;
		DropoutOp dropout;
	}
}

// DeviceOp methods

fn void DeviceOp.set_input_pattern(&self, usz idx, char[] pattern)
{
	assert(idx < 2, "set_input_pattern: idx out of bounds");
	self.in_pat[idx].clear();
	self.in_pat[idx].push_all(pattern);
}

fn void DeviceOp.set_output_pattern(&self, char[] pattern)
{
	self.out_pat.clear();
	self.out_pat.push_all(pattern);
}

fn char[] DeviceOp.get_input_pattern(&self, usz idx)
{
	return self.in_pat[idx].array_view();
}

fn char[] DeviceOp.get_output_pattern(&self)
{
	return self.out_pat.array_view();
}
