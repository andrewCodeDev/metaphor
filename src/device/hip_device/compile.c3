//! HIP Graph Compiler — Entry Point & Boundary Dispatch
//!
//! Compiles CAS IR entries to HIP execution units.
//!
//! Pipeline:
//! 1. Check for boundary ops via payload (softmax, conv, transfer, etc.)
//! 2. Translate CAS expression to HipOps
//! 3. Build einsum patterns for hipTENSOR
//! 4. Register execution unit via HipExecutableCallback
//!
//! Mirrors src/device/cuda_device/compile.c3 for AMD/HIP.

module metaphor::device::hip_device::compile;

import metaphor::cas;
import metaphor::device::hip_device;
import metaphor::device::hip_device::types;
import metaphor::device::hip_device::callbacks;
import metaphor::device::hip_device::hip_executable_callback;
import metaphor::device::common::operations;
import metaphor::device::common::op_payload;
import metaphor::tensor::tensor_core;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import std::core::mem::allocator;

// CAS ↔ Operations Enum Adapters

/// Convert CAS UnaryOp to operations UnaryOp.
fn operations::UnaryOp cas_unary_to_op(cas::UnaryOp op)
{
	switch (op)
	{
		case ABS:       return ABS;
		case SQRT:      return SQRT;
		case SQR:       return SQR;
		case RECIP:     return RECIP;
		case EXP:       return EXP;
		case LOG:       return LOG;
		case SIN:       return SIN;
		case COS:       return COS;
		case SIGN:      return SIGN;
		case RELU:      return RELU;
		case SIGMOID:   return SIGMOID;
		case TANH:      return TANH;
		case GELU:      return GELU;
		case SILU:      return SILU;
		case SOFTPLUS:  return SOFTPLUS;
		case HEAVISIDE: return HEAVISIDE;
		case LOGIC_NOT: return NOT;
		default:        unreachable("cas_unary_to_op: unhandled CAS unary op");
	}
}

/// Convert CAS BinaryOp to operations BinaryOp.
fn operations::BinaryOp cas_binary_to_op(cas::BinaryOp op)
{
	switch (op)
	{
		case DIV:       return DIV;
		case SUB:       return SUB;
		case MAX:       return MAX;
		case MIN:       return MIN;
		case LOGIC_AND: return AND;
		case LOGIC_OR:  return OR;
		case LOGIC_XOR: return XOR;
		default:        unreachable("cas_binary_to_op: unhandled CAS binary op");
	}
}

/// Convert CAS ReduceOp to operations BinaryOp.
fn operations::BinaryOp cas_reduce_to_binary(cas::ReduceOp op)
{
	switch (op)
	{
		case ADD: return ADD;
		case MUL: return MUL;
		case MAX: return MAX;
		case MIN: return MIN;
	}
}

// Main Entry Point

<*
 Compile a single IR entry to an execution unit.
 Called after CAS mapper has built the IR.
 @require entry != null : "entry must not be null"
 @require entry.expr != null : "entry must have an expression"
 @require entry.output != null : "entry must have an output tensor"
 @require device != null : "device must not be null"
 @require g != null : "graph must not be null"
*>
fn void compile(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
)
{
	void* output = (void*)entry.output;
	assert(output != null, "IR entry output resolved to null");

	// Skip if already compiled
	if (g.exec_seq.contains_target(output))
	{
		return;
	}

	// 1. Check payload for boundary ops
	if (entry.payload != null)
	{
		compile_boundary_op(device, g, entry, entry.payload);
		return;
	}

	// 2. Translate CAS expression to HipOps
	compile_cas_expr(device, g, entry);
}

// Boundary Op Dispatch

/// Dispatch boundary operations from IR payload to specialized HIP callbacks.
<*
 @require device != null : "device must not be null"
 @require g != null : "graph must not be null"
 @require entry != null : "entry must not be null"
 @require payload != null : "payload must not be null"
*>
fn void compile_boundary_op(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	op_payload::OpPayload* payload,
)
{
	void* output = (void*)entry.output;

	switch (payload.tag())
	{
		case SOFTMAX:
			compile_softmax(device, g, entry, &payload.softmax, output);

		case RMS_NORM:
			compile_rms_norm(device, g, entry, &payload.rms_norm, output);

		case CONV:
			compile_conv_fwd(device, g, entry, &payload.conv, output);

		case CONV_TRANSPOSE:
			compile_conv_bwd_data(device, g, entry, &payload.conv_transpose, output);

		case CONV_WEIGHT_GRAD:
			compile_conv_bwd_filter(device, g, entry, &payload.conv, output);

		case TRANSFER:
			compile_transfer(device, g, entry, output);

		case SLICE:
			compile_slice(device, g, entry, &payload.slice, output);

		case CONCAT:
			compile_concat(device, g, entry, &payload.concat, output);

		case STACK:
			compile_stack(device, g, entry, &payload.stack, output);

		case GATHER:
			compile_gather(device, g, entry, &payload.gather, output);

		case SCATTER_ADD:
			compile_scatter_add(device, g, entry, &payload.scatter_add, output);

		case CAST:
			compile_cast(device, g, entry, output);

		case RESHAPE:
			compile_reshape(g, entry, output);

		case REDUCE_MAXMIN_BACKWARD:
			compile_reduce_maxmin_bwd(device, g, entry, &payload.reduce_maxmin_backward, output);

		case SELECTIVE_SCAN:
			compile_selective_scan(device, g, entry, &payload.selective_scan, output);

		case CAUSAL_CONV1D:
			compile_causal_conv1d(device, g, entry, &payload.causal_conv1d, output);

		case DROPOUT:
			compile_dropout(device, g, entry, &payload.dropout, output);

		// These fall through to CAS expression compilation
		case EINSUM:     compile_cas_expr(device, g, entry);
		case REDUCE:     compile_cas_expr(device, g, entry);
		case EXPAND:     compile_cas_expr(device, g, entry);
		case MAP:        compile_cas_expr(device, g, entry);
		case SELECT:     compile_cas_expr(device, g, entry);
		case PERMUTE:    compile_cas_expr(device, g, entry);

		// Shape queries should not reach HIP compiler
		case DIM:        unreachable("shape query 'dim' should not reach HIP compiler");
		case NUMEL:      unreachable("shape query 'numel' should not reach HIP compiler");

		default:         unreachable("compile_boundary_op: unhandled payload tag");
	}
}

// Boundary Op Helpers

fn void compile_softmax(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::SoftmaxPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = SOFTMAX;
	hip_op.result = output;
	hip_op.softmax.axis = p.axis;
	hip_op.softmax.is_log = p.is_log;

	if (p.is_backward)
	{
		assert(entry.inputs.len >= 2, "softmax backward requires 2 inputs (y_val, y_grad)");
		hip_op.inputs[0] = (void*)entry.inputs[0];
		hip_op.inputs[1] = (void*)entry.inputs[1];
		hip_op.input_count = 2;
	}
	else
	{
		assert(entry.inputs.len >= 1, "softmax forward requires 1 input");
		hip_op.inputs[0] = (void*)entry.inputs[0];
		hip_op.input_count = 1;
	}

	register_hip_op(device, g, entry, hip_op);
}

fn void compile_rms_norm(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::RmsNormPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.result = output;
	hip_op.rms_norm.eps = (float)p.eps;

	switch (p.op_type)
	{
		case FORWARD:
			hip_op.op_type = RMS_NORM;
			assert(entry.inputs.len >= 2);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.input_count = 2;

		case BACKWARD_X:
			hip_op.op_type = RMS_NORM_BWD_X;
			assert(entry.inputs.len >= 3);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.inputs[2] = (void*)entry.inputs[2];
			hip_op.input_count = 3;

		case BACKWARD_WEIGHT:
			hip_op.op_type = RMS_NORM_BWD_WEIGHT;
			assert(entry.inputs.len >= 2);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.input_count = 2;
	}

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 2 : "conv forward requires 2 inputs (input, weight)"
*>
fn void compile_conv_fwd(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConvPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = CONTRACTION;
	hip_op.result = output;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.inputs[1] = (void*)entry.inputs[1];
	hip_op.input_count = 2;

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 2 : "conv_transpose requires 2 inputs"
*>
fn void compile_conv_bwd_data(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConvTransposePayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = CONTRACTION;
	hip_op.result = output;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.inputs[1] = (void*)entry.inputs[1];
	hip_op.input_count = 2;

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 2 : "conv_weight_grad requires 2 inputs"
*>
fn void compile_conv_bwd_filter(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConvPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = CONTRACTION;
	hip_op.result = output;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.inputs[1] = (void*)entry.inputs[1];
	hip_op.input_count = 2;

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 1 : "transfer requires 1 input"
*>
fn void compile_transfer(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = TRANSFER;
	hip_op.result = output;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.input_count = 1;
	hip_op.transfer.direction = entry.payload.transfer.direction;

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 1 : "slice requires 1 input"
*>
fn void compile_slice(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::SlicePayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = SLICE;
	hip_op.result = output;
	hip_op.slice.ndim = p.ndim;
	hip_op.slice.is_scatter = p.is_scatter;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.input_count = 1;

	usz ndim = (usz)p.ndim;
	for (usz i = 0; i < ndim && i < types::MAX_SLICE_DIMS; i++)
	{
		hip_op.slice.offsets[i] = p.offsets[i];
		hip_op.slice.steps[i] = (ulong)p.steps[i];
	}

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= (usz)p.num_inputs : "concat: not enough IR inputs for declared num_inputs"
 @require output != null : "concat: output must not be null"
*>
fn void compile_concat(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConcatPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = CONCAT;
	hip_op.result = output;
	hip_op.concat.dim = p.dim;
	hip_op.concat.num_inputs = p.num_inputs;

	usz num_inputs = (usz)p.num_inputs;
	if (num_inputs > 8) num_inputs = 8;

	for (usz i = 0; i <= num_inputs && i < 9; i++)
	{
		hip_op.concat.offsets[i] = p.offsets[i];
	}
	for (usz i = 0; i < num_inputs; i++)
	{
		assert(i < entry.inputs.len, "concat: input index out of bounds");
		hip_op.concat.multi_inputs[i] = (void*)entry.inputs[i];
	}

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= (usz)p.num_inputs : "stack: not enough IR inputs for declared num_inputs"
 @require output != null : "stack: output must not be null"
*>
fn void compile_stack(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::StackPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = STACK;
	hip_op.result = output;
	hip_op.stack.dim = p.dim;
	hip_op.stack.num_inputs = p.num_inputs;

	usz num_inputs = (usz)p.num_inputs;
	if (num_inputs > 8) num_inputs = 8;

	for (usz i = 0; i < num_inputs; i++)
	{
		assert(i < entry.inputs.len, "stack: input index out of bounds");
		hip_op.stack.multi_inputs[i] = (void*)entry.inputs[i];
	}

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 2 : "gather requires 2 inputs (data, indices)"
*>
fn void compile_gather(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::GatherPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = GATHER;
	hip_op.result = output;
	hip_op.gather.dim = p.params.dim;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.input_count = 1;
	hip_op.gather.indices_tensor = (void*)entry.inputs[1];

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 2 : "scatter_add requires 2 inputs (data, indices)"
*>
fn void compile_scatter_add(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ScatterAddPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = SCATTER_ADD;
	hip_op.result = output;
	hip_op.scatter_add.dim = p.params.dim;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.input_count = 1;
	hip_op.scatter_add.indices_tensor = (void*)entry.inputs[1];

	register_hip_op(device, g, entry, hip_op);
}

<*
 @require entry.inputs.len >= 1 : "cast requires 1 input"
*>
fn void compile_cast(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = CAST;
	hip_op.result = output;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.input_count = 1;

	register_hip_op(device, g, entry, hip_op);
}

// Reshape — zero-copy, device-agnostic

struct ReshapeCallback (execution_sequence::Executable)
{
	char _pad;
}

fn void ReshapeCallback.execute(&self, execution_sequence::TensorCorePtr[] sources,
	execution_sequence::TensorCorePtr target) @dynamic
{
	tensor_core::TensorCore* src = (tensor_core::TensorCore*)sources[0];
	tensor_core::TensorCore* tgt = (tensor_core::TensorCore*)target;
	src.share_data(tgt);
}

fn void ReshapeCallback.destroy(&self) @dynamic {}

<*
 @require entry.inputs.len >= 1 : "reshape requires 1 input"
*>
fn void compile_reshape(
	graph::Graph* g,
	cas::IREntry* entry,
	void* output,
)
{
	ReshapeCallback* cb = allocator::new(&graph::graph_alloc, ReshapeCallback);
	*cb = {};

	execution_sequence::TensorCorePtr[1] source_cores = { entry.inputs[0] };
	g.add_execution_unit(
		(any)cb,
		&source_cores,
		(tensor_core::TensorCore*)output,
	);
}

<*
 @require entry.inputs.len >= 2 : "reduce_maxmin_backward requires 2 inputs"
*>
fn void compile_reduce_maxmin_bwd(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ReduceMaxMinBackwardPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = REDUCE_MAXMIN_BACKWARD;
	hip_op.result = output;
	hip_op.reduce_maxmin_bwd.row = p.row;
	hip_op.reduce_maxmin_bwd.col = p.col;
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.inputs[1] = (void*)entry.inputs[1];
	hip_op.input_count = 2;

	register_hip_op(device, g, entry, hip_op);
}

fn void compile_selective_scan(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::SelectiveScanPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.result = output;

	switch (p.op_type)
	{
		case FORWARD:
			hip_op.op_type = SELECTIVE_SCAN;
			assert(entry.inputs.len >= 2);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.input_count = 2;

		case BACKWARD_A_BAR:
			hip_op.op_type = SELECTIVE_SCAN_BWD_A;
			assert(entry.inputs.len >= 3);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.inputs[2] = (void*)entry.inputs[2];
			hip_op.input_count = 3;

		case BACKWARD_B_BAR_X:
			hip_op.op_type = SELECTIVE_SCAN_BWD_BX;
			assert(entry.inputs.len >= 2);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.input_count = 2;
	}

	register_hip_op(device, g, entry, hip_op);
}

fn void compile_causal_conv1d(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::CausalConv1dPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.result = output;

	switch (p.op_type)
	{
		case FORWARD:
			hip_op.op_type = CAUSAL_CONV1D;
			assert(entry.inputs.len >= 3);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.inputs[2] = (void*)entry.inputs[2];
			hip_op.input_count = 3;

		case BACKWARD_X:
			hip_op.op_type = CAUSAL_CONV1D_BWD_X;
			assert(entry.inputs.len >= 2);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.input_count = 2;

		case BACKWARD_WEIGHT:
			hip_op.op_type = CAUSAL_CONV1D_BWD_WEIGHT;
			assert(entry.inputs.len >= 2);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.inputs[1] = (void*)entry.inputs[1];
			hip_op.input_count = 2;

		case BACKWARD_BIAS:
			hip_op.op_type = CAUSAL_CONV1D_BWD_BIAS;
			assert(entry.inputs.len >= 1);
			hip_op.inputs[0] = (void*)entry.inputs[0];
			hip_op.input_count = 1;
	}

	register_hip_op(device, g, entry, hip_op);
}

fn void compile_dropout(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::DropoutPayload* p,
	void* output,
)
{
	types::HipOp hip_op;
	hip_op.op_type = DROPOUT;
	hip_op.result = output;
	hip_op.dropout.p = p.p;
	hip_op.dropout.seed = p.seed;
	hip_op.dropout.is_bwd = p.is_backward;

	assert(entry.inputs.len >= 1, "dropout requires 1 input");
	hip_op.inputs[0] = (void*)entry.inputs[0];
	hip_op.input_count = 1;

	register_hip_op(device, g, entry, hip_op);
}
