//! CUDA Graph Compiler — Entry Point & Boundary Dispatch
//!
//! Compiles CAS IR entries to CUDA execution units.
//!
//! Pipeline:
//! 1. Check for boundary ops via payload (softmax, conv, transfer, etc.)
//! 2. Translate CAS expression to CudaOps
//! 3. Build einsum patterns for cuTENSOR
//! 4. Register execution unit via CudaExecutableCallback
//!

module metaphor::device::cuda_device::compile;

import metaphor::cas;
import metaphor::device::cuda_device;
import metaphor::device::cuda_device::types;
import metaphor::device::cuda_device::callbacks;
import metaphor::device::cuda_device::cuda_executable_callback;
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
	cuda_device::CudaDevice* device,
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

	// 2. Translate CAS expression to CudaOps
	compile_cas_expr(device, g, entry);
}

// Boundary Op Dispatch

/// Dispatch boundary operations from IR payload to specialized CUDA callbacks.
<*
 @require device != null : "device must not be null"
 @require g != null : "graph must not be null"
 @require entry != null : "entry must not be null"
 @require payload != null : "payload must not be null"
*>
fn void compile_boundary_op(
	cuda_device::CudaDevice* device,
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

		// Shape queries should not reach CUDA compiler
		case DIM:        unreachable("shape query 'dim' should not reach CUDA compiler");
		case NUMEL:      unreachable("shape query 'numel' should not reach CUDA compiler");

		default:         unreachable("compile_boundary_op: unhandled payload tag");
	}
}

// Boundary Op Helpers

fn void compile_softmax(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::SoftmaxPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = SOFTMAX;
	cuda_op.result = output;
	cuda_op.softmax.axis = p.axis;
	cuda_op.softmax.is_log = p.is_log;

	if (p.is_backward)
	{
		assert(entry.inputs.len >= 2, "softmax backward requires 2 inputs (y_val, y_grad)");
		cuda_op.inputs[0] = (void*)entry.inputs[0];
		cuda_op.inputs[1] = (void*)entry.inputs[1];
		cuda_op.input_count = 2;
	}
	else
	{
		assert(entry.inputs.len >= 1, "softmax forward requires 1 input");
		cuda_op.inputs[0] = (void*)entry.inputs[0];
		cuda_op.input_count = 1;
	}

	register_cuda_op(device, g, entry, cuda_op);
}

fn void compile_rms_norm(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::RmsNormPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.result = output;
	cuda_op.rms_norm.eps = (float)p.eps;

	switch (p.op_type)
	{
		case FORWARD:
			cuda_op.op_type = RMS_NORM;
			assert(entry.inputs.len >= 2, "rms_norm forward requires 2 inputs");
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.input_count = 2;

		case BACKWARD_X:
			cuda_op.op_type = RMS_NORM_BWD_X;
			assert(entry.inputs.len >= 3, "rms_norm backward_x requires 3 inputs");
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.inputs[2] = (void*)entry.inputs[2];
			cuda_op.input_count = 3;

		case BACKWARD_WEIGHT:
			cuda_op.op_type = RMS_NORM_BWD_WEIGHT;
			assert(entry.inputs.len >= 2, "rms_norm backward_weight requires 2 inputs");
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.input_count = 2;
	}

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 2 : "conv forward requires 2 inputs (input, weight)"
*>
fn void compile_conv_fwd(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConvPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = CONTRACTION;
	cuda_op.result = output;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.inputs[1] = (void*)entry.inputs[1];
	cuda_op.input_count = 2;

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 2 : "conv_transpose requires 2 inputs"
*>
fn void compile_conv_bwd_data(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConvTransposePayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = CONTRACTION;
	cuda_op.result = output;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.inputs[1] = (void*)entry.inputs[1];
	cuda_op.input_count = 2;

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 2 : "conv_weight_grad requires 2 inputs"
*>
fn void compile_conv_bwd_filter(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConvPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = CONTRACTION;
	cuda_op.result = output;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.inputs[1] = (void*)entry.inputs[1];
	cuda_op.input_count = 2;

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 1 : "transfer requires 1 input"
*>
fn void compile_transfer(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = TRANSFER;
	cuda_op.result = output;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.input_count = 1;
	cuda_op.transfer.direction = entry.payload.transfer.direction;

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 1 : "slice requires 1 input"
*>
fn void compile_slice(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::SlicePayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = SLICE;
	cuda_op.result = output;
	cuda_op.slice.ndim = p.ndim;
	cuda_op.slice.is_scatter = p.is_scatter;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.input_count = 1;

	usz ndim = (usz)p.ndim;
	for (usz i = 0; i < ndim && i < types::MAX_SLICE_DIMS; i++)
	{
		cuda_op.slice.offsets[i] = p.offsets[i];
		cuda_op.slice.steps[i] = (ulong)p.steps[i];
	}

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= (usz)p.num_inputs : "concat: not enough IR inputs for declared num_inputs"
 @require output != null : "concat: output must not be null"
*>
fn void compile_concat(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ConcatPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = CONCAT;
	cuda_op.result = output;
	cuda_op.concat.dim = p.dim;
	cuda_op.concat.num_inputs = p.num_inputs;

	usz num_inputs = (usz)p.num_inputs;
	if (num_inputs > 8) num_inputs = 8;

	for (usz i = 0; i <= num_inputs && i < 9; i++)
	{
		cuda_op.concat.offsets[i] = p.offsets[i];
	}
	for (usz i = 0; i < num_inputs; i++)
	{
		assert(i < entry.inputs.len, "concat: input index out of bounds");
		cuda_op.concat.multi_inputs[i] = (void*)entry.inputs[i];
	}

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= (usz)p.num_inputs : "stack: not enough IR inputs for declared num_inputs"
 @require output != null : "stack: output must not be null"
*>
fn void compile_stack(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::StackPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = STACK;
	cuda_op.result = output;
	cuda_op.stack.dim = p.dim;
	cuda_op.stack.num_inputs = p.num_inputs;

	usz num_inputs = (usz)p.num_inputs;
	if (num_inputs > 8) num_inputs = 8;

	for (usz i = 0; i < num_inputs; i++)
	{
		assert(i < entry.inputs.len, "stack: input index out of bounds");
		cuda_op.stack.multi_inputs[i] = (void*)entry.inputs[i];
	}

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 2 : "gather requires 2 inputs (data, indices)"
*>
fn void compile_gather(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::GatherPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = GATHER;
	cuda_op.result = output;
	cuda_op.gather.dim = p.params.dim;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.input_count = 1;
	cuda_op.gather.indices_tensor = (void*)entry.inputs[1];

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 2 : "scatter_add requires 2 inputs (data, indices)"
*>
fn void compile_scatter_add(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ScatterAddPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = SCATTER_ADD;
	cuda_op.result = output;
	cuda_op.scatter_add.dim = p.params.dim;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.input_count = 1;
	cuda_op.scatter_add.indices_tensor = (void*)entry.inputs[1];

	register_cuda_op(device, g, entry, cuda_op);
}

<*
 @require entry.inputs.len >= 1 : "cast requires 1 input"
*>
fn void compile_cast(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = CAST;
	cuda_op.result = output;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.input_count = 1;

	register_cuda_op(device, g, entry, cuda_op);
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
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::ReduceMaxMinBackwardPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = REDUCE_MAXMIN_BACKWARD;
	cuda_op.result = output;
	cuda_op.reduce_maxmin_bwd.row = p.row;
	cuda_op.reduce_maxmin_bwd.col = p.col;
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.inputs[1] = (void*)entry.inputs[1];
	cuda_op.input_count = 2;

	register_cuda_op(device, g, entry, cuda_op);
}

fn void compile_selective_scan(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::SelectiveScanPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.result = output;

	switch (p.op_type)
	{
		case FORWARD:
			cuda_op.op_type = SELECTIVE_SCAN;
			assert(entry.inputs.len >= 2);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.input_count = 2;

		case BACKWARD_A_BAR:
			cuda_op.op_type = SELECTIVE_SCAN_BWD_A;
			assert(entry.inputs.len >= 3);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.inputs[2] = (void*)entry.inputs[2];
			cuda_op.input_count = 3;

		case BACKWARD_B_BAR_X:
			cuda_op.op_type = SELECTIVE_SCAN_BWD_BX;
			assert(entry.inputs.len >= 2);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.input_count = 2;
	}

	register_cuda_op(device, g, entry, cuda_op);
}

fn void compile_causal_conv1d(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::CausalConv1dPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.result = output;

	switch (p.op_type)
	{
		case FORWARD:
			cuda_op.op_type = CAUSAL_CONV1D;
			assert(entry.inputs.len >= 3);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.inputs[2] = (void*)entry.inputs[2];
			cuda_op.input_count = 3;

		case BACKWARD_X:
			cuda_op.op_type = CAUSAL_CONV1D_BWD_X;
			assert(entry.inputs.len >= 2);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.input_count = 2;

		case BACKWARD_WEIGHT:
			cuda_op.op_type = CAUSAL_CONV1D_BWD_WEIGHT;
			assert(entry.inputs.len >= 2);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.inputs[1] = (void*)entry.inputs[1];
			cuda_op.input_count = 2;

		case BACKWARD_BIAS:
			cuda_op.op_type = CAUSAL_CONV1D_BWD_BIAS;
			assert(entry.inputs.len >= 1);
			cuda_op.inputs[0] = (void*)entry.inputs[0];
			cuda_op.input_count = 1;
	}

	register_cuda_op(device, g, entry, cuda_op);
}

fn void compile_dropout(
	cuda_device::CudaDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	operations::DropoutPayload* p,
	void* output,
)
{
	types::CudaOp cuda_op;
	cuda_op.op_type = DROPOUT;
	cuda_op.result = output;
	cuda_op.dropout.p = p.p;
	cuda_op.dropout.seed = p.seed;
	cuda_op.dropout.is_bwd = p.is_backward;

	assert(entry.inputs.len >= 1, "dropout requires 1 input");
	cuda_op.inputs[0] = (void*)entry.inputs[0];
	cuda_op.input_count = 1;

	register_cuda_op(device, g, entry, cuda_op);
}
