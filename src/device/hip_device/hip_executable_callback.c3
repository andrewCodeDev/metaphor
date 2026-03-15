//! HIP Executable Callback
//!
//! Self-contained callback for executing HIP operations. Stores a list of
//! HipOps and dispatches each to the appropriate backend (hipTENSOR, JIT,
//! MIOpen, custom kernels). Tensor pointers are updated from sources/target
//! via bindings at the start of each execute() call.
//!
//! This is the HIP equivalent of a host executable callback.
//!
//! Port of src/device/cuda_device/cuda_executable_callback.c3 for AMD/HIP.

module metaphor::device::hip_device::hip_executable_callback;
import metaphor::device::hip_device;
import metaphor::device::hip_device::types;
import metaphor::device::hip_device::executor;
import metaphor::device::common::device_binding;
import metaphor::graph::execution_sequence;

// TensorCore pointer alias

alias TensorCorePtr = execution_sequence::TensorCorePtr;

// Binding — imported from common device_binding module

alias Binding = device_binding::Binding;
alias BindingSourceKind = device_binding::BindingSourceKind;

// HipExecutableCallback

/// Self-contained callback for executing HIP operations.
/// Tensor pointers are updated from sources/target via bindings
/// at the start of each execute() call.
struct HipExecutableCallback (Executable)
{
	/// Device reference for dispatch.
	HipDevice* device;

	/// Operations to execute in topological order.
	HipOp[] ops;

	/// Bindings describe how to update tensor pointers from sources/targets.
	Binding[] bindings;
}

// Executable implementation

<*
 @require sources.len > 0 || self.bindings.len == 0 : "execute: sources is empty but bindings exist"
*>
fn void HipExecutableCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	// Apply bindings: update tensor slots from sources/target
	for (usz i = 0; i < self.bindings.len; i++)
	{
		Binding* binding = &self.bindings[i];
		TensorCorePtr tensor = null;

		switch (binding.kind)
		{
			case INPUT:
				if (binding.source_idx < (uint)sources.len)
				{
					tensor = sources[binding.source_idx];
				}
			case OUTPUT:
				tensor = target;
			case INTERMEDIATE:
				if (binding.source_idx < (uint)self.ops.len)
				{
					tensor = self.ops[binding.source_idx].result;
				}
		}

		if (tensor == null) { continue; }

		switch (binding.slot.kind)
		{
			case RESULT:
				self.ops[binding.op_idx].result = tensor;
			case JIT_INPUT:
				usz jit_idx = (usz)binding.slot.index;
				self.ops[binding.op_idx].cas_expr.jit_inputs[jit_idx] = tensor;
			case MULTI_INPUT:
				usz multi_idx = (usz)binding.slot.index;
				if (self.ops[binding.op_idx].op_type == types::OpType.ELEMENTWISE_FUSED_CHAIN)
				{
					self.ops[binding.op_idx].fused_chain.chain_inputs[multi_idx] = tensor;
				}
				else if (self.ops[binding.op_idx].op_type == types::OpType.CONCAT)
				{
					self.ops[binding.op_idx].concat.multi_inputs[multi_idx] = tensor;
				}
				else if (self.ops[binding.op_idx].op_type == types::OpType.STACK)
				{
					self.ops[binding.op_idx].stack.multi_inputs[multi_idx] = tensor;
				}
			case INPUT:
				assert(binding.op_idx < (uint)self.ops.len,
					"binding: op_idx out of bounds");
				self.ops[binding.op_idx].inputs[(usz)binding.slot.index] = tensor;
			case INDICES:
				if (self.ops[binding.op_idx].op_type == types::OpType.GATHER)
				{
					self.ops[binding.op_idx].gather.indices_tensor = tensor;
				}
				else
				{
					self.ops[binding.op_idx].scatter_add.indices_tensor = tensor;
				}
		}
	}

	// Validate all ops have non-null results after binding resolution
	for (usz i = 0; i < self.ops.len; i++)
	{
		assert(self.ops[i].result != null,
			"execute: op result is null after bindings");
	}

	// Execute all ops
	for (usz i = 0; i < self.ops.len; i++)
	{
		executor::execute_single_op(self.device, &self.ops[i]);
	}
}

fn void HipExecutableCallback.destroy(&self) @dynamic
{
	// TODO: Free ops, bindings slices
	// These are allocator-managed and freed when the graph is torn down
}
