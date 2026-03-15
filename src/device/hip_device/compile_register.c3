//! HIP Graph Compiler — Registration & Fusion
//!
//! Registers compiled HipOps as execution units in the sequence,
//! and applies fusion optimization to merge consecutive elementwise ops.
//!
//! Part of the compile module.

module metaphor::device::hip_device::compile;

import metaphor::cas;
import metaphor::device::hip_device;
import metaphor::device::hip_device::types;
import metaphor::device::hip_device::hip_executable_callback;
import metaphor::device::hip_device::executor;
import metaphor::device::common::device_binding;
import metaphor::device::common::operations;
import metaphor::tensor::tensor_core;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import std::core::mem::allocator;

// Binding builder — imported from common device_binding module

alias BindingBuilder = device_binding::BindingBuilder;

/// Find which entry input index matches a given tensor pointer.
fn uint find_input_idx(void* tensor, tensor_core::TensorCore*[] inputs)
{
	for (usz i = 0; i < inputs.len; i++)
	{
		if ((void*)inputs[i] == tensor) return (uint)i;
	}
	unreachable("find_input_idx: tensor not found in inputs");
}

/// Find which input index matches a given tensor pointer in a void*[] list.
fn uint find_input_idx_v(void* tensor, void*[] inputs, usz count)
{
	for (usz i = 0; i < count; i++)
	{
		if (inputs[i] == tensor) return (uint)i;
	}
	unreachable("find_input_idx_v: tensor not found in inputs");
}

// Registration — wraps HipOps in HipExecutableCallback

/// Register a single HipOp from a boundary dispatch.
fn void register_hip_op(
	hip_device::HipDevice* device,
	graph::Graph* g,
	cas::IREntry* entry,
	types::HipOp hip_op,
)
{
	Allocator alloc = g.exec_seq.allocator;

	// Allocate ops array (1 element)
	HipOp[] ops = allocator::alloc_array(alloc, HipOp, 1);
	ops[0] = hip_op;

	// JIT source handling: if cached_handle is already resolved (via JitCache),
	// skip source duplication and compilation. Otherwise fall back to old path.
	if ((void*)ops[0].cached_handle == null && ops[0].cas_expr.jit_source != null && ops[0].cas_expr.jit_source_len > 0)
	{
		char[] heap_src = allocator::alloc_array(alloc, char, ops[0].cas_expr.jit_source_len + 1);
		heap_src[:ops[0].cas_expr.jit_source_len] = ops[0].cas_expr.jit_source[:ops[0].cas_expr.jit_source_len];
		heap_src[ops[0].cas_expr.jit_source_len] = 0;
		ops[0].cas_expr.jit_source = heap_src.ptr;
		ops[0].cached_handle = executor::get_or_load_kernel(ops[0].cas_expr.jit_source, ops[0].cas_expr.jit_source_len);
	}

	// Copy inputs
	execution_sequence::TensorCorePtr[] inputs_copy = {};
	if (entry.inputs.len > 0)
	{
		inputs_copy = allocator::alloc_array(alloc, execution_sequence::TensorCorePtr, entry.inputs.len);
		for (usz i = 0; i < entry.inputs.len; i++)
		{
			inputs_copy[i] = entry.inputs[i];
		}
	}

	// Build bindings
	BindingBuilder bb;
	bb.len = 0;

	// Input slot bindings
	for (usz slot = 0; slot < (usz)hip_op.input_count; slot++)
	{
		if (hip_op.inputs[slot] != null)
		{
			uint src_idx = find_input_idx(hip_op.inputs[slot], entry.inputs);
			bb.add(0, { INPUT, (char)slot }, INPUT, src_idx);
		}
	}

	// Multi-input bindings for concat/stack
	if (hip_op.op_type == types::OpType.CONCAT)
	{
		for (usz i = 0; i < (usz)hip_op.concat.num_inputs; i++)
		{
			if (hip_op.concat.multi_inputs[i] != null)
			{
				uint src_idx = find_input_idx(hip_op.concat.multi_inputs[i], entry.inputs);
				bb.add(0, { MULTI_INPUT, (char)i }, INPUT, src_idx);
			}
		}
	}
	else if (hip_op.op_type == types::OpType.STACK)
	{
		for (usz i = 0; i < (usz)hip_op.stack.num_inputs; i++)
		{
			if (hip_op.stack.multi_inputs[i] != null)
			{
				uint src_idx = find_input_idx(hip_op.stack.multi_inputs[i], entry.inputs);
				bb.add(0, { MULTI_INPUT, (char)i }, INPUT, src_idx);
			}
		}
	}

	// Indices tensor binding for gather/scatter_add
	if (hip_op.op_type == types::OpType.GATHER && hip_op.gather.indices_tensor != null)
	{
		uint src_idx = find_input_idx(hip_op.gather.indices_tensor, entry.inputs);
		bb.add(0, { INDICES, 0 }, INPUT, src_idx);
	}
	else if (hip_op.op_type == types::OpType.SCATTER_ADD && hip_op.scatter_add.indices_tensor != null)
	{
		uint src_idx = find_input_idx(hip_op.scatter_add.indices_tensor, entry.inputs);
		bb.add(0, { INDICES, 0 }, INPUT, src_idx);
	}

	// Output binding
	bb.add(0, { RESULT, 0 }, OUTPUT, 0);

	// Copy bindings to heap
	hip_executable_callback::Binding[] bindings = allocator::alloc_array(alloc, hip_executable_callback::Binding, bb.len);
	bindings[..] = bb.buf[:bb.len];

	// Allocate callback
	hip_executable_callback::HipExecutableCallback* callback = allocator::alloc(alloc, hip_executable_callback::HipExecutableCallback);
	*callback = {
		.device = device,
		.ops = ops,
		.bindings = bindings,
	};

	// Register via graph to get default_persistence promotion
	g.add_execution_unit(
		(any)(Executable)callback,
		inputs_copy,
		entry.output,
	);
}

/// Register a HipOp with inputs collected from the op itself.
/// Used for CAS expression-derived ops.
fn void register_hip_op_with_inputs(
	hip_device::HipDevice* device,
	graph::Graph* g,
	types::HipOp hip_op,
)
{
	Allocator alloc = g.exec_seq.allocator;

	// Collect unique input tensors from the op
	void*[32] input_tensors;
	usz input_count = 0;

	for (usz i = 0; i < (usz)hip_op.input_count; i++)
	{
		if (hip_op.inputs[i] != null)
		{
			if (!ptr_in_list(hip_op.inputs[i], input_tensors[:input_count]))
			{
				input_tensors[input_count] = hip_op.inputs[i];
				input_count++;
			}
		}
	}

	// Multi-inputs for concat/stack
	if (hip_op.op_type == types::OpType.CONCAT)
	{
		for (usz i = 0; i < (usz)hip_op.concat.num_inputs; i++)
		{
			if (hip_op.concat.multi_inputs[i] != null)
			{
				if (!ptr_in_list(hip_op.concat.multi_inputs[i], input_tensors[:input_count]))
				{
					input_tensors[input_count] = hip_op.concat.multi_inputs[i];
					input_count++;
				}
			}
		}
	}
	else if (hip_op.op_type == types::OpType.STACK)
	{
		for (usz i = 0; i < (usz)hip_op.stack.num_inputs; i++)
		{
			if (hip_op.stack.multi_inputs[i] != null)
			{
				if (!ptr_in_list(hip_op.stack.multi_inputs[i], input_tensors[:input_count]))
				{
					input_tensors[input_count] = hip_op.stack.multi_inputs[i];
					input_count++;
				}
			}
		}
	}

	// Indices tensor
	if (hip_op.op_type == types::OpType.GATHER && hip_op.gather.indices_tensor != null)
	{
		if (!ptr_in_list(hip_op.gather.indices_tensor, input_tensors[:input_count]))
		{
			input_tensors[input_count] = hip_op.gather.indices_tensor;
			input_count++;
		}
	}
	else if (hip_op.op_type == types::OpType.SCATTER_ADD && hip_op.scatter_add.indices_tensor != null)
	{
		if (!ptr_in_list(hip_op.scatter_add.indices_tensor, input_tensors[:input_count]))
		{
			input_tensors[input_count] = hip_op.scatter_add.indices_tensor;
			input_count++;
		}
	}

	// Fused chain inputs
	if (hip_op.op_type == types::OpType.ELEMENTWISE_FUSED_CHAIN)
	{
		for (usz i = 0; i < (usz)hip_op.fused_chain.inputs_len; i++)
		{
			if (hip_op.fused_chain.chain_inputs[i] != null)
			{
				if (!ptr_in_list(hip_op.fused_chain.chain_inputs[i], input_tensors[:input_count]))
				{
					input_tensors[input_count] = hip_op.fused_chain.chain_inputs[i];
					input_count++;
				}
			}
		}
	}

	// CAS JIT inputs
	if (hip_op.op_type == types::OpType.ELEMENTWISE_CAS_EXPR)
	{
		for (usz i = 0; i < (usz)hip_op.cas_expr.jit_num_inputs; i++)
		{
			if (hip_op.cas_expr.jit_inputs[i] != null)
			{
				if (!ptr_in_list(hip_op.cas_expr.jit_inputs[i], input_tensors[:input_count]))
				{
					input_tensors[input_count] = hip_op.cas_expr.jit_inputs[i];
					input_count++;
				}
			}
		}
	}

	// Allocate ops
	HipOp[] ops = allocator::alloc_array(alloc, HipOp, 1);
	ops[0] = hip_op;

	// JIT source handling: if cached_handle is already resolved (via JitCache),
	// skip source duplication and compilation. Otherwise fall back to old path.
	if ((void*)ops[0].cached_handle == null && ops[0].cas_expr.jit_source != null && ops[0].cas_expr.jit_source_len > 0)
	{
		char[] heap_src = allocator::alloc_array(alloc, char, ops[0].cas_expr.jit_source_len + 1);
		heap_src[:ops[0].cas_expr.jit_source_len] = ops[0].cas_expr.jit_source[:ops[0].cas_expr.jit_source_len];
		heap_src[ops[0].cas_expr.jit_source_len] = 0;
		ops[0].cas_expr.jit_source = heap_src.ptr;
		ops[0].cached_handle = executor::get_or_load_kernel(ops[0].cas_expr.jit_source, ops[0].cas_expr.jit_source_len);
	}

	// Copy inputs to heap
	execution_sequence::TensorCorePtr[] inputs_copy = {};
	if (input_count > 0)
	{
		inputs_copy = allocator::alloc_array(alloc, execution_sequence::TensorCorePtr, input_count);
		for (usz i = 0; i < input_count; i++)
		{
			inputs_copy[i] = (execution_sequence::TensorCorePtr)input_tensors[i];
		}
	}

	// Build bindings
	BindingBuilder bb;
	bb.len = 0;

	// Input slot bindings
	for (usz slot = 0; slot < (usz)hip_op.input_count; slot++)
	{
		if (hip_op.inputs[slot] != null)
		{
			uint src_idx = find_input_idx_v(hip_op.inputs[slot], &input_tensors, input_count);
			bb.add(0, { INPUT, (char)slot }, INPUT, src_idx);
		}
	}

	// Fused chain input bindings
	if (hip_op.op_type == types::OpType.ELEMENTWISE_FUSED_CHAIN)
	{
		for (usz fused_slot = 0; fused_slot < (usz)hip_op.fused_chain.inputs_len; fused_slot++)
		{
			if (hip_op.fused_chain.chain_inputs[fused_slot] != null)
			{
				uint src_idx = find_input_idx_v(hip_op.fused_chain.chain_inputs[fused_slot], &input_tensors, input_count);
				bb.add(0, { MULTI_INPUT, (char)fused_slot }, INPUT, src_idx);
			}
		}
	}

	// CAS expression JIT input bindings
	if (hip_op.op_type == types::OpType.ELEMENTWISE_CAS_EXPR)
	{
		for (usz jit_slot = 0; jit_slot < (usz)hip_op.cas_expr.jit_num_inputs; jit_slot++)
		{
			if (hip_op.cas_expr.jit_inputs[jit_slot] != null)
			{
				uint src_idx = find_input_idx_v(hip_op.cas_expr.jit_inputs[jit_slot], &input_tensors, input_count);
				bb.add(0, { JIT_INPUT, (char)jit_slot }, INPUT, src_idx);
			}
		}
	}

	// Multi-input bindings for concat/stack
	if (hip_op.op_type == types::OpType.CONCAT)
	{
		for (usz i = 0; i < (usz)hip_op.concat.num_inputs; i++)
		{
			if (hip_op.concat.multi_inputs[i] != null)
			{
				uint src_idx = find_input_idx_v(hip_op.concat.multi_inputs[i], &input_tensors, input_count);
				bb.add(0, { MULTI_INPUT, (char)i }, INPUT, src_idx);
			}
		}
	}
	else if (hip_op.op_type == types::OpType.STACK)
	{
		for (usz i = 0; i < (usz)hip_op.stack.num_inputs; i++)
		{
			if (hip_op.stack.multi_inputs[i] != null)
			{
				uint src_idx = find_input_idx_v(hip_op.stack.multi_inputs[i], &input_tensors, input_count);
				bb.add(0, { MULTI_INPUT, (char)i }, INPUT, src_idx);
			}
		}
	}

	// Output binding
	bb.add(0, { RESULT, 0 }, OUTPUT, 0);

	// Copy bindings
	hip_executable_callback::Binding[] bindings = allocator::alloc_array(alloc, hip_executable_callback::Binding, bb.len);
	bindings[..] = bb.buf[:bb.len];

	// Allocate callback
	hip_executable_callback::HipExecutableCallback* callback = allocator::alloc(alloc, hip_executable_callback::HipExecutableCallback);
	*callback = {
		.device = device,
		.ops = ops,
		.bindings = bindings,
	};

	// Register via graph to get default_persistence promotion
	g.add_execution_unit(
		(any)(Executable)callback,
		inputs_copy,
		(tensor_core::TensorCore*)hip_op.result,
	);
}

fn bool ptr_in_list(void* ptr, void*[] list)
{
	for (usz i = 0; i < list.len; i++)
	{
		if (list[i] == ptr) return true;
	}
	return false;
}

// Fusion Pass

/// Fusion optimization: combines consecutive elementwise ops into fused chains.
fn void apply_fusion_pass(types::HipOp* ops_buf, usz* ops_count)
{
	if (*ops_count < 2) return;

	const usz MAX_FUSED_INPUTS = 6;

	// Count consumers for each op's result
	void*[MAX_COMPILE_OPS] results;
	usz[MAX_COMPILE_OPS] consumer_counts;
	for (usz i = 0; i < *ops_count; i++)
	{
		results[i] = ops_buf[i].result;
		consumer_counts[i] = 0;
	}

	// Count how many times each result is consumed as input
	for (usz i = 0; i < *ops_count; i++)
	{
		for (usz j = 0; j < (usz)ops_buf[i].input_count; j++)
		{
			if (ops_buf[i].inputs[j] != null)
			{
				for (usz k = 0; k < *ops_count; k++)
				{
					if (results[k] == ops_buf[i].inputs[j])
					{
						consumer_counts[k]++;
						break;
					}
				}
			}
		}
	}

	// Find fusible chains
	const usz MAX_CHAINS = 16;
	usz[MAX_CHAINS] chain_starts;
	usz[MAX_CHAINS] chain_lens;
	usz num_chains = 0;

	bool[MAX_COMPILE_OPS] in_chain;
	for (usz i = 0; i < MAX_COMPILE_OPS; i++) in_chain[i] = false;

	usz i = 0;
	while (i < *ops_count)
	{
		if (in_chain[i] || !is_fusible_elementwise(&ops_buf[i]))
		{
			i++;
			continue;
		}

		// Check if first input comes from a producer in this batch
		bool first_input_is_internal = false;
		if (ops_buf[i].inputs[0] != null)
		{
			for (usz k = 0; k < *ops_count; k++)
			{
				if (results[k] == ops_buf[i].inputs[0])
				{
					first_input_is_internal = true;
					break;
				}
			}
		}
		if (first_input_is_internal)
		{
			i++;
			continue;
		}

		// Try to extend chain forward
		usz chain_len = 1;
		void* current_result = ops_buf[i].result;
		usz j = i + 1;
		usz input_count = 1;

		if (ops_buf[i].op_type == types::OpType.ELEMENTWISE_BINARY) input_count++;

		while (j < *ops_count)
		{
			types::HipOp* next_op = &ops_buf[j];
			if (!is_fusible_elementwise(next_op)) break;
			if (next_op.inputs[0] != current_result) break;
			if (next_op.op_type == types::OpType.ELEMENTWISE_BINARY && input_count >= MAX_FUSED_INPUTS) break;
			if (consumer_counts[find_result_idx(results[:*ops_count], current_result)] != 1) break;

			chain_len++;
			if (next_op.op_type == types::OpType.ELEMENTWISE_BINARY) input_count++;
			current_result = next_op.result;
			j++;
		}

		if (chain_len >= 2 && num_chains < MAX_CHAINS)
		{
			chain_starts[num_chains] = i;
			chain_lens[num_chains] = chain_len;
			num_chains++;
			for (usz k = i; k < i + chain_len; k++) in_chain[k] = true;
		}

		i++;
	}

	if (num_chains == 0) return;

	// Build new ops list with fused chains
	types::HipOp[MAX_COMPILE_OPS] new_ops;
	usz new_count = 0;
	usz chain_idx = 0;

	i = 0;
	while (i < *ops_count)
	{
		if (chain_idx < num_chains && chain_starts[chain_idx] == i)
		{
			usz len = chain_lens[chain_idx];
			new_ops[new_count] = build_fused_op(&ops_buf[i], len);
			new_count++;
			i += len;
			chain_idx++;
		}
		else
		{
			new_ops[new_count] = ops_buf[i];
			new_count++;
			i++;
		}
	}

	// Copy back
	for (usz k = 0; k < new_count; k++) ops_buf[k] = new_ops[k];
	*ops_count = new_count;
}

fn usz find_result_idx(void*[] results, void* target)
{
	for (usz i = 0; i < results.len; i++)
	{
		if (results[i] == target) return i;
	}
	unreachable("find_result_idx: target not found in results");
}

fn types::HipOp build_fused_op(types::HipOp* chain, usz chain_len)
{
	types::HipOp fused;
	fused.op_type = ELEMENTWISE_FUSED_CHAIN;
	fused.result = chain[chain_len - 1].result;

	fused.fused_chain.chain_inputs[0] = chain[0].inputs[0];
	fused.fused_chain.inputs_len = 1;

	for (usz idx = 0; idx < chain_len; idx++)
	{
		types::ChainOp chain_op;
		if (chain[idx].op_type == types::OpType.ELEMENTWISE_UNARY)
		{
			chain_op.is_unary = true;
			chain_op.unary = (CInt)chain[idx].elem_unary.map_op;
		}
		else
		{
			chain_op.is_unary = false;
			chain_op.binary = (CInt)chain[idx].elem_binary.binary_op;
			if (chain[idx].inputs[1] != null)
			{
				if ((usz)fused.fused_chain.inputs_len < types::MAX_FUSED_CHAIN)
				{
					fused.fused_chain.chain_inputs[(usz)fused.fused_chain.inputs_len] = chain[idx].inputs[1];
					fused.fused_chain.inputs_len++;
				}
			}
		}
		fused.fused_chain.ops[idx] = chain_op;
	}
	fused.fused_chain.ops_len = (char)chain_len;

	return fused;
}

fn bool is_fusible_elementwise(types::HipOp* op)
{
	return op.op_type == types::OpType.ELEMENTWISE_UNARY
		|| op.op_type == types::OpType.ELEMENTWISE_BINARY;
}
