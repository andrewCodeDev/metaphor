//! Graph Compilation with Device-Agnostic IR
//!
//! Walk depth-first from target tensors, build IR for each device subgraph,
//! then dispatch to devices for compilation.
//!
//! Flow:
//! 1. Walk backward, detect device boundaries
//! 2. Build IR (IREntry) for each device subgraph via CAS mapper
//! 3. For each IR: device.compile(ir)
//!

module metaphor::graph;
import metaphor::tensor::tensor_core;
import metaphor::cas;
import metaphor::device::common::operations;
import metaphor::device::device_reference;

// compile_graph — top-level entry point

/// Compile the graph for the given observable tensors.
/// Builds IR via CAS mapper, dispatches to device compilers.
fn void compile_graph(Graph* g, tensor_core::TensorCore*[] observables)
{
	for (usz i = 0; i < observables.len; i++)
	{
		compile_tensor(g, observables[i]);
	}
	g.exec_seq.finalize_closures();
}

/// Compile a single tensor and all its dependencies.
fn void compile_tensor(Graph* g, tensor_core::TensorCore* target)
{
	cas::IREntry[] irs = map_graph(&g.symbolic_seq, target, g);

	for (usz i = 0; i < irs.len; i++)
	{
		compile_ir_entry(g, &irs[i]);
	}
}

/// Compile a single IR entry by dispatching to its device.
fn void compile_ir_entry(Graph* g, cas::IREntry* entry)
{
	assert(!g.has_execution_unit(entry.output), "IR entry already has execution unit");

	// For TRANSFER ops, dispatch to the non-host device since it knows
	// how to do cross-device memcpy (HtoD, DtoH, DtoD).
	if (entry.payload != null && entry.payload.tag() == operations::OpTag.TRANSFER)
	{
		device_reference::DeviceReference src_dev = entry.inputs[0].device;
		device_reference::DeviceReference dst_dev = entry.output.device;
		if (!src_dev.is_host())
		{
			src_dev.compile(g, entry);
		}
		else
		{
			dst_dev.compile(g, entry);
		}
		return;
	}

	// Dispatch to device's compile function via the device interface.
	entry.output.device.compile(g, entry);
}
