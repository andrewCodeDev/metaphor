//! Graph — Execution Unit Integration
//!
//! Wrappers that delegate to the graph's ExecutionSequence.
//! Separate file to keep graph.c3 focused on core graph structure.

module metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor::tensor_core;

// Execution Unit wrappers

/// Returns true if the given tensor has a compiled execution unit.
fn bool Graph.has_execution_unit(&self, tensor_core::TensorCore* core)
{
	return self.exec_seq.contains_target(core);
}

/// Execute the computation graph for a target tensor.
/// If not yet compiled, compile first (via CAS mapper + device compile).
/// Then execute via the execution sequence.
fn void Graph.execute_node(&self, tensor_core::TensorCore* core)
{
	if (self.has_execution_unit(core))
	{
		self.exec_seq.execute_for_target(core);
	}
	else
	{
		// Compile first, then execute
		tensor_core::TensorCore*[1] observables = { core };
		compile_graph(self, &observables);
		self.exec_seq.execute_for_target(core);
	}
}

/// Append a new execution unit to the graph's execution sequence.
/// Source dependencies are resolved internally from source_cores.
/// Auto-promotes target tensor from NONE to graph's default_persistence.
fn execution_sequence::ExecutionUnit* Graph.add_execution_unit(
	&self,
	any operation,
	execution_sequence::TensorCorePtr[] source_cores,
	tensor_core::TensorCore* target,
)
{
	// Auto-promote from NONE to the graph's default persistence.
	// If the user has already called .stable() or .unstable(),
	// persistence is above NONE and promote() is a no-op.
	target.flags.persistence.promote(self.flags.default_persistence);
	return self.exec_seq.append(
		operation,
		source_cores,
		target,
	);
}
