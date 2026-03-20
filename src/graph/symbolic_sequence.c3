//! Symbolic Sequence
//!
//! Maps TensorCores to their SymbolicNodes — one node per tensor.
//! Used for CAS mapping, backward graph construction, and fusion identification.

module metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::utils::hash_map;
import std::core::mem::allocator;

// SymbolicSequence — maps TensorCore* → SymbolicNode*

alias NodeMap = hash_map::HashMap {uptr, SymbolicNode*};

struct SymbolicSequence
{
	NodeMap map;
}

fn void SymbolicSequence.init(&self, Allocator allocator)
{
	self.map.init(allocator);
}

fn void SymbolicSequence.free(&self)
{
	self.map.free();
}

fn void SymbolicSequence.clear_retaining_capacity(&self)
{
	self.map.clear();
}

/// Register a symbolic node for its result tensor.
/// Panics if the tensor already has a node (one op per tensor).
fn void SymbolicSequence.put(&self, SymbolicNode* node)
{
	uptr key = (uptr)node.result.core();
	assert(!self.map.has_key(key), "Duplicate symbolic node for tensor");
	self.map.set(key, node);
}

/// Returns true if the tensor has a registered node.
fn bool SymbolicSequence.contains(&self, tensor_core::TensorCore* tc)
{
	return self.map.has_key((uptr)tc);
}

/// Look up the node for a tensor.
fn SymbolicNode*? SymbolicSequence.get(&self, tensor_core::TensorCore* tc)
{
	return self.map.get((uptr)tc);
}

/// Total node count.
fn uint SymbolicSequence.num_nodes(&self)
{
	return (uint)self.map.len();
}

/// Reassigns a node from one tensor core to another.
/// Updates the node's result to point to to_core.
/// Returns true if the node was found and reassigned.
fn bool SymbolicSequence.reassign(&self,
	tensor_core::TensorCore* from_core,
	tensor_core::TensorCore* to_core)
{
	uptr from_key = (uptr)from_core;

	SymbolicNode*? node = self.get(from_core);
	if (catch node) return false;

	// Remove from old key
	self.map.remove(from_key)!!;

	// Update node to point to to_core
	node.result = tensor::tensor_from_core(to_core);

	// Put under new key
	self.map.set((uptr)to_core, node);
	return true;
}

/// Removes a tensor's node from the sequence.
/// Returns true if a node was removed.
fn bool SymbolicSequence.remove(&self, tensor_core::TensorCore* tc)
{
	if (!self.map.has_key((uptr)tc)) return false;
	self.map.remove((uptr)tc)!!;
	return true;
}
