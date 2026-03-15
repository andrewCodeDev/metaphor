//! Symbolic Chain Map
//!
//! Maps TensorCores to doubly-linked chains of SymbolicNodes.
//! Each chain is a linked list of operations targeting the same tensor,
//! used for CAS mapping, backward graph construction, and fusion identification.
//!

module metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::utils::hash_map;
import std::core::mem::allocator;

// Direction

enum Direction : char
{
	FWD,
	BWD,
}

// SymbolicChain — doubly-linked list of SymbolicNodes

struct SymbolicChain
{
	SymbolicNode* first;
	SymbolicNode* last;
}

/// Append a node to the end of the chain.
fn void SymbolicChain.append(&self, SymbolicNode* node)
{
	node.prev = self.last;
	node.next = null;
	if (self.last != null)
	{
		self.last.next = node;
	}
	else
	{
		self.first = node;
	}
	self.last = node;
}

/// Count nodes in the chain.
fn usz SymbolicChain.len(&self)
{
	usz count = 0;
	SymbolicNode* it = self.first;
	while (it != null)
	{
		count++;
		it = it.next;
	}
	return count;
}

// ChainIterator — walks a chain forward or backward

struct ChainIterator
{
	SymbolicNode* pos;
	Direction dir;
}

fn SymbolicNode* ChainIterator.next(&self)
{
	SymbolicNode* current = self.pos;
	if (current == null) return null;
	switch (self.dir)
	{
		case FWD: self.pos = current.next;
		case BWD: self.pos = current.prev;
	}
	return current;
}

// SymbolicChainMap — maps TensorCore* → SymbolicChain

alias ChainMap = hash_map::HashMap {uptr, SymbolicChain};

struct SymbolicChainMap
{
	ChainMap map;
	uint count;
}

fn void SymbolicChainMap.init(&self, Allocator allocator)
{
	self.map.init(allocator);
	self.count = 0;
}

fn void SymbolicChainMap.free(&self)
{
	self.map.free();
}

fn void SymbolicChainMap.clear_retaining_capacity(&self)
{
	self.map.clear();
	self.count = 0;
}

/// Append a symbolic node to the chain for its result tensor.
fn void SymbolicChainMap.put_append(&self, SymbolicNode* node)
{
	uptr key = (uptr)node.result.core();

	// Get existing chain or create new one
	if (try chain_ptr = self.map.get_ref(key))
	{
		chain_ptr.append(node);
	}
	else
	{
		SymbolicChain chain;
		chain.first = null;
		chain.last = null;
		chain.append(node);
		self.map.set(key, chain);
	}
	self.count += 1;
}

/// Returns true if the map contains an entry for the given tensor.
fn bool SymbolicChainMap.contains(&self, tensor_core::TensorCore* tc)
{
	return self.map.has_key((uptr)tc);
}

/// Look up the chain for a tensor. Returns null if not found.
fn SymbolicChain* SymbolicChainMap.lookup(&self, tensor_core::TensorCore* tc)
{
	if (try ptr = self.map.get_ref((uptr)tc))
	{
		return ptr;
	}
	return null;
}

/// Creates an iterator over a tensor's symbolic chain.
/// Returns null if the tensor has no chain.
fn ChainIterator? SymbolicChainMap.chain_iterator(&self, tensor_core::TensorCore* tc, Direction dir)
{
	SymbolicChain* chain = self.lookup(tc);
	if (chain == null) return NOT_FOUND~;

	SymbolicNode* start;
	switch (dir)
	{
		case FWD: start = chain.first;
		case BWD: start = chain.last;
	}
	return { .pos = start, .dir = dir };
}

/// Returns the first node in a tensor's chain, or null.
fn SymbolicNode* SymbolicChainMap.get_first_node(&self, tensor_core::TensorCore* tc)
{
	SymbolicChain* chain = self.lookup(tc);
	if (chain == null) return null;
	return chain.first;
}

/// Returns the number of symbolic nodes in a tensor's chain, or 0.
fn usz SymbolicChainMap.chain_len(&self, tensor_core::TensorCore* tc)
{
	SymbolicChain* chain = self.lookup(tc);
	if (chain == null) return 0;
	return chain.len();
}

/// Total node count across all chains.
fn uint SymbolicChainMap.num_nodes(&self)
{
	return self.count;
}

/// Returns true if the map is empty.
fn bool SymbolicChainMap.is_empty(&self)
{
	return self.count == 0;
}

/// Reassigns a chain from one tensor core to another.
/// Updates all node results to point to to_core.
/// Returns true if the chain was found and reassigned.
fn bool SymbolicChainMap.reassign_chain(&self,
	tensor_core::TensorCore* from_core,
	tensor_core::TensorCore* to_core)
{
	uptr from_key = (uptr)from_core;
	SymbolicChain* chain = self.lookup(from_core);
	if (chain == null) return false;

	// Copy chain data before removal
	SymbolicChain chain_copy = *chain;

	// Remove from old key
	self.map.remove(from_key)!!;

	// Update all nodes to point to to_core
	tensor::Tensor to_tensor = tensor::tensor_from_core(to_core);
	SymbolicNode* node = chain_copy.first;
	while (node != null)
	{
		node.result = to_tensor;
		node = node.next;
	}

	// Put chain under new key
	self.map.set((uptr)to_core, chain_copy);
	return true;
}

/// Removes a tensor's chain entry from the map.
/// Returns the number of nodes that were in the chain.
fn usz SymbolicChainMap.remove_tensor(&self, tensor_core::TensorCore* tc)
{
	SymbolicChain* chain = self.lookup(tc);
	if (chain == null) return 0;
	usz removed = chain.len();
	self.count -= (uint)removed;
	self.map.remove((uptr)tc)!!;
	return removed;
}

/// Collect all symbolic nodes across all chains into a buffer.
/// Returns the number of nodes collected.
fn usz SymbolicChainMap.collect_all_nodes(&self, SymbolicNode*[] out_buf)
{
	usz idx = 0;

	// Walk the SwissTable backing store directly
	char* ctrl = self.map.table.ctrl;
	if (ctrl == null) return 0;

	usz num_buckets = self.map.table.bucket_mask + 1;

	for (usz b = 0; b < num_buckets; b++)
	{
		// Only process full slots (high bit clear = occupied)
		if ((ctrl[b] & 0x80) != 0) continue;

		SymbolicChain* chain = &self.map.table.data[b].value;
		SymbolicNode* it = chain.first;
		while (it != null)
		{
			if (idx < out_buf.len)
			{
				out_buf[idx] = it;
				idx++;
			}
			it = it.next;
		}
	}
	return idx;
}

faultdef NOT_FOUND;
