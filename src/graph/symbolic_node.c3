//! Symbolic Node
//!
//! Represents a single symbolic operation in the computation graph.
//! Links to other nodes targeting the same result tensor via prev/next
//! pointers forming a doubly-linked chain per tensor.

module metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::device::common::op_payload;

const usz CHILDREN_CAPACITY = 8;

// SymbolicNode

struct SymbolicNode
{
	SymbolicNode* prev;
	SymbolicNode* next;
	Children children;
	op_payload::OpPayload payload;
	Tensor result;
	uint seq;  // topological ordering — assigned by register_op
}

// Children — fixed-capacity buffer for child tensor references

<*
 Tracks versions to detect invalid inplace mutations that would
 invalidate the backward pass.
*>
struct Children
{
	Tensor[CHILDREN_CAPACITY] buffer;
	uint[CHILDREN_CAPACITY] versions;
	usz len;
}

<*
 Creates Children from a slice of tensors.
 Captures the current version of each child for later validation.
*>
fn Children children_from_slice(Tensor[] tensors)
{
	assert(CHILDREN_CAPACITY >= tensors.len);
	Children c;
	c.len = tensors.len;
	for (usz i = 0; i < tensors.len; i++)
	{
		c.buffer[i] = tensors[i];
		c.versions[i] = tensors[i].core().version;
	}
	return c;
}

fn Tensor[] Children.slice(&self)
{
	return self.buffer[:self.len];
}

<*
 Validates that child versions haven't changed since capture.
 Panics if an inplace mutation invalidated the backward pass.
*>
fn void Children.check(&self)
{
	for (usz i = 0; i < self.len; i++)
	{
		uint old_version = self.versions[i];
		uint new_version = self.buffer[i].core().version;
		assert(old_version == new_version,
			"Version mismatch: tensor was modified in-place after operation was scheduled");
	}
}
