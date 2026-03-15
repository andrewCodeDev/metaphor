//! Device Binding Types
//!
//! Shared types for describing how tensor pointers are wired from
//! execution unit sources/targets into operation slots. Used by both
//! HIP and CUDA executable callbacks.

module metaphor::device::common::device_binding;

/// Source kind for tensor pointer bindings.
enum BindingSourceKind : char
{
	INPUT,         // from sources[] at source_idx
	OUTPUT,        // from target
	INTERMEDIATE,  // from result of ops[source_idx]
}

/// Which field on DeviceOp a tensor pointer should be written to.
enum SlotKind : char
{
	INPUT,         // inputs[index]
	MULTI_INPUT,   // fused_chain_inputs[index]
	JIT_INPUT,     // jit_inputs[index]
	INDICES,       // indices_tensor (index ignored)
	RESULT,        // result (index ignored)
}

struct Slot
{
	SlotKind kind;
	char index;
}

/// Describes how to update a tensor slot in an operation.
struct Binding
{
	uint op_idx;
	Slot slot;
	BindingSourceKind kind;
	uint source_idx;
}

/// Stack-local accumulator for building Binding arrays during compilation.
struct BindingBuilder
{
	Binding[64] buf;
	usz len;
}

fn void BindingBuilder.add(&self, uint op_idx, Slot slot, BindingSourceKind kind, uint source_idx)
{
	assert(self.len < 64, "BindingBuilder overflow");
	self.buf[self.len] = { .op_idx = op_idx, .slot = slot, .kind = kind, .source_idx = source_idx };
	self.len++;
}
