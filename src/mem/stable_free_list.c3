//! Stable Free List
//!
//! Arena-backed pool with intrusive free list. Items returned to the free list
//! remain valid (data is preserved) — this enables idempotent double-free and
//! simplifies error handling when lifetimes are contextually dependent.
//!
//! The Wrapper struct places `data` first so that Type* and Wrapper* share the
//! same address, making casts trivial with no offset calculation.

module metaphor::mem::stable_free_list <Type>;
import std::core::mem::allocator;

const usz PAGE_SIZE = 4096;

struct Wrapper
{
	Type data;
	Wrapper* next_free;
}

struct StableFreeList
{
	DynamicArenaAllocator arena;
	Wrapper* free_head;
}

// Lifecycle

fn StableFreeList* StableFreeList.init(&self, Allocator backing)
{
	self.arena.init(backing, PAGE_SIZE);
	self.free_head = null;
	return self;
}

fn void StableFreeList.free(&self)
{
	self.arena.free();
}

// Core operations

<*
 Allocates or reuses a slot from the pool.
 Returns a pointer to the data field of the wrapper.
*>
fn Type*? StableFreeList.create(&self)
{
	Wrapper* wrapper;
	if (self.free_head)
	{
		wrapper = self.free_head;
		self.free_head = wrapper.next_free;
	}
	else
	{
		wrapper = allocator::new_try(&self.arena, Wrapper)!;
	}
	return &wrapper.data;
}

<*
 Returns a slot to the free list. Data remains valid in memory.
*>
fn void StableFreeList.destroy(&self, Type* ptr)
{
	Wrapper* wrapper = (Wrapper*)ptr;
	wrapper.next_free = self.free_head;
	self.free_head = wrapper;
}

<*
 Resets the arena and clears the free list.
 All previously returned pointers become invalid.
*>
fn void StableFreeList.reset(&self)
{
	self.arena.reset();
	self.free_head = null;
}
