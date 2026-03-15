//! Caching Allocator — high-level wrapper around BlockPool
//!
//! Provides dtype-aware allocation for tensor memory. Delegates to a
//! BlockPool buddy allocator for the actual memory management.
//!

module metaphor::device::host_device::caching_allocator;
import metaphor::device::host_device::block_pool;
import metaphor::rtti;

// Faults

faultdef DEVICE_OOM;

// CachingAllocator

struct CachingAllocator
{
	BlockPool pool;
}

// Lifecycle

fn CachingAllocator*? CachingAllocator.init(&self, usz pool_size)
{
	self.pool.init(pool_size)!;
	return self;
}

fn void CachingAllocator.deinit(&self)
{
	self.pool.deinit();
}

// Alloc / Free / Reset

fn char[]? CachingAllocator.alloc(&self, Datatype dtype, usz len)
{
	usz byte_size = len * dtype.size_of();
	if (byte_size == 0)
	{
		return {};
	}

	if (try result = self.pool.alloc(byte_size))
	{
		return result;
	}
	return DEVICE_OOM~;
}

fn void CachingAllocator.free(&self, char[] slice)
{
	if (slice.len == 0) return;
	self.pool.free(slice);
}

fn void CachingAllocator.reset(&self)
{
	self.pool.reset();
}

// Memory tracking

fn usz CachingAllocator.mem_used(&self) => self.pool.mem_used();

fn usz CachingAllocator.mem_reserved(&self) => self.pool.mem_reserved();

fn usz CachingAllocator.mem_peak(&self) => self.pool.mem_peak();
