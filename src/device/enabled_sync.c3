//! Enabled Sync Primitives
//!
//! Synchronization primitives for device execution.
//! Each device type has its own sync mechanism:
//! - Host: atomic counter (increment/decrement/spin)
//! - CUDA: CudaEvent (record/synchronize/query) — added later
//!

module metaphor::device::sync;
import std::core::mem::allocator;

// SyncPrimitive interface — unified API across device sync types

interface SyncPrimitive
{
	/// Signal that work has started.
	fn void begin();

	/// Signal that work has completed.
	fn void complete();

	/// Block until all pending work is complete.
	fn void await_ready();

	/// Non-blocking check if work is complete.
	fn bool is_ready();

	/// Reset to ready state.
	fn void reset();
}

// HostSync — host-side atomic counter synchronization

<*
 Host synchronization primitive using a shared atomic counter.

 When counter is null: always ready (no pending work).
 When counter is non-null: work is pending if counter > 0.

 The counter pointer is shared — copying a HostSync shares the sync state.
 This enables cross-graph tensor sharing: when TensorCore.sync is copied,
 both copies point to the same underlying counter.
*>
struct HostSync (SyncPrimitive)
{
	/// Pointer to shared atomic counter. null means "ready" (no pending work).
	uint* counter;
}

fn void HostSync.init_empty(&self)
{
	self.counter = null;
}

fn void HostSync.init_with_counter(&self, Allocator alloc)
{
	uint* c = allocator::new(alloc, uint);
	*c = 0;
	self.counter = c;
}

fn void HostSync.begin(&self) @dynamic
{
	if (self.counter != null)
	{
		// Simple non-atomic increment — sufficient for single-threaded host
		(*self.counter)++;
	}
}

fn void HostSync.complete(&self) @dynamic
{
	if (self.counter != null)
	{
		(*self.counter)--;
	}
}

fn void HostSync.await_ready(&self) @dynamic
{
	if (self.counter == null) { return; }
	// Spin until counter reaches 0
	while (*self.counter != 0)
	{
		// Spin hint — yields to CPU
		asm("pause");
	}
}

fn bool HostSync.is_ready(&self) @dynamic
{
	if (self.counter == null) { return true; }
	return *self.counter == 0;
}

fn void HostSync.reset(&self) @dynamic
{
	if (self.counter != null)
	{
		*self.counter = 0;
	}
}

/// Free the allocated counter. Called by the device that owns it during deinit.
fn void HostSync.free_counter(&self, Allocator alloc)
{
	if (self.counter != null)
	{
		allocator::free(alloc, self.counter);
	}
}
