//! Independent event/tracing module
//!
//! Fires lifecycle callbacks for tensor operations: alloc, free, execute,
//! offload, restore. Gated behind ENABLE_EVENTS — when disabled, the fire!
//! macro expands to nothing (zero overhead).

module metaphor::events;
import std::core::mem::allocator;
import metaphor::utils::hash_map;
import metaphor::tensor::tensor_core;

alias TensorCorePtr = tensor_core::TensorCore*;

enum EventKind : char {
	ALLOC,
	FREE,
	EXECUTE,
	OFFLOAD,
	RESTORE,
}

alias EventFn = fn void(TensorCorePtr tc, EventKind kind);

macro void fire(tc, kind) {
	$if $feature(ENABLE_EVENTS):
		fire_impl(tc, kind);
	$endif
}

// --- Implementation below: only meaningful when ENABLE_EVENTS is set ---

alias PtrToEventMap = hash_map::HashMap {uptr, EventFn};

PtrToEventMap event_map;
bool initialized = false;
EventFn global_hook = null;

fn void ensure_init() @private
{
	if (!initialized)
	{
		allocator::LibcAllocator alloc;
		event_map.init(&alloc);
		initialized = true;
	}
}

/// Register a per-tensor event callback.
fn void register(TensorCorePtr tc, EventFn callback)
{
	ensure_init();
	event_map.set((uptr)tc, callback);
}

/// Remove a per-tensor event callback.
fn void unregister(TensorCorePtr tc)
{
	ensure_init();
	event_map.remove((uptr)tc)!!;
}

/// Remove all registered per-tensor events.
fn void clear()
{
	if (initialized)
	{
		event_map.clear();
	}
}

/// Set a global hook that fires for every event.
fn void set_global_hook(EventFn hook)
{
	global_hook = hook;
}

fn void fire_impl(TensorCorePtr tc, EventKind kind) @private @inline
{
	// Global hook first
	if (global_hook != null)
	{
		global_hook(tc, kind);
	}

	// Per-tensor callbacks
	if (!initialized) { return; }
	if (try callback = event_map.get((uptr)tc))
	{
		callback(tc, kind);
	}
}

fn void shutdown() @finalizer
{
	if (initialized)
	{
		event_map.free();
		initialized = false;
	}
	global_hook = null;
}
