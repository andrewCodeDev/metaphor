//! Execution Sequence
//!
//! A pointer-based execution graph. Each ExecutionUnit stores direct
//! pointers to its dependency units and embeds its tensor cores inline.
//!
//! Key properties:
//! - Units heap-allocated, owned by the target_to_unit_map
//! - Dependencies are direct ExecutionUnit* pointers (no integer indices)
//! - Tensor cores embedded inline (no separate map lookups)
//! - RefCounts embedded inline (no parallel array)

module metaphor::graph::execution_sequence;
import std::core::mem::allocator;
import std::core::mem;

// Use our SwissTable-backed HashMap
import metaphor::utils::hash_map;
import metaphor::utils::swisstable;
import metaphor::tensor::tensor_core;
import metaphor::graph;
import metaphor::events;
import metaphor::device::device_reference;
import std::io;
import std::collections::list;

alias TensorCorePtr = tensor_core::TensorCore*;

// Interfaces — C3 dynamic dispatch

<*
 Executable: the core interface that every execution operation must implement.
 Device backends create concrete types implementing this interface.
*>
interface Executable {
	fn void execute(TensorCorePtr[] sources, TensorCorePtr target);
	fn void destroy();
}

// RefCount — packed visit flag + consumer count for DFS execution

bitstruct RefCount : uint
{
	uint count : 0..30;
	bool visited : 31;
}

// Persistence — tensor persistence level (mirrors TensorCore.Persistence)

alias Persistence = tensor_core::Persistence;

// ExecutionUnit — a single unit in the graph

struct ExecutionUnit
{
	/// Type-erased operation (implements Executable).
	any operation;

	/// Direct dependency units — units this unit depends on.
	ExecutionUnit*[] direct_sources;

	/// Consumer units — units that depend on this unit's output.
	/// Inverse of direct_sources, computed in finalize_closures.
	list::List{ExecutionUnit*} consumers;

	/// Tensor cores (embedded inline — no separate map).
	TensorCorePtr[] source_cores;
	TensorCorePtr target;

	/// Ref count for DFS execution (embedded inline — no parallel array).
	RefCount ref_count;

	/// Bit vector of transitive leaf dependencies (indices into LeafRegistry).
	/// Built in finalize_closures(). Used to compute fingerprints.
	list::List{ulong} leaf_bits;

	/// Fingerprint of leaf mutation state at last execution.
	/// XOR of hash(leaf_index, mutation_counter) for each leaf in leaf_bits.
	/// Mismatch with current state means a dependency changed — must re-execute.
	ulong fingerprint;

	/// Whether this unit has ever been executed.
	bool has_executed;
}

// LeafRegistry — flat array mapping leaf index ↔ pointer + mutation counter

struct LeafEntry
{
	uptr ptr;          // leaf TensorCore pointer
	ulong mutations;   // incremented on each load/fill/bind
}

struct LeafRegistry
{
	list::List{LeafEntry} entries;
	hash_map::HashMap{uptr, usz} ptr_to_index;  // leaf ptr → index
	Allocator allocator;
}

fn void LeafRegistry.init(&self, Allocator alloc)
{
	self.entries.init(alloc, initial_capacity: 64);
	self.ptr_to_index.init(alloc);
	self.allocator = alloc;
}

fn void LeafRegistry.free(&self)
{
	self.entries.free();
	self.ptr_to_index.free();
}

fn void LeafRegistry.clear(&self)
{
	self.entries.clear();
	self.ptr_to_index.clear();
}

/// Register a leaf and return its index. Deduplicates.
fn usz LeafRegistry.register_leaf(&self, uptr leaf_ptr)
{
	if (try idx = self.ptr_to_index.get(leaf_ptr))
	{
		return idx;
	}
	usz idx = self.entries.size;
	self.entries.push({ .ptr = leaf_ptr, .mutations = 0 });
	self.ptr_to_index.set(leaf_ptr, idx);
	return idx;
}

/// Increment mutation counter for a leaf. Returns false if not registered.
fn bool LeafRegistry.mutate(&self, uptr leaf_ptr)
{
	if (try idx = self.ptr_to_index.get(leaf_ptr))
	{
		(&self.entries[idx]).mutations++;
		return true;
	}
	return false;
}

/// Number of ulong words needed for a bit vector covering all leaves.
fn usz LeafRegistry.bitvec_words(&self) => (self.entries.size + 63) / 64;

fn void ExecutionUnit.add_consumer(&self, ExecutionUnit* consumer)
{
	self.consumers.push(consumer);
}

// HashMap aliases

alias PtrToUnitPtrMap = hash_map::HashMap {uptr, ExecutionUnit*};

// OffloadBuffer — CPU-side copy of device tensor data
alias OffloadBuffer = char[];

alias PtrToOffloadMap = hash_map::HashMap {uptr, OffloadBuffer};

// ExecutionSequence — the main sequence of execution units

struct ExecutionSequence
{
	/// Maps target tensor core pointer → unit that writes it.
	/// The map owns the heap-allocated ExecutionUnit pointers.
	PtrToUnitPtrMap target_to_unit_map;

	/// Maps tensor core pointer → CPU-side offload buffer.
	/// Presence in this map means the tensor's data has been offloaded.
	PtrToOffloadMap offload_buffers;

	/// Registry of all leaf tensors referenced by the graph.
	/// Maps leaf pointers to indices and tracks mutation counters.
	LeafRegistry leaf_registry;

	/// Set when a tensor is destroyed while compiled. Blocks new appends.
	bool corrupted;

	/// Allocator used for all internal allocations.
	Allocator allocator;

}

// Execution statistics (module-level)

ulong stats_executed = 0;
ulong stats_skipped = 0;
ulong stats_offloaded = 0;
ulong stats_restored = 0;
ulong stats_freed = 0;

fn void reset_stats()
{
	stats_executed = 0;
	stats_skipped = 0;
	stats_offloaded = 0;
	stats_restored = 0;
	stats_freed = 0;
}

// Fingerprint helpers

/// Hash a (leaf_index, mutation_counter) pair into a contribution for XOR fingerprint.
/// Uses a simple mixing function to ensure different indices/counters produce distinct values.
fn ulong leaf_hash(usz leaf_index, ulong mutations) @inline
{
	ulong x = ((ulong)leaf_index * 0x9E3779B97F4A7C15) ^ (mutations * 0x517CC1B727220A95);
	// murmurhash3 finalizer
	x ^= x >> 30;
	x *= 0xBF58476D1CE4E5B9;
	x ^= x >> 27;
	x *= 0x94D049BB133111EB;
	x ^= x >> 31;
	return x;
}

/// Compute the current fingerprint for a unit from its leaf_bits and the registry.
fn ulong compute_fingerprint(ulong[] leaf_bits, LeafRegistry* reg) @inline
{
	ulong fp = 0;
	usz words = leaf_bits.len;
	for (usz w = 0; w < words; w++)
	{
		ulong bits = leaf_bits[w];
		while (bits != 0)
		{
			usz bit = (usz)$$ctz(bits);  // count trailing zeros → bit index
			usz leaf_idx = w * 64 + bit;
			fp ^= leaf_hash(leaf_idx, reg.entries[leaf_idx].mutations);
			bits &= bits - 1;  // clear lowest set bit
		}
	}
	return fp;
}

// Lifecycle

fn ExecutionSequence* ExecutionSequence.init(&self, Allocator alloc)
{
	self.corrupted = false;
	self.allocator = alloc;
	self.target_to_unit_map.init(alloc);
	self.offload_buffers.init(alloc);
	self.leaf_registry.init(alloc);
	return self;
}

fn void ExecutionSequence.destroy_all_units(&self)
{
	swisstable::SwissTableValueIterator {uptr, ExecutionUnit*} it = self.target_to_unit_map.value_iter();
	while (try unit = it.next())
	{
		// Free GPU data for the target tensor. Only execution sequence
		// targets (STABLE/UNSTABLE) live here — never LEAFs (inputs/weights).
		// Exported tensors already have data=null (export steals the pointer),
		// so release_data() no-ops for them. Refcount-aware via the graph's
		// data_refcounts map (still valid — freed after exec_seq in Graph.deinit).
		tensor_core::TensorCore* tc = unit.target;
		tc.release_data();

		Executable e = (Executable)unit.operation;
		if (&e.destroy) { e.destroy(); }

		allocator::free(self.allocator, unit.direct_sources);
		unit.consumers.free();
		allocator::free(self.allocator, unit.source_cores);
		unit.leaf_bits.free();
		allocator::free(self.allocator, unit);
	}
}

fn void ExecutionSequence.free(&self)
{
	self.destroy_all_units();

	// Free all offload buffers
	self.free_offload_buffers();
	self.offload_buffers.free();

	self.leaf_registry.free();
	self.target_to_unit_map.free();
}

<*
 Clears all entries.
*>
fn void ExecutionSequence.clear_retaining_capacity(&self)
{
	self.destroy_all_units();

	self.target_to_unit_map.clear();
	self.leaf_registry.clear();

	// Sequence is now empty, corruption is resolved
	self.corrupted = false;
}

// Accessors

fn usz ExecutionSequence.len(&self) @inline => self.target_to_unit_map.len();

fn bool ExecutionSequence.is_empty(&self) @inline => self.target_to_unit_map.len() == 0;

fn bool ExecutionSequence.contains_target(&self, TensorCorePtr core)
{
	return self.target_to_unit_map.has_key((uptr)core);
}

fn ExecutionUnit*? ExecutionSequence.get_unit_for_target(&self, TensorCorePtr core)
{
	return self.target_to_unit_map.get((uptr)core);
}

// Corruption

<*
 Marks the sequence as corrupted because a compiled tensor was destroyed.
*>
fn void ExecutionSequence.mark_corrupted(&self, TensorCorePtr core)
{
	self.target_to_unit_map.remove((uptr)core)!!;

	// If no targets remain, units have nowhere to write — clear.
	if (self.target_to_unit_map.len() == 0)
	{
		self.clear_retaining_capacity();
	}
	else
	{
		self.corrupted = true;
	}
}

// Offloading — DtoH/HtoD tensor data management

<*
 Offload a tensor's device data to a CPU-side buffer.
 Copies data DtoH, frees device memory, and stores the buffer in the map.
 For host devices this is a no-op (data is already on CPU).
 Cannot offload leaf tensors.
*>
fn void ExecutionSequence.offload_tensor(&self, TensorCorePtr core)
{
	tensor_core::TensorCore* tc = core;

	// Only stable tensors can be offloaded
	assert(tc.flags.persistence >= STABLE,
		"Can only offload stable tensors");

	// Skip if host device — data is already on CPU
	if (tc.device.is_host()) { return; }

	// Skip if no data to offload
	if (!tc.has_data()) { return; }

	// Only support dense tensors for offloading
	assert(tc.state.class_ == DENSE,
		"Can only offload dense tensors");

	usz byte_size = (usz)tc.state.dense.byte_size;
	char[] device_data = ((char*)tc.state.dense.data)[:byte_size];

	// Allocate or reuse CPU buffer
	OffloadBuffer buf;
	if (try existing = self.offload_buffers.get((uptr)core))
	{
		buf = existing;
		// Realloc if size changed
		if (buf.len != byte_size)
		{
			allocator::free(self.allocator, buf);
			buf = allocator::alloc_array(self.allocator, char, byte_size);
		}
	}
	else
	{
		buf = allocator::alloc_array(self.allocator, char, byte_size);
	}

	// DtoH copy
	char[] host_dst = buf;
	tc.device.mem_transfer(device_data, host_dst, DTOH);

	// Store in map
	self.offload_buffers.set((uptr)core, buf);

	// Free device memory (refcount-aware)
	tc.release_data();
	stats_offloaded++;
}

<*
 Restore a tensor's data from the offload buffer back to the device.
 Allocates device memory and copies HtoD. Keeps the CPU buffer for reuse
 on the next offload cycle.
 Called automatically by dfs_execute when a source with no data is found
 in the offload buffer map.
*>
fn void ExecutionSequence.restore_tensor(&self, TensorCorePtr core) @private
{
	tensor_core::TensorCore* tc = core;

	OffloadBuffer buf = self.offload_buffers.get((uptr)core)!!;

	// Allocate device memory
	tc.alloc_data();

	char[] host_src = buf;
	char[] device_dst = ((char*)tc.state.dense.data)[:buf.len];

	// HtoD copy
	tc.device.mem_transfer(host_src, device_dst, HTOD);

	// Keep CPU buffer and map entry for reuse on next offload cycle
	stats_restored++;
}

<*
 Free all CPU-side offload buffers. Called during ExecutionSequence.free().
*>
fn void ExecutionSequence.free_offload_buffers(&self) @private
{
	// Iterate over all offload buffer entries and free CPU memory
	swisstable::SwissTableIterator {uptr, OffloadBuffer} it = self.offload_buffers.iter();
	while (try entry = it.next())
	{
		if (entry.value.len > 0)
		{
			allocator::free(self.allocator, entry.value);
		}
	}
}

// Append


<*
 Appends a new execution unit to the sequence.
 Source dependencies are resolved internally from source_cores via the
 target_to_unit_map. No integer indices needed.
 Returns a pointer to the new unit.
 @require target_core.flags.persistence != NONE
     : "tensor must have persistence before entering execution sequence"
*>
fn ExecutionUnit* ExecutionSequence.append(
	&self,
	any operation,
	TensorCorePtr[] source_cores,
	TensorCorePtr target_core,
)
{
	assert(!self.corrupted,
		"Cannot add to corrupted execution sequence. A compiled tensor was destroyed.");

	// Resolve source units: look up which source_cores have producing units
	ExecutionUnit*[64] src_buf;
	usz src_count = 0;
	for (usz i = 0; i < source_cores.len; i++)
	{
		if (try src_unit = self.target_to_unit_map.get((uptr)source_cores[i]))
		{
			src_buf[src_count] = src_unit;
			src_count++;
		}
	}

	// Heap-allocate the unit
	ExecutionUnit* unit = allocator::new(self.allocator, ExecutionUnit);

	*unit = {
		.operation = operation,
		.direct_sources = allocator::clone_slice(self.allocator, src_buf[:src_count]),
		.source_cores = allocator::clone_slice(self.allocator, source_cores),
		.target = target_core,
		.ref_count = { .count = 0, .visited = false },
	};
	unit.consumers.init(self.allocator, initial_capacity: 4);
	unit.leaf_bits.init(self.allocator, initial_capacity: 0);

	// Register target → unit mapping
	self.target_to_unit_map.set((uptr)target_core, unit);

	return unit;
}

// Finalize Closures — compute dependency relationships

<*
 Computes consumer relationships (inverse of direct_sources)
 and rebuilds leaf bit vectors for fingerprinting.
*>
fn void ExecutionSequence.finalize_closures(&self)
{
	// First pass: compute consumers (inverse of direct_sources)
	swisstable::SwissTableValueIterator {uptr, ExecutionUnit*} it2 = self.target_to_unit_map.value_iter();
	while (try consumer = it2.next())
	{
		for (usz s = 0; s < consumer.direct_sources.len; s++)
		{
			ExecutionUnit* source = consumer.direct_sources[s];
			source.add_consumer(consumer);
		}
	}

	// Third pass: register leaves and rebuild bit vectors.
	// Clear old bitvecs — leaf count may have changed since last compile.
	swisstable::SwissTableValueIterator {uptr, ExecutionUnit*} it_clear = self.target_to_unit_map.value_iter();
	while (try unit = it_clear.next())
	{
		unit.leaf_bits.clear();
	}

	// Discover and register all leaf source_cores.
	swisstable::SwissTableValueIterator {uptr, ExecutionUnit*} it3 = self.target_to_unit_map.value_iter();
	while (try unit = it3.next())
	{
		for (usz s = 0; s < unit.source_cores.len; s++)
		{
			tensor_core::TensorCore* src_tc = unit.source_cores[s];
			if (src_tc.flags.persistence == LEAF)
			{
				self.leaf_registry.register_leaf((uptr)unit.source_cores[s]);
			}
		}
	}

	// Build bit vectors via DFS propagation.
	// Each unit's leaf_bits = union of its direct leaf sources + union of its source units' leaf_bits.
	// We use a recursive helper with memoization (leaf_bits.len > 0 means done).
	swisstable::SwissTableValueIterator {uptr, ExecutionUnit*} it4 = self.target_to_unit_map.value_iter();
	while (try unit = it4.next())
	{
		self.build_leaf_bits(unit);
	}
}

// Leaf bit vector construction

/// Recursively build the leaf_bits for a unit.
/// Memoized: if leaf_bits.size > 0, return immediately.
fn void ExecutionSequence.build_leaf_bits(&self, ExecutionUnit* unit) @private
{
	// Already built (memoization)
	if (unit.leaf_bits.size > 0) { return; }

	usz words = self.leaf_registry.bitvec_words();
	if (words == 0) { return; }  // no leaves at all

	// Reserve and zero-fill
	for (usz i = 0; i < words; i++) { unit.leaf_bits.push(0); }

	// Set bits for direct leaf sources
	for (usz s = 0; s < unit.source_cores.len; s++)
	{
		tensor_core::TensorCore* src_tc = unit.source_cores[s];
		if (src_tc.flags.persistence == LEAF)
		{
			if (try idx = self.leaf_registry.ptr_to_index.get((uptr)unit.source_cores[s]))
			{
				*(&unit.leaf_bits[idx / 64]) |= (ulong)1 << (idx % 64);
			}
		}
	}

	// Union in transitive leaf deps from source units
	for (usz s = 0; s < unit.direct_sources.len; s++)
	{
		ExecutionUnit* src_unit = unit.direct_sources[s];
		self.build_leaf_bits(src_unit);  // recurse (memoized)
		if (src_unit.leaf_bits.size > 0)
		{
			usz src_words = src_unit.leaf_bits.size;
			usz union_words = words < src_words ? words : src_words;
			for (usz w = 0; w < union_words; w++)
			{
				*(&unit.leaf_bits[w]) |= src_unit.leaf_bits[w];
			}
		}
	}
}

// Leaf mutation

/// Called when a leaf tensor is mutated (load/fill/bind).
/// Increments the leaf's mutation counter in the registry.
fn void ExecutionSequence.notify_leaf_mutation(&self, uptr leaf_ptr)
{
	self.leaf_registry.mutate(leaf_ptr);
}

// Execute For Target — DFS post-order execution

<*
 Execute only the units needed to produce a specific tensor.
 Uses two-phase DFS:
 1. compute_ref_counts — DFS from target, counts incoming edges.
 2. dfs_execute — DFS post-order execution with fingerprint-based skip.
*>
fn void ExecutionSequence.execute_for_target(&self, TensorCorePtr core)
{
	ExecutionUnit*? unit_opt = self.target_to_unit_map.get((uptr)core);
	if (catch unit_opt) { return; }
	ExecutionUnit* root = (ExecutionUnit*)unit_opt;

	// Reset ref counts for all units
	swisstable::SwissTableValueIterator {uptr, ExecutionUnit*} it = self.target_to_unit_map.value_iter();
	while (try unit = it.next())
	{
		unit.ref_count = { .count = 0, .visited = false };
	}

	// Phase 1: Compute reference counts via DFS
	self.compute_ref_counts(root);

	// Phase 2: DFS post-order execution with fingerprint-based skip
	self.dfs_execute(root);
}

// Phase 1: DFS ref count computation

fn void ExecutionSequence.compute_ref_counts(&self, ExecutionUnit* unit) @private
{
	// Already traversed this subtree
	if (unit.ref_count.visited) { return; }

	tensor_core::TensorCore* tc_unit = unit.target;

	// Fingerprint frontier: has data, has executed before, and fingerprint matches → fresh, stop
	if (tc_unit.has_data() && unit.has_executed && unit.leaf_bits.size > 0)
	{
		ulong current_fp = compute_fingerprint(unit.leaf_bits.array_view(), &self.leaf_registry);
		if (current_fp == unit.fingerprint) { return; }
	}

	// Persistence frontiers
	Persistence p = tc_unit.flags.persistence;
	// Leaf: external input, always valid — stop
	if (p == LEAF) { return; }
	// Detached: acts as a leaf boundary — stop recursion
	if (!tc_unit.flags.attached) { return; }

	// Offload frontier: tensor will be restored, not recomputed — stop recursion
	if (tc_unit.flags.offload && !tc_unit.has_data() && self.offload_buffers.has_key((uptr)unit.target))
	{
		unit.ref_count.visited = true;
		return;
	}

	// Mark as visited
	unit.ref_count.visited = true;

	// Recurse on dependencies, incrementing their ref counts
	for (usz s = 0; s < unit.direct_sources.len; s++)
	{
		ExecutionUnit* source = unit.direct_sources[s];

		// Each edge increments the dependency's ref count
		source.ref_count.count += 1;

		self.compute_ref_counts(source);
	}
}

// Phase 2: DFS post-order execution with ref-count-based freeing

fn void ExecutionSequence.dfs_execute(
	&self,
	ExecutionUnit* unit,
) @private
{
	TensorCorePtr target = unit.target;
	tensor_core::TensorCore* tc = target;

	// Fingerprint frontier: has data, has executed, fingerprint matches → skip
	if (tc.has_data() && unit.has_executed && unit.leaf_bits.size > 0)
	{
		ulong current_fp = compute_fingerprint(unit.leaf_bits.array_view(), &self.leaf_registry);
		if (current_fp == unit.fingerprint)
		{
			stats_skipped++;
			return;
		}
	}

	// Persistence frontiers
	Persistence p = tc.flags.persistence;
	// Leaf: external input, always valid — skip
	if (p == LEAF)
	{
		stats_skipped++;
		return;
	}
	// Detached: acts as a leaf boundary — skip (must already have data)
	if (!tc.flags.attached)
	{
		stats_skipped++;
		return;
	}

	// Offload restore: if this tensor was offloaded, restore from CPU and skip re-execution
	if (tc.flags.offload && !tc.has_data() && self.offload_buffers.has_key((uptr)target))
	{
		self.restore_tensor(target);
		events::fire(target, EventKind.RESTORE);
		// Update fingerprint — restored data is current
		if (unit.leaf_bits.size > 0)
		{
			unit.fingerprint = compute_fingerprint(unit.leaf_bits.array_view(), &self.leaf_registry);
		}
		unit.has_executed = true;
		return;
	}

	// Recurse into dependencies first (post-order)
	for (usz s = 0; s < unit.direct_sources.len; s++)
	{
		ExecutionUnit* source = unit.direct_sources[s];
		self.dfs_execute(source);
	}

	// Ensure all sources have data before execution.
	// Sources may have been freed by ref-count decrements in sibling paths.
	// Also auto-restore any sources that were offloaded to CPU.
	for (usz s = 0; s < unit.source_cores.len; s++)
	{
		TensorCorePtr src = unit.source_cores[s];
		tensor_core::TensorCore* src_tc = src;

		if (!src_tc.has_data() && self.offload_buffers.has_key((uptr)src))
		{
			self.restore_tensor(src);
			events::fire(src, EventKind.RESTORE);
		}

		if (try src_unit = self.target_to_unit_map.get((uptr)src))
		{
			if (!src_tc.has_data())
			{
				self.dfs_execute(src_unit);
			}
		}
	}

	// Allocate output memory if needed
	if (!tc.has_data())
	{
		tc.alloc_data();
	}

	Executable e = (Executable)unit.operation;
	e.execute(unit.source_cores, target);

	// Update fingerprint to current leaf state
	if (unit.leaf_bits.size > 0)
	{
		unit.fingerprint = compute_fingerprint(unit.leaf_bits.array_view(), &self.leaf_registry);
	}
	unit.has_executed = true;
	stats_executed++;

	// Fire event callback if registered
	events::fire(target, EventKind.EXECUTE);

	// Ref-count freeing: decrement deps, free when count hits 0
	for (usz s = 0; s < unit.direct_sources.len; s++)
	{
		ExecutionUnit* dep = unit.direct_sources[s];
		TensorCorePtr dep_target = dep.target;

		// Don't free leaf or stable tensors (unless offload-flagged)
		{
			tensor_core::TensorCore* tc_dep = dep_target;
			Persistence dp = tc_dep.flags.persistence;
			if (dp == LEAF) { continue; }
			if (dp == STABLE)
			{
				if (!tc_dep.flags.offload) { continue; }
				if (tc_dep.device.is_host()) { continue; }
			}
		}

		// Decrement ref count
		if (dep.ref_count.count > 0)
		{
			dep.ref_count.count -= 1;
			if (dep.ref_count.count == 0)
			{
				tensor_core::TensorCore* tc_freed = dep_target;
				if (tc_freed.has_data())
				{
					// Invariant: STABLE tensors must never be freed here
					assert(tc_freed.flags.persistence != STABLE,
						"BUG: freeing STABLE tensor via ref-count!");
					if (tc_freed.flags.offload)
					{
						self.offload_tensor(dep_target);
						events::fire(dep_target, EventKind.OFFLOAD);
					}
					else
					{
						tc_freed.release_data();
					}
					// Data freed — invalidate fingerprint so next visit recomputes
					dep.has_executed = false;
					stats_freed++;
				}
			}
		}
	}
}
