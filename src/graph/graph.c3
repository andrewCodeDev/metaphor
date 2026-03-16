//! Graph
//!
//! Central coordinator that owns all TensorCores, tracks operations via
//! symbolic chain maps, and manages gradients. Single source of truth for
//! the computation. Operations build the graph lazily; collect() triggers
//! compilation and execution.
//!
//! Uses a module-level global graph pointer. All tensor operations implicitly
//! use the current graph. Use `@subgraph()` to temporarily swap in a clean,
//! isolated graph for a scoped block (like `@pool` for allocators).

module metaphor::graph;
import metaphor::mem::stable_free_list;
import metaphor::utils::hash_map;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::dense_core;
import metaphor::tensor::shape;
import metaphor::device::device_reference;
import metaphor::rtti;
import metaphor::graph::execution_sequence;
import metaphor::graph::op_cache;
import metaphor::device::common::op_payload;
import metaphor::tensor::tensor_ops;
import metaphor::logging;
import std::core::mem::allocator;
import metaphor::settings;

import std::io;
import std::sort;
import metaphor::utils::swisstable;

alias DataRefcountMap = hash_map::HashMap {uptr, uint};

// Type aliases (internal — use fully-qualified names for tensor/graph types
// to avoid namespace collisions when other modules import graph)

alias SymbolicNodePool = stable_free_list::StableFreeList {SymbolicNode};
alias TensorCorePool = stable_free_list::StableFreeList {tensor_core::TensorCore};
alias GradientCoreMap = hash_map::HashMap {uptr, tensor_core::TensorCore*};

// Flags

struct Flags
{
	bool eager_teardown;
	bool track_ops;
	bool allow_recompute;
	bool symbolic_recompute;
	bool op_caching;
	tensor_core::Persistence default_persistence;
}

const Flags DEFAULT_FLAGS = {
	.eager_teardown = true,
	.track_ops = true,
	.allow_recompute = true,
	.symbolic_recompute = true,
	.op_caching = false,
	.default_persistence = tensor_core::Persistence.UNSTABLE,
};

struct NormalizerFlags
{
	bool chain_fusion;
	bool algebraic_simplification;
}

const NormalizerFlags DEFAULT_NORMALIZER_FLAGS = {
	.chain_fusion = true,
	.algebraic_simplification = true,
};

// Counts

struct Counts
{
	uint tensors;
}

// Global Graph State

/// Module-level allocator for the global graph and its internals.
/// Uses libc directly so test tracking allocators don't see graph
/// allocations as leaks (the graph outlives every individual test).
allocator::LibcAllocator graph_alloc;

/// Thread-local pointer to the current active graph.
/// All tensor operations implicitly use this graph.
tlocal Graph* current_graph;

/// Thread-local root graph pointer, set once by init(). Never changes.
/// Used by export() to move tensors out of subgraphs.
tlocal Graph* root_graph;

/// Initialize the global graph with explicit flags.
/// If not called, get() will lazily initialize with defaults.
fn void init(Flags flags = DEFAULT_FLAGS)
{
	if (current_graph != null) return;  // already initialized
	current_graph = allocator::new(&graph_alloc, Graph);
	current_graph.init(&graph_alloc, flags);
	root_graph = current_graph;
}

/// Tear down the global graph and release all resources.
fn void deinit()
{
	if (root_graph == null) return;  // nothing to tear down
	root_graph.deinit();
	allocator::free(&graph_alloc, root_graph);
	root_graph = null;
	current_graph = null;
}

/// Get the current active graph. Lazily initializes on first access.
fn Graph* get()
{
	if (current_graph == null)
	{
		init();
	}
	return current_graph;
}

enum DefaultPersistence
{
	UNSTABLE,
	STABLE,
}

/// Set the default persistence for the current graph.
/// Affects auto-promotion of tensors entering the execution sequence.
/// Can be called at any time — only affects future tensor promotions.
/// Only UNSTABLE and STABLE are valid defaults — NONE and LEAF are
/// structural properties set per-tensor, not graph-wide defaults.
fn void set_default_persistence(DefaultPersistence p)
{
	switch (p)
	{
		case UNSTABLE: get().flags.default_persistence = tensor_core::Persistence.UNSTABLE;
		case STABLE:   get().flags.default_persistence = tensor_core::Persistence.STABLE;
	}
}

/// Create or swap in a subgraph for the duration of the body.
///
/// With no graph pointer (default): creates a temporary graph, destroys on exit.
/// With a user-owned graph pointer: swaps it in, does NOT destroy on exit.
///
/// Usage (temporary):
///   graph::@subgraph()
///   {
///       // all tensor ops here use a fresh subgraph
///   };
///
/// Usage (persistent, user-owned):
///   Graph my_graph;
///   my_graph.init(&alloc, graph::DEFAULT_FLAGS);
///   graph::@subgraph(&my_graph)
///   {
///       // ops use my_graph, which persists after scope exit
///   };
macro void @subgraph(Flags flags = DEFAULT_FLAGS, Graph* dst = null; @body())
{
	Graph* parent = current_graph;
	Graph sub;
	Graph* active;
	if (dst != null)
	{
		active = dst;
	}
	else
	{
		Flags sub_flags = flags;
		if (parent != null)
		{
			sub_flags.default_persistence = parent.flags.default_persistence;
		}
		sub.init(&graph_alloc, sub_flags);
		active = &sub;
	}
	current_graph = active;
	defer
	{
		if (dst == null) { sub.deinit(); }
		current_graph = parent;
	}
	@body();
}

/// Export a tensor from the current (sub)graph to the root global graph.
/// Steals the data pointer — the source tensor becomes empty.
/// Returns a new leaf tensor on the global graph.
///
/// Usage inside @subgraph():
///   Tensor result;
///   graph::@subgraph()
///   {
///       Tensor tmp = tensor::empty(.F32, dev, &dims);
///       // ... compute ...
///       result = graph::export(tmp);
///   };
///   // result is a leaf on the global graph
fn tensor::Tensor export(tensor::Tensor t)
{
	assert(root_graph != null, "No global graph. Call graph::init() first.");
	assert(current_graph != root_graph, "Already on the global graph, nothing to export.");

	tensor_core::TensorCore* src = t.core();

	// Create a new core on the root graph with same properties
	tensor_core::TensorCore* dst = root_graph.create_tensor_core(
		src.shape().slice(),
		src.dtype(),
		src.device,
	);

	// Steal the data — move pointer, zero out source so subgraph teardown
	// won't free it.
	dst.state = src.state;
	switch (src.state.class_)
	{
		case DENSE:
			src.state.dense.data = null;
			src.state.dense.byte_size = 0;
		case SPARSE: src.state.sparse.data = {};
	}

	// Mark as leaf on the global graph
	dst.flags.persistence.promote(tensor_core::Persistence.LEAF);

	return tensor::tensor_from_core(dst);
}

// Graph

struct Graph
{
	Allocator allocator;

	// -- symbolic node internals --
	SymbolicNodePool symbolic_node_pool;
	SymbolicChainMap symbolic_chain_map;

	// -- gradient internals --
	GradientCoreMap gradient_core_map;

	// -- tensor internals --
	TensorCorePool tensor_pool;

	// -- data sharing refcounts (data_ptr → refcount) --
	DataRefcountMap data_refcounts;

	// -- execution internals --
	execution_sequence::ExecutionSequence exec_seq;

	// -- CSE cache --
	op_cache::OpCache cse_cache;

	// -- graph meta-data --
	Direction direction;
	Counts counts;

	/// Creation context watermark for forward vs backward tensors.
	uint creation_context;


	Flags flags;
	NormalizerFlags normalizer_flags;
	usz max_collect_depth;
	uint next_node_seq;
}

// Lifecycle

fn Graph* Graph.init(&self, Allocator alloc, Flags flags = DEFAULT_FLAGS)
{
	self.allocator = alloc;
	self.flags = flags;
	self.normalizer_flags = DEFAULT_NORMALIZER_FLAGS;
	self.max_collect_depth = 16;
	self.symbolic_node_pool.init(alloc);
	self.symbolic_chain_map.init(alloc);
	self.tensor_pool.init(alloc);
	self.data_refcounts.init(alloc);
	self.gradient_core_map.init(alloc);
	self.exec_seq.init(alloc);
	self.cse_cache.init(alloc);

	return self;
}

fn void Graph.deinit(&self)
{
	// Free GPU data for all execution sequence targets (STABLE/UNSTABLE computed tensors).
	// Must happen before exec_seq.free() destroys units and before data_refcounts.free().
	// LEAFs are never exec targets (register_op asserts this), so weights/caches are safe.
	// Exported tensors have data=null (export steals the pointer), so release_data() no-ops.
	{
		swisstable::SwissTableIterator {uptr, execution_sequence::ExecutionUnit*} it =
			self.exec_seq.target_to_unit_map.iter();
		while (try entry = it.next())
		{
			tensor_core::TensorCore* tc = (tensor_core::TensorCore*)entry.key;
			tc.release_data();
		}
	}

	self.cse_cache.free();
	self.exec_seq.free();
	self.symbolic_chain_map.free();
	self.symbolic_node_pool.free();
	self.gradient_core_map.free();
	self.data_refcounts.free();
	self.tensor_pool.free();
}

/// Release GPU data for all non-LEAF execution targets and reset their
/// has_executed flags.  Forces the next forward pass to allocate fresh
/// buffers and re-execute every op — useful for diagnosing stale-data bugs.
fn void Graph.release_computed_data(&self)
{
	swisstable::SwissTableIterator {uptr, execution_sequence::ExecutionUnit*} it =
		self.exec_seq.target_to_unit_map.iter();
	while (try entry = it.next())
	{
		tensor_core::TensorCore* tc = (tensor_core::TensorCore*)entry.key;
		if (tc.flags.persistence == tensor_core::Persistence.LEAF) { continue; }
		if (tc.has_data())
		{
			tc.release_data();
		}
		entry.value.has_executed = false;
	}
}

fn void Graph.reset(&self)
{
	self.symbolic_chain_map.clear_retaining_capacity();
	self.symbolic_node_pool.reset();
	self.exec_seq.clear_retaining_capacity();
	self.cse_cache.clear();
}

// Data Sharing Refcounts

/// Share data from this tensor core to target (for reshape).
/// Manages refcounting via the graph's data pointer map.
/// Defined in graph module because it needs access to Graph's refcount map.
fn void tensor_core::TensorCore.share_data(&self, tensor_core::TensorCore* tgt)
{
	assert(self.has_data(), "share_data: source has no data");
	Graph* g = (Graph*)self.graph;

	// If target already has data (e.g., pre-allocated by executor), free it
	if (tgt.has_data())
	{
		tgt.release_data();
	}

	uptr key = (uptr)self.data_bytes().ptr;

	// Promote source to tracked refcount if not already in map
	if (try rc_ptr = g.data_refcounts.get_ref(key))
	{
		*rc_ptr += 1;
	}
	else
	{
		// First share — source counts as 1, target makes 2
		g.data_refcounts.set(key, 2);
	}

	// Copy the data pointer
	self.copy_data_ptr(tgt);
}

/// Free tensor data with refcount awareness.
/// If shared (in refcount map), decrements count and clears our pointer.
/// When count reaches 1, removes from map — the remaining holder will
/// see a cache miss and free normally as sole owner.
/// Defined in graph module because it needs access to Graph's refcount map.
fn void tensor_core::TensorCore.release_data(&self)
{
	if (!self.has_data()) { return; }

	Graph* g = (Graph*)self.graph;
	uptr key = (uptr)self.data_bytes().ptr;

	if (try rc_ptr = g.data_refcounts.get_ref(key))
	{
		assert(*rc_ptr > 1, "data_refcount should have been removed at 1");
		*rc_ptr -= 1;
		if (*rc_ptr == 1)
		{
			// Down to one remaining holder — remove from map
			g.data_refcounts.remove(key)!!;
		}
		// Clear our pointer without deallocating
		self.clear_data_pointers();
	}
	else
	{
		// Not in map — sole owner, free directly
		self.free_data();
	}
}

// Gradient / Backward

<*
 Get-or-create a gradient core for the given tensor core.
 Reuses existing gradient if one is already mapped.
*>
fn tensor_core::TensorCore* Graph.ensure_gradient_core(&self, tensor_core::TensorCore* core)
{
	uptr key = (uptr)core;
	if (try existing = self.gradient_core_map.get(key))
	{
		return existing;
	}
	// Create new gradient core with same shape/dtype/device
	tensor_core::TensorCore* grad = self.create_tensor_core(
		core.shape().slice(), core.dtype(), core.device);
	// Inherit persistence from parent (capped at STABLE — gradients are never LEAF).
	// Creates fusion boundaries at checkpoint locations during backward pass,
	// mirroring forward-pass execution and persistence boundaries.
	if (core.flags.persistence.gte(tensor_core::Persistence.STABLE))
	{
		grad.flags.persistence.promote(tensor_core::Persistence.STABLE);
	}
	else if (core.flags.persistence.gte(tensor_core::Persistence.UNSTABLE))
	{
		grad.flags.persistence.promote(tensor_core::Persistence.UNSTABLE);
	}
	if (settings::grad_uses_parent_flags)
	{
		grad.flags.attached = core.flags.attached;
	}
	// Inherit label from parent with prime suffix
	if (core.label.len > 0)
	{
		usz new_len = core.label.len + 1;
		char[] grad_label = allocator::alloc_array(self.allocator, char, new_len);
		grad_label[:core.label.len] = core.label[..];
		grad_label[core.label.len] = '\'';
		grad.label = grad_label;
	}
	// Leaf tensors get eagerly allocated + zeroed gradients
	if (core.flags.persistence == tensor_core::Persistence.LEAF)
	{
		grad.alloc_data();
		grad.fill(0.0);
	}
	self.gradient_core_map.set(key, grad);
	return grad;
}

<*
 Look up the gradient core for a tensor. Returns null if none exists.
*>
fn tensor_core::TensorCore* Graph.get_gradient_core(&self, tensor_core::TensorCore* core)
{
	if (try val = self.gradient_core_map.get((uptr)core))
	{
		return val;
	}
	return null;
}

<*
 Directly assign a gradient core for a tensor, propagating persistence.
*>
fn void Graph.set_gradient_core(&self, tensor_core::TensorCore* core, tensor_core::TensorCore* grad_core)
{
	if (core.flags.persistence.gte(tensor_core::Persistence.STABLE))
	{
		grad_core.flags.persistence.promote(tensor_core::Persistence.STABLE);
	}
	self.gradient_core_map.set((uptr)core, grad_core);
}

<*
 Backward pass: propagate gradients from loss through the computation graph.
 Collects all symbolic nodes, sorts by seq descending (reverse topological),
 then dispatches symbolic_backward for each child that requires grad.
*>
fn void Graph.backward(&self, tensor_core::TensorCore* loss)
{
	logging::@debug(GRAPH, "loss_core=%p shape_len=%d dtype=%d",
		loss, (int)loss.shape().len, (int)loss.dtype());

	// Bump creation_context to odd (backward context)
	self.creation_context += 1;
	defer self.creation_context += 1;
	self.direction = Direction.BWD;
	defer self.direction = Direction.FWD;

	// Collect all nodes into a temporary buffer
	uint num_nodes = self.symbolic_chain_map.num_nodes();
	if (num_nodes == 0) return;

	// Stack-allocate for small graphs, heap for large
	const usz STACK_LIMIT = 256;
	SymbolicNode*[STACK_LIMIT] stack_buf;
	SymbolicNode*[] buf;
	bool heap_allocated = false;

	if (num_nodes <= STACK_LIMIT)
	{
		buf = stack_buf[:num_nodes];
	}
	else
	{
		buf = ((SymbolicNode**)mem::malloc(num_nodes * SymbolicNode*.sizeof))[:num_nodes];
		heap_allocated = true;
	}
	defer if (heap_allocated) { mem::free(buf.ptr); };

	usz collected = self.symbolic_chain_map.collect_all_nodes(buf);

	// Sort by seq descending (reverse topological order)
	sort_nodes_descending(buf[:collected]);

	logging::@debug(GRAPH, "entering backward: num_nodes=%d collected=%d", (int)num_nodes, (int)collected);

	// Diagnostic counters
	usz bwd_visited = 0;
	usz bwd_no_grad = 0;
	usz bwd_grad_set = 0;
	usz bwd_failed = 0;
	usz bwd_detached = 0;

	// Process each node
	for (usz n = 0; n < collected; n++)
	{
		SymbolicNode* node = buf[n];

		// Skip detached tensors
		if (!node.result.attached()) { bwd_detached++; continue; }

		tensor::Tensor[] children = node.children.slice();

		// For each child that requires grad
		for (usz i = 0; i < children.len; i++)
		{
			if (!children[i].requires_grad())
			{
				bwd_no_grad++;
				continue;
			}

			bwd_visited++;
			tensor_core::TensorCore* child_core = children[i].core();

			logging::@debug(GRAPH, "node seq=%d result=%p child[%d]=%p",
				(int)node.seq, node.result.core(), (int)i, child_core);

			// Compute symbolic gradient
			tensor::Tensor grad;
			if (try g = node.payload.symbolic_backward(i, children, node.result))
			{
				grad = g;
				bwd_grad_set++;
				logging::@debug(GRAPH, "grad=%p shape_len=%d",
					grad.core(), (int)grad.core().shape().len);
			}
			else
			{
				bwd_failed++;
				logging::@debug(GRAPH, "bwd FAILED: seq=%d tag=%d child_idx=%d",
					(int)node.seq, (int)node.payload.tag(), (int)i);
				continue;
			}

			// Check if gradient already exists for child
			tensor_core::TensorCore* existing_grad = self.get_gradient_core(child_core);

			if (existing_grad == null)
			{
				// Case 1: No existing gradient — set directly
				self.set_gradient_core(child_core, grad.core());
				if (child_core.flags.persistence.gte(tensor_core::Persistence.STABLE))
				{
					grad.core().flags.persistence.promote(tensor_core::Persistence.STABLE);
				}
			}
			else if (existing_grad == grad.core())
			{
				// Same tensor — skip
				continue;
			}
			else if (!self.symbolic_chain_map.contains(existing_grad))
			{
				// Case 2: Gradient exists but has no computation chain.
				// Created by x.grad() but never computed. Reassign chain.
				tensor_core::TensorCore* defunct = grad.core();
				self.symbolic_chain_map.reassign_chain(defunct, existing_grad);
				if (child_core.flags.persistence.gte(tensor_core::Persistence.STABLE))
				{
					existing_grad.flags.persistence.promote(tensor_core::Persistence.STABLE);
				}
			}
			else
			{
				// Case 3: Gradient exists and has been computed — accumulate.
				// Create a new sum tensor rather than modifying the existing
				// gradient's chain (which would break boundary ops like
				// expand/reduce that the CAS mapper compiles independently).
				tensor::Tensor child_grad_tensor = tensor::tensor_from_core(existing_grad);
				tensor::Tensor accumulated = child_grad_tensor.add(grad)!!;
				self.set_gradient_core(child_core, accumulated.core());
				// Copy flags from existing gradient so the accumulated tensor
				// inherits boundary properties (persistence, attached).
				accumulated.core().flags.persistence = existing_grad.flags.persistence;
				accumulated.core().flags.attached = existing_grad.flags.attached;
			}
		}
	}

	logging::@debug(GRAPH, "bwd visited=%d detached=%d no_grad=%d grad_set=%d failed=%d",
		(int)bwd_visited, (int)bwd_detached, (int)bwd_no_grad, (int)bwd_grad_set, (int)bwd_failed);
}

<*
 Sort symbolic nodes by seq descending.
*>
fn void sort_nodes_descending(SymbolicNode*[] nodes)
{
	sort::quicksort(nodes, fn int(SymbolicNode* a, SymbolicNode* b) {
		if (a.seq > b.seq) return -1;
		if (a.seq < b.seq) return 1;
		return 0;
	});
}

// Tensor Core Creation

struct CreateOptions
{
	bool requires_grad;
	char[] label;
}

<*
 Creates a new TensorCore managed by this graph.
 Allocates from the tensor pool but does NOT allocate device data.
 Caller must call alloc_data() if data is needed.
*>
fn tensor_core::TensorCore* Graph.create_tensor_core(&self,
	ulong[] shape_slice,
	rtti::Datatype dtype,
	device_reference::DeviceReference device,
	CreateOptions opts = {})
{
	// Normalize all-ones shapes to scalar (0-dim).
	// A shape like [1] or [1,1] has exactly 1 element — it's a scalar.
	// Canonicalizing here prevents every kernel from needing broadcast edge cases.
	bool all_ones = shape_slice.len > 0;
	for (usz i = 0; i < shape_slice.len; i++)
	{
		if (shape_slice[i] != 1) { all_ones = false; break; }
	}
	if (all_ones) { shape_slice = {}; }

	tensor_core::TensorCore* core = self.tensor_pool.create()!!;

	dense_core::DenseCore dc = dense_core::dense_core_empty(shape_slice, dtype, dense_core::DEFAULT_DENSE_CONFIG);

	*core = {
		.graph = self,
		.device = device,
		.flags = {
			.alive = true,
			.attached = true,
			.grad_operand = false,
			._requires_grad = opts.requires_grad,
			.persistence = tensor_core::Persistence.NONE,
		},
		.label = opts.label,
		.consumers = 0,
		.version = 0,
		.creation_context = self.creation_context,
		.state = tensor_core::core_state_dense(dc),
	};

	self.counts.tensors += 1;
	return core;
}

// Leaf Mutation

/// Notify the graph that a leaf tensor was mutated.
/// Increments the leaf's mutation counter in the execution sequence's registry.
fn void Graph.notify_leaf_mutation(&self, tensor_core::TensorCore* leaf)
{
	self.exec_seq.notify_leaf_mutation((uptr)leaf);
}

/// Notify that a leaf tensor was mutated outside the normal tensor API.
/// Use this when writing to a leaf via raw mem_copy or other low-level ops.
/// The tensor must be a LEAF — only leaves are tracked as mutation points.
fn void notify_mutation(tensor::Tensor t)
{
	tensor_core::TensorCore* c = t.core();
	assert(c.flags.persistence == LEAF, "notify_mutation: only LEAF tensors are mutation points");
	Graph* g = (Graph*)c.graph;
	g.notify_leaf_mutation(c);
}

// Operation Registration

<*
 Registers an operation as a SymbolicNode in the graph.
 Creates the node, links it to the result tensor's compute chain,
 and propagates gradient requirements to children.
*>
fn void Graph.register_op(&self,
	op_payload::OpPayload payload,
	tensor::Tensor[] args,
	tensor::Tensor result)
{
	if (!self.flags.track_ops || !settings::track_ops) return;

	// Leaf tensors must never have operations registered on them.
	assert(!result.core().flags.persistence.gte(tensor_core::Persistence.LEAF),
		"Cannot assign operations to a leaf tensor");

	logging::@debug(GRAPH, "result=%p num_children=%d op_tag=%d",
		result.core(), (int)args.len, (int)payload.tag());

	// Create symbolic node
	SymbolicNode* node = self.symbolic_node_pool.create()!!;
	*node = {
		.children = children_from_slice(args),
		.payload = payload,
		.result = result,
		.seq = self.next_node_seq,
	};
	self.next_node_seq += 1;

	// Add to chain map
	self.symbolic_chain_map.put_append(node);

	// Track consumer counts
	for (usz i = 0; i < args.len; i++)
	{
		args[i].core().consumers += 1;
	}

	// Propagate gradient requirements
	bool any_grad = false;
	for (usz i = 0; i < args.len; i++)
	{
		if (args[i].requires_grad())
		{
			any_grad = true;
			break;
		}
	}
	result.core().flags._requires_grad = any_grad;

	if (any_grad)
	{
		for (usz i = 0; i < args.len; i++)
		{
			args[i].core().flags.grad_operand = true;
		}
	}
}
