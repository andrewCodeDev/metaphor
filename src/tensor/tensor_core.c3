//! Tensor Core
//!
//! Actual tensor storage owned by the Graph. Contains shape, dtype, device
//! reference, and format-specific storage (dense or sparse). Has flags for
//! requires_grad, attached, alive, and persistence. Leaf detection is structural:
//! tensors with no entry in the symbolic sequence are leaves.
//!

module metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::tensor::dense_core;
import metaphor::tensor::sparse_core;
import metaphor::device::device_reference;
import metaphor::device::sync;
import metaphor::rtti;
import std::core::mem::allocator;

// Persistence — tensor lifetime level

<*
 Tensor persistence level — determines compilation and data lifetime behavior.
 Higher values = stronger persistence.

 Compilation:
 - NONE: can be fused into other kernels
 - UNSTABLE, STABLE, LEAF: compilation boundaries

 Data lifetime:
 - NONE: freed immediately after use
 - UNSTABLE: freed after execution tree completes
 - STABLE: preserved across execution boundaries
 - LEAF: user-managed, never freed by the system
*>
enum Persistence : char
{
	NONE,
	UNSTABLE,
	STABLE,
	LEAF,
}

fn bool Persistence.lt(&self, Persistence other) @inline
{
	return (char)*self < (char)other;
}

fn bool Persistence.lte(&self, Persistence other) @inline
{
	return (char)*self <= (char)other;
}

fn bool Persistence.gt(&self, Persistence other) @inline
{
	return (char)*self > (char)other;
}

fn bool Persistence.gte(&self, Persistence other) @inline
{
	return (char)*self >= (char)other;
}

<*
 Monotonic promote: only increases persistence, never demotes.
 Used by system internals (gradient inheritance, compilation boundaries,
 leaf creation, auto-promotion at execution time).
 @require floor != NONE : "promote: NONE is not a valid promotion target"
*>
fn void Persistence.promote(&self, Persistence floor)
{
	if (floor.gte(*self))
	{
		*self = floor;
	}
}

<*
 Direct assignment within user-controllable range [UNSTABLE, STABLE].
 Used by user-facing .stable() / .unstable() APIs.
 @require level == UNSTABLE || level == STABLE
     : "assign: can only assign UNSTABLE or STABLE"
*>
fn void Persistence.assign(&self, Persistence level)
{
	*self = level;
}

// TensorCoreFlags — packed lifecycle/autodiff flags

struct TensorCoreFlags
{
	/// Whether the tensor is still valid and usable.
	bool alive;

	/// Traversal control for backward pass.
	/// If unattached, reversal stops at this tensor.
	bool attached;

	/// True if this node is the child of a gradient-requiring operation.
	/// Prevents premature freeing before dependents collect.
	bool grad_operand;

	/// Whether backward should initialize a gradient.
	/// Use requires_grad() instead — accounts for runtime flag.
	bool _requires_grad;

	/// Tensor persistence level.
	Persistence persistence;

	/// When true, data is offloaded to CPU instead of discarded when freed.
	/// Set via Tensor.offload() — implies STABLE persistence.
	bool offload;
}

const TensorCoreFlags DEFAULT_FLAGS = {
	.alive = true,
	.attached = true,
	.grad_operand = false,
	._requires_grad = false,
	.persistence = NONE,
	.offload = false,
};

// CoreState — dense or sparse storage

enum CoreClass : char
{
	DENSE,
	SPARSE,
}

struct CoreState
{
	CoreClass class_;
	union
	{
		DenseCore dense;
		SparseCore sparse;
	}
}

fn CoreState core_state_dense(DenseCore d)
{
	CoreState s;
	s.class_ = CoreClass.DENSE;
	s.dense = d;
	return s;
}

fn CoreState core_state_sparse(SparseCore sp)
{
	CoreState s;
	s.class_ = CoreClass.SPARSE;
	s.sparse = sp;
	return s;
}

// TensorCore

struct TensorCore
{
	/// Owning graph (opaque until Graph is fully ported).
	void* graph;

	/// Device reference for dispatch.
	DeviceReference device;

	/// Shared flags for lifecycle and autodiff.
	TensorCoreFlags flags;

	/// Optional label for debugging/lookup.
	char[] label;

	/// Synchronization primitive for pending operations.
	SyncPrimitive sync;

	/// Number of operations consuming this tensor as input.
	uint consumers;

	/// Version counter for inplace operation ordering.
	uint version;

	/// Creation context watermark (even=forward, odd=backward).
	uint creation_context;

	/// Format-specific storage.
	CoreState state;
}

// Debug counters
ulong tc_alloc_count = 0;
ulong tc_alloc_bytes = 0;

fn void reset_tc_alloc_stats()
{
	tc_alloc_count = 0;
	tc_alloc_bytes = 0;
}

fn void TensorCore.increment_version(&self)
{
	self.version++;
}

/// Access shape through DenseCore. Asserts DENSE state.
fn Shape* TensorCore.shape(&self)
{
	assert(self.state.class_ == CoreClass.DENSE, "TensorCore.shape() requires DENSE state");
	return &self.state.dense.shape;
}

/// Access dtype through DenseCore. Asserts DENSE state.
fn rtti::Datatype TensorCore.dtype(&self)
{
	assert(self.state.class_ == CoreClass.DENSE, "TensorCore.dtype() requires DENSE state");
	return self.state.dense.datatype();
}

/// Access dtype — panics for non-DENSE tensors.
fn rtti::Datatype TensorCore.dtype_raw(&self)
{
	if (self.state.class_ == CoreClass.DENSE) return self.state.dense.datatype();
	unreachable("dtype_raw called on non-DENSE tensor");
}

/// Get tensor rank (number of dimensions).
fn usz TensorCore.rank(&self)
{
	if (self.state.class_ == CoreClass.DENSE) return self.state.dense.shape.len;
	unreachable("rank called on non-DENSE tensor");
}

fn ZString TensorCore.label_or_null(&self)
{
	if (self.label.len > 0) { return (ZString)self.label.ptr; }
	return null;
}

fn bool TensorCore.has_data(&self)
{
	switch (self.state.class_)
	{
		case DENSE:  return self.state.dense.data != null;
		case SPARSE: return self.state.sparse.data.len > 0;
	}
}

/// Return the raw data bytes for this tensor core.
fn char[] TensorCore.data_bytes(&self)
{
	switch (self.state.class_)
	{
		case DENSE:
			assert(self.state.dense.data != null, "data_bytes: dense tensor has no data");
			return self.state.dense.data[0..self.state.dense.byte_size - 1];
		case SPARSE:
			assert(self.state.sparse.data.len > 0, "data_bytes: sparse tensor has no data");
			return self.state.sparse.data;
	}
}

/// Allocate data buffer using device cache.
fn void TensorCore.alloc_data(&self)
{
	if (self.has_data()) { return; }

	switch (self.state.class_)
	{
		case DENSE:
		{
			rtti::Datatype dt = self.dtype();
			usz n = self.shape().size();
			usz bytes = n * dt.size_of();
			tc_alloc_count++;
			tc_alloc_bytes += (ulong)bytes;

			char[] buf = self.device.mem_alloc(dt, n);
			self.state.dense.data = buf.ptr;
			self.state.dense.byte_size = (ulong)buf.len;
		}
		case SPARSE:
		{
			rtti::Datatype dt = self.dtype();
			usz bytes = self.state.sparse.nnz * dt.size_of();
			tc_alloc_count++;
			tc_alloc_bytes += (ulong)bytes;

			self.state.sparse.data = self.device.mem_alloc(dt, self.state.sparse.nnz);
		}
	}
}

/// Free tensor data to the device memory pool (sole ownership only).
/// For shared data (reshape), use release_data() instead.
fn void TensorCore.free_data(&self)
{
	if (!self.has_data()) { return; }

	self.dealloc_data();
	self.clear_data_pointers();
}

/// Actually free device memory. Dispatches to core specialization.
fn void TensorCore.dealloc_data(&self) @private
{
	switch (self.state.class_)
	{
		case DENSE:
			if (self.state.dense.data != null)
			{
				char[] slice = ((char*)self.state.dense.data)[:self.state.dense.byte_size];
				self.device.mem_free(slice);
			}
		case SPARSE:
			if (self.state.sparse.data.len > 0)
			{
				self.device.mem_free(self.state.sparse.data);
			}
	}
}

/// Null out data pointers without freeing. Dispatches to core specialization.
fn void TensorCore.clear_data_pointers(&self)
{
	switch (self.state.class_)
	{
		case DENSE:
			self.state.dense.data = null;
			self.state.dense.byte_size = 0;
		case SPARSE:
			self.state.sparse.data = {};
	}
}

/// Copy data pointer from this tensor to target (no ownership transfer).
/// Used by Graph.share_tensor_data() after refcount setup.
fn void TensorCore.copy_data_ptr(&self, TensorCore* target)
{
	switch (self.state.class_)
	{
		case DENSE:
			assert(target.state.class_ == CoreClass.DENSE);
			target.state.dense.data = self.state.dense.data;
			target.state.dense.byte_size = self.state.dense.byte_size;
		case SPARSE:
			assert(target.state.class_ == CoreClass.SPARSE);
			target.state.sparse.data = self.state.sparse.data;
	}
}

/// Zero-fill the tensor data buffer.
/// Dispatches through the device interface so it works on GPU memory too.
fn void TensorCore.fill_zero(&self)
{
	switch (self.state.class_)
	{
		case DENSE:
			if (self.state.dense.data != null)
			{
				char[] bytes = ((char*)self.state.dense.data)[:self.state.dense.byte_size];
				self.device.mem_fill(self.dtype(), bytes, 0.0);
			}
		case SPARSE:
			// Sparse data is always host-side index data
			for (usz i = 0; i < self.state.sparse.data.len; i++)
			{
				self.state.sparse.data[i] = 0;
			}
	}
}

/// Fill all elements of a dense tensor with a scalar float value.
/// Dispatches through the device interface so it works on GPU memory too.
fn void TensorCore.fill(&self, float value)
{
	assert(self.state.class_ == CoreClass.DENSE, "TensorCore.fill() requires DENSE state");
	if (self.state.dense.data == null) return;

	char[] bytes = ((char*)self.state.dense.data)[:self.state.dense.byte_size];
	self.device.mem_fill(self.dtype(), bytes, (double)value);
}

/// Release tensor core state without freeing device memory.
/// Device memory is arena-managed. Only frees allocator-managed metadata.
fn void TensorCore.deinit(&self, Allocator alloc)
{
	if (self.label.len > 0)
	{
		allocator::free(alloc, self.label);
		self.label = {};
	}

	switch (self.state.class_)
	{
		case DENSE:
			self.state.dense.deinit();
		case SPARSE:
			self.state.sparse.deinit(alloc);
	}
}

/// Returns true if this tensor requires its own execution unit.
fn bool TensorCore.is_compilation_boundary(&self)
{
	return self.flags.persistence.gte(UNSTABLE);
}

/// Returns true if this tensor needs compilation (not a leaf).
fn bool TensorCore.needs_compilation(&self)
{
	return self.flags.persistence.lt(LEAF);
}
