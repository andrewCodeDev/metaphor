//! Tensor Handle
//!
//! Lightweight handle wrapping a TensorCore pointer. Every tensor belongs
//! to one Graph. Multiple handles can reference the same core.
//!
//! for railroad-style error propagation, C3 uses native optionals:
//!
//!   C3:   a.add(b)!.mul(c)!.neg()!  // errors propagate via rethrow operator
//!
//! Factory/operation methods return Tensor? (optional). Callers chain with !
//! (rethrow) or !! (force-unwrap) to propagate faults.
//!

module metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::tensor::tensor_ops;
import metaphor::device::device_reference;
import metaphor::device::common::index_expr;
import metaphor::graph;
import metaphor::rtti;
import metaphor::device::common::operations;
import metaphor::tensor::display;
import metaphor::settings;
import metaphor::events;
import std::core::mem::allocator;
import std::io;

// Faults — tensor error conditions
//
//   error{ @"incompatible-shape", @"incompatible-dtype", ... }
//
// In C3, faults are used as excuses in optional return types.
// A method returning Tensor? can fail with any of these.

faultdef INCOMPATIBLE_SHAPE;
faultdef INCOMPATIBLE_DTYPE;
faultdef UNSUPPORTED_SHAPE;
faultdef AMBIGUOUS_GRAPH;
faultdef AMBIGUOUS_DEVICE;
faultdef WRONG_DIRECTION;
faultdef DEPENDENCY_LOOP;
faultdef ALLOCATION_FAILURE;
faultdef COMPILATION_FAILURE;
faultdef EXECUTION_FAILED;
faultdef INVALID_EXPR;
faultdef TOO_MANY_INPUTS;
faultdef DEVICE_OOM;
faultdef UNINITIALIZED_LEAF;

// Tensor — handle to a TensorCore

struct Tensor
{
	TensorCore* _core;
}

/// A new tensor node has been added to the computation graph.
alias GraphTensor = Tensor;

/// The same tensor handle, returned for method chaining. No graph mutation.
alias Self = Tensor;

/// Get the underlying TensorCore.
fn TensorCore* Tensor.core(&self) @inline
{
	return self._core;
}

/// Wrap a TensorCore pointer in a Tensor handle.
fn Tensor tensor_from_core(TensorCore* c) @inline
{
	return { ._core = c };
}

/// Sentinel value for optional Tensor arguments.
/// Can be used as a default parameter value; check with `t._core != null`.
const Tensor NULL_TENSOR = { ._core = null };

// Factory Functions — create tensors on the current global graph

/// Creates an empty tensor with allocated (but uninitialized) data.
/// The tensor is marked as a leaf (user-provided input).
fn Tensor? empty(rtti::Datatype dtype, DeviceReference device, ulong[] shape_slice)
{
	graph::Graph* g = graph::get();
	TensorCore* tc = g.create_tensor_core(shape_slice, dtype, device);
	tc.alloc_data();
	events::fire(tc, events::EventKind.ALLOC);
	tc.flags.persistence.promote(tensor_core::Persistence.LEAF);
	return tensor_from_core(tc);
}

/// Creates a tensor filled with zeros.
fn Tensor? zeros(rtti::Datatype dtype, DeviceReference device, ulong[] shape_slice)
{
	Tensor t = empty(dtype, device, shape_slice)!;
	device.mem_fill(dtype, t.value_bytes(), 0.0);
	return t;
}

/// Creates a tensor filled with ones.
fn Tensor? ones(rtti::Datatype dtype, DeviceReference device, ulong[] shape_slice)
{
	Tensor t = empty(dtype, device, shape_slice)!;
	device.mem_fill(dtype, t.value_bytes(), 1.0);
	return t;
}

// Accessors

/// Data type of stored values.
fn rtti::Datatype Tensor.dtype(&self)
{
	return self._core.dtype();
}

/// Tensor shape.
fn Shape* Tensor.shape(&self)
{
	return self._core.shape();
}

/// Total number of elements.
fn usz Tensor.len(&self)
{
	return (usz)self._core.shape().size();
}

/// Device this tensor is on.
fn DeviceReference Tensor.device(&self)
{
	return self._core.device;
}

/// Owning graph.
fn graph::Graph* Tensor.graph(&self)
{
	return (graph::Graph*)self._core.graph;
}

/// Non-blocking check if computation has completed.
fn bool Tensor.is_ready(&self)
{
	if (&self._core.sync)
	{
		return self._core.sync.is_ready();
	}
	return true;
}

/// Check if tensor has allocated data.
fn bool Tensor.has_data(&self)
{
	return self._core.has_data();
}

/// Increment version counter (for inplace ops).
fn void Tensor.increment_version(&self)
{
	self._core.increment_version();
}

/// Get version counter.
fn uint Tensor.get_version(&self)
{
	return self._core.version;
}

/// Check if this tensor requires gradient computation.
fn bool Tensor.requires_grad(&self)
{
	return self._core.flags._requires_grad && settings::grad_enabled;
}

/// Enable gradient computation. Returns self for chaining.
fn Self Tensor.enable_grad(&self)
{
	self._core.flags._requires_grad = true;
	return *self;
}

/// Disable gradient computation. Returns self for chaining.
fn Self Tensor.disable_grad(&self)
{
	self._core.flags._requires_grad = false;
	return *self;
}

/// Fill tensor with random values from the given distribution.
/// Uses the global RNG (seed with sampling::seed()). Returns self for chaining.
fn Self Tensor.fill_random(&self,
	device_reference::RandType rand_type,
	device_reference::Scale scale = device_reference::DEFAULT_SCALE)
{
	char[] bytes = self.value_bytes();
	assert(bytes.len > 0, "fill_random: tensor data not allocated");
	self._core.device.mem_random(self.dtype(), bytes, rand_type, scale);
	return *self;
}

/// Returns true if this tensor is attached to its graph.
fn bool Tensor.attached(&self)
{
	return self._core.flags.attached;
}

/// Attach tensor to its graph for gradient tracking.
fn Self Tensor.attach(&self)
{
	self._core.flags.attached = true;
	return *self;
}
/// Detach tensor from the computation graph.
/// After detaching, backward() will not traverse through this tensor
/// and persistence is promoted to leaf.
fn Self Tensor.detach(&self)
{
	self._core.flags.attached = false;
	return *self;
}

/// Mark tensor as stable (compilation boundary, data preserved across executions).
fn Self Tensor.stable(&self)
{
	self._core.flags.persistence.assign(tensor_core::Persistence.STABLE);
	return *self;
}

/// Set a debug label on this tensor. Returns self for chaining.
fn Self Tensor.set_label(&self, char[] label)
{
	self._core.label = label;
	return *self;
}

/// Set a formatted debug label on this tensor. Returns self for chaining.
/// Usage: t.fmt_label("layer_%d_x", i)
fn Self Tensor.fmt_label(&self, String fmt, int arg)
{
	char[128] buf;
	char[] formatted = io::bprintf(&buf, fmt, arg)!!;
	// Copy to graph allocator so it lives as long as the graph
	// Allocate len+1 for null terminator so (ZString) casts work safely
	graph::Graph* g = (graph::Graph*)self._core.graph;
	char[] label = allocator::alloc_array(g.allocator, char, formatted.len + 1);
	label[:formatted.len] = formatted[..];
	label[formatted.len] = 0;
	self._core.label = label[:formatted.len];  // slice excludes null, but null is there in memory
	return *self;
}

/// Mark tensor as unstable (compilation boundary, data may be freed after execution).
fn Self Tensor.unstable(&self)
{
	self._core.flags.persistence.assign(tensor_core::Persistence.UNSTABLE);
	return *self;
}

// Factory Functions — additional constructors

/// Creates a tensor filled with a constant float value.
/// Dispatches through the device interface so it works on GPU memory too.
fn Tensor? constant(rtti::Datatype dtype, DeviceReference device, ulong[] shape_slice, float value)
{
	Tensor t = empty(dtype, device, shape_slice)!;
	t.fill(value);
	return t;
}

/// Creates a tensor from typed host data (slice, fixed array pointer, or scalar pointer).
/// Infers dtype from the element type at compile time.
macro Tensor? @from_data(DeviceReference device, ulong[] shape_slice, data)
{
	$if $defined(data.ptr):
		// Slice — has .ptr and .len
		const DT = rtti::from_type($typeof(data[0]));
		usz byte_len = data.len * $typeof(data[0]).sizeof;
		Tensor t = empty(DT, device, shape_slice)!;
		char[] dst = t.value_bytes();
		assert(byte_len <= dst.len, "from_data: source data too large");
		device.mem_transfer(((char*)data.ptr)[:byte_len], dst[:byte_len], device_reference::TransferDirection.HTOD);
		return t;
	$else
		// Pointer to fixed array
		const DT = rtti::from_type($typeof((*data)[0]));
		usz byte_len = (*data).len * $typeof((*data)[0]).sizeof;
		Tensor t = empty(DT, device, shape_slice)!;
		char[] dst = t.value_bytes();
		assert(byte_len <= dst.len, "from_data: source data too large");
		device.mem_transfer(((char*)data)[:byte_len], dst[:byte_len], device_reference::TransferDirection.HTOD);
		return t;
	$endif
}


/// Creates a tensor filled with random values from the given distribution.
/// Uses the global RNG (seed with sampling::seed()).
fn Tensor? random(rtti::Datatype dtype, DeviceReference device, ulong[] shape_slice,
	device_reference::RandType rand_type,
	device_reference::Scale scale = device_reference::DEFAULT_SCALE)
{
	Tensor t = empty(dtype, device, shape_slice)!;
	char[] bytes = t.value_bytes();
	assert(bytes.len > 0, "random: tensor data not allocated");
	device.mem_random(dtype, bytes, rand_type, scale);
	return t;
}

// Data Access

/// Returns raw data pointer, or null if no data allocated.
fn void* Tensor.data_ptr(&self)
{
	if (self._core.state.class_ == tensor_core::CoreClass.DENSE)
	{
		return self._core.state.dense.data;
	}
	return null;
}

/// Returns raw data as a byte slice, or empty if not allocated.
fn char[] Tensor.value_bytes(&self)
{
	if (self._core.state.class_ == tensor_core::CoreClass.DENSE)
	{
		void* p = self._core.state.dense.data;
		if (p == null) { return {}; }
		usz bytes = self.shape().size() * self.dtype().size_of();
		return ((char*)p)[:bytes];
	}
	return {};
}

/// Free tensor data to the device memory pool (refcount-aware).
fn void Tensor.free_data(&self)
{
	self._core.release_data();
	events::fire(self._core, events::EventKind.FREE);
}

// Tensor Operations — forwarding to tensor_ops module

/// Element-wise addition: a + b.
fn GraphTensor? Tensor.add(&self, Tensor other)
{
	return tensor_ops::add(*self, other);
}

/// Element-wise subtraction: a - b.
fn GraphTensor? Tensor.sub(&self, Tensor other)
{
	return tensor_ops::sub(*self, other);
}

/// Element-wise multiplication: a * b.
fn GraphTensor? Tensor.mul(&self, Tensor other)
{
	return tensor_ops::mul(*self, other);
}

/// Element-wise division: a / b.
fn GraphTensor? Tensor.div(&self, Tensor other)
{
	return tensor_ops::div(*self, other);
}

/// Negation: -a.
fn GraphTensor? Tensor.neg(&self) { return tensor_ops::neg(*self); }

/// Absolute value.
fn GraphTensor? Tensor.abs(&self) { return tensor_ops::abs(*self); }

/// Square: a*a.
fn GraphTensor? Tensor.sqr(&self) { return tensor_ops::sqr(*self); }

/// Square root.
fn GraphTensor? Tensor.sqrt(&self) { return tensor_ops::sqrt(*self); }

/// Reciprocal: 1/a.
fn GraphTensor? Tensor.recip(&self) { return tensor_ops::recip(*self); }

/// Exponential.
fn GraphTensor? Tensor.exp(&self) { return tensor_ops::exp(*self); }

/// Natural logarithm.
fn GraphTensor? Tensor.log(&self) { return tensor_ops::log(*self); }

/// Cosine.
fn GraphTensor? Tensor.cos(&self) { return tensor_ops::cos(*self); }

/// Sine.
fn GraphTensor? Tensor.sin(&self) { return tensor_ops::sin(*self); }

/// Hyperbolic tangent.
fn GraphTensor? Tensor.tanh(&self) { return tensor_ops::tanh(*self); }

/// ReLU activation.
fn GraphTensor? Tensor.relu(&self) { return tensor_ops::relu(*self); }

/// Sigmoid activation.
fn GraphTensor? Tensor.sigmoid(&self) { return tensor_ops::sigmoid(*self); }

/// Reduction with index expression.
fn GraphTensor? Tensor.reduce(&self, char[] expr)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	return tensor_ops::reduce(*self, &parsed);
}

/// Expansion with index expression and new dimension sizes.
fn GraphTensor? Tensor.expand(&self, char[] expr, ulong[] sizes)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	return tensor_ops::expand(*self, &parsed, { .sizes = sizes });
}

/// Binary operation with index pattern (einsum).
fn GraphTensor? Tensor.einsum(&self, Tensor other, char[] expr,
	operations::BinaryOp inner = operations::BinaryOp.MUL,
	operations::BinaryOp outer = operations::BinaryOp.ADD)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	return tensor_ops::einsum(*self, other, &parsed, { .inner = inner, .outer = outer });
}

/// Returns a u64 scalar tensor containing the total number of elements.
fn GraphTensor? Tensor.numel(&self)
{
	return tensor_ops::numel_op(*self);
}

/// Dispatches computation and waits for completion.
/// Marks the tensor as executable+stable, triggers compilation if needed,
/// then executes via the graph's execution sequence.
fn Self? Tensor.collect(&self, bool sync = true)
{
	TensorCore* c = self._core;
	graph::Graph* g = self.graph();

	c.flags.persistence.assign(tensor_core::Persistence.STABLE);

	g.execute_node(c);

	if (sync)
	{
		self.device().sync();
	}

	return *self;
}

/// Mark tensor for CPU offloading during execution.
/// When ref count hits 0, data is saved to a CPU buffer instead of discarded,
/// and auto-restored when needed as a source for a downstream operation.
/// Implies stable() — only stable tensors can be offloaded.
/// No-op for host devices (data is already on CPU).
fn Self Tensor.offload(&self)
{
	self.stable();
	if (!self._core.device.is_host())
	{
		self._core.flags.offload = true;
	}
	return *self;
}

/// Executes backward pass from this tensor (typically a loss).
/// Seeds the gradient with 1.0 (dL/dL = 1), then builds gradient
/// computation chains for all tensors that require_grad.
/// Does NOT execute gradients — call x.grad().collect() to compute them.
fn void Tensor.backward(&self)
{
	TensorCore* c = self._core;
	graph::Graph* g = self.graph();

	// Get or create gradient core for the loss tensor
	TensorCore* loss_grad = g.ensure_gradient_core(c);

	// Seed with 1.0: allocate data if needed, then fill via device interface
	if (!loss_grad.has_data())
	{
		loss_grad.alloc_data();
		events::fire(loss_grad, events::EventKind.ALLOC);
	}
	rtti::Datatype dt = loss_grad.dtype();
	usz n = loss_grad.shape().size();
	usz bytes = n * dt.size_of();
	char[] grad_bytes = ((char*)loss_grad.state.dense.data)[:bytes];
	loss_grad.device.mem_fill(dt, grad_bytes, 1.0);

	// Run the backward pass on the graph
	g.backward(c);
}

/// Returns the gradient tensor for this tensor.
/// Creates a new gradient core if one doesn't exist.
fn Tensor Tensor.grad(&self)
{
	TensorCore* c = self._core;
	graph::Graph* g = self.graph();
	TensorCore* grad_core = g.ensure_gradient_core(c);
	return tensor_from_core(grad_core);
}

/// Fill tensor data with a scalar value.
/// Allocates data if not already present.
/// Dispatches through the device interface so it works on GPU memory too.
fn Self Tensor.fill(&self, float value)
{
	TensorCore* c = self._core;
	if (!c.has_data())
	{
		c.alloc_data();
		events::fire(c, events::EventKind.ALLOC);
	}
	c.device.mem_fill(self.dtype(), self.value_bytes(), (double)value);

	if (c.flags.persistence == LEAF)
	{
		graph::Graph* g = (graph::Graph*)c.graph;
		g.notify_leaf_mutation(c);
	}

	return *self;
}

/// Seed tensor with a value, but only if it doesn't already have data.
/// Used for initializing gradients.
fn Self Tensor.seed(&self, float value)
{
	TensorCore* c = self._core;
	if (c.has_data()) return *self;
	return self.fill(value);
}

// Gather / Scatter

/// Index select / embedding lookup.
///
/// Selects elements from self along gather_dim using indices tensor.
///   For input [A, B, C] and indices [X, Y] at dim=1:
///   output shape is [A, X, Y, C]
fn GraphTensor? Tensor.gather(&self, usz gather_dim, Tensor indices)
{
	return tensor_ops::gather(*self, gather_dim, indices);
}

// Slice

/// Create a slice view of this tensor.
/// ranges: slice specification for each dimension.
/// offset: number of leading dimensions to skip (implicitly full-range).
///   slice({ { .end = half } }, offset: 3) on a rank-4 tensor slices only dim 3.
fn GraphTensor? Tensor.slice(&self, operations::SliceRange[] ranges, usz offset = 0)
{
	return tensor_ops::slice(*self, ranges, offset);
}

// Cat / Stack

/// Concatenate tensors along an existing dimension.
fn GraphTensor? cat(Tensor[] inputs, char concat_dim)
{
	return tensor_ops::cat(inputs, concat_dim);
}

/// Stack tensors along a new dimension.
fn GraphTensor? stack(Tensor[] inputs, char stack_dim)
{
	return tensor_ops::stack(inputs, stack_dim);
}

// Squeeze / Unsqueeze

/// Remove a dimension of size 1.
fn GraphTensor? Tensor.squeeze(&self, usz dim)
{
	return tensor_ops::squeeze(*self, dim);
}

/// Insert a new dimension of size 1 at the specified position.
/// Zero-copy: delegates to reshape.
///
///   [2, 3].unsqueeze(0) -> [1, 2, 3]
///   [2, 3].unsqueeze(1) -> [2, 1, 3]
///   [2, 3].unsqueeze(2) -> [2, 3, 1]
fn GraphTensor? Tensor.unsqueeze(&self, usz at)
{
	usz rank = self.shape().len;
	assert(at <= rank, "unsqueeze: position out of range");

	ulong[8] new_shape;
	usz input_idx = 0;
	for (usz d = 0; d < rank + 1; d++)
	{
		if (d == at)
		{
			new_shape[d] = 1;
		}
		else
		{
			new_shape[d] = self.shape().get(input_idx);
			input_idx++;
		}
	}

	return self.reshape(new_shape[:rank + 1]);
}

// Transpose

/// Swap two dimensions of a tensor.
fn GraphTensor? Tensor.transpose(&self, usz dim1, usz dim2)
{
	usz rank = self.shape().len;

	// Build permute pattern: "ijk->ikj" for swapping dims 1 and 2 in rank-3 tensor
	char[17] pattern;
	usz idx = 0;

	// Input pattern
	for (usz i = 0; i < rank; i++)
	{
		pattern[idx] = (char)('i' + i);
		idx += 1;
	}
	pattern[idx] = '-'; idx += 1;
	pattern[idx] = '>'; idx += 1;

	// Output pattern with dims swapped
	for (usz i = 0; i < rank; i++)
	{
		usz out_dim;
		if (i == dim1) { out_dim = dim2; }
		else if (i == dim2) { out_dim = dim1; }
		else { out_dim = i; }
		pattern[idx] = (char)('i' + out_dim);
		idx += 1;
	}

	index_expr::IndexExpr parsed = index_expr::index_expr_parse(pattern[:idx])!;
	return tensor_ops::permute(*self, &parsed);
}

// Cast

/// Cast tensor to a different dtype.
fn GraphTensor? Tensor.cast(&self, rtti::Datatype target_dtype)
{
	return tensor_ops::cast(*self, target_dtype);
}

// Transfer

/// Move tensor to a different device. No-op if already on target device.
fn GraphTensor? Tensor.to(&self, DeviceReference target_dev)
{
	return tensor_ops::to(*self, target_dev);
}

// Reshape

/// Reshape tensor to a new shape (must have same total elements).
/// Zero-copy: shares the underlying data buffer via refcounting.
fn GraphTensor? Tensor.reshape(&self, ulong[] new_shape)
{
	return tensor_ops::reshape(*self, new_shape);
}

// Linear Algebra Helpers

/// Matrix multiplication with batch broadcasting: C[...,i,k] = A[...,i,j] * B[j,k]
fn GraphTensor? Tensor.matmul(&self, Tensor other)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse("...ij,jk->...ik")!;
	return tensor_ops::einsum(*self, other, &parsed);
}

/// Matrix-vector multiplication: y[...,i] = A[...,i,j] * x[j]
fn GraphTensor? Tensor.matvec(&self, Tensor vec)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse("...ij,j->...i")!;
	return tensor_ops::einsum(*self, vec, &parsed);
}

/// Vector dot product: scalar = x[i] * y[i]
fn GraphTensor? Tensor.dot(&self, Tensor other)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse("i,i->")!;
	return tensor_ops::einsum(*self, other, &parsed);
}

/// Outer product: C[i,j] = x[i] * y[j]
fn GraphTensor? Tensor.outer(&self, Tensor other)
{
	index_expr::IndexExpr parsed = index_expr::index_expr_parse("i,j->ij")!;
	return tensor_ops::einsum(*self, other, &parsed);
}

// Scalar helpers

/// Create a scalar tensor with the same dtype/device/graph as this tensor.
fn GraphTensor? Tensor.scalar_like(&self, double value)
{
	ulong[1] scalar_shape = { 1 };
	return constant(self.dtype(), self.device(), &scalar_shape, (float)value);
}

/// Element-wise maximum.
fn GraphTensor? Tensor.max(&self, Tensor other) { return tensor_ops::max(*self, other); }

/// Element-wise minimum.
fn GraphTensor? Tensor.min(&self, Tensor other) { return tensor_ops::min(*self, other); }

/// SiLU activation.
fn GraphTensor? Tensor.silu(&self) { return tensor_ops::silu(*self); }

/// Softplus: log(1 + exp(x)), numerically stable.
fn GraphTensor? Tensor.softplus(&self) { return tensor_ops::softplus(*self); }

/// GELU activation.
fn GraphTensor? Tensor.gelu(&self) { return tensor_ops::gelu(*self); }

/// Heaviside step function.
fn GraphTensor? Tensor.heaviside(&self) { return tensor_ops::heaviside(*self); }

/// Element-wise clamp: result = max(lower, min(upper, self)).
fn GraphTensor? Tensor.clamp(&self, Tensor* lower = null, Tensor* upper = null)
{
	Tensor t = *self;
	if (upper != null) { t = tensor_ops::min(t, *upper)!; }
	if (lower != null) { t = tensor_ops::max(t, *lower)!; }
	return t;
}

/// Bind an external slice directly as the tensor's backing storage (zero-copy).
/// Validates that the element type matches the tensor's dtype and that the
/// slice length matches the tensor's element count.
/// The tensor becomes a view and will NOT free the bound buffer.
/// Advances graph generation so downstream ops re-execute with new data.
///
/// WARNING: Does not work with CUDA graphs — they capture the original pointer.
macro Tensor.bind(&self, data)
{
	var c = self._core;
	assert(c.state.class_ == tensor_core::CoreClass.DENSE, "bind: requires DENSE tensor");

	var $Type = $typeof(data[0]);
	assert($Type.sizeof == c.dtype().size_of(), "bind: element type size does not match tensor dtype");

	usz expected = c.shape().size();
	assert(data.len == expected, "bind: slice length does not match tensor size");

	c.state.dense.data = (void*)data.ptr;
	c.state.dense.byte_size = (ulong)(data.len * $Type.sizeof);

	if (c.flags.persistence == LEAF)
	{
		graph::Graph* g = (graph::Graph*)c.graph;
		g.notify_leaf_mutation(c);
	}

	return *self;
}

/// Write value(s) into a tensor at an element offset.
/// Handles scalars (float via mem_fill, int via raw bytes), arrays, and slices.
/// count: for float scalars, fill this many elements (default 1).
macro Tensor.set(&self, value, usz offset = 0, usz count = 1)
{
	var c = self._core;
	char[] dst = self.value_bytes();
	assert(dst.len > 0, "set: tensor data not allocated");
	usz dtype_size = c.dtype().size_of();
	usz byte_offset = offset * dtype_size;

	$if $defined(value.ptr):
		// Slice — has .ptr and .len
		usz elem_size = $typeof(value[0]).sizeof;
		assert(elem_size == dtype_size, "set: element size must match tensor dtype");
		usz byte_len = value.len * elem_size;
		assert(byte_offset + byte_len <= dst.len, "set: write exceeds tensor bounds");
		char[] src = ((char*)value.ptr)[:byte_len];
		c.device.mem_transfer(src, dst[byte_offset:byte_len], device_reference::TransferDirection.HTOD);
	$else
		$if $defined(value.len):
			// Fixed array — has .len but no .ptr
			usz elem_size = $typeof(value[0]).sizeof;
			assert(elem_size == dtype_size, "set: element size must match tensor dtype");
			usz byte_len = value.len * elem_size;
			assert(byte_offset + byte_len <= dst.len, "set: write exceeds tensor bounds");
			var tmp = value;
			char[] src = ((char*)&tmp)[:byte_len];
			c.device.mem_transfer(src, dst[byte_offset:byte_len], device_reference::TransferDirection.HTOD);
		$else
			// Scalar
			usz byte_len = count * dtype_size;
			assert(byte_offset + byte_len <= dst.len, "set: write exceeds tensor bounds");
			$if $typeof(value).kindof == TypeKind.FLOAT:
				c.device.mem_fill(c.dtype(), dst[byte_offset:byte_len], (double)value);
			$else
				assert(count == 1, "set: count > 1 not supported for integer scalars");
				assert($typeof(value).sizeof == dtype_size, "set: scalar size must match tensor dtype");
				const BYTE_LEN = $typeof(value).sizeof;
				var temp = value;
				char[] src = ((char*)&temp)[:BYTE_LEN];
				c.device.mem_transfer(src, dst[byte_offset:BYTE_LEN], device_reference::TransferDirection.HTOD);
			$endif
		$endif
	$endif

	if (c.flags.persistence == LEAF)
	{
		graph::Graph* g = (graph::Graph*)c.graph;
		g.notify_leaf_mutation(c);
	}

	return *self;
}

/// Read value(s) from a tensor at an element offset, transferring from device.
/// Handles slices, pointers to arrays, and pointers to scalars.
/// This is a synchronizing operation (DtoH transfer per call), so avoid using it in hot-loops.
///
/// Usage:
///   float val;       tensor.get(&val);          // single element at index 0
///   float val;       tensor.get(&val, index: 5); // single element at index 5
///   float[10] buf;   tensor.get(&buf);           // 10 elements starting at index 0
///   float[] slice;   tensor.get(slice);           // fill slice from tensor
macro Tensor.get(&self, target, usz index = 0)
{
	var c = self._core;
	char[] src = self.value_bytes();
	assert(src.len > 0, "get: tensor has no data");
	usz dtype_size = c.dtype().size_of();
	usz byte_offset = index * dtype_size;

	$if $defined(target.ptr):
		// Slice — has .ptr and .len (distinguishes from pointer-to-array which auto-derefs .len)
		usz elem_size = $typeof(target[0]).sizeof;
		assert(elem_size == dtype_size, "get: element size must match tensor dtype");
		usz byte_len = target.len * elem_size;
		assert(byte_offset + byte_len <= src.len, "get: read exceeds tensor bounds");
		char[] dst = ((char*)target.ptr)[:byte_len];
		c.device.mem_transfer(src[byte_offset:byte_len], dst, device_reference::TransferDirection.DTOH);
	$else
		$if $defined((*target).len):
			// Pointer to array — deref to get length
			usz elem_size = $typeof((*target)[0]).sizeof;
			assert(elem_size == dtype_size, "get: element size must match tensor dtype");
			usz byte_len = (*target).len * elem_size;
			assert(byte_offset + byte_len <= src.len, "get: read exceeds tensor bounds");
			char[] dst = ((char*)target)[:byte_len];
			c.device.mem_transfer(src[byte_offset:byte_len], dst, device_reference::TransferDirection.DTOH);
		$else
			// Pointer to scalar — single element
			usz elem_size = $typeof(*target).sizeof;
			assert(elem_size == dtype_size, "get: pointed-to size must match tensor dtype");
			assert(byte_offset + dtype_size <= src.len, "get: read exceeds tensor bounds");
			char[] dst = ((char*)target)[:dtype_size];
			c.device.mem_transfer(src[byte_offset:dtype_size], dst, device_reference::TransferDirection.DTOH);
		$endif
	$endif
}

/// Pretty-print tensor contents in numpy-style format to stderr.
fn void Tensor.print(&self)
{
	display::print(self._core);
}

// Convolution

/// 2D convolution.
///   Input:  [N, C_in, H, W]
///   Weight: [C_out, C_in/groups, kH, kW]
///   Output: [N, C_out, H_out, W_out]
fn GraphTensor? Tensor.conv2d(&self, Tensor weight,
	operations::ConvParams params = operations::DEFAULT_CONV_PARAMS)
{
	return tensor_ops::conv2d(*self, weight, params);
}

// Dimension query

/// Returns the size of dimension `axis` as a u64 scalar tensor.
fn GraphTensor? Tensor.dim(&self, usz axis)
{
	return tensor_ops::dim_op(*self, axis);
}

/// Ternary selection using this tensor as condition.
///   result[i] = self[i] ? positive[i] : negative[i]
fn GraphTensor? Tensor.select(&self, Tensor positive, Tensor negative)
{
	return tensor_ops::select(*self, positive, negative);
}

// Operator Overloads

fn Tensor Tensor.op_add(self, Tensor other) @operator(+)
{
	return self.add(other)!!;
}

fn Tensor Tensor.op_add_scalar(self, float scalar) @operator(+)
{
	return self.add(self.scalar_like(scalar)!!)!!;
}

fn Tensor Tensor.op_add_scalar_r(self, float scalar) @operator_r(+)
{
	Tensor s = self.scalar_like(scalar)!!;
	return s.add(self)!!;
}

fn Tensor Tensor.op_sub(self, Tensor other) @operator(-)
{
	return self.sub(other)!!;
}

fn Tensor Tensor.op_sub_scalar(self, float scalar) @operator(-)
{
	return self.sub(self.scalar_like(scalar)!!)!!;
}

fn Tensor Tensor.op_sub_scalar_r(self, float scalar) @operator_r(-)
{
	Tensor s = self.scalar_like(scalar)!!;
	return s.sub(self)!!;
}

fn Tensor Tensor.op_mul(self, Tensor other) @operator(*)
{
	return self.mul(other)!!;
}

fn Tensor Tensor.op_mul_scalar(self, float scalar) @operator(*)
{
	return self.mul(self.scalar_like(scalar)!!)!!;
}

fn Tensor Tensor.op_mul_scalar_r(self, float scalar) @operator_r(*)
{
	Tensor s = self.scalar_like(scalar)!!;
	return s.mul(self)!!;
}

fn Tensor Tensor.op_div(self, Tensor other) @operator(/)
{
	return self.div(other)!!;
}

fn Tensor Tensor.op_div_scalar(self, float scalar) @operator(/)
{
	return self.div(self.scalar_like(scalar)!!)!!;
}

fn Tensor Tensor.op_div_scalar_r(self, float scalar) @operator_r(/)
{
	Tensor s = self.scalar_like(scalar)!!;
	return s.div(self)!!;
}

fn Tensor Tensor.op_neg(self) @operator(-)
{
	return self.neg()!!;
}
