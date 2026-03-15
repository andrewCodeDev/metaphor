//! Dense Core
//!
//! Storage for dense (contiguous) tensors. Contains element count, strides,
//! and an optional data buffer. Data is lazily allocated.
//!
//! C-interoperable layout: data pointer, byte_size, num_elements, dtype,
//! embedded Shape, flat strides array. Matches dense_core.h exactly.
//!

module metaphor::tensor::dense_core;
import metaphor::tensor::shape;
import metaphor::rtti;

// DenseConfig — initialization options

struct DenseConfig
{
	/// Custom strides (in elements). If null (len=0), computes contiguous row-major.
	ulong[] strides;
}

const DenseConfig DEFAULT_DENSE_CONFIG = { .strides = {} };

// DenseCore — C-interoperable tensor storage

struct DenseCore
{
	/// Device/host memory pointer. Null if lazily initialized.
	void* data;

	/// Buffer size in bytes.
	ulong byte_size;

	/// Total number of elements in the tensor.
	ulong num_elements;

	/// Datatype ordinal stored as ulong (rtti::Datatype cast to ulong).
	ulong dtype;

	/// Embedded shape — all Shape methods work directly.
	Shape shape;

	/// Row-major strides in elements. ndim shared with shape.len.
	ulong[8] strides;
}

/// Initialize dense core with strides computed from shape. Data is NOT allocated.
fn DenseCore dense_core_empty(ulong[] shape_slice, rtti::Datatype dtype, DenseConfig config)
{
	Shape s = shape::shape_init(shape_slice);

	Strides st;
	if (config.strides.len > 0)
	{
		Shape stride_shape = shape::shape_init(config.strides);
		st.buffer = stride_shape.buffer;
		st.len = config.strides.len;
	}
	else
	{
		st = s.strides();
	}

	return {
		.data = null,
		.byte_size = 0,
		.num_elements = s.size(),
		.dtype = (ulong)dtype,
		.shape = s,
		.strides = st.buffer,
	};
}

/// Return the rtti::Datatype from the stored ulong ordinal.
fn rtti::Datatype DenseCore.datatype(&self)
{
	return (rtti::Datatype)(char)self.dtype;
}

/// Construct a Strides struct from the flat strides array + shape.len.
fn Strides DenseCore.get_strides(&self)
{
	Strides st;
	st.buffer = self.strides;
	st.len = self.shape.len;
	return st;
}

/// Clear state without freeing device memory (arena-managed).
fn void DenseCore.deinit(&self)
{
	self.data = null;
	self.byte_size = 0;
	self.num_elements = 0;
}
