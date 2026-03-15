//! Sparse Core
//!
//! Storage for sparse tensors in COO (coordinate) format.
//! Indices are eagerly allocated; data may be lazy.
//!

module metaphor::tensor::sparse_core;
import std::core::mem::allocator;

// SparseConfig — initialization options

struct SparseConfig
{
	/// Source indices to copy (one slice per dimension).
	ulong[][] indices;
}

// SparseCore

struct SparseCore
{
	/// Number of stored non-zero elements.
	ulong nnz;

	/// Index arrays for COO format.
	/// Length: `order` (one slice per dimension).
	/// Each inner slice has length `nnz`.
	///
	/// Backed by a single contiguous allocation.
	/// The outer slice contains views into this buffer.
	ulong[][] indices;

	/// Non-zero values stored contiguously (nnz * dtype.size_of() bytes).
	/// Null if not yet materialized.
	char[] data;
}

/// Initialize sparse core with indices. Data is NOT allocated.
fn SparseCore sparse_core_empty(Allocator alloc, ulong[] shape_slice, SparseConfig config)
{
	usz tensor_order = shape_slice.len;
	assert(config.indices.len == tensor_order, "indices count must match shape rank");
	assert(tensor_order > 0, "zero-order tensors not supported");

	for (usz i = 0; i < shape_slice.len; i++)
	{
		assert(shape_slice[i] > 0, "zero-sized dimensions not supported");
	}

	ulong nnz = config.indices[0].len;
	assert(nnz > 0, "sparse tensor must have at least one element");

	for (usz i = 1; i < config.indices.len; i++)
	{
		assert(config.indices[i].len == nnz, "all index slices must have same length (nnz)");
	}

	// Allocate flat buffer for all index data
	ulong[] flat_indices = allocator::alloc_array(alloc, ulong, tensor_order * nnz);

	// Allocate slice-of-slices to view into flat buffer
	ulong[][] indices = allocator::alloc_array(alloc, ulong[], tensor_order);

	// Populate flat buffer and assign slices
	for (usz dim = 0; dim < tensor_order; dim++)
	{
		usz start = dim * nnz;
		indices[dim] = flat_indices[start : nnz];
		// Copy source indices
		for (usz j = 0; j < nnz; j++)
		{
			indices[dim][j] = config.indices[dim][j];
		}
	}

	return {
		.nnz = nnz,
		.indices = indices,
		.data = {},
	};
}

/// Free allocator-managed index metadata.
/// Device memory (data) is arena-managed; not freed here.
fn void SparseCore.deinit(&self, Allocator alloc)
{
	if (self.indices.len > 0)
	{
		// Free flat index buffer via first slice's pointer
		ulong* flat_ptr = self.indices[0].ptr;
		usz flat_len = self.indices.len * self.nnz;
		allocator::free(alloc, flat_ptr[:flat_len]);

		// Free slice-of-slices
		allocator::free(alloc, self.indices);
	}

	self.indices = {};
	self.data = {};
	self.nnz = 0;
}
