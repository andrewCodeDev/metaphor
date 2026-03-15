/*
 * memory/gather.cu - Index-based gather operation for CUDA
 *
 * gather(input, dim, indices) -> output
 * where output[pre..., idx_coords..., post...] = input[pre..., indices[idx_coords...], post...]
 *
 * Example:
 *   input shape [10, 20, 30], indices shape [5, 7], dim=1
 *   output shape [10, 5, 7, 30]
 *   output[i, j, k, l] = input[i, indices[j, k], l]
 *
 * NOTE: All parameters are passed by value to ensure CUDA graph capture compatibility.
 * No cudaMalloc/cudaFree operations during kernel execution.
 */

#ifndef __MEMORY_GATHER_CU__
#define __MEMORY_GATHER_CU__

#include "../core/assert.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"

#define MAX_GATHER_DIMS 8

/* ============================================================================
 * Gather Parameters Structure (passed by value)
 * ============================================================================ */

struct GatherParams {
    len_t in_shape[MAX_GATHER_DIMS];      // Input tensor shape
    len_t in_strides[MAX_GATHER_DIMS];    // Input tensor strides
    len_t idx_shape[MAX_GATHER_DIMS];     // Indices tensor shape
    len_t idx_strides[MAX_GATHER_DIMS];   // Indices tensor strides
    len_t out_shape[MAX_GATHER_DIMS];     // Output tensor shape
    len_t in_ndim;                         // Input number of dimensions
    len_t idx_ndim;                        // Indices number of dimensions
    len_t out_ndim;                        // Output number of dimensions
    len_t gather_dim;                      // Dimension to gather from
    len_t numel;                           // Total output elements
};

/* ============================================================================
 * Scatter-Add Parameters Structure (backward for gather)
 * ============================================================================ */

struct ScatterAddParams {
    len_t out_shape[MAX_GATHER_DIMS];     // Output (original input) shape
    len_t out_strides[MAX_GATHER_DIMS];   // Output strides
    len_t idx_shape[MAX_GATHER_DIMS];     // Indices tensor shape
    len_t idx_strides[MAX_GATHER_DIMS];   // Indices tensor strides
    len_t grad_shape[MAX_GATHER_DIMS];    // Gradient tensor shape
    len_t out_ndim;                        // Output number of dimensions
    len_t idx_ndim;                        // Indices number of dimensions
    len_t grad_ndim;                       // Gradient number of dimensions
    len_t gather_dim;                      // Dimension that was gathered
    len_t grad_numel;                      // Total gradient elements
};

/* ============================================================================
 * Helper: Compute strides from shape (row-major)
 * ============================================================================ */

static void gather_compute_strides(const len_t* shape, len_t ndim, len_t* strides) {
    len_t stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        strides[d] = stride;
        stride *= shape[d];
    }
}

/* ============================================================================
 * Gather Kernel
 *
 * For each output position:
 * 1. Decompose into (pre_coords, idx_coords, post_coords)
 * 2. Look up index value from indices[idx_coords]
 * 3. Read from input[pre_coords, index_value, post_coords]
 * ============================================================================ */

template<typename T, typename IdxT>
__global__ void kernel_gather(
    T* __restrict__ out,
    const T* __restrict__ in,
    const IdxT* __restrict__ indices,
    GatherParams params
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.numel) return;

    // Decompose flat output index into multi-index
    len_t out_coords[MAX_GATHER_DIMS];
    len_t idx = i;
    for (int d = params.out_ndim - 1; d >= 0; d--) {
        out_coords[d] = idx % params.out_shape[d];
        idx /= params.out_shape[d];
    }

    // Output layout: [pre_dims..., idx_dims..., post_dims...]
    // where pre_dims are input dims before gather_dim
    //       idx_dims are indices dims
    //       post_dims are input dims after gather_dim

    // Extract idx_coords from output coords (middle section)
    len_t idx_offset = 0;
    for (len_t d = 0; d < params.idx_ndim; d++) {
        len_t out_d = params.gather_dim + d;
        idx_offset += out_coords[out_d] * params.idx_strides[d];
    }

    // Look up the index value
    IdxT index_value = indices[idx_offset];

    // Compute input offset
    // pre_dims: output coords [0..gather_dim) map directly to input
    // gather_dim: use index_value
    // post_dims: output coords [gather_dim + idx_ndim..) map to input [gather_dim + 1..)
    len_t in_offset = 0;

    // Pre-dimensions
    for (len_t d = 0; d < params.gather_dim; d++) {
        in_offset += out_coords[d] * params.in_strides[d];
    }

    // Gather dimension - use the looked up index
    in_offset += index_value * params.in_strides[params.gather_dim];

    // Post-dimensions
    for (len_t d = params.gather_dim + 1; d < params.in_ndim; d++) {
        // Output coord for this input dim is at gather_dim + idx_ndim + (d - gather_dim - 1)
        len_t out_d = params.gather_dim + params.idx_ndim + (d - params.gather_dim - 1);
        in_offset += out_coords[out_d] * params.in_strides[d];
    }

    out[i] = in[in_offset];
}

/* ============================================================================
 * Scatter-Add Kernel (backward for gather)
 *
 * For each gradient position, atomically add to the corresponding input position.
 * This is the reverse of gather - we need to accumulate gradients.
 * ============================================================================ */

template<typename T, typename IdxT>
__global__ void kernel_scatter_add(
    T* __restrict__ out,           // Output gradient (shape of original input)
    const T* __restrict__ grad,    // Incoming gradient (shape of gather output)
    const IdxT* __restrict__ indices,
    ScatterAddParams params
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grad_numel) return;

    // Decompose flat gradient index into multi-index
    len_t grad_coords[MAX_GATHER_DIMS];
    len_t idx = i;
    for (int d = params.grad_ndim - 1; d >= 0; d--) {
        grad_coords[d] = idx % params.grad_shape[d];
        idx /= params.grad_shape[d];
    }

    // Extract idx_coords from grad coords (middle section)
    len_t idx_offset = 0;
    for (len_t d = 0; d < params.idx_ndim; d++) {
        len_t grad_d = params.gather_dim + d;
        idx_offset += grad_coords[grad_d] * params.idx_strides[d];
    }

    // Look up the index value
    IdxT index_value = indices[idx_offset];

    // Compute output offset (same logic as gather but reversed)
    len_t out_offset = 0;

    // Pre-dimensions
    for (len_t d = 0; d < params.gather_dim; d++) {
        out_offset += grad_coords[d] * params.out_strides[d];
    }

    // Gather dimension - use the looked up index
    out_offset += index_value * params.out_strides[params.gather_dim];

    // Post-dimensions
    for (len_t d = params.gather_dim + 1; d < params.out_ndim; d++) {
        len_t grad_d = params.gather_dim + params.idx_ndim + (d - params.gather_dim - 1);
        out_offset += grad_coords[grad_d] * params.out_strides[d];
    }

    // Atomic add to handle multiple indices pointing to the same location
    atomicAdd(&out[out_offset], grad[i]);
}

/* ============================================================================
 * Extern C API
 * ============================================================================ */

/*
 * Execute gather operation.
 *
 * out: Output DenseCore
 * in: Input DenseCore to gather from
 * indices: Index DenseCore (i64)
 * gather_dim: Dimension to gather from
 * stream: CUDA stream
 */
extern "C" void cuda_gather(
    CudaDeviceHandle device,
    DenseCore* out,
    const DenseCore* in,
    const DenseCore* indices,
    len_t gather_dim,
    StreamHandle stream_handle
) {
    len_t in_ndim  = in->shape.len;
    len_t idx_ndim = indices->shape.len;
    len_t out_ndim = out->shape.len;
    len_t numel    = out->num_elements;

    LOG_DEBUG("cuda_gather: gather_dim=%lu in_ndim=%lu idx_ndim=%lu out_ndim=%lu numel=%lu",
              gather_dim, in_ndim, idx_ndim, out_ndim, numel);

    CUstream stream = static_cast<CUstream>(stream_handle.ptr);

    // Build parameters struct
    GatherParams params = {};
    params.gather_dim = gather_dim;
    params.in_ndim = in_ndim;
    params.idx_ndim = idx_ndim;
    params.out_ndim = out_ndim;
    params.numel = numel;

    for (len_t d = 0; d < in_ndim && d < MAX_GATHER_DIMS; d++) {
        params.in_shape[d] = in->shape.buffer[d];
    }
    for (len_t d = 0; d < idx_ndim && d < MAX_GATHER_DIMS; d++) {
        params.idx_shape[d] = indices->shape.buffer[d];
    }
    for (len_t d = 0; d < out_ndim && d < MAX_GATHER_DIMS; d++) {
        params.out_shape[d] = out->shape.buffer[d];
    }

    gather_compute_strides(params.in_shape, in_ndim, params.in_strides);
    gather_compute_strides(params.idx_shape, idx_ndim, params.idx_strides);

    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    Dtype dtype = dense_core_dtype(out);

    // Dispatch based on data type (indices assumed i64)
    if (dtype == DTYPE_F32) {
        kernel_gather<float, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(out->data),
            static_cast<const float*>(in->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else if (dtype == DTYPE_F64) {
        kernel_gather<double, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(out->data),
            static_cast<const double*>(in->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else if (dtype == DTYPE_F16) {
        kernel_gather<f16, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<f16*>(out->data),
            static_cast<const f16*>(in->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else if (dtype == DTYPE_BF16) {
        kernel_gather<__nv_bfloat16, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out->data),
            static_cast<const __nv_bfloat16*>(in->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else {
        SYSTEM_EXIT("Unsupported dtype for gather");
    }
}

/*
 * Execute scatter-add operation (backward for gather).
 *
 * out: Output DenseCore (zeros, same shape as original input)
 * grad: Gradient DenseCore (same shape as gather output)
 * indices: Index DenseCore (i64)
 * gather_dim: Dimension that was gathered
 * stream: CUDA stream
 */
extern "C" void cuda_scatter_add(
    CudaDeviceHandle device,
    DenseCore* out,
    const DenseCore* grad,
    const DenseCore* indices,
    len_t gather_dim,
    StreamHandle stream_handle
) {
    len_t out_ndim   = out->shape.len;
    len_t idx_ndim   = indices->shape.len;
    len_t grad_ndim  = grad->shape.len;
    len_t grad_numel = grad->num_elements;

    LOG_DEBUG("cuda_scatter_add: gather_dim=%lu out_ndim=%lu idx_ndim=%lu grad_ndim=%lu grad_numel=%lu",
              gather_dim, out_ndim, idx_ndim, grad_ndim, grad_numel);

    CUstream stream = static_cast<CUstream>(stream_handle.ptr);

    // Build parameters struct
    ScatterAddParams params = {};
    params.gather_dim = gather_dim;
    params.out_ndim = out_ndim;
    params.idx_ndim = idx_ndim;
    params.grad_ndim = grad_ndim;
    params.grad_numel = grad_numel;

    for (len_t d = 0; d < out_ndim && d < MAX_GATHER_DIMS; d++) {
        params.out_shape[d] = out->shape.buffer[d];
    }
    for (len_t d = 0; d < idx_ndim && d < MAX_GATHER_DIMS; d++) {
        params.idx_shape[d] = indices->shape.buffer[d];
    }
    for (len_t d = 0; d < grad_ndim && d < MAX_GATHER_DIMS; d++) {
        params.grad_shape[d] = grad->shape.buffer[d];
    }

    gather_compute_strides(params.out_shape, out_ndim, params.out_strides);
    gather_compute_strides(params.idx_shape, idx_ndim, params.idx_strides);

    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (grad_numel + block_size - 1) / block_size;

    Dtype dtype = dense_core_dtype(out);

    if (dtype == DTYPE_F32) {
        kernel_scatter_add<float, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(out->data),
            static_cast<const float*>(grad->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else if (dtype == DTYPE_F64) {
        kernel_scatter_add<double, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(out->data),
            static_cast<const double*>(grad->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else if (dtype == DTYPE_F16) {
        kernel_scatter_add<f16, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<f16*>(out->data),
            static_cast<const f16*>(grad->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else if (dtype == DTYPE_BF16) {
        kernel_scatter_add<__nv_bfloat16, int64_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out->data),
            static_cast<const __nv_bfloat16*>(grad->data),
            static_cast<const int64_t*>(indices->data),
            params
        );
    } else {
        SYSTEM_EXIT("Unsupported dtype for scatter_add");
    }
}

#endif /* __MEMORY_GATHER_CU__ */
