/*
 * memory/concat_stack.cu - Concat and Stack operations for CUDA
 *
 * Concat: Join tensors along an existing dimension
 * Stack: Join tensors along a new dimension
 *
 * NOTE: All parameters are passed by value to ensure CUDA graph capture compatibility.
 * No cudaMalloc/cudaFree operations during kernel execution.
 */

#ifndef __MEMORY_CONCAT_STACK_CU__
#define __MEMORY_CONCAT_STACK_CU__

#include "../core/assert.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"

#define MAX_CONCAT_INPUTS 8

/* ============================================================================
 * Concat Parameters (passed by value)
 * ============================================================================ */

struct ConcatParams {
    len_t out_shape[CUDA_MAX_DIMS];
    len_t base_shape[CUDA_MAX_DIMS];       // Shape of first input (reference for non-concat dims)
    len_t offsets[MAX_CONCAT_INPUTS + 1];  // Cumulative offsets along concat dim
    len_t concat_sizes[MAX_CONCAT_INPUTS]; // Size of concat dim for each input
    const void* inputs[MAX_CONCAT_INPUTS]; // Input data pointers
    len_t concat_dim;
    len_t ndim;
    len_t numel;
    len_t num_inputs;
};

/* ============================================================================
 * Stack Parameters (passed by value)
 * ============================================================================ */

struct StackParams {
    len_t out_shape[CUDA_MAX_DIMS];
    len_t in_strides[CUDA_MAX_DIMS];
    const void* inputs[MAX_CONCAT_INPUTS]; // Input data pointers
    len_t stack_dim;
    len_t out_ndim;
    len_t in_ndim;
    len_t numel;
    len_t num_inputs;
};

/* ============================================================================
 * Concat Kernel
 *
 * For each output position, determine which input tensor it belongs to
 * based on the coordinate along the concat dimension.
 *
 * IMPORTANT: Each input may have a different size along the concat dimension,
 * which affects strides. We compute per-input strides on-the-fly using the
 * input's actual concat dimension size.
 * ============================================================================ */

template<typename T>
__global__ void kernel_concat(
    T* __restrict__ out,
    ConcatParams params
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.numel) return;

    // Compute multi-index from flat output index
    len_t idx = i;
    len_t coords[CUDA_MAX_DIMS];

    for (int d = params.ndim - 1; d >= 0; d--) {
        coords[d] = idx % params.out_shape[d];
        idx /= params.out_shape[d];
    }

    // Find which input this position belongs to based on concat dim coordinate
    len_t concat_coord = coords[params.concat_dim];
    len_t input_idx = 0;
    for (len_t n = 0; n < params.num_inputs; n++) {
        if (concat_coord < params.offsets[n + 1]) {
            input_idx = n;
            break;
        }
    }

    // Adjust coordinate relative to this input's start
    len_t local_concat_coord = concat_coord - params.offsets[input_idx];

    // Compute input strides on-the-fly using this input's concat dimension size
    // Row-major: stride[d] = product of dims[d+1..ndim]
    // Use base_shape for all dims except concat_dim, where we use concat_sizes[input_idx]
    len_t in_offset = 0;
    len_t stride = 1;
    for (int d = params.ndim - 1; d >= 0; d--) {
        len_t coord = (d == params.concat_dim) ? local_concat_coord : coords[d];
        in_offset += coord * stride;

        // Update stride for next dimension
        len_t dim_size = (d == params.concat_dim)
            ? params.concat_sizes[input_idx]
            : params.base_shape[d];
        stride *= dim_size;
    }

    const T* input = static_cast<const T*>(params.inputs[input_idx]);
    out[i] = input[in_offset];
}

/* ============================================================================
 * Stack Kernel
 *
 * For each output position, use the stack dimension index to select
 * which input tensor, then compute the position within that input.
 * ============================================================================ */

template<typename T>
__global__ void kernel_stack(
    T* __restrict__ out,
    StackParams params
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.numel) return;

    // Compute multi-index from flat output index
    len_t idx = i;
    len_t coords[CUDA_MAX_DIMS];

    for (int d = params.out_ndim - 1; d >= 0; d--) {
        coords[d] = idx % params.out_shape[d];
        idx /= params.out_shape[d];
    }

    // The stack dimension index tells us which input to use
    len_t input_idx = coords[params.stack_dim];

    // Compute input offset (skip the stack dimension)
    len_t in_offset = 0;
    int in_d = 0;
    for (int d = 0; d < params.out_ndim; d++) {
        if (d == params.stack_dim) continue;
        in_offset += coords[d] * params.in_strides[in_d];
        in_d++;
    }

    const T* input = static_cast<const T*>(params.inputs[input_idx]);
    out[i] = input[in_offset];
}

/* ============================================================================
 * Stride Computation Helper
 * ============================================================================ */

static void concat_compute_strides(const len_t* shape, len_t ndim, len_t* strides) {
    len_t stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        strides[d] = stride;
        stride *= shape[d];
    }
}

/* ============================================================================
 * Extern C API
 * ============================================================================ */

/*
 * Execute concat operation.
 *
 * out: Output tensor (concatenated result)
 * inputs: Array of DenseCore pointers (max 8)
 * num_inputs: Number of input tensors
 * offsets: Cumulative offsets along concat dimension (size = num_inputs + 1)
 * concat_dim: Dimension to concatenate along
 * stream: CUDA stream for async execution
 */
extern "C" void cuda_concat(
    CudaDeviceHandle device,
    DenseCore* out,
    const DenseCore* const* inputs,
    len_t num_inputs,
    const len_t* offsets,
    len_t concat_dim,
    StreamHandle stream_handle
) {
    len_t ndim = inputs[0]->shape.len;
    Dtype dtype = dense_core_dtype(out);

    LOG_DEBUG("cuda_concat: num_inputs=%lu concat_dim=%lu ndim=%lu numel=%lu",
              num_inputs, concat_dim, ndim, out->num_elements);

    CUstream stream = static_cast<CUstream>(stream_handle.ptr);

    // Build parameters struct (passed by value to kernel)
    ConcatParams params = {};
    params.concat_dim = concat_dim;
    params.ndim = ndim;
    params.numel = out->num_elements;
    params.num_inputs = num_inputs;

    for (len_t d = 0; d < ndim; d++) {
        params.out_shape[d] = out->shape.buffer[d];
        params.base_shape[d] = inputs[0]->shape.buffer[d];
    }
    for (len_t n = 0; n <= num_inputs && n <= MAX_CONCAT_INPUTS; n++) {
        params.offsets[n] = offsets[n];
    }
    for (len_t n = 0; n < num_inputs && n < MAX_CONCAT_INPUTS; n++) {
        params.inputs[n] = inputs[n]->data;
        // Compute each input's concat dimension size from cumulative offsets
        params.concat_sizes[n] = offsets[n + 1] - offsets[n];
    }

    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (out->num_elements + block_size - 1) / block_size;

    if (dtype == DTYPE_F32) {
        kernel_concat<float><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(out->data),
            params
        );
    } else if (dtype == DTYPE_F64) {
        kernel_concat<double><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(out->data),
            params
        );
    } else if (dtype == DTYPE_F16) {
        kernel_concat<f16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<f16*>(out->data),
            params
        );
    } else if (dtype == DTYPE_BF16) {
        kernel_concat<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out->data),
            params
        );
    } else {
        SYSTEM_EXIT("Unsupported dtype for concat");
    }
}

/*
 * Execute stack operation.
 *
 * out: Output tensor (stacked result)
 * inputs: Array of DenseCore pointers (max 8)
 * num_inputs: Number of input tensors
 * stack_dim: Dimension to stack along (new dimension)
 * stream: CUDA stream for async execution
 */
extern "C" void cuda_stack(
    CudaDeviceHandle device,
    DenseCore* out,
    const DenseCore* const* inputs,
    len_t num_inputs,
    len_t stack_dim,
    StreamHandle stream_handle
) {
    len_t in_ndim = inputs[0]->shape.len;
    Dtype dtype = dense_core_dtype(out);

    LOG_DEBUG("cuda_stack: num_inputs=%lu stack_dim=%lu in_ndim=%lu numel=%lu",
              num_inputs, stack_dim, in_ndim, out->num_elements);

    CUstream stream = static_cast<CUstream>(stream_handle.ptr);

    // Build parameters struct (passed by value to kernel)
    StackParams params = {};
    params.stack_dim = stack_dim;
    params.out_ndim = in_ndim + 1;  // Output has one more dimension
    params.in_ndim = in_ndim;
    params.numel = out->num_elements;
    params.num_inputs = num_inputs;

    for (len_t d = 0; d < params.out_ndim; d++) {
        params.out_shape[d] = out->shape.buffer[d];
    }
    for (len_t n = 0; n < num_inputs && n < MAX_CONCAT_INPUTS; n++) {
        params.inputs[n] = inputs[n]->data;
    }
    concat_compute_strides(inputs[0]->shape.buffer, in_ndim, params.in_strides);

    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (out->num_elements + block_size - 1) / block_size;

    if (dtype == DTYPE_F32) {
        kernel_stack<float><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(out->data),
            params
        );
    } else if (dtype == DTYPE_F64) {
        kernel_stack<double><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(out->data),
            params
        );
    } else if (dtype == DTYPE_F16) {
        kernel_stack<f16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<f16*>(out->data),
            params
        );
    } else if (dtype == DTYPE_BF16) {
        kernel_stack<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out->data),
            params
        );
    } else {
        SYSTEM_EXIT("Unsupported dtype for stack");
    }
}

#endif /* __MEMORY_CONCAT_STACK_CU__ */
