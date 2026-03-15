/*
 * memory/slice.cu - Strided copy operations for slice forward/backward
 *
 * Forward (gather): Copy from strided source positions to contiguous output
 * Backward (scatter): Copy from contiguous gradient to strided output positions
 *
 * NOTE: All parameters are passed by value to ensure CUDA graph capture compatibility.
 * No cudaMalloc/cudaFree operations during kernel execution.
 */

#ifndef __MEMORY_SLICE_CU__
#define __MEMORY_SLICE_CU__

#include "../core/assert.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"

/* ============================================================================
 * Slice Parameters Structure (passed by value)
 * ============================================================================ */

struct SliceGatherParams {
    len_t out_shape[CUDA_MAX_DIMS];
    len_t in_strides[CUDA_MAX_DIMS];
    len_t offsets[CUDA_MAX_DIMS];
    len_t steps[CUDA_MAX_DIMS];
    len_t ndim;
    len_t numel;
};

struct SliceScatterParams {
    len_t out_shape[CUDA_MAX_DIMS];
    len_t grad_shape[CUDA_MAX_DIMS];
    len_t grad_strides[CUDA_MAX_DIMS];
    len_t offsets[CUDA_MAX_DIMS];
    len_t steps[CUDA_MAX_DIMS];
    len_t ndim;
    len_t numel;
};

/* ============================================================================
 * Forward Kernel: Strided Gather (Slice)
 *
 * For each output position, compute the corresponding input position:
 *   in_idx[d] = offsets[d] + steps[d] * out_idx[d]
 * ============================================================================ */

template<typename T>
__global__ void kernel_slice_gather(
    T* __restrict__ out,
    const T* __restrict__ in,
    SliceGatherParams params
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.numel) return;

    /* Compute multi-index from flat output index */
    len_t idx = i;
    len_t in_offset = 0;

    for (int d = params.ndim - 1; d >= 0; d--) {
        len_t coord = idx % params.out_shape[d];
        idx /= params.out_shape[d];

        /* Apply slice transformation: in_coord = offset + step * out_coord */
        len_t in_coord = params.offsets[d] + params.steps[d] * coord;
        in_offset += in_coord * params.in_strides[d];
    }

    out[i] = in[in_offset];
}

/* ============================================================================
 * Backward Kernel: Scatter (Slice Gradient)
 *
 * For each output position (original tensor shape), check if it maps to
 * a valid input position (gradient tensor):
 *   - Must be >= offset[d]
 *   - Must be < offset[d] + step[d] * grad_shape[d]
 *   - (pos - offset[d]) must be divisible by step[d]
 *
 * If valid: grad_idx[d] = (out_idx[d] - offset[d]) / step[d]
 * If invalid: output 0
 * ============================================================================ */

template<typename T>
__global__ void kernel_slice_scatter(
    T* __restrict__ out,
    const T* __restrict__ grad,
    SliceScatterParams params
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.numel) return;

    /* Compute multi-index from flat output index */
    len_t idx = i;
    len_t grad_offset = 0;
    bool in_bounds = true;

    for (int d = params.ndim - 1; d >= 0; d--) {
        len_t coord = idx % params.out_shape[d];
        idx /= params.out_shape[d];

        len_t offset = params.offsets[d];
        len_t step = params.steps[d];
        len_t grad_dim = params.grad_shape[d];

        /* Check bounds: coord must be in [offset, offset + step * grad_dim) */
        len_t end = offset + step * grad_dim;
        if (coord < offset || coord >= end) {
            in_bounds = false;
            break;
        }

        /* Check alignment: (coord - offset) must be divisible by step */
        len_t rel = coord - offset;
        if (step > 1 && (rel % step) != 0) {
            in_bounds = false;
            break;
        }

        /* Compute gradient index */
        len_t grad_coord = rel / step;
        grad_offset += grad_coord * params.grad_strides[d];
    }

    out[i] = in_bounds ? grad[grad_offset] : T(0);
}

/* ============================================================================
 * Stride Computation Helper
 * ============================================================================ */

static void compute_strides(const len_t* shape, len_t ndim, len_t* strides) {
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
 * Execute slice forward pass (gather from strided positions).
 *
 * out: Output tensor (contiguous, slice result shape)
 * in: Input tensor (source tensor)
 * offsets: Start offset for each dimension
 * steps: Step size for each dimension
 * stream: CUDA stream for async execution
 */
extern "C" void cuda_slice_forward(
    CudaDeviceHandle device,
    DenseCore* out,
    const DenseCore* in,
    const len_t* offsets,
    const len_t* steps,
    StreamHandle stream_handle
) {
    len_t ndim = in->shape.len;

    LOG_DEBUG("cuda_slice_forward: ndim=%lu out_numel=%lu in_numel=%lu",
              ndim, out->num_elements, in->num_elements);

    CUstream stream = static_cast<CUstream>(stream_handle.ptr);

    /* Build parameters struct (passed by value to kernel) */
    SliceGatherParams params = {};
    params.ndim = ndim;
    params.numel = out->num_elements;

    /* Copy shape arrays */
    for (len_t d = 0; d < ndim; d++) {
        params.out_shape[d] = out->shape.buffer[d];
        params.offsets[d] = offsets[d];
        params.steps[d] = steps[d];
    }

    /* Compute input strides */
    compute_strides(in->shape.buffer, ndim, params.in_strides);

    /* Launch kernel */
    const int block_size = 256;
    const int num_blocks = (out->num_elements + block_size - 1) / block_size;

    Dtype dtype = dense_core_dtype(out);

    if (dtype == DTYPE_F32) {
        kernel_slice_gather<float><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(out->data),
            static_cast<const float*>(in->data),
            params
        );
    } else if (dtype == DTYPE_F64) {
        kernel_slice_gather<double><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(out->data),
            static_cast<const double*>(in->data),
            params
        );
    } else if (dtype == DTYPE_F16) {
        kernel_slice_gather<f16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<f16*>(out->data),
            static_cast<const f16*>(in->data),
            params
        );
    } else if (dtype == DTYPE_BF16) {
        kernel_slice_gather<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out->data),
            static_cast<const __nv_bfloat16*>(in->data),
            params
        );
    } else {
        SYSTEM_EXIT("Unsupported dtype for slice_forward");
    }
}

/*
 * Execute slice backward pass (scatter gradients to strided positions).
 *
 * out: Output tensor (original tensor shape, will contain scattered gradients)
 * grad: Gradient tensor (slice result shape, contiguous)
 * offsets: Start offset for each dimension
 * steps: Step size for each dimension
 * stream: CUDA stream for async execution
 */
extern "C" void cuda_slice_backward(
    CudaDeviceHandle device,
    DenseCore* out,
    const DenseCore* grad,
    const len_t* offsets,
    const len_t* steps,
    StreamHandle stream_handle
) {
    len_t ndim = out->shape.len;

    LOG_DEBUG("cuda_slice_backward: ndim=%lu out_numel=%lu grad_numel=%lu",
              ndim, out->num_elements, grad->num_elements);

    CUstream stream = static_cast<CUstream>(stream_handle.ptr);

    /* Build parameters struct (passed by value to kernel) */
    SliceScatterParams params = {};
    params.ndim = ndim;
    params.numel = out->num_elements;

    /* Copy shape arrays */
    for (len_t d = 0; d < ndim; d++) {
        params.out_shape[d] = out->shape.buffer[d];
        params.grad_shape[d] = grad->shape.buffer[d];
        params.offsets[d] = offsets[d];
        params.steps[d] = steps[d];
    }

    /* Compute gradient strides */
    compute_strides(grad->shape.buffer, ndim, params.grad_strides);

    /* Launch kernel */
    const int block_size = 256;
    const int num_blocks = (out->num_elements + block_size - 1) / block_size;

    Dtype dtype = dense_core_dtype(out);

    if (dtype == DTYPE_F32) {
        kernel_slice_scatter<float><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(out->data),
            static_cast<const float*>(grad->data),
            params
        );
    } else if (dtype == DTYPE_F64) {
        kernel_slice_scatter<double><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(out->data),
            static_cast<const double*>(grad->data),
            params
        );
    } else if (dtype == DTYPE_F16) {
        kernel_slice_scatter<f16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<f16*>(out->data),
            static_cast<const f16*>(grad->data),
            params
        );
    } else if (dtype == DTYPE_BF16) {
        kernel_slice_scatter<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out->data),
            static_cast<const __nv_bfloat16*>(grad->data),
            params
        );
    } else {
        SYSTEM_EXIT("Unsupported dtype for slice_backward");
    }
}

#endif /* __MEMORY_SLICE_CU__ */
