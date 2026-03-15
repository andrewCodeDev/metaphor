/*
 * memory/slice.cpp - Strided copy operations for slice forward/backward
 *
 * Forward (gather): Copy from strided source positions to contiguous output
 * Backward (scatter): Copy from contiguous gradient to strided output positions
 *
 * NOTE: All parameters are passed by value to ensure HIP graph capture compatibility.
 * No hipMalloc/hipFree operations during kernel execution.
 */

#ifndef __MEMORY_SLICE_CU__
#define __MEMORY_SLICE_CU__

#include "../core/assert.h"
#include "../core/dtype_dispatch.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"

/* ============================================================================
 * Slice Parameters Structure (passed by value)
 * ============================================================================ */

struct SliceGatherParams {
    len_t out_shape[HIP_MAX_DIMS];
    len_t in_strides[HIP_MAX_DIMS];
    len_t offsets[HIP_MAX_DIMS];
    len_t steps[HIP_MAX_DIMS];
    len_t ndim;
    len_t numel;
};

struct SliceScatterParams {
    len_t out_shape[HIP_MAX_DIMS];
    len_t grad_shape[HIP_MAX_DIMS];
    len_t grad_strides[HIP_MAX_DIMS];
    len_t offsets[HIP_MAX_DIMS];
    len_t steps[HIP_MAX_DIMS];
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
 * stream: HIP stream for async execution
 */
extern "C" void hip_slice_forward(
    HipDeviceHandle device,
    DenseCore* out,
    const DenseCore* in,
    const len_t* offsets,
    const len_t* steps,
    StreamHandle stream_handle
) {
    len_t ndim = in->shape.len;

    LOG_DEBUG("hip_slice_forward: ndim=%lu out_numel=%lu in_numel=%lu",
              ndim, out->num_elements, in->num_elements);

    hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);

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

    DISPATCH_DTYPE(dtype,
        kernel_slice_gather<scalar_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<scalar_t*>(out->data),
            static_cast<const scalar_t*>(in->data), params));
}

/*
 * Execute slice backward pass (scatter gradients to strided positions).
 *
 * out: Output tensor (original tensor shape, will contain scattered gradients)
 * grad: Gradient tensor (slice result shape, contiguous)
 * offsets: Start offset for each dimension
 * steps: Step size for each dimension
 * stream: HIP stream for async execution
 */
extern "C" void hip_slice_backward(
    HipDeviceHandle device,
    DenseCore* out,
    const DenseCore* grad,
    const len_t* offsets,
    const len_t* steps,
    StreamHandle stream_handle
) {
    len_t ndim = out->shape.len;

    LOG_DEBUG("hip_slice_backward: ndim=%lu out_numel=%lu grad_numel=%lu",
              ndim, out->num_elements, grad->num_elements);

    hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);

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

    DISPATCH_DTYPE(dtype,
        kernel_slice_scatter<scalar_t><<<num_blocks, block_size, 0, stream>>>(
            static_cast<scalar_t*>(out->data),
            static_cast<const scalar_t*>(grad->data), params));
}

#endif /* __MEMORY_SLICE_CU__ */
