/*
 * nn/rms_norm.cpp - RMS Normalization forward and backward
 *
 * Implements RMS normalization with f32 internal accumulation for numerical
 * stability when using f16/bf16 inputs. This prevents overflow in the backward
 * pass where division by small values (rms^2) can exceed f16 range.
 *
 * Formula:
 *   rms = sqrt(mean(x^2) + eps)
 *   y = (x / rms) * weight
 *
 * Backward:
 *   grad_weight = sum over (batch, seq) of (grad_y * x / rms)
 *   grad_x = (1/rms) * (grad_y * weight - (x/rms) * mean(grad_y * weight * x / rms))
 */

#ifndef __NN_RMS_NORM_H__
#define __NN_RMS_NORM_H__

#include "utils.cpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

/* ============================================================================
 * Type conversion helpers for f32 accumulation
 * ============================================================================ */

template<typename T>
__device__ __forceinline__ f32 to_f32(T x);

template<>
__device__ __forceinline__ f32 to_f32(f32 x) { return x; }

template<>
__device__ __forceinline__ f32 to_f32(f64 x) { return static_cast<f32>(x); }

template<>
__device__ __forceinline__ f32 to_f32(f16 x) { return __half2float(x); }

template<>
__device__ __forceinline__ f32 to_f32(bf16 x) { return static_cast<f32>(x); }

template<typename T>
__device__ __forceinline__ T from_f32(f32 x);

template<>
__device__ __forceinline__ f32 from_f32(f32 x) { return x; }

template<>
__device__ __forceinline__ f64 from_f32(f32 x) { return static_cast<f64>(x); }

template<>
__device__ __forceinline__ f16 from_f32(f32 x) { return __float2half(x); }

template<>
__device__ __forceinline__ bf16 from_f32(f32 x) { return bf16(x); }

/* ============================================================================
 * Warp-level reduction
 * ============================================================================ */

__device__ __forceinline__ f32 warp_reduce_sum(f32 val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffffffffffffULL, val, offset);
    }
    return val;
}

/* ============================================================================
 * RMS Norm Forward Kernel
 *
 * Each block processes one row (batch*seq position).
 * Uses f32 accumulation for mean(x^2) computation.
 * ============================================================================ */

template<typename T>
__global__ void rms_norm_forward_kernel(
    const T* __restrict__ x,      // [batch*seq, d_model]
    const T* __restrict__ weight, // [d_model]
    T* __restrict__ y,            // [batch*seq, d_model]
    f32* __restrict__ rms_out,    // [batch*seq] - saved for backward
    len_t d_model,
    f32 eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const T* x_row = x + row * d_model;
    T* y_row = y + row * d_model;

    // Step 1: Compute sum of squares in f32
    f32 sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += block_size) {
        f32 val = to_f32(x_row[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);

    // Block reduction via shared memory
    __shared__ f32 shared[32]; // One per warp
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum_sq = (lane_id < (block_size + 31) / 32) ? shared[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    // Broadcast rms to all threads
    __shared__ f32 rms_shared;
    if (tid == 0) {
        f32 mean_sq = sum_sq / static_cast<f32>(d_model);
        rms_shared = sqrtf(mean_sq + eps);
        rms_out[row] = rms_shared;
    }
    __syncthreads();

    f32 rms = rms_shared;
    f32 inv_rms = 1.0f / rms;

    // Step 2: Normalize and scale
    for (int i = tid; i < d_model; i += block_size) {
        f32 x_val = to_f32(x_row[i]);
        f32 w_val = to_f32(weight[i]);
        f32 y_val = (x_val * inv_rms) * w_val;
        y_row[i] = from_f32<T>(y_val);
    }
}

/* ============================================================================
 * RMS Norm Backward Kernel for grad_x
 *
 * Each block processes one row.
 * Computes RMS internally from x (no need to save from forward pass).
 * grad_x = (1/rms) * (grad_y * weight - (x/rms) * mean(grad_y * weight * x / rms))
 * ============================================================================ */

template<typename T>
__global__ void rms_norm_backward_x_kernel(
    const T* __restrict__ grad_y,   // [batch*seq, d_model]
    const T* __restrict__ x,        // [batch*seq, d_model]
    const T* __restrict__ weight,   // [d_model]
    T* __restrict__ grad_x,         // [batch*seq, d_model]
    len_t d_model,
    f32 eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const T* grad_y_row = grad_y + row * d_model;
    const T* x_row = x + row * d_model;
    T* grad_x_row = grad_x + row * d_model;

    // Step 1: Compute sum of squares for RMS (recompute from x)
    f32 sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += block_size) {
        f32 val = to_f32(x_row[i]);
        sum_sq += val * val;
    }

    // Warp reduction for sum_sq
    sum_sq = warp_reduce_sum(sum_sq);

    // Block reduction via shared memory
    __shared__ f32 shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (block_size + 31) / 32) ? shared[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    // Broadcast RMS to all threads
    __shared__ f32 inv_rms_shared;
    if (tid == 0) {
        f32 mean_sq = sum_sq / static_cast<f32>(d_model);
        f32 rms_val = sqrtf(mean_sq + eps);
        inv_rms_shared = 1.0f / rms_val;
    }
    __syncthreads();

    f32 inv_rms = inv_rms_shared;

    // Step 2: Compute sum of (grad_y * weight * x / rms) for the correction term
    // This is: sum(g * h) where g = grad_y * weight, h = x / rms
    f32 sum_gh = 0.0f;
    for (int i = tid; i < d_model; i += block_size) {
        f32 gy = to_f32(grad_y_row[i]);
        f32 w = to_f32(weight[i]);
        f32 xv = to_f32(x_row[i]);
        f32 g = gy * w;
        f32 h = xv * inv_rms;
        sum_gh += g * h;
    }

    // Warp reduction
    sum_gh = warp_reduce_sum(sum_gh);

    // Block reduction via shared memory (reuse shared)
    if (lane_id == 0) {
        shared[warp_id] = sum_gh;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_gh = (lane_id < (block_size + 31) / 32) ? shared[lane_id] : 0.0f;
        sum_gh = warp_reduce_sum(sum_gh);
    }

    // Broadcast mean to all threads
    __shared__ f32 mean_gh_shared;
    if (tid == 0) {
        mean_gh_shared = sum_gh / static_cast<f32>(d_model);
    }
    __syncthreads();

    f32 mean_gh = mean_gh_shared;

    // Step 3: Compute grad_x = (1/rms) * (g - h * mean_gh)
    for (int i = tid; i < d_model; i += block_size) {
        f32 gy = to_f32(grad_y_row[i]);
        f32 w = to_f32(weight[i]);
        f32 xv = to_f32(x_row[i]);
        f32 g = gy * w;
        f32 h = xv * inv_rms;
        f32 gx = inv_rms * (g - h * mean_gh);
        grad_x_row[i] = from_f32<T>(gx);
    }
}

/* ============================================================================
 * RMS Computation Kernel (helper for backward passes)
 *
 * Computes RMS for each row into a buffer. One block per row.
 * ============================================================================ */

template<typename T>
__global__ void rms_norm_compute_rms_kernel(
    const T* __restrict__ x,        // [batch*seq, d_model]
    f32* __restrict__ rms_out,      // [batch*seq]
    len_t d_model,
    f32 eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const T* x_row = x + row * d_model;

    // Compute sum of squares in f32
    f32 sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += block_size) {
        f32 val = to_f32(x_row[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);

    // Block reduction via shared memory
    __shared__ f32 shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (block_size + 31) / 32) ? shared[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    if (tid == 0) {
        f32 mean_sq = sum_sq / static_cast<f32>(d_model);
        rms_out[row] = sqrtf(mean_sq + eps);
    }
}

/* ============================================================================
 * RMS Norm Backward Kernel for grad_weight
 *
 * Reduction over batch*seq dimension.
 * Uses pre-computed RMS values from compute_rms kernel.
 * grad_weight[i] = sum over rows of (grad_y[row, i] * x[row, i] / rms[row])
 * ============================================================================ */

template<typename T>
__global__ void rms_norm_backward_weight_kernel(
    const T* __restrict__ grad_y,   // [batch*seq, d_model]
    const T* __restrict__ x,        // [batch*seq, d_model]
    const f32* __restrict__ rms,    // [batch*seq]
    T* __restrict__ grad_weight,    // [d_model]
    len_t num_rows,
    len_t d_model
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model) return;

    f32 sum = 0.0f;
    for (int row = 0; row < num_rows; row++) {
        f32 gy = to_f32(grad_y[row * d_model + i]);
        f32 xv = to_f32(x[row * d_model + i]);
        f32 inv_rms = 1.0f / rms[row];
        sum += gy * xv * inv_rms;
    }

    grad_weight[i] = from_f32<T>(sum);
}

/* ============================================================================
 * C Interface
 * ============================================================================ */

#define LAUNCH_RMS_NORM_FWD(T) do { \
    const int block_size = (d_model < 1024) ? ((d_model + 31) / 32 * 32) : 1024; \
    rms_norm_forward_kernel<T><<<num_rows, block_size, 0, stream>>>( \
        static_cast<const T*>(x), \
        static_cast<const T*>(weight), \
        static_cast<T*>(y), \
        static_cast<f32*>(rms_out), \
        d_model, \
        eps \
    ); \
} while(0)

extern "C" void hip_nn_rms_norm_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* x,
    const void* weight,
    void* y,
    void* rms_out,
    len_t num_rows,
    len_t d_model,
    f32 eps
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_RMS_NORM_FWD(f32);  break;
        case DTYPE_F64:  LAUNCH_RMS_NORM_FWD(f64);  break;
        case DTYPE_F16:  LAUNCH_RMS_NORM_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_RMS_NORM_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for rms_norm_forward");
    }
}

#define LAUNCH_RMS_NORM_BWD_X(T) do { \
    const int block_size = (d_model < 1024) ? ((d_model + 31) / 32 * 32) : 1024; \
    rms_norm_backward_x_kernel<T><<<num_rows, block_size, 0, stream>>>( \
        static_cast<const T*>(grad_y), \
        static_cast<const T*>(x), \
        static_cast<const T*>(weight), \
        static_cast<T*>(grad_x), \
        d_model, \
        eps \
    ); \
} while(0)

extern "C" void hip_nn_rms_norm_backward_x(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad_y,
    const void* x,
    const void* weight,
    void* grad_x,
    len_t num_rows,
    len_t d_model,
    f32 eps
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_RMS_NORM_BWD_X(f32);  break;
        case DTYPE_F64:  LAUNCH_RMS_NORM_BWD_X(f64);  break;
        case DTYPE_F16:  LAUNCH_RMS_NORM_BWD_X(f16);  break;
        case DTYPE_BF16: LAUNCH_RMS_NORM_BWD_X(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for rms_norm_backward_x");
    }
}

#define LAUNCH_RMS_NORM_COMPUTE_RMS(T) do { \
    const int block_size = (d_model < 1024) ? ((d_model + 31) / 32 * 32) : 1024; \
    rms_norm_compute_rms_kernel<T><<<num_rows, block_size, 0, stream>>>( \
        static_cast<const T*>(x), \
        static_cast<f32*>(rms_out), \
        d_model, \
        eps \
    ); \
} while(0)

extern "C" void hip_nn_rms_norm_compute_rms(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* x,
    void* rms_out,
    len_t num_rows,
    len_t d_model,
    f32 eps
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_RMS_NORM_COMPUTE_RMS(f32);  break;
        case DTYPE_F64:  LAUNCH_RMS_NORM_COMPUTE_RMS(f64);  break;
        case DTYPE_F16:  LAUNCH_RMS_NORM_COMPUTE_RMS(f16);  break;
        case DTYPE_BF16: LAUNCH_RMS_NORM_COMPUTE_RMS(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for rms_norm_compute_rms");
    }
}

#define LAUNCH_RMS_NORM_BWD_W(T) do { \
    const int block_size = 256; \
    const int num_blocks = (d_model + block_size - 1) / block_size; \
    rms_norm_backward_weight_kernel<T><<<num_blocks, block_size, 0, stream>>>( \
        static_cast<const T*>(grad_y), \
        static_cast<const T*>(x), \
        static_cast<const f32*>(rms), \
        static_cast<T*>(grad_weight), \
        num_rows, \
        d_model \
    ); \
} while(0)

extern "C" void hip_nn_rms_norm_backward_weight(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad_y,
    const void* x,
    const void* rms,
    void* grad_weight,
    len_t num_rows,
    len_t d_model
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_RMS_NORM_BWD_W(f32);  break;
        case DTYPE_F64:  LAUNCH_RMS_NORM_BWD_W(f64);  break;
        case DTYPE_F16:  LAUNCH_RMS_NORM_BWD_W(f16);  break;
        case DTYPE_BF16: LAUNCH_RMS_NORM_BWD_W(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for rms_norm_backward_weight");
    }
}

#endif /* __NN_RMS_NORM_H__ */
