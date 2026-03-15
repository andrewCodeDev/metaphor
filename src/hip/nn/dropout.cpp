/*
 * nn/dropout.cpp - Inverted dropout forward and backward (HIP)
 *
 * Forward: generate Bernoulli mask from seed, apply y = x * mask / (1-p)
 * Backward: x_grad = y_grad * mask / (1-p)
 *
 * Uses Philox-style counter-based PRNG for reproducible mask generation.
 * The same seed + element index always produces the same mask bit,
 * so the backward pass can regenerate the mask without storing it.
 */

#ifndef __NN_DROPOUT_H__
#define __NN_DROPOUT_H__

#include "utils.cpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

/* ============================================================================
 * Philox-style hash for reproducible per-element randomness
 *
 * Maps (seed, index) -> uniform float in [0, 1).
 * Two rounds of multiply-xor-shift gives good statistical quality.
 * ============================================================================ */

__device__ __forceinline__ float philox_uniform(unsigned long long seed, unsigned long long idx) {
    unsigned long long key = seed;
    unsigned long long ctr = idx;

    // Round 1
    ctr ^= key;
    ctr *= 0x9E3779B97F4A7C15ULL;
    ctr ^= ctr >> 27;

    // Round 2
    ctr *= 0xBF58476D1CE4E5B9ULL;
    ctr ^= ctr >> 31;
    ctr *= 0x94D049BB133111EBULL;
    ctr ^= ctr >> 32;

    // Convert to [0, 1)
    return (float)(ctr >> 40) / (float)(1ULL << 24);
}

/* ============================================================================
 * Type conversion helpers (same as rms_norm.cpp)
 * ============================================================================ */

template<typename T>
__device__ __forceinline__ float to_float(T x);

template<> __device__ __forceinline__ float to_float(float x)  { return x; }
template<> __device__ __forceinline__ float to_float(double x) { return (float)x; }
template<> __device__ __forceinline__ float to_float(f16 x)    { return __half2float(x); }
template<> __device__ __forceinline__ float to_float(bf16 x)   { return (float)x; }

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<> __device__ __forceinline__ float  from_float(float x)  { return x; }
template<> __device__ __forceinline__ double from_float(float x)  { return (double)x; }
template<> __device__ __forceinline__ f16    from_float(float x)  { return __float2half(x); }
template<> __device__ __forceinline__ bf16   from_float(float x)  { return bf16(x); }

/* ============================================================================
 * Dropout Forward Kernel
 *
 * For each element i:
 *   mask[i] = (philox_uniform(seed, i) >= p) ? 1 : 0
 *   y[i] = x[i] * mask[i] / (1 - p)
 * ============================================================================ */

template<typename T>
__global__ void dropout_forward_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    unsigned char* __restrict__ mask,
    unsigned long long n,
    float p,
    unsigned long long seed
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = philox_uniform(seed, i);
    unsigned char keep = (r >= p) ? 1 : 0;
    mask[i] = keep;

    if (keep) {
        float scale = 1.0f / (1.0f - p);
        y[i] = from_float<T>(to_float(x[i]) * scale);
    } else {
        y[i] = from_float<T>(0.0f);
    }
}

/* ============================================================================
 * Dropout Backward Kernel
 *
 * x_grad[i] = y_grad[i] * mask[i] / (1 - p)
 * ============================================================================ */

template<typename T>
__global__ void dropout_backward_kernel(
    const T* __restrict__ y_grad,
    const unsigned char* __restrict__ mask,
    T* __restrict__ x_grad,
    unsigned long long n,
    float p
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (mask[i]) {
        float scale = 1.0f / (1.0f - p);
        x_grad[i] = from_float<T>(to_float(y_grad[i]) * scale);
    } else {
        x_grad[i] = from_float<T>(0.0f);
    }
}

/* ============================================================================
 * C Interface
 * ============================================================================ */

#define LAUNCH_DROPOUT_FWD(T) do { \
    const int block_size = 256; \
    const int num_blocks = (int)((n + block_size - 1) / block_size); \
    dropout_forward_kernel<T><<<num_blocks, block_size, 0, stream>>>( \
        static_cast<const T*>(x), \
        static_cast<T*>(y), \
        static_cast<unsigned char*>(mask), \
        n, p, seed \
    ); \
} while(0)

extern "C" void hip_nn_dropout_fwd(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* x,
    void* y,
    void* mask,
    unsigned long long n,
    float p,
    unsigned long long seed
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_DROPOUT_FWD(f32);  break;
        case DTYPE_F64:  LAUNCH_DROPOUT_FWD(f64);  break;
        case DTYPE_F16:  LAUNCH_DROPOUT_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_DROPOUT_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for dropout_fwd");
    }
}

#define LAUNCH_DROPOUT_BWD(T) do { \
    const int block_size = 256; \
    const int num_blocks = (int)((n + block_size - 1) / block_size); \
    dropout_backward_kernel<T><<<num_blocks, block_size, 0, stream>>>( \
        static_cast<const T*>(y_grd), \
        static_cast<const unsigned char*>(mask), \
        static_cast<T*>(x_grd), \
        n, p \
    ); \
} while(0)

extern "C" void hip_nn_dropout_bwd(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* y_grd,
    const void* mask,
    void* x_grd,
    unsigned long long n,
    float p
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_DROPOUT_BWD(f32);  break;
        case DTYPE_F64:  LAUNCH_DROPOUT_BWD(f64);  break;
        case DTYPE_F16:  LAUNCH_DROPOUT_BWD(f16);  break;
        case DTYPE_BF16: LAUNCH_DROPOUT_BWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for dropout_bwd");
    }
}

#endif /* __NN_DROPOUT_H__ */
