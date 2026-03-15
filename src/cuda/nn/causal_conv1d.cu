/*
 * nn/causal_conv1d.cu - Depthwise causal 1D convolution forward and backward
 *
 * Based on the causal conv1d from the Mamba architecture:
 *   Albert Gu and Tri Dao, "Mamba: Linear-Time Sequence Modeling
 *   with Selective State Spaces", 2023
 *   https://github.com/state-spaces/mamba
 *   Licensed under Apache 2.0
 *
 * Computes depthwise causal conv1d (cross-correlation, no kernel flip):
 *   out[b,e,t] = bias[e] + sum_{k=0}^{K-1} weight[e,0,k] * x[b,e,t-(K-1)+k]
 *
 * where positions before t=0 are zero-padded (causal).
 *
 * Input layout:
 *   x:      [B, E, L]   - input activations
 *   weight: [E, 1, K]   - depthwise conv weights
 *   bias:   [E]          - per-channel bias
 *   output: [B, E, L]   - result
 *
 * Forward:         out[b,e,t] = bias[e] + sum_k weight[e,0,k] * x[b,e,t-(K-1)+k]
 * Backward x:      grad_x[b,e,t] = sum_k weight[e,0,K-1-k] * grad[b,e,t-(K-1)+k]  (flipped correlation)
 * Backward weight: grad_weight[e,0,k] = sum_{b,t} grad[b,e,t] * x[b,e,t-(K-1)+k]
 * Backward bias:   grad_bias[e] = sum_{b,t} grad[b,e,t]
 */

#ifndef __NN_CAUSAL_CONV1D_H__
#define __NN_CAUSAL_CONV1D_H__

#include "utils.cu"

/* ============================================================================
 * Forward Kernel
 *
 * Grid:  dim3(batch, ceil(channels/256))
 * Block: 256 threads
 * Each thread handles one (b, e) pair, loops over L with K taps.
 * ============================================================================ */

template<typename T>
__global__ void causal_conv1d_forward_kernel(
    const T* __restrict__ x,       // [B, E, L]
    const T* __restrict__ weight,  // [E, 1, K]
    const T* __restrict__ bias,    // [E]
    T* __restrict__ output,        // [B, E, L]
    len_t channels,
    len_t seq_len,
    len_t kernel_size
) {
    const len_t b = blockIdx.x;
    const len_t e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= channels) return;

    const float b_val = static_cast<float>(bias[e]);

    for (len_t t = 0; t < seq_len; t++) {
        float acc = b_val;
        for (len_t k = 0; k < kernel_size; k++) {
            // Input position: t - (K-1) + k
            len_t in_t = t - (kernel_size - 1) + k;
            // Causal: positions before 0 are zero
            if (in_t < seq_len) {  // unsigned: underflow wraps to huge value
                // Check it didn't underflow (in_t >= 0 in signed terms)
                if (t + k >= kernel_size - 1) {
                    float x_val = static_cast<float>(x[b * channels * seq_len + e * seq_len + in_t]);
                    float w_val = static_cast<float>(weight[e * kernel_size + k]);
                    acc += x_val * w_val;
                }
            }
        }
        output[b * channels * seq_len + e * seq_len + t] = static_cast<T>(acc);
    }
}

/* ============================================================================
 * Backward X Kernel
 *
 * grad_x[b,e,t] = sum_{k=0}^{K-1} weight[e,0,k] * grad[b,e,t+k-(K-1)+K-1-k]
 *
 * Simplified: for each output position t, we need to find which forward
 * outputs used x[b,e,t] and sum their contributions.
 * x[b,e,t] contributes to out[b,e,t'] for t' in [t, t+K-1] with weight[e,0,t'-t+K-1-(K-1)] = weight[e,0,t'-t]
 * Wait, let's re-derive:
 *   out[b,e,t'] = sum_k weight[e,0,k] * x[b,e,t'-(K-1)+k]
 * x[b,e,t] appears when t'-(K-1)+k = t, i.e. k = t - t' + K - 1
 * Valid when 0 <= k < K, so t' in [t, t+K-1], k = t - t' + K - 1
 * grad_x[b,e,t] = sum_{t'=t}^{min(t+K-1, L-1)} grad[b,e,t'] * weight[e,0,t-t'+K-1]
 * ============================================================================ */

template<typename T>
__global__ void causal_conv1d_backward_x_kernel(
    const T* __restrict__ grad,    // [B, E, L]
    const T* __restrict__ weight,  // [E, 1, K]
    T* __restrict__ grad_x,        // [B, E, L]
    len_t channels,
    len_t seq_len,
    len_t kernel_size
) {
    const len_t b = blockIdx.x;
    const len_t e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= channels) return;

    for (len_t t = 0; t < seq_len; t++) {
        float acc = 0.0f;
        // x[b,e,t] contributes to out at positions t' in [t, t+K-1]
        for (len_t dk = 0; dk < kernel_size; dk++) {
            len_t t_prime = t + dk;
            if (t_prime < seq_len) {
                // weight index = t - t' + K - 1 = K - 1 - dk
                float g_val = static_cast<float>(grad[b * channels * seq_len + e * seq_len + t_prime]);
                float w_val = static_cast<float>(weight[e * kernel_size + (kernel_size - 1 - dk)]);
                acc += g_val * w_val;
            }
        }
        grad_x[b * channels * seq_len + e * seq_len + t] = static_cast<T>(acc);
    }
}

/* ============================================================================
 * Backward Weight Kernel
 *
 * grad_weight[e,0,k] = sum_{b,t} grad[b,e,t] * x[b,e,t-(K-1)+k]
 * where t-(K-1)+k >= 0 and t-(K-1)+k < L
 *
 * Grid:  dim3(ceil(channels/256))
 * Block: 256 threads
 * Each thread accumulates over batch and seq_len for one channel.
 * ============================================================================ */

template<typename T>
__global__ void causal_conv1d_backward_weight_kernel(
    const T* __restrict__ grad,    // [B, E, L]
    const T* __restrict__ x,       // [B, E, L]
    T* __restrict__ grad_weight,   // [E, 1, K]
    len_t batch,
    len_t channels,
    len_t seq_len,
    len_t kernel_size
) {
    const len_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= channels) return;

    for (len_t k = 0; k < kernel_size; k++) {
        float acc = 0.0f;
        for (len_t b = 0; b < batch; b++) {
            for (len_t t = 0; t < seq_len; t++) {
                // Input position for this tap
                len_t in_t = t - (kernel_size - 1) + k;
                if (in_t < seq_len && t + k >= kernel_size - 1) {
                    float g_val = static_cast<float>(grad[b * channels * seq_len + e * seq_len + t]);
                    float x_val = static_cast<float>(x[b * channels * seq_len + e * seq_len + in_t]);
                    acc += g_val * x_val;
                }
            }
        }
        grad_weight[e * kernel_size + k] = static_cast<T>(acc);
    }
}

/* ============================================================================
 * Backward Bias Kernel
 *
 * grad_bias[e] = sum_{b,t} grad[b,e,t]
 *
 * Grid:  dim3(ceil(channels/256))
 * Block: 256 threads
 * ============================================================================ */

template<typename T>
__global__ void causal_conv1d_backward_bias_kernel(
    const T* __restrict__ grad,    // [B, E, L]
    T* __restrict__ grad_bias,     // [E]
    len_t batch,
    len_t channels,
    len_t seq_len
) {
    const len_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= channels) return;

    float acc = 0.0f;
    for (len_t b = 0; b < batch; b++) {
        for (len_t t = 0; t < seq_len; t++) {
            acc += static_cast<float>(grad[b * channels * seq_len + e * seq_len + t]);
        }
    }
    grad_bias[e] = static_cast<T>(acc);
}

/* ============================================================================
 * C Interface — Forward
 * ============================================================================ */

#define LAUNCH_CCONV1D_FWD(T) do { \
    const int threads = 256; \
    dim3 grid(batch, (channels + threads - 1) / threads); \
    causal_conv1d_forward_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(x), \
        static_cast<const T*>(weight), \
        static_cast<const T*>(bias), \
        static_cast<T*>(output), \
        channels, seq_len, kernel_size \
    ); \
} while(0)

extern "C" void cuda_nn_causal_conv1d_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* x,
    const void* weight,
    const void* bias,
    void* output,
    len_t batch,
    len_t channels,
    len_t seq_len,
    len_t kernel_size
) {
    cudaStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_CCONV1D_FWD(f32);  break;
        case DTYPE_F64:  LAUNCH_CCONV1D_FWD(f64);  break;
        case DTYPE_F16:  LAUNCH_CCONV1D_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_CCONV1D_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for causal_conv1d_forward");
    }
}

/* ============================================================================
 * C Interface — Backward X
 * ============================================================================ */

#define LAUNCH_CCONV1D_BWD_X(T) do { \
    const int threads = 256; \
    dim3 grid(batch, (channels + threads - 1) / threads); \
    causal_conv1d_backward_x_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(grad), \
        static_cast<const T*>(weight), \
        static_cast<T*>(grad_x), \
        channels, seq_len, kernel_size \
    ); \
} while(0)

extern "C" void cuda_nn_causal_conv1d_backward_x(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad,
    const void* weight,
    void* grad_x,
    len_t batch,
    len_t channels,
    len_t seq_len,
    len_t kernel_size
) {
    cudaStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_CCONV1D_BWD_X(f32);  break;
        case DTYPE_F64:  LAUNCH_CCONV1D_BWD_X(f64);  break;
        case DTYPE_F16:  LAUNCH_CCONV1D_BWD_X(f16);  break;
        case DTYPE_BF16: LAUNCH_CCONV1D_BWD_X(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for causal_conv1d_backward_x");
    }
}

/* ============================================================================
 * C Interface — Backward Weight
 * ============================================================================ */

#define LAUNCH_CCONV1D_BWD_W(T) do { \
    const int threads = 256; \
    dim3 grid((channels + threads - 1) / threads); \
    causal_conv1d_backward_weight_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(grad), \
        static_cast<const T*>(x), \
        static_cast<T*>(grad_weight), \
        batch, channels, seq_len, kernel_size \
    ); \
} while(0)

extern "C" void cuda_nn_causal_conv1d_backward_weight(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad,
    const void* x,
    void* grad_weight,
    len_t batch,
    len_t channels,
    len_t seq_len,
    len_t kernel_size
) {
    cudaStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_CCONV1D_BWD_W(f32);  break;
        case DTYPE_F64:  LAUNCH_CCONV1D_BWD_W(f64);  break;
        case DTYPE_F16:  LAUNCH_CCONV1D_BWD_W(f16);  break;
        case DTYPE_BF16: LAUNCH_CCONV1D_BWD_W(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for causal_conv1d_backward_weight");
    }
}

/* ============================================================================
 * C Interface — Backward Bias
 * ============================================================================ */

#define LAUNCH_CCONV1D_BWD_B(T) do { \
    const int threads = 256; \
    dim3 grid((channels + threads - 1) / threads); \
    causal_conv1d_backward_bias_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(grad), \
        static_cast<T*>(grad_bias), \
        batch, channels, seq_len \
    ); \
} while(0)

extern "C" void cuda_nn_causal_conv1d_backward_bias(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad,
    void* grad_bias,
    len_t batch,
    len_t channels,
    len_t seq_len
) {
    cudaStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_CCONV1D_BWD_B(f32);  break;
        case DTYPE_F64:  LAUNCH_CCONV1D_BWD_B(f64);  break;
        case DTYPE_F16:  LAUNCH_CCONV1D_BWD_B(f16);  break;
        case DTYPE_BF16: LAUNCH_CCONV1D_BWD_B(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for causal_conv1d_backward_bias");
    }
}

#endif /* __NN_CAUSAL_CONV1D_H__ */
