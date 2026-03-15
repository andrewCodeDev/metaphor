/*
 * nn/selective_scan.cpp - Fused selective scan (SSM recurrence) forward and backward
 *
 * Computes the linear recurrence:
 *   h[t] = a_bar[t] * h[t-1] + b_bar_x[t]
 *
 * for all timesteps in parallel across (batch, d_inner) dimensions,
 * with a sequential loop over d_state inside each block.
 *
 * Input layout: [batch, seq_len, d_inner, d_state] (row-major)
 *
 * Forward:  h[t] = a_bar[t] * h[t-1] + b_bar_x[t], h[-1] = 0
 * Backward: dh[t] = grad_output[t] + a_bar[t+1] * dh[t+1]
 *           grad_b_bar_x[t] = dh[t]
 *           grad_a_bar[t]   = dh[t] * h[t-1]  (0 for t=0)
 *
 * Based on the selective scan kernel design from:
 *   Mamba: Linear-Time Sequence Modeling with Selective State Spaces
 *   Albert Gu and Tri Dao, 2023
 *   https://github.com/state-spaces/mamba
 *   Licensed under Apache 2.0
 *
 * Simplified: a_bar and b_bar_x are pre-computed by the tensor graph,
 * so this kernel only implements the raw recurrence.
 */

#ifndef __NN_SELECTIVE_SCAN_H__
#define __NN_SELECTIVE_SCAN_H__

#include "utils.cpp"

/* ============================================================================
 * Forward Kernel
 *
 * Grid:  dim3(batch, d_inner)
 * Block: 1 thread (sequential over seq_len, loop over d_state)
 *
 * Each thread handles one (batch, d_inner) lane, scanning seq_len steps
 * for all d_state indices. This is compute-light but eliminates ~86k
 * kernel launches from the graph-level Blelloch scan.
 * ============================================================================ */

template<typename T>
__global__ void selective_scan_forward_kernel(
    const T* __restrict__ a_bar,    // [batch, seq_len, d_inner, d_state]
    const T* __restrict__ b_bar_x,  // [batch, seq_len, d_inner, d_state]
    T* __restrict__ h_out,          // [batch, seq_len, d_inner, d_state]
    len_t seq_len,
    len_t d_inner,
    len_t d_state
) {
    const len_t b = blockIdx.x;       // batch index
    const len_t di = blockIdx.y * blockDim.x + threadIdx.x;  // d_inner index
    if (di >= d_inner) return;

    const len_t inner_stride = d_inner * d_state;  // stride for one timestep

    for (len_t ds = 0; ds < d_state; ds++) {
        float running = 0.0f;

        for (len_t t = 0; t < seq_len; t++) {
            // Index: [b, t, di, ds]
            len_t idx = b * seq_len * inner_stride + t * inner_stride + di * d_state + ds;

            float a = static_cast<float>(a_bar[idx]);
            float bx = static_cast<float>(b_bar_x[idx]);

            running = a * running + bx;
            h_out[idx] = static_cast<T>(running);
        }
    }
}

/* ============================================================================
 * Backward Kernel: grad_a_bar (child 0)
 *
 * Reverse scan computing dh[t], then:
 *   grad_a_bar[t] = dh[t] * h[t-1]  (0 for t=0)
 * ============================================================================ */

template<typename T>
__global__ void selective_scan_backward_a_bar_kernel(
    const T* __restrict__ grad_output,  // [batch, seq_len, d_inner, d_state]
    const T* __restrict__ a_bar,        // [batch, seq_len, d_inner, d_state]
    const T* __restrict__ h,            // [batch, seq_len, d_inner, d_state]
    T* __restrict__ grad_a_bar,         // [batch, seq_len, d_inner, d_state]
    len_t seq_len,
    len_t d_inner,
    len_t d_state
) {
    const len_t b = blockIdx.x;
    const len_t di = blockIdx.y * blockDim.x + threadIdx.x;
    if (di >= d_inner) return;

    const len_t inner_stride = d_inner * d_state;

    for (len_t ds = 0; ds < d_state; ds++) {
        float dh = 0.0f;

        for (len_t t_rev = 0; t_rev < seq_len; t_rev++) {
            len_t t = seq_len - 1 - t_rev;
            len_t idx = b * seq_len * inner_stride + t * inner_stride + di * d_state + ds;

            dh = static_cast<float>(grad_output[idx]) + dh;

            if (t > 0) {
                len_t prev_idx = b * seq_len * inner_stride + (t - 1) * inner_stride + di * d_state + ds;
                grad_a_bar[idx] = static_cast<T>(dh * static_cast<float>(h[prev_idx]));
            } else {
                grad_a_bar[idx] = static_cast<T>(0.0f);
            }

            dh = static_cast<float>(a_bar[idx]) * dh;
        }
    }
}

/* ============================================================================
 * Backward Kernel: grad_b_bar_x (child 1)
 *
 * Reverse scan computing dh[t], then:
 *   grad_b_bar_x[t] = dh[t]
 * ============================================================================ */

template<typename T>
__global__ void selective_scan_backward_b_bar_x_kernel(
    const T* __restrict__ grad_output,  // [batch, seq_len, d_inner, d_state]
    const T* __restrict__ a_bar,        // [batch, seq_len, d_inner, d_state]
    T* __restrict__ grad_b_bar_x,       // [batch, seq_len, d_inner, d_state]
    len_t seq_len,
    len_t d_inner,
    len_t d_state
) {
    const len_t b = blockIdx.x;
    const len_t di = blockIdx.y * blockDim.x + threadIdx.x;
    if (di >= d_inner) return;

    const len_t inner_stride = d_inner * d_state;

    for (len_t ds = 0; ds < d_state; ds++) {
        float dh = 0.0f;

        for (len_t t_rev = 0; t_rev < seq_len; t_rev++) {
            len_t t = seq_len - 1 - t_rev;
            len_t idx = b * seq_len * inner_stride + t * inner_stride + di * d_state + ds;

            dh = static_cast<float>(grad_output[idx]) + dh;
            grad_b_bar_x[idx] = static_cast<T>(dh);
            dh = static_cast<float>(a_bar[idx]) * dh;
        }
    }
}

/* ============================================================================
 * C Interface
 * ============================================================================ */

#define LAUNCH_SSCAN_FWD(T) do { \
    const int threads = 256; \
    dim3 grid(batch, (d_inner + threads - 1) / threads); \
    selective_scan_forward_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(a_bar), \
        static_cast<const T*>(b_bar_x), \
        static_cast<T*>(h_out), \
        seq_len, d_inner, d_state \
    ); \
} while(0)

extern "C" void hip_nn_selective_scan_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* a_bar,
    const void* b_bar_x,
    void* h_out,
    len_t batch,
    len_t seq_len,
    len_t d_inner,
    len_t d_state
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSCAN_FWD(f32);  break;
        case DTYPE_F64:  LAUNCH_SSCAN_FWD(f64);  break;
        case DTYPE_F16:  LAUNCH_SSCAN_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSCAN_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for selective_scan_forward");
    }
}

#define LAUNCH_SSCAN_BWD_A_BAR(T) do { \
    const int threads = 256; \
    dim3 grid(batch, (d_inner + threads - 1) / threads); \
    selective_scan_backward_a_bar_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(grad_output), \
        static_cast<const T*>(a_bar), \
        static_cast<const T*>(h), \
        static_cast<T*>(grad_a_bar), \
        seq_len, d_inner, d_state \
    ); \
} while(0)

extern "C" void hip_nn_selective_scan_backward_a_bar(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad_output,
    const void* a_bar,
    const void* h,
    void* grad_a_bar,
    len_t batch,
    len_t seq_len,
    len_t d_inner,
    len_t d_state
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSCAN_BWD_A_BAR(f32);  break;
        case DTYPE_F64:  LAUNCH_SSCAN_BWD_A_BAR(f64);  break;
        case DTYPE_F16:  LAUNCH_SSCAN_BWD_A_BAR(f16);  break;
        case DTYPE_BF16: LAUNCH_SSCAN_BWD_A_BAR(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for selective_scan_backward_a_bar");
    }
}

#define LAUNCH_SSCAN_BWD_B_BAR_X(T) do { \
    const int threads = 256; \
    dim3 grid(batch, (d_inner + threads - 1) / threads); \
    selective_scan_backward_b_bar_x_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(grad_output), \
        static_cast<const T*>(a_bar), \
        static_cast<T*>(grad_b_bar_x), \
        seq_len, d_inner, d_state \
    ); \
} while(0)

extern "C" void hip_nn_selective_scan_backward_b_bar_x(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad_output,
    const void* a_bar,
    void* grad_b_bar_x,
    len_t batch,
    len_t seq_len,
    len_t d_inner,
    len_t d_state
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSCAN_BWD_B_BAR_X(f32);  break;
        case DTYPE_F64:  LAUNCH_SSCAN_BWD_B_BAR_X(f64);  break;
        case DTYPE_F16:  LAUNCH_SSCAN_BWD_B_BAR_X(f16);  break;
        case DTYPE_BF16: LAUNCH_SSCAN_BWD_B_BAR_X(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for selective_scan_backward_b_bar_x");
    }
}

#endif /* __NN_SELECTIVE_SCAN_H__ */
