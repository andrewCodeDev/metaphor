/*
 * nn/softmax_kernels.h — Custom row-wise softmax kernels for HIP
 *
 * Warp-level persistent softmax (rows <= 2048) ported from PyTorch.
 * Block-level fallback softmax (rows > 2048) written for this project.
 *
 * All kernels accumulate in f32 internally for numerical stability with
 * f16/bf16 inputs. Cast on load, cast on store.
 *
 * ============================================================================
 * The warp-level kernels (softmax_warp_forward, softmax_warp_backward, and
 * the dispatch_softmax_forward/backward functions) are derived from:
 *
 *   PyTorch — aten/src/ATen/native/cuda/PersistentSoftmax.cuh
 *   https://github.com/pytorch/pytorch
 *
 * Original copyright and license:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#ifndef __NN_SOFTMAX_KERNELS_H__
#define __NN_SOFTMAX_KERNELS_H__

#include <cfloat>
#include <limits>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * Constants
 *
 * RDNA3 (gfx1100) uses wave32 by default (no -mwavefrontsize64).
 * ============================================================================ */

#ifndef METAPHOR_WARP_SIZE
#define METAPHOR_WARP_SIZE 32
#endif

/* ============================================================================
 * Warp-level reduction helpers (from PyTorch PersistentSoftmax.cuh)
 * ============================================================================ */

namespace softmax_detail {

template<typename T>
struct Add {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct Max {
    __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? b : a; }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            acc_t b = __shfl_xor(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

static inline int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

/* ============================================================================
 * Warp-level forward softmax kernel (rows <= 2048)
 *
 * Ported from PyTorch PersistentSoftmax.cuh — stripped masking support.
 * One warp processes WARP_BATCH rows. All math in acc_t (float).
 * ============================================================================ */

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(
    output_t *dst, const input_t *src,
    int batch_size, int stride, int element_count)
{
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < METAPHOR_WARP_SIZE) ? next_power_of_two : METAPHOR_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    int local_idx = threadIdx.x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    /* Load data from global memory into registers */
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0; i < WARP_BATCH; ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = static_cast<acc_t>(src[i * element_count + it * WARP_SIZE]);
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    /* Compute row max */
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
        }
    }
    softmax_detail::warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, softmax_detail::Max>(max_value);

    /* Compute exp(x - max) and sum */
    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        sum[i] = acc_t(0);
    }
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        #pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            if (is_log_softmax) {
                sum[i] += std::exp(elements[i][it] - max_value[i]);
            } else {
                elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                sum[i] += elements[i][it];
            }
        }
    }
    softmax_detail::warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, softmax_detail::Add>(sum);

    /* Store result */
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i * element_count + it * WARP_SIZE] =
                        static_cast<output_t>(elements[i][it] - max_value[i] - sum[i]);
                } else if (sum[i] == acc_t(0)) {
                    dst[i * element_count + it * WARP_SIZE] =
                        static_cast<output_t>(std::numeric_limits<acc_t>::quiet_NaN());
                } else {
                    dst[i * element_count + it * WARP_SIZE] =
                        static_cast<output_t>(elements[i][it] / sum[i]);
                }
            } else {
                break;
            }
        }
    }
}

/* ============================================================================
 * Warp-level backward softmax kernel (rows <= 1024)
 *
 * Ported from PyTorch PersistentSoftmax.cuh — stripped masking support.
 * ============================================================================ */

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(
    output_t *gradInput, const input_t *grad, const input_t *output,
    int batch_size, int stride, int element_count)
{
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < METAPHOR_WARP_SIZE) ? next_power_of_two : METAPHOR_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    int local_idx = threadIdx.x % WARP_SIZE;
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    /* Load data */
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0; i < WARP_BATCH; ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = static_cast<acc_t>(grad[i * element_count + it * WARP_SIZE]);
                output_reg[i][it] = static_cast<acc_t>(output[i * element_count + it * WARP_SIZE]);
            } else {
                grad_reg[i][it] = acc_t(0);
                output_reg[i][it] = acc_t(0);
            }
        }
    }

    /* Compute sum(grad * output) for softmax, or sum(grad) for log-softmax */
    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        sum[i] = acc_t(0);
    }
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        #pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            if (is_log_softmax) {
                sum[i] += grad_reg[i][it];
            } else {
                sum[i] += grad_reg[i][it] * output_reg[i][it];
            }
        }
    }
    softmax_detail::warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, softmax_detail::Add>(sum);

    /* Store gradients */
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    gradInput[i * element_count + it * WARP_SIZE] =
                        static_cast<output_t>(grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i * element_count + it * WARP_SIZE] =
                        static_cast<output_t>(output_reg[i][it] * (grad_reg[i][it] - sum[i]));
                }
            }
        }
    }
}

/* ============================================================================
 * Dispatch functions for warp-level kernels (from PyTorch)
 *
 * Selects the correct log2_elements template instantiation at runtime.
 * ============================================================================ */

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_warp_softmax_forward(
    output_t *dst, const input_t *src,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    hipStream_t stream)
{
    if (softmax_elements == 0) return;

    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    int warp_size = METAPHOR_WARP_SIZE;
    warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    constexpr int threads_per_block = 128;
    int warps_per_block = threads_per_block / warp_size;
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    #define LAUNCH_WARP_SOFTMAX_FWD(L2E) case L2E: \
        softmax_warp_forward<input_t, output_t, acc_t, L2E, is_log_softmax> \
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements); \
        break;

    switch (log2_elements) {
        LAUNCH_WARP_SOFTMAX_FWD(0)   // 1
        LAUNCH_WARP_SOFTMAX_FWD(1)   // 2
        LAUNCH_WARP_SOFTMAX_FWD(2)   // 4
        LAUNCH_WARP_SOFTMAX_FWD(3)   // 8
        LAUNCH_WARP_SOFTMAX_FWD(4)   // 16
        LAUNCH_WARP_SOFTMAX_FWD(5)   // 32
        LAUNCH_WARP_SOFTMAX_FWD(6)   // 64
        LAUNCH_WARP_SOFTMAX_FWD(7)   // 128
        LAUNCH_WARP_SOFTMAX_FWD(8)   // 256
        LAUNCH_WARP_SOFTMAX_FWD(9)   // 512
        LAUNCH_WARP_SOFTMAX_FWD(10)  // 1024
        LAUNCH_WARP_SOFTMAX_FWD(11)  // 2048
        default: break;
    }
    #undef LAUNCH_WARP_SOFTMAX_FWD
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_warp_softmax_backward(
    output_t *grad_input, const input_t *grad, const input_t *output,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    hipStream_t stream)
{
    if (softmax_elements == 0) return;

    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    int warp_size = METAPHOR_WARP_SIZE;
    warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    constexpr int threads_per_block = 128;
    int warps_per_block = threads_per_block / warp_size;
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    #define LAUNCH_WARP_SOFTMAX_BWD(L2E) case L2E: \
        softmax_warp_backward<input_t, output_t, acc_t, L2E, is_log_softmax> \
            <<<blocks, threads, 0, stream>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements); \
        break;

    switch (log2_elements) {
        LAUNCH_WARP_SOFTMAX_BWD(0)   // 1
        LAUNCH_WARP_SOFTMAX_BWD(1)   // 2
        LAUNCH_WARP_SOFTMAX_BWD(2)   // 4
        LAUNCH_WARP_SOFTMAX_BWD(3)   // 8
        LAUNCH_WARP_SOFTMAX_BWD(4)   // 16
        LAUNCH_WARP_SOFTMAX_BWD(5)   // 32
        LAUNCH_WARP_SOFTMAX_BWD(6)   // 64
        LAUNCH_WARP_SOFTMAX_BWD(7)   // 128
        LAUNCH_WARP_SOFTMAX_BWD(8)   // 256
        LAUNCH_WARP_SOFTMAX_BWD(9)   // 512
        LAUNCH_WARP_SOFTMAX_BWD(10)  // 1024
        default: break;
    }
    #undef LAUNCH_WARP_SOFTMAX_BWD
}

/* ============================================================================
 * Block-level softmax kernels (for rows > 2048 forward / > 1024 backward)
 *
 * Simple 3-pass shared-memory reduction. One block per row.
 * Not ported from PyTorch — written for this project.
 * ============================================================================ */

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    /* Warp-level reduction */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset, 32);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    /* First warp reduces across warps */
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_xor(val, offset, 32);
        }
    }
    /* Broadcast via shared memory */
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor(val, offset, 32);
        val = val > other ? val : other;
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : -INFINITY;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_xor(val, offset, 32);
            val = val > other ? val : other;
        }
    }
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

template<typename T>
__global__ void block_softmax_forward(const T* x, T* y, int n) {
    const int row = blockIdx.x;
    const T* x_row = x + (int64_t)row * n;
    T* y_row = y + (int64_t)row * n;

    /* Pass 1: find max */
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_max = fmaxf(thread_max, static_cast<float>(x_row[i]));
    }
    float row_max = block_reduce_max(thread_max);

    /* Pass 2: exp(x - max) and sum */
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += expf(static_cast<float>(x_row[i]) - row_max);
    }
    float row_sum = block_reduce_sum(thread_sum);

    /* Pass 3: normalize */
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y_row[i] = static_cast<T>(expf(static_cast<float>(x_row[i]) - row_max) * inv_sum);
    }
}

template<typename T>
__global__ void block_log_softmax_forward(const T* x, T* y, int n) {
    const int row = blockIdx.x;
    const T* x_row = x + (int64_t)row * n;
    T* y_row = y + (int64_t)row * n;

    /* Pass 1: find max */
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_max = fmaxf(thread_max, static_cast<float>(x_row[i]));
    }
    float row_max = block_reduce_max(thread_max);

    /* Pass 2: sum of exp(x - max) */
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += expf(static_cast<float>(x_row[i]) - row_max);
    }
    float row_sum = block_reduce_sum(thread_sum);
    float log_sum = logf(row_sum);

    /* Pass 3: y = x - max - log(sum) */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y_row[i] = static_cast<T>(static_cast<float>(x_row[i]) - row_max - log_sum);
    }
}

template<typename T>
__global__ void block_softmax_backward(
    const T* y_val, const T* y_grd, T* x_grd, int n)
{
    const int row = blockIdx.x;
    const T* yv = y_val + (int64_t)row * n;
    const T* yg = y_grd + (int64_t)row * n;
    T* xg = x_grd + (int64_t)row * n;

    /* sum(y * dy) */
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += static_cast<float>(yv[i]) * static_cast<float>(yg[i]);
    }
    float dot = block_reduce_sum(thread_sum);

    /* dx = y * (dy - dot) */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float y_f = static_cast<float>(yv[i]);
        float dy_f = static_cast<float>(yg[i]);
        xg[i] = static_cast<T>(y_f * (dy_f - dot));
    }
}

template<typename T>
__global__ void block_log_softmax_backward(
    const T* y_val, const T* y_grd, T* x_grd, int n)
{
    const int row = blockIdx.x;
    const T* yv = y_val + (int64_t)row * n;
    const T* yg = y_grd + (int64_t)row * n;
    T* xg = x_grd + (int64_t)row * n;

    /* sum(dy) */
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += static_cast<float>(yg[i]);
    }
    float sum_dy = block_reduce_sum(thread_sum);

    /* dx = dy - exp(y) * sum(dy) */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float y_f = static_cast<float>(yv[i]);
        float dy_f = static_cast<float>(yg[i]);
        xg[i] = static_cast<T>(dy_f - expf(y_f) * sum_dy);
    }
}

} /* namespace softmax_detail */

/* ============================================================================
 * Top-level dispatch: picks warp-level or block-level kernel
 *
 * Warp-level: rows <= 2048 (fwd) / <= 1024 (bwd) — register-only, fast
 * Block-level: rows > 2048 (fwd) / > 1024 (bwd) — shared memory, general
 * ============================================================================ */

template<typename T>
static void custom_softmax_forward(
    const T* x, T* y, int m, int n, bool is_log, hipStream_t stream)
{
    using namespace softmax_detail;

    if (n <= 2048) {
        if (is_log) {
            dispatch_warp_softmax_forward<T, T, float, true>(y, x, n, n, m, stream);
        } else {
            dispatch_warp_softmax_forward<T, T, float, false>(y, x, n, n, m, stream);
        }
    } else {
        constexpr int BLOCK = 256;
        if (is_log) {
            block_log_softmax_forward<T><<<m, BLOCK, 0, stream>>>(x, y, n);
        } else {
            block_softmax_forward<T><<<m, BLOCK, 0, stream>>>(x, y, n);
        }
    }
}

template<typename T>
static void custom_softmax_backward(
    const T* y_val, const T* y_grd, T* x_grd, int m, int n,
    bool is_log, hipStream_t stream)
{
    using namespace softmax_detail;

    if (n <= 1024) {
        if (is_log) {
            dispatch_warp_softmax_backward<T, T, float, true>(x_grd, y_grd, y_val, n, n, m, stream);
        } else {
            dispatch_warp_softmax_backward<T, T, float, false>(x_grd, y_grd, y_val, n, n, m, stream);
        }
    } else {
        constexpr int BLOCK = 256;
        if (is_log) {
            block_log_softmax_backward<T><<<m, BLOCK, 0, stream>>>(y_val, y_grd, x_grd, n);
        } else {
            block_softmax_backward<T><<<m, BLOCK, 0, stream>>>(y_val, y_grd, x_grd, n);
        }
    }
}

#endif /* __NN_SOFTMAX_KERNELS_H__ */
