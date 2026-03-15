/*
 * nn/sampling.cu - GPU sampling operations for token generation
 *
 * Implements temperature scaling, top-k, top-p, and multinomial sampling on GPU.
 * Designed for autoregressive text generation (single token at a time).
 */

#ifndef __NN_SAMPLING_H__
#define __NN_SAMPLING_H__

#include "utils.cu"
#include <cub/cub.cuh>
#include <curand_kernel.h>

/* ============================================================================
 * Temperature Scaling Kernel
 *
 * Scales logits by 1/temperature in-place.
 * Lower temperature = sharper distribution, higher = more uniform.
 * ============================================================================ */

template<typename T>
__global__ void temperature_kernel(
    T* __restrict__ logits,
    len_t n,
    T inv_temperature
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        logits[i] *= inv_temperature;
    }
}

/* ============================================================================
 * Repetition Penalty Kernel
 *
 * Applies penalty to tokens that appear in context.
 * Positive logits are divided by penalty, negative are multiplied.
 * ============================================================================ */

template<typename T>
__global__ void repetition_penalty_kernel(
    T* __restrict__ logits,
    const uint32_t* __restrict__ context,
    len_t context_len,
    len_t vocab_size,
    T penalty
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < context_len) {
        uint32_t token = context[i];
        if (token < vocab_size) {
            T val = logits[token];
            if (val > T(0)) {
                logits[token] = val / penalty;
            } else {
                logits[token] = val * penalty;
            }
        }
    }
}

/* ============================================================================
 * Softmax Kernel (numerically stable)
 *
 * Single-block implementation for vocab-sized vectors.
 * Computes max, exp(x - max), sum, normalize.
 * ============================================================================ */

template<typename T>
__global__ void softmax_kernel(
    T* __restrict__ data,
    len_t n
) {
    extern __shared__ char smem[];
    T* sdata = reinterpret_cast<T*>(smem);

    // Phase 1: Find max
    T thread_max = -INFINITY;
    for (len_t i = threadIdx.x; i < n; i += blockDim.x) {
        thread_max = max(thread_max, data[i]);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Block reduce for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    T max_val = sdata[0];
    __syncthreads();

    // Phase 2: exp(x - max) and sum
    T thread_sum = T(0);
    for (len_t i = threadIdx.x; i < n; i += blockDim.x) {
        T val = exp(data[i] - max_val);
        data[i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Block reduce for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    T sum_val = sdata[0];
    __syncthreads();

    // Phase 3: Normalize
    T inv_sum = T(1) / sum_val;
    for (len_t i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] *= inv_sum;
    }
}

/* ============================================================================
 * Top-K Selection using CUB Radix Sort
 *
 * Sorts (value, index) pairs and keeps top-k.
 * Outputs the sorted probabilities and original indices.
 * ============================================================================ */

__global__ void init_indices_kernel(
    uint32_t* __restrict__ indices,
    len_t n
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = static_cast<uint32_t>(i);
    }
}

/* ============================================================================
 * Cumulative Sum for Top-P
 * ============================================================================ */

template<typename T>
__global__ void cumsum_kernel(
    const T* __restrict__ probs,
    T* __restrict__ cumsum,
    len_t n
) {
    // Single-threaded cumsum (fine for vocab size, runs on one thread)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        T sum = T(0);
        for (len_t i = 0; i < n; i++) {
            sum += probs[i];
            cumsum[i] = sum;
        }
    }
}

/* ============================================================================
 * Find Top-P Cutoff
 *
 * Returns the index where cumulative probability >= p.
 * ============================================================================ */

template<typename T>
__global__ void find_topp_cutoff_kernel(
    const T* __restrict__ cumsum,
    len_t n,
    T p,
    len_t* __restrict__ cutoff
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (len_t i = 0; i < n; i++) {
            if (cumsum[i] >= p) {
                *cutoff = i + 1;
                return;
            }
        }
        *cutoff = n;
    }
}

/* ============================================================================
 * Multinomial Sample Kernel
 *
 * Samples one token from probability distribution using random value.
 * ============================================================================ */

template<typename T>
__global__ void multinomial_sample_kernel(
    const T* __restrict__ probs,
    const uint32_t* __restrict__ indices,
    len_t n,
    T random_val,
    uint32_t* __restrict__ result
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        T cumsum = T(0);
        for (len_t i = 0; i < n; i++) {
            cumsum += probs[i];
            if (random_val < cumsum) {
                *result = indices[i];
                return;
            }
        }
        // Fallback to last token
        *result = indices[n - 1];
    }
}

/* ============================================================================
 * Argmax Kernel (for greedy decoding)
 * ============================================================================ */

template<typename T>
__global__ void argmax_kernel(
    const T* __restrict__ data,
    len_t n,
    uint32_t* __restrict__ result
) {
    extern __shared__ char smem[];
    T* smax = reinterpret_cast<T*>(smem);
    uint32_t* sidx = reinterpret_cast<uint32_t*>(smem + blockDim.x * sizeof(T));

    T thread_max = -INFINITY;
    uint32_t thread_idx = 0;

    for (len_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (data[i] > thread_max) {
            thread_max = data[i];
            thread_idx = static_cast<uint32_t>(i);
        }
    }

    smax[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_idx;
    __syncthreads();

    // Block reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (smax[threadIdx.x + s] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *result = sidx[0];
    }
}

/* ============================================================================
 * Renormalize Probabilities after Top-P cutoff
 * ============================================================================ */

template<typename T>
__global__ void renormalize_kernel(
    T* __restrict__ probs,
    len_t cutoff
) {
    extern __shared__ char smem[];
    T* sdata = reinterpret_cast<T*>(smem);

    // Sum probabilities
    T thread_sum = T(0);
    for (len_t i = threadIdx.x; i < cutoff; i += blockDim.x) {
        thread_sum += probs[i];
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Block reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    T sum_val = sdata[0];
    __syncthreads();

    // Normalize
    T inv_sum = T(1) / sum_val;
    for (len_t i = threadIdx.x; i < cutoff; i += blockDim.x) {
        probs[i] *= inv_sum;
    }
}

/* ============================================================================
 * Extern C API
 * ============================================================================ */

extern "C" void cuda_sampling_temperature(
    Dtype dtype,
    StreamHandle stream_handle,
    void* logits,
    len_t n,
    float temperature
) {
    cudaStream_t stream = cast_stream(stream_handle);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    if (dtype == DTYPE_F32) {
        float inv_temp = 1.0f / temperature;
        temperature_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(logits), n, inv_temp
        );
    } else {
        double inv_temp = 1.0 / static_cast<double>(temperature);
        temperature_kernel<double><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(logits), n, inv_temp
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_repetition_penalty(
    Dtype dtype,
    StreamHandle stream_handle,
    void* logits,
    const uint32_t* context,
    len_t context_len,
    len_t vocab_size,
    float penalty
) {
    cudaStream_t stream = cast_stream(stream_handle);
    const int block_size = 256;
    const int num_blocks = (context_len + block_size - 1) / block_size;

    if (dtype == DTYPE_F32) {
        repetition_penalty_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(logits), context, context_len, vocab_size, penalty
        );
    } else {
        repetition_penalty_kernel<double><<<num_blocks, block_size, 0, stream>>>(
            static_cast<double*>(logits), context, context_len, vocab_size,
            static_cast<double>(penalty)
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_softmax(
    Dtype dtype,
    StreamHandle stream_handle,
    void* data,
    len_t n
) {
    cudaStream_t stream = cast_stream(stream_handle);
    const int block_size = 256;
    size_t smem_size = block_size * sizeof(float);

    if (dtype == DTYPE_F32) {
        softmax_kernel<float><<<1, block_size, smem_size, stream>>>(
            static_cast<float*>(data), n
        );
    } else {
        smem_size = block_size * sizeof(double);
        softmax_kernel<double><<<1, block_size, smem_size, stream>>>(
            static_cast<double*>(data), n
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_argmax(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* data,
    len_t n,
    uint32_t* result
) {
    cudaStream_t stream = cast_stream(stream_handle);
    const int block_size = 256;
    size_t smem_size = block_size * (sizeof(float) + sizeof(uint32_t));

    if (dtype == DTYPE_F32) {
        argmax_kernel<float><<<1, block_size, smem_size, stream>>>(
            static_cast<const float*>(data), n, result
        );
    } else {
        smem_size = block_size * (sizeof(double) + sizeof(uint32_t));
        argmax_kernel<double><<<1, block_size, smem_size, stream>>>(
            static_cast<const double*>(data), n, result
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_init_indices(
    StreamHandle stream_handle,
    uint32_t* indices,
    len_t n
) {
    cudaStream_t stream = cast_stream(stream_handle);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    init_indices_kernel<<<num_blocks, block_size, 0, stream>>>(indices, n);
    CUDA_ASSERT(cudaPeekAtLastError());
}

/*
 * Top-K selection using CUB radix sort (descending).
 *
 * Sorts (key, value) pairs where keys are negated probabilities (for descending order)
 * and values are indices. After sort, the top-k elements are at the front.
 *
 * probs: Input probabilities (modified in-place to sorted order)
 * indices: Output indices (sorted by probability descending)
 * temp_storage: Workspace buffer
 * temp_storage_bytes: Size of workspace
 * n: Vocabulary size
 * k: Number of top elements to keep
 */
extern "C" size_t cuda_sampling_topk_workspace_size(len_t n) {
    size_t temp_storage_bytes = 0;
    // Query required workspace size
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes,
        static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<int>(n)
    );
    return temp_storage_bytes;
}

extern "C" void cuda_sampling_topk(
    Dtype dtype,
    StreamHandle stream_handle,
    void* probs,
    uint32_t* indices,
    void* probs_out,
    uint32_t* indices_out,
    void* temp_storage,
    size_t temp_storage_bytes,
    len_t n
) {
    cudaStream_t stream = cast_stream(stream_handle);

    if (dtype == DTYPE_F32) {
        cub::DeviceRadixSort::SortPairsDescending(
            temp_storage, temp_storage_bytes,
            static_cast<float*>(probs), static_cast<float*>(probs_out),
            indices, indices_out,
            static_cast<int>(n),
            0, sizeof(float) * 8,  // Sort all bits
            stream
        );
    } else {
        cub::DeviceRadixSort::SortPairsDescending(
            temp_storage, temp_storage_bytes,
            static_cast<double*>(probs), static_cast<double*>(probs_out),
            indices, indices_out,
            static_cast<int>(n),
            0, sizeof(double) * 8,
            stream
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_multinomial(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* probs,
    const uint32_t* indices,
    len_t n,
    float random_val,
    uint32_t* result
) {
    cudaStream_t stream = cast_stream(stream_handle);

    if (dtype == DTYPE_F32) {
        multinomial_sample_kernel<float><<<1, 1, 0, stream>>>(
            static_cast<const float*>(probs), indices, n, random_val, result
        );
    } else {
        multinomial_sample_kernel<double><<<1, 1, 0, stream>>>(
            static_cast<const double*>(probs), indices, n,
            static_cast<double>(random_val), result
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_cumsum(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* probs,
    void* cumsum,
    len_t n
) {
    cudaStream_t stream = cast_stream(stream_handle);

    if (dtype == DTYPE_F32) {
        cumsum_kernel<float><<<1, 1, 0, stream>>>(
            static_cast<const float*>(probs),
            static_cast<float*>(cumsum), n
        );
    } else {
        cumsum_kernel<double><<<1, 1, 0, stream>>>(
            static_cast<const double*>(probs),
            static_cast<double*>(cumsum), n
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_find_topp_cutoff(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* cumsum,
    len_t n,
    float p,
    len_t* cutoff
) {
    cudaStream_t stream = cast_stream(stream_handle);

    if (dtype == DTYPE_F32) {
        find_topp_cutoff_kernel<float><<<1, 1, 0, stream>>>(
            static_cast<const float*>(cumsum), n, p, cutoff
        );
    } else {
        find_topp_cutoff_kernel<double><<<1, 1, 0, stream>>>(
            static_cast<const double*>(cumsum), n, static_cast<double>(p), cutoff
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_sampling_renormalize(
    Dtype dtype,
    StreamHandle stream_handle,
    void* probs,
    len_t cutoff
) {
    cudaStream_t stream = cast_stream(stream_handle);
    const int block_size = 256;
    size_t smem_size = block_size * sizeof(float);

    if (dtype == DTYPE_F32) {
        renormalize_kernel<float><<<1, block_size, smem_size, stream>>>(
            static_cast<float*>(probs), cutoff
        );
    } else {
        smem_size = block_size * sizeof(double);
        renormalize_kernel<double><<<1, block_size, smem_size, stream>>>(
            static_cast<double*>(probs), cutoff
        );
    }
    CUDA_ASSERT(cudaPeekAtLastError());
}

#endif /* __NN_SAMPLING_H__ */
