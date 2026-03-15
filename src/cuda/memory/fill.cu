/*
 * memory/fill.cu - Memory fill, sequence, and random operations
 */


#ifndef __MEMORY_FILL_CU__
#define __MEMORY_FILL_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"

/* ============================================================================
 * cuRAND-based Random Number Generation
 *
 * Uses cuRAND for GPU random number generation. Supports F32, F64, F16, BF16.
 * F16/BF16 are generated as F32 and converted.
 *
 * Note: cuRAND uniform generates [0,1), we transform to [-1,1] for compatibility.
 * ============================================================================ */

#define CURAND_BLOCK_SIZE 256

/* Convert F32 to target type */
template <typename T>
__device__ __forceinline__ T convert_from_f32(f32 val);

template <> __device__ __forceinline__ f32 convert_from_f32<f32>(f32 val) { return val; }
template <> __device__ __forceinline__ f16 convert_from_f32<f16>(f32 val) { return __float2half(val); }
template <> __device__ __forceinline__ bf16 convert_from_f32<bf16>(f32 val) { return __float2bfloat16(val); }

/* Convert F64 to target type */
template <typename T>
__device__ __forceinline__ T convert_from_f64(f64 val);

template <> __device__ __forceinline__ f64 convert_from_f64<f64>(f64 val) { return val; }

/* Kernel to convert and transform uniform [0,1) -> [-1,1] */
template <typename T>
__global__ void convert_uniform_kernel(const f32* src, T* dst, len_t n) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = convert_from_f32<T>(2.0f * src[i] - 1.0f);
    }
}

/* Kernel to convert normal values (no transform needed) */
template <typename T>
__global__ void convert_normal_kernel(const f32* src, T* dst, len_t n) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = convert_from_f32<T>(src[i]);
    }
}

/* F64 uniform transform (in-place, no conversion needed) */
__global__ void transform_uniform_f64_kernel(f64* data, len_t n) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 2.0 * data[i] - 1.0;
    }
}

/* Generate random F32 values into temp buffer, then convert to target type */
template <typename T>
void generate_random_via_f32(
    curandGenerator_t gen,
    T* out,
    len_t n,
    RandType op,
    cudaStream_t stream
) {
    f32* tmp;
    CUDA_ASSERT(cudaMalloc(&tmp, n * sizeof(f32)));

    const len_t blocks = (n + CURAND_BLOCK_SIZE - 1) / CURAND_BLOCK_SIZE;

    if (op == RAND_UNIFORM) {
        CURAND_ASSERT(curandGenerateUniform(gen, tmp, n));
        convert_uniform_kernel<T><<<blocks, CURAND_BLOCK_SIZE, 0, stream>>>(tmp, out, n);
    } else {
        CURAND_ASSERT(curandGenerateNormal(gen, tmp, n, 0.0f, 1.0f));
        convert_normal_kernel<T><<<blocks, CURAND_BLOCK_SIZE, 0, stream>>>(tmp, out, n);
    }

    CUDA_ASSERT(cudaFree(tmp));
}

/* Generate scaled normal F32 values into temp buffer, then convert */
template <typename T>
void generate_scaled_normal_via_f32(
    curandGenerator_t gen,
    T* out,
    len_t n,
    f32 scale,
    cudaStream_t stream
) {
    f32* tmp;
    CUDA_ASSERT(cudaMalloc(&tmp, n * sizeof(f32)));

    CURAND_ASSERT(curandGenerateNormal(gen, tmp, n, 0.0f, scale));

    const len_t blocks = (n + CURAND_BLOCK_SIZE - 1) / CURAND_BLOCK_SIZE;
    convert_normal_kernel<T><<<blocks, CURAND_BLOCK_SIZE, 0, stream>>>(tmp, out, n);

    CUDA_ASSERT(cudaFree(tmp));
}

/* ============================================================================
 * Fill Operations
 * ============================================================================ */

extern "C" void cuda_mem_fill(
    CudaDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    HostPtr value
) {
    cudaStream_t stream = unwrap_device(device)->stream;

    switch (dtype) {
        case DTYPE_F32: {
            f32* iter = static_cast<f32*>(unwrap(data));
            f32 val = *static_cast<const f32*>(unwrap(value));
            thrust::fill(thrust::cuda::par.on(stream), iter, iter + n, val);
            break;
        }
        case DTYPE_F64: {
            f64* iter = static_cast<f64*>(unwrap(data));
            f64 val = *static_cast<const f64*>(unwrap(value));
            thrust::fill(thrust::cuda::par.on(stream), iter, iter + n, val);
            break;
        }
        case DTYPE_F16: {
            f16* iter = static_cast<f16*>(unwrap(data));
            f32 f32_val = *static_cast<const f32*>(unwrap(value));
            f16 val = __float2half(f32_val);
            thrust::fill(thrust::cuda::par.on(stream), iter, iter + n, val);
            break;
        }
        case DTYPE_BF16: {
            bf16* iter = static_cast<bf16*>(unwrap(data));
            f32 f32_val = *static_cast<const f32*>(unwrap(value));
            bf16 val = __float2bfloat16(f32_val);
            thrust::fill(thrust::cuda::par.on(stream), iter, iter + n, val);
            break;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for mem_fill");
    }
}

/* ============================================================================
 * Sequence Operations
 * ============================================================================ */

extern "C" void cuda_mem_sequence(
    CudaDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    HostPtr init,
    HostPtr step
) {
    cudaStream_t stream = unwrap_device(device)->stream;

    switch (dtype) {
        case DTYPE_F32: {
            f32* iter = static_cast<f32*>(unwrap(data));
            f32 init_val = *static_cast<const f32*>(unwrap(init));
            f32 step_val = *static_cast<const f32*>(unwrap(step));
            thrust::sequence(thrust::cuda::par.on(stream), iter, iter + n, init_val, step_val);
            break;
        }
        case DTYPE_F64: {
            f64* iter = static_cast<f64*>(unwrap(data));
            f64 init_val = *static_cast<const f64*>(unwrap(init));
            f64 step_val = *static_cast<const f64*>(unwrap(step));
            thrust::sequence(thrust::cuda::par.on(stream), iter, iter + n, init_val, step_val);
            break;
        }
        case DTYPE_F16:
            SYSTEM_EXIT("F16 sequence not supported - use cast from f32");
        case DTYPE_BF16:
            SYSTEM_EXIT("BF16 sequence not supported - use cast from f32");
        default:
            SYSTEM_EXIT("Unsupported dtype for mem_sequence");
    }
}

/* ============================================================================
 * Random Fill Operations (cuRAND-based)
 * ============================================================================ */

extern "C" void cuda_mem_random(
    CudaDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    RandType op,
    unsigned seed
) {
    cudaStream_t stream = unwrap_device(device)->stream;

    curandGenerator_t gen;
    CURAND_ASSERT(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ASSERT(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_ASSERT(curandSetStream(gen, stream));

    switch (dtype) {
        case DTYPE_F32:
            generate_random_via_f32<f32>(gen, static_cast<f32*>(unwrap(data)), n, op, stream);
            break;
        case DTYPE_F64: {
            /* F64 has native cuRAND double support - no temp buffer needed */
            f64* out = static_cast<f64*>(unwrap(data));
            if (op == RAND_UNIFORM) {
                CURAND_ASSERT(curandGenerateUniformDouble(gen, out, n));
                const len_t blocks = (n + CURAND_BLOCK_SIZE - 1) / CURAND_BLOCK_SIZE;
                transform_uniform_f64_kernel<<<blocks, CURAND_BLOCK_SIZE, 0, stream>>>(out, n);
            } else {
                CURAND_ASSERT(curandGenerateNormalDouble(gen, out, n, 0.0, 1.0));
            }
            break;
        }
        case DTYPE_F16:
            generate_random_via_f32<f16>(gen, static_cast<f16*>(unwrap(data)), n, op, stream);
            break;
        case DTYPE_BF16:
            generate_random_via_f32<bf16>(gen, static_cast<bf16*>(unwrap(data)), n, op, stream);
            break;
        default:
            CURAND_ASSERT(curandDestroyGenerator(gen));
            SYSTEM_EXIT("Unsupported dtype for mem_random");
    }

    CURAND_ASSERT(curandDestroyGenerator(gen));
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_mem_random_scaled_normal(
    CudaDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    f64 scale,
    unsigned seed
) {
    cudaStream_t stream = unwrap_device(device)->stream;

    curandGenerator_t gen;
    CURAND_ASSERT(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ASSERT(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_ASSERT(curandSetStream(gen, stream));

    switch (dtype) {
        case DTYPE_F32:
            /* F32 can generate directly with scale as stddev */
            CURAND_ASSERT(curandGenerateNormal(gen, static_cast<f32*>(unwrap(data)), n, 0.0f, static_cast<f32>(scale)));
            break;
        case DTYPE_F64:
            CURAND_ASSERT(curandGenerateNormalDouble(gen, static_cast<f64*>(unwrap(data)), n, 0.0, scale));
            break;
        case DTYPE_F16:
            generate_scaled_normal_via_f32<f16>(gen, static_cast<f16*>(unwrap(data)), n, static_cast<f32>(scale), stream);
            break;
        case DTYPE_BF16:
            generate_scaled_normal_via_f32<bf16>(gen, static_cast<bf16*>(unwrap(data)), n, static_cast<f32>(scale), stream);
            break;
        default:
            CURAND_ASSERT(curandDestroyGenerator(gen));
            SYSTEM_EXIT("Unsupported dtype for mem_random_scaled_normal");
    }

    CURAND_ASSERT(curandDestroyGenerator(gen));
    CUDA_ASSERT(cudaPeekAtLastError());
}

/* ============================================================================
 * Gather (Take) Operations
 * ============================================================================ */

template <typename T>
void mem_take_impl(
    const void* src,
    const len_t* idxs,
    len_t idxs_len,
    void* dst,
    cudaStream_t stream
) {
    const T* src_ptr = static_cast<const T*>(src);
    T* dst_ptr = static_cast<T*>(dst);
    thrust::counting_iterator<len_t> stencil(0);

    thrust::transform(
        thrust::cuda::par.on(stream),
        stencil, stencil + idxs_len, dst_ptr,
        [=] __device__(len_t i) -> T { return src_ptr[idxs[i]]; }
    );
}

extern "C" void cuda_mem_take(
    CudaDeviceHandle device,
    Dtype dtype,
    DevicePtr src,
    len_t src_len,
    DevicePtr idxs,
    len_t idxs_len,
    DevicePtr dst
) {
    cudaStream_t stream = unwrap_device(device)->stream;
    const len_t* idx_ptr = static_cast<const len_t*>(unwrap(idxs));

    switch (dtype) {
        case DTYPE_F32:
            mem_take_impl<f32>(unwrap(src), idx_ptr, idxs_len, unwrap(dst), stream);
            break;
        case DTYPE_F64:
            mem_take_impl<f64>(unwrap(src), idx_ptr, idxs_len, unwrap(dst), stream);
            break;
        case DTYPE_F16:
            mem_take_impl<f16>(unwrap(src), idx_ptr, idxs_len, unwrap(dst), stream);
            break;
        case DTYPE_BF16:
            mem_take_impl<bf16>(unwrap(src), idx_ptr, idxs_len, unwrap(dst), stream);
            break;
        default:
            SYSTEM_EXIT("Unsupported dtype for mem_take");
    }
}

#endif /* __MEMORY_FILL_CU__ */
