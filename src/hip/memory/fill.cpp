/*
 * memory/fill.cpp - Memory fill, sequence, and random operations
 */


#ifndef __MEMORY_FILL_CU__
#define __MEMORY_FILL_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/dtype_dispatch.h"
#include "../core/includes.h"
#include "../interop.h"

/* ============================================================================
 * hipRAND-based Random Number Generation
 *
 * Uses hipRAND for GPU random number generation. Supports F32, F64, F16, BF16.
 * F16/BF16 are generated as F32 and converted.
 *
 * Note: hipRAND uniform generates [0,1), we transform to [-1,1] for compatibility.
 * ============================================================================ */

#define HIPRAND_BLOCK_SIZE 256

/* Convert F32 to target type */
template <typename T>
__device__ __forceinline__ T convert_from_f32(f32 val);

template <> __device__ __forceinline__ f32 convert_from_f32<f32>(f32 val) { return val; }
template <> __device__ __forceinline__ f16 convert_from_f32<f16>(f32 val) { return __float2half(val); }
template <> __device__ __forceinline__ bf16 convert_from_f32<bf16>(f32 val) { return bf16(val); }

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
    hiprandGenerator_t gen,
    T* out,
    len_t n,
    RandType op,
    hipStream_t stream
) {
    f32* tmp;
    HIP_ASSERT(hipMalloc(&tmp, n * sizeof(f32)));

    const len_t blocks = (n + HIPRAND_BLOCK_SIZE - 1) / HIPRAND_BLOCK_SIZE;

    if (op == RAND_UNIFORM) {
        HIPRAND_ASSERT(hiprandGenerateUniform(gen, tmp, n));
        convert_uniform_kernel<T><<<blocks, HIPRAND_BLOCK_SIZE, 0, stream>>>(tmp, out, n);
    } else {
        HIPRAND_ASSERT(hiprandGenerateNormal(gen, tmp, n, 0.0f, 1.0f));
        convert_normal_kernel<T><<<blocks, HIPRAND_BLOCK_SIZE, 0, stream>>>(tmp, out, n);
    }

    HIP_ASSERT(hipFree(tmp));
}

/* Generate scaled normal F32 values into temp buffer, then convert */
template <typename T>
void generate_scaled_normal_via_f32(
    hiprandGenerator_t gen,
    T* out,
    len_t n,
    f32 scale,
    hipStream_t stream
) {
    f32* tmp;
    HIP_ASSERT(hipMalloc(&tmp, n * sizeof(f32)));

    HIPRAND_ASSERT(hiprandGenerateNormal(gen, tmp, n, 0.0f, scale));

    const len_t blocks = (n + HIPRAND_BLOCK_SIZE - 1) / HIPRAND_BLOCK_SIZE;
    convert_normal_kernel<T><<<blocks, HIPRAND_BLOCK_SIZE, 0, stream>>>(tmp, out, n);

    HIP_ASSERT(hipFree(tmp));
}

/* ============================================================================
 * Fill Operations
 * ============================================================================ */

extern "C" void hip_mem_fill(
    HipDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    HostPtr value
) {
    hipStream_t stream = unwrap_device(device)->stream;

    switch (dtype) {
        case DTYPE_F32: {
            f32* iter = static_cast<f32*>(unwrap(data));
            f32 val = *static_cast<const f32*>(unwrap(value));
            thrust::fill(thrust::hip::par.on(stream), iter, iter + n, val);
            break;
        }
        case DTYPE_F64: {
            f64* iter = static_cast<f64*>(unwrap(data));
            f64 val = *static_cast<const f64*>(unwrap(value));
            thrust::fill(thrust::hip::par.on(stream), iter, iter + n, val);
            break;
        }
        case DTYPE_F16: {
            f16* iter = static_cast<f16*>(unwrap(data));
            /* C3 passes a float* — convert to f16 on the host side */
            f16 val = __float2half(*static_cast<const f32*>(unwrap(value)));
            thrust::fill(thrust::hip::par.on(stream), iter, iter + n, val);
            break;
        }
        case DTYPE_BF16: {
            bf16* iter = static_cast<bf16*>(unwrap(data));
            /* C3 passes a float* — convert to bf16 on the host side */
            bf16 val = bf16(*static_cast<const f32*>(unwrap(value)));
            thrust::fill(thrust::hip::par.on(stream), iter, iter + n, val);
            break;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for mem_fill");
    }
}

/* ============================================================================
 * Scalar Broadcast (device-to-device)
 *
 * Reads a single element from src and fills dst with that value.
 * Entirely on-device — no host round-trip.
 * ============================================================================ */

template <typename T>
__global__ void kernel_broadcast_scalar(const T* __restrict__ src, T* __restrict__ dst, len_t n) {
    const T val = src[0];
    for (len_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = val;
    }
}

extern "C" void hip_mem_broadcast_scalar(
    HipDeviceHandle device,
    Dtype dtype,
    DevicePtr src,
    DevicePtr dst,
    len_t n
) {
    hipStream_t stream = unwrap_device(device)->stream;
    const int block = 256;
    const int grid = std::min((n + block - 1) / block, (len_t)65535);

    DISPATCH_DTYPE(dtype,
        kernel_broadcast_scalar<scalar_t><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(unwrap(src)),
            static_cast<scalar_t*>(unwrap(dst)), n));
}

/* ============================================================================
 * Sequence Operations
 * ============================================================================ */

extern "C" void hip_mem_sequence(
    HipDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    HostPtr init,
    HostPtr step
) {
    hipStream_t stream = unwrap_device(device)->stream;

    switch (dtype) {
        case DTYPE_F32: {
            f32* iter = static_cast<f32*>(unwrap(data));
            f32 init_val = *static_cast<const f32*>(unwrap(init));
            f32 step_val = *static_cast<const f32*>(unwrap(step));
            thrust::sequence(thrust::hip::par.on(stream), iter, iter + n, init_val, step_val);
            break;
        }
        case DTYPE_F64: {
            f64* iter = static_cast<f64*>(unwrap(data));
            f64 init_val = *static_cast<const f64*>(unwrap(init));
            f64 step_val = *static_cast<const f64*>(unwrap(step));
            thrust::sequence(thrust::hip::par.on(stream), iter, iter + n, init_val, step_val);
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
 * Random Fill Operations (hipRAND-based)
 * ============================================================================ */

extern "C" void hip_mem_random(
    HipDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    RandType op,
    unsigned seed
) {
    hipStream_t stream = unwrap_device(device)->stream;

    hiprandGenerator_t gen;
    HIPRAND_ASSERT(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_ASSERT(hiprandSetPseudoRandomGeneratorSeed(gen, seed));
    HIPRAND_ASSERT(hiprandSetStream(gen, stream));

    switch (dtype) {
        case DTYPE_F32:
            generate_random_via_f32<f32>(gen, static_cast<f32*>(unwrap(data)), n, op, stream);
            break;
        case DTYPE_F64: {
            /* F64 has native hipRAND double support - no temp buffer needed */
            f64* out = static_cast<f64*>(unwrap(data));
            if (op == RAND_UNIFORM) {
                HIPRAND_ASSERT(hiprandGenerateUniformDouble(gen, out, n));
                const len_t blocks = (n + HIPRAND_BLOCK_SIZE - 1) / HIPRAND_BLOCK_SIZE;
                transform_uniform_f64_kernel<<<blocks, HIPRAND_BLOCK_SIZE, 0, stream>>>(out, n);
            } else {
                HIPRAND_ASSERT(hiprandGenerateNormalDouble(gen, out, n, 0.0, 1.0));
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
            HIPRAND_ASSERT(hiprandDestroyGenerator(gen));
            SYSTEM_EXIT("Unsupported dtype for mem_random");
    }

    HIPRAND_ASSERT(hiprandDestroyGenerator(gen));
    HIP_ASSERT(hipPeekAtLastError());
}

extern "C" void hip_mem_random_scaled_normal(
    HipDeviceHandle device,
    Dtype dtype,
    DevicePtr data,
    len_t n,
    f64 scale,
    unsigned seed
) {
    hipStream_t stream = unwrap_device(device)->stream;

    hiprandGenerator_t gen;
    HIPRAND_ASSERT(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_ASSERT(hiprandSetPseudoRandomGeneratorSeed(gen, seed));
    HIPRAND_ASSERT(hiprandSetStream(gen, stream));

    switch (dtype) {
        case DTYPE_F32:
            /* F32 can generate directly with scale as stddev */
            HIPRAND_ASSERT(hiprandGenerateNormal(gen, static_cast<f32*>(unwrap(data)), n, 0.0f, static_cast<f32>(scale)));
            break;
        case DTYPE_F64:
            HIPRAND_ASSERT(hiprandGenerateNormalDouble(gen, static_cast<f64*>(unwrap(data)), n, 0.0, scale));
            break;
        case DTYPE_F16:
            generate_scaled_normal_via_f32<f16>(gen, static_cast<f16*>(unwrap(data)), n, static_cast<f32>(scale), stream);
            break;
        case DTYPE_BF16:
            generate_scaled_normal_via_f32<bf16>(gen, static_cast<bf16*>(unwrap(data)), n, static_cast<f32>(scale), stream);
            break;
        default:
            HIPRAND_ASSERT(hiprandDestroyGenerator(gen));
            SYSTEM_EXIT("Unsupported dtype for mem_random_scaled_normal");
    }

    HIPRAND_ASSERT(hiprandDestroyGenerator(gen));
    HIP_ASSERT(hipPeekAtLastError());
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
    hipStream_t stream
) {
    const T* src_ptr = static_cast<const T*>(src);
    T* dst_ptr = static_cast<T*>(dst);
    thrust::counting_iterator<len_t> stencil(0);

    thrust::transform(
        thrust::hip::par.on(stream),
        stencil, stencil + idxs_len, dst_ptr,
        [=] __device__(len_t i) -> T { return src_ptr[idxs[i]]; }
    );
}

extern "C" void hip_mem_take(
    HipDeviceHandle device,
    Dtype dtype,
    DevicePtr src,
    len_t src_len,
    DevicePtr idxs,
    len_t idxs_len,
    DevicePtr dst
) {
    hipStream_t stream = unwrap_device(device)->stream;
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
