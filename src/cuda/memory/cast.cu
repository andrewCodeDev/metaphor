/*
 * memory/cast.cu - Dtype casting kernels
 *
 * Converts tensor data between different dtypes (f32, f64, f16, bf16).
 * Uses thrust::transform for efficient GPU-parallel conversion.
 */

#ifndef __MEMORY_CAST_CU__
#define __MEMORY_CAST_CU__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"

/* ============================================================================
 * Template Cast Implementation
 * ============================================================================ */

template <typename SrcT, typename DstT>
static void cast_impl(
    cudaStream_t stream,
    const void* src,
    void* dst,
    len_t n
) {
    const SrcT* src_iter = static_cast<const SrcT*>(src);
    DstT* dst_iter = static_cast<DstT*>(dst);

    thrust::transform(
        thrust::cuda::par.on(stream),
        src_iter, src_iter + n,
        dst_iter,
        [] __device__ (SrcT a) -> DstT { return static_cast<DstT>(a); }
    );
}

/* ============================================================================
 * Dispatch Macros
 *
 * Generate all source-to-destination type combinations.
 * ============================================================================ */

#define DISPATCH_DST(stream, src_type, src_enum, src, dst, n) \
    case src_enum: \
        switch (dst_dtype) { \
            case DTYPE_F32:  cast_impl<src_type, f32>(stream, src, dst, n); return; \
            case DTYPE_F64:  cast_impl<src_type, f64>(stream, src, dst, n); return; \
            case DTYPE_F16:  cast_impl<src_type, f16>(stream, src, dst, n); return; \
            case DTYPE_BF16: cast_impl<src_type, bf16>(stream, src, dst, n); return; \
            default: SYSTEM_EXIT("Unsupported destination dtype for cast"); \
        }

/* ============================================================================
 * Public API
 * ============================================================================ */

extern "C" void cuda_mem_cast(
    StreamHandle w,
    Dtype src_dtype,
    Dtype dst_dtype,
    const void* src,
    void* dst,
    len_t n
) {
    cudaStream_t stream = cast_stream(w.ptr);

    // No-op if same dtype - just copy
    if (src_dtype == dst_dtype) {
        size_t element_size;
        switch (src_dtype) {
            case DTYPE_F32:  element_size = sizeof(f32); break;
            case DTYPE_F64:  element_size = sizeof(f64); break;
            case DTYPE_F16:  element_size = sizeof(f16); break;
            case DTYPE_BF16: element_size = sizeof(bf16); break;
            default: SYSTEM_EXIT("Unsupported dtype for cast");
        }
        CUDA_ASSERT(cudaMemcpyAsync(dst, src, n * element_size, cudaMemcpyDeviceToDevice, stream));
        return;
    }

    switch (src_dtype) {
        DISPATCH_DST(stream, f32, DTYPE_F32, src, dst, n)
        DISPATCH_DST(stream, f64, DTYPE_F64, src, dst, n)
        DISPATCH_DST(stream, f16, DTYPE_F16, src, dst, n)
        DISPATCH_DST(stream, bf16, DTYPE_BF16, src, dst, n)
        default:
            SYSTEM_EXIT("Unsupported source dtype for cast");
    }
}

#undef DISPATCH_DST

#endif /* __MEMORY_CAST_CU__ */
