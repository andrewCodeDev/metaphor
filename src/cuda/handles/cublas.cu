/*
 * handles/cublas.cu - Standalone cuBLAS handle management
 *
 * For cases where a cuBLAS handle is needed without the full device context.
 */


#ifndef __HANDLES_CUBLAS_CU__
#define __HANDLES_CUBLAS_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * cuBLAS Handle Wrapper
 * ============================================================================ */

/* CublasHandle is defined in interop.h */

static inline cublasHandle_t unwrap_cublas(CublasHandle h) {
    return static_cast<cublasHandle_t>(h.ptr);
}

static inline CublasHandle wrap_cublas(cublasHandle_t h) {
    return {.ptr = h};
}

/* ============================================================================
 * cuBLAS Lifecycle
 * ============================================================================ */

extern "C" CublasHandle cuda_cublas_create() {
    cublasHandle_t handle;
    CUBLAS_ASSERT(cublasCreate(&handle));
    return wrap_cublas(handle);
}

extern "C" void cuda_cublas_destroy(CublasHandle handle) {
    CUBLAS_ASSERT(cublasDestroy(unwrap_cublas(handle)));
}

/* ============================================================================
 * cuBLAS Stream Binding
 * ============================================================================ */

extern "C" void cuda_cublas_set_stream(CublasHandle handle, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    CUBLAS_ASSERT(cublasSetStream(unwrap_cublas(handle), cuda_stream));
}

extern "C" void* cuda_cublas_get_stream(CublasHandle handle) {
    cudaStream_t stream;
    CUBLAS_ASSERT(cublasGetStream(unwrap_cublas(handle), &stream));
    return stream;
}

/* ============================================================================
 * cuBLAS Math Mode
 * ============================================================================ */

extern "C" void cuda_cublas_set_math_mode(CublasHandle handle, unsigned mode) {
    CUBLAS_ASSERT(cublasSetMathMode(unwrap_cublas(handle), static_cast<cublasMath_t>(mode)));
}

#endif /* __HANDLES_CUBLAS_CU__ */
