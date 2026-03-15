/*
 * blas/utils.cu - BLAS utility functions and types
 */

#ifndef __BLAS_UTILS_H__
#define __BLAS_UTILS_H__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"

/* ============================================================================
 * Handle Casts
 *
 * These inline functions cast opaque handle types to their CUDA equivalents.
 * StreamHandle and CublasHandle are defined in interop.h
 * ============================================================================ */

static inline cudaStream_t cast_stream(StreamHandle w) {
    return static_cast<cudaStream_t>(w.ptr);
}

static inline cublasHandle_t cast_cublas(CublasHandle w) {
    return static_cast<cublasHandle_t>(w.ptr);
}

static inline cudaStream_t stream_from_cublas(CublasHandle w) {
    cudaStream_t stream;
    CUBLAS_ASSERT(cublasGetStream(cast_cublas(w), &stream));
    return stream;
}

#endif /* __BLAS_UTILS_H__ */
