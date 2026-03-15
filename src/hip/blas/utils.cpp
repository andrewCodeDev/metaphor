/*
 * blas/utils.cpp - BLAS utility functions and types
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
 * These inline functions cast opaque handle types to their HIP equivalents.
 * StreamHandle and HipblasHandle are defined in interop.h
 * ============================================================================ */

static inline hipStream_t cast_stream(StreamHandle w) {
    return static_cast<hipStream_t>(w.ptr);
}

static inline hipblasHandle_t cast_hipblas(HipblasHandle w) {
    return static_cast<hipblasHandle_t>(w.ptr);
}

static inline hipStream_t stream_from_hipblas(HipblasHandle w) {
    hipStream_t stream;
    HIPBLAS_ASSERT(hipblasGetStream(cast_hipblas(w), &stream));
    return stream;
}

#endif /* __BLAS_UTILS_H__ */
