/*
 * handles/hipblas.cpp - Standalone hipBLAS handle management
 *
 * For cases where a hipBLAS handle is needed without the full device context.
 */


#ifndef __HANDLES_HIPBLAS_CPP__
#define __HANDLES_HIPBLAS_CPP__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * hipBLAS Handle Wrapper
 * ============================================================================ */

/* HipblasHandle is defined in interop.h */

static inline hipblasHandle_t unwrap_hipblas(HipblasHandle h) {
    return static_cast<hipblasHandle_t>(h.ptr);
}

static inline HipblasHandle wrap_hipblas(hipblasHandle_t h) {
    return {.ptr = h};
}

/* ============================================================================
 * hipBLAS Lifecycle
 * ============================================================================ */

extern "C" HipblasHandle hip_hipblas_create() {
    hipblasHandle_t handle;
    HIPBLAS_ASSERT(hipblasCreate(&handle));
    return wrap_hipblas(handle);
}

extern "C" void hip_hipblas_destroy(HipblasHandle handle) {
    HIPBLAS_ASSERT(hipblasDestroy(unwrap_hipblas(handle)));
}

/* ============================================================================
 * hipBLAS Stream Binding
 * ============================================================================ */

extern "C" void hip_hipblas_set_stream(HipblasHandle handle, void* stream) {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    HIPBLAS_ASSERT(hipblasSetStream(unwrap_hipblas(handle), hip_stream));
}

extern "C" void* hip_hipblas_get_stream(HipblasHandle handle) {
    hipStream_t stream;
    HIPBLAS_ASSERT(hipblasGetStream(unwrap_hipblas(handle), &stream));
    return stream;
}

/* ============================================================================
 * hipBLAS Math Mode
 * ============================================================================ */

extern "C" void hip_hipblas_set_math_mode(HipblasHandle handle, unsigned mode) {
    HIPBLAS_ASSERT(hipblasSetMathMode(unwrap_hipblas(handle), static_cast<hipblasMath_t>(mode)));
}

#endif /* __HANDLES_HIPBLAS_CPP__ */
