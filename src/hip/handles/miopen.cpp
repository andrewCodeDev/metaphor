/*
 * handles/miopen.cpp - Standalone MIOpen handle management
 *
 * For cases where a MIOpen handle is needed without the full device context.
 */


#ifndef __HANDLES_MIOPEN_CPP__
#define __HANDLES_MIOPEN_CPP__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * MIOpen Handle Wrapper
 * ============================================================================ */

/* MiopenHandle is defined in interop.h */

static inline miopenHandle_t unwrap_miopen(MiopenHandle h) {
    return static_cast<miopenHandle_t>(h.ptr);
}

static inline MiopenHandle wrap_miopen(miopenHandle_t h) {
    return {.ptr = h};
}

/* ============================================================================
 * MIOpen Lifecycle
 * ============================================================================ */

extern "C" MiopenHandle hip_miopen_create() {
    miopenHandle_t handle;
    MIOPEN_ASSERT(miopenCreate(&handle));
    return wrap_miopen(handle);
}

extern "C" void hip_miopen_destroy(MiopenHandle handle) {
    MIOPEN_ASSERT(miopenDestroy(unwrap_miopen(handle)));
}

/* ============================================================================
 * MIOpen Stream Binding
 * ============================================================================ */

extern "C" void hip_miopen_set_stream(MiopenHandle handle, void* stream) {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    MIOPEN_ASSERT(miopenSetStream(unwrap_miopen(handle), hip_stream));
}

extern "C" void* hip_miopen_get_stream(MiopenHandle handle) {
    hipStream_t stream;
    MIOPEN_ASSERT(miopenGetStream(unwrap_miopen(handle), &stream));
    return stream;
}

#endif /* __HANDLES_MIOPEN_CPP__ */
