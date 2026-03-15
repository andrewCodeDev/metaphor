/*
 * nn/utils.cpp - Neural network utility functions and types (MIOpen)
 */

#ifndef __NN_UTILS_H__
#define __NN_UTILS_H__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"

/* ============================================================================
 * Handle Casts
 *
 * StreamHandle, HipblasHandle, MiopenHandle defined in interop.h
 * cast_stream and cast_hipblas defined in blas/utils.cpp
 * ============================================================================ */

static inline miopenHandle_t cast_miopen(MiopenHandle w) {
    return static_cast<miopenHandle_t>(w.ptr);
}

static inline hipStream_t stream_from_miopen(MiopenHandle w) {
    hipStream_t stream;
    MIOPEN_ASSERT(miopenGetStream(cast_miopen(w), &stream));
    return stream;
}

/* ============================================================================
 * MIOpen Data Type Mapping
 *
 * NOTE: F16 uses miopenHalf, BF16 uses miopenBFloat16.
 * MIOpen supports FP16 on gfx900+ and BF16 on gfx90a+.
 * ============================================================================ */

static inline miopenDataType_t miopen_dtype(Dtype id) {
    switch (id) {
        case DTYPE_F32:  return miopenFloat;
        case DTYPE_F64:  return miopenDouble;
        case DTYPE_F16:  return miopenHalf;
        case DTYPE_BF16: return miopenBFloat16;
        default:         return miopenFloat;  /* fallback */
    }
}

/* ============================================================================
 * Softmax Algorithm Mapping
 * SoftmaxType defined in interop.h
 * ============================================================================ */

static inline miopenSoftmaxAlgorithm_t miopen_softmax_algo(SoftmaxType op) {
    switch (op) {
        case SMAX_FAST: return MIOPEN_SOFTMAX_FAST;
        case SMAX_MAX: return MIOPEN_SOFTMAX_ACCURATE;  /* MAX maps to accurate */
        case SMAX_LOG: return MIOPEN_SOFTMAX_LOG;
        default: return MIOPEN_SOFTMAX_ACCURATE;
    }
}

/* ReduxType defined in interop.h */

#endif /* __NN_UTILS_H__ */
