/*
 * core/cast.h - Internal type cast helpers
 *
 * Converts interop wrapper types to internal HIP types.
 * Internal only - not exposed across the language boundary.
 */

#ifndef METAPHOR_HIP_CAST_H
#define METAPHOR_HIP_CAST_H

#include "../interop.h"
#include "includes.h"

/* ============================================================================
 * Primitive Type Aliases
 * ============================================================================ */

typedef unsigned char u8;
typedef float f32;
typedef double f64;
typedef __half f16;
typedef hip_bfloat16 bf16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;

/* ============================================================================
 * Device/Host Pointer Unwrap
 * ============================================================================ */

inline void* unwrap(DevicePtr p) { return p.ptr; }
inline void* unwrap(HostPtr p) { return p.ptr; }
inline DevicePtr wrap_device(void* p) { return {.ptr = p}; }
inline HostPtr wrap_host(void* p) { return {.ptr = p}; }

/* ============================================================================
 * Stream Casts
 *
 * Note: hipStream_t and hipStream_t are different types in HIP.
 * hipStream_t is Runtime API, hipStream_t is Driver API.
 * They are interchangeable but need explicit casts.
 * ============================================================================ */

inline hipStream_t cast_stream(void* ptr) {
    return static_cast<hipStream_t>(ptr);
}

inline hipStream_t cast_hipstream(void* ptr) {
    return static_cast<hipStream_t>(ptr);
}

/* ============================================================================
 * Handle Casts
 * ============================================================================ */

inline hipblasHandle_t cast_hipblas(void* ptr) {
    return static_cast<hipblasHandle_t>(ptr);
}

inline miopenHandle_t cast_miopen(void* ptr) {
    return static_cast<miopenHandle_t>(ptr);
}

/* ============================================================================
 * hipTENSOR Type Conversions
 * ============================================================================ */

inline hiptensorDataType_t to_hiptensor_dtype(Dtype id) {
    switch (id) {
        case DTYPE_F32: return HIPTENSOR_R_32F;
        case DTYPE_F64: return HIPTENSOR_R_64F;
        case DTYPE_F16: return HIPTENSOR_R_16F;
        case DTYPE_BF16: return HIPTENSOR_R_16BF;
        default:
            fprintf(stderr, "Invalid dtype for hipTENSOR\n");
            abort();
    }
}

/*
 * NOTE: F16/BF16 hipTENSOR compute types
 *
 * hipTENSOR supports F16 data with different compute precisions:
 * - HIPTENSOR_COMPUTE_16F: FP16 compute (faster, less accurate)
 * - HIPTENSOR_COMPUTE_DESC_32F: FP32 compute (mixed precision, recommended)
 * - HIPTENSOR_COMPUTE_TF32: TF32 compute (Ampere+, good balance)
 *
 * For BF16, similar options exist with HIPTENSOR_COMPUTE_16BF.
 * Current implementation: Uses FP32 accumulation for F16/BF16 for better accuracy.
 */
inline hiptensorComputeDescriptor_t to_hiptensor_compute(Dtype id) {
    switch (id) {
        case DTYPE_F32: return HIPTENSOR_COMPUTE_DESC_32F;
        case DTYPE_F64: return HIPTENSOR_COMPUTE_DESC_64F;
        case DTYPE_F16: return HIPTENSOR_COMPUTE_DESC_32F;  /* FP32 accumulation for accuracy */
        case DTYPE_BF16: return HIPTENSOR_COMPUTE_DESC_32F; /* FP32 accumulation for accuracy */
        default:
            fprintf(stderr, "Invalid dtype for hipTENSOR compute\n");
            abort();
    }
}

inline hiptensorOperator_t to_hiptensor_op(BinaryOp op) {
    switch (op) {
        case BINARY_ADD: return HIPTENSOR_OP_ADD;
        case BINARY_MIN: return HIPTENSOR_OP_MIN;
        case BINARY_MAX: return HIPTENSOR_OP_MAX;
        case BINARY_MUL: return HIPTENSOR_OP_MUL;
        default:
            fprintf(stderr, "Invalid binary op for hipTENSOR\n");
            abort();
    }
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

inline len_t product(const len_t* dims, len_t size) {
    len_t result = 1;
    for (len_t i = 0; i < size; ++i) {
        result *= dims[i];
    }
    return result;
}

#endif /* METAPHOR_HIP_CAST_H */
