/*
 * core/cast.h - Internal type cast helpers
 *
 * Converts interop wrapper types to internal CUDA types.
 * Internal only - not exposed across the language boundary.
 */

#ifndef METAPHOR_CUDA_CAST_H
#define METAPHOR_CUDA_CAST_H

#include "../interop.h"
#include "includes.h"

/* ============================================================================
 * Primitive Type Aliases
 * ============================================================================ */

typedef unsigned char u8;
typedef float f32;
typedef double f64;
typedef __half f16;
typedef __nv_bfloat16 bf16;
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
 * Note: cudaStream_t and CUstream are different types in CUDA.
 * cudaStream_t is Runtime API, CUstream is Driver API.
 * They are interchangeable but need explicit casts.
 * ============================================================================ */

inline cudaStream_t cast_stream(void* ptr) {
    return static_cast<cudaStream_t>(ptr);
}

inline CUstream cast_custream(void* ptr) {
    return static_cast<CUstream>(ptr);
}

/* ============================================================================
 * Handle Casts
 * ============================================================================ */

inline cublasHandle_t cast_cublas(void* ptr) {
    return static_cast<cublasHandle_t>(ptr);
}

inline cudnnHandle_t cast_cudnn(void* ptr) {
    return static_cast<cudnnHandle_t>(ptr);
}

/* ============================================================================
 * cuTENSOR Type Conversions
 * ============================================================================ */

inline cutensorDataType_t to_cutensor_dtype(Dtype id) {
    switch (id) {
        case DTYPE_F32: return CUTENSOR_R_32F;
        case DTYPE_F64: return CUTENSOR_R_64F;
        case DTYPE_F16: return CUTENSOR_R_16F;
        case DTYPE_BF16: return CUTENSOR_R_16BF;
        default:
            fprintf(stderr, "Invalid dtype for cuTENSOR\n");
            abort();
    }
}

/*
 * NOTE: F16/BF16 cuTENSOR compute types
 *
 * cuTENSOR supports F16 data with different compute precisions:
 * - CUTENSOR_COMPUTE_DESC_16F: FP16 compute (faster, less accurate)
 * - CUTENSOR_COMPUTE_DESC_32F: FP32 compute (mixed precision, recommended)
 * - CUTENSOR_COMPUTE_DESC_TF32: TF32 compute (Ampere+, good balance)
 *
 * For BF16, similar options exist with CUTENSOR_COMPUTE_DESC_16BF.
 * Current implementation: Uses FP32 accumulation for F16/BF16 for better accuracy.
 */
inline cutensorComputeDescriptor_t to_cutensor_compute(Dtype id) {
    switch (id) {
        case DTYPE_F32: return CUTENSOR_COMPUTE_DESC_32F;
        case DTYPE_F64: return CUTENSOR_COMPUTE_DESC_64F;
        case DTYPE_F16: return CUTENSOR_COMPUTE_DESC_32F;  /* FP32 accumulation for accuracy */
        case DTYPE_BF16: return CUTENSOR_COMPUTE_DESC_32F; /* FP32 accumulation for accuracy */
        default:
            fprintf(stderr, "Invalid dtype for cuTENSOR compute\n");
            abort();
    }
}

inline cutensorOperator_t to_cutensor_op(BinaryOp op) {
    switch (op) {
        case BINARY_ADD: return CUTENSOR_OP_ADD;
        case BINARY_MIN: return CUTENSOR_OP_MIN;
        case BINARY_MAX: return CUTENSOR_OP_MAX;
        case BINARY_MUL: return CUTENSOR_OP_MUL;
        default:
            fprintf(stderr, "Invalid binary op for cuTENSOR\n");
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

#define WARP_SIZE 32

#endif /* METAPHOR_CUDA_CAST_H */
