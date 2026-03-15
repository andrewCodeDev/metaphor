/*
 * nn/utils.cu - Neural network utility functions and types
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
 * StreamHandle, CublasHandle, CudnnHandle defined in interop.h
 * cast_stream and cast_cublas defined in blas/utils.cu
 * ============================================================================ */

static inline cudnnHandle_t cast_cudnn(CudnnHandle w) {
    return static_cast<cudnnHandle_t>(w.ptr);
}

static inline cudaStream_t stream_from_cudnn(CudnnHandle w) {
    cudaStream_t stream;
    CUDNN_ASSERT(cudnnGetStream(cast_cudnn(w), &stream));
    return stream;
}

/* ============================================================================
 * cuDNN Data Type Mapping
 *
 * NOTE: F16 uses CUDNN_DATA_HALF, BF16 uses CUDNN_DATA_BFLOAT16.
 * cuDNN supports FP16 tensor operations on SM 5.3+ and BF16 on SM 8.0+.
 * ============================================================================ */

static inline cudnnDataType_t cudnn_dtype(Dtype id) {
    switch (id) {
        case DTYPE_F32:  return CUDNN_DATA_FLOAT;
        case DTYPE_F64:  return CUDNN_DATA_DOUBLE;
        case DTYPE_F16:  return CUDNN_DATA_HALF;
        case DTYPE_BF16: return CUDNN_DATA_BFLOAT16;
        default:         return CUDNN_DATA_FLOAT;  /* fallback */
    }
}

/* ============================================================================
 * Softmax Algorithm Mapping
 * SoftmaxType defined in interop.h
 * ============================================================================ */

static inline cudnnSoftmaxAlgorithm_t cudnn_softmax_algo(SoftmaxType op) {
    switch (op) {
        case SMAX_FAST: return CUDNN_SOFTMAX_FAST;
        case SMAX_MAX: return CUDNN_SOFTMAX_ACCURATE;  /* MAX maps to accurate */
        case SMAX_LOG: return CUDNN_SOFTMAX_LOG;
        default: return CUDNN_SOFTMAX_ACCURATE;
    }
}

/* ReduxType defined in interop.h */

#endif /* __NN_UTILS_H__ */
