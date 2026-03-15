/*
 * core/assert.h - CUDA error checking macros
 *
 * Assert macros for CUDA, cuBLAS, cuDNN, cuTENSOR, and Driver API.
 * Crashes with file:line on failure - most CUDA errors are unrecoverable.
 */

#ifndef METAPHOR_CUDA_ASSERT_H
#define METAPHOR_CUDA_ASSERT_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <cutensor.h>

/* ============================================================================
 * CUDA Runtime API
 * ============================================================================ */

#define CUDA_ASSERT(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * CUDA Driver API
 * ============================================================================ */

#define CUDRIVER_ASSERT(expr) \
    do { \
        CUresult _err = (expr); \
        if (_err != CUDA_SUCCESS) { \
            const char* _msg = NULL; \
            cuGetErrorString(_err, &_msg); \
            fprintf(stderr, "CUDA Driver error at %s:%d: %s\n", \
                __FILE__, __LINE__, _msg ? _msg : "unknown"); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * cuBLAS
 * ============================================================================ */

#define CUBLAS_ASSERT(expr) \
    do { \
        cublasStatus_t _err = (expr); \
        if (_err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", \
                __FILE__, __LINE__, (int)_err); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * cuDNN
 * ============================================================================ */

#define CUDNN_ASSERT(expr) \
    do { \
        cudnnStatus_t _err = (expr); \
        if (_err != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudnnGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * cuTENSOR
 * ============================================================================ */

#define CUTENSOR_ASSERT(expr) \
    do { \
        cutensorStatus_t _err = (expr); \
        if (_err != CUTENSOR_STATUS_SUCCESS) { \
            fprintf(stderr, "cuTENSOR error at %s:%d: %s\n", \
                __FILE__, __LINE__, cutensorGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * cuRAND
 * ============================================================================ */

#define CURAND_ASSERT(expr) \
    do { \
        curandStatus_t _err = (expr); \
        if (_err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND error at %s:%d: status %d\n", \
                __FILE__, __LINE__, (int)_err); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * Generic Invariant Check
 * ============================================================================ */

#define CHECK_INVARIANT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "Invariant failed at %s:%d: %s\n", \
                __FILE__, __LINE__, (msg)); \
            abort(); \
        } \
    } while(0)

#define SYSTEM_EXIT(msg) \
    do { \
        fprintf(stderr, "Fatal error at %s:%d: %s\n", \
            __FILE__, __LINE__, (msg)); \
        abort(); \
    } while(0)

#endif /* METAPHOR_CUDA_ASSERT_H */
