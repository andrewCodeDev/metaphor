/*
 * core/assert.h - HIP error checking macros
 *
 * Assert macros for HIP, hipBLAS, MIOpen, hipTENSOR, and Driver API.
 * Crashes with file:line on failure - most HIP errors are unrecoverable.
 */

#ifndef METAPHOR_HIP_ASSERT_H
#define METAPHOR_HIP_ASSERT_H

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>
#include <hiprand/hiprand.h>
#include <hiptensor/hiptensor.h>

/* ============================================================================
 * HIP Runtime API
 * ============================================================================ */

#define HIP_ASSERT(expr) \
    do { \
        hipError_t _err = (expr); \
        if (_err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * HIP Driver API
 * ============================================================================ */

#define HIPDRIVER_ASSERT(expr) \
    do { \
        hipError_t _err = (expr); \
        if (_err != hipSuccess) { \
            const char* _msg = NULL; \
            hipError_t _str_err = hipDrvGetErrorString(_err, &_msg); \
            if (_str_err != hipSuccess || _msg == NULL) { \
                fprintf(stderr, "HIP Driver error at %s:%d: code %d " \
                    "(could not retrieve error string)\n", \
                    __FILE__, __LINE__, (int)_err); \
            } else { \
                fprintf(stderr, "HIP Driver error at %s:%d: %s\n", \
                    __FILE__, __LINE__, _msg); \
            } \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * hipBLAS
 * ============================================================================ */

#define HIPBLAS_ASSERT(expr) \
    do { \
        hipblasStatus_t _err = (expr); \
        if (_err != HIPBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "hipBLAS error at %s:%d: status %d\n", \
                __FILE__, __LINE__, (int)_err); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * MIOpen
 * ============================================================================ */

#define MIOPEN_ASSERT(expr) \
    do { \
        miopenStatus_t _err = (expr); \
        if (_err != miopenStatusSuccess) { \
            fprintf(stderr, "MIOpen error at %s:%d: %s\n", \
                __FILE__, __LINE__, miopenGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * hipTENSOR
 * ============================================================================ */

#define HIPTENSOR_ASSERT(expr) \
    do { \
        hiptensorStatus_t _err = (expr); \
        if (_err != HIPTENSOR_STATUS_SUCCESS) { \
            fprintf(stderr, "hipTENSOR error at %s:%d: %s\n", \
                __FILE__, __LINE__, hiptensorGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

/* ============================================================================
 * hipRAND
 * ============================================================================ */

#define HIPRAND_ASSERT(expr) \
    do { \
        hiprandStatus_t _err = (expr); \
        if (_err != HIPRAND_STATUS_SUCCESS) { \
            fprintf(stderr, "hipRAND error at %s:%d: status %d\n", \
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

#endif /* METAPHOR_HIP_ASSERT_H */
