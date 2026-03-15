/*
 * handles/cudnn.cu - Standalone cuDNN handle management
 *
 * For cases where a cuDNN handle is needed without the full device context.
 */


#ifndef __HANDLES_CUDNN_CU__
#define __HANDLES_CUDNN_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * cuDNN Handle Wrapper
 * ============================================================================ */

/* CudnnHandle is defined in interop.h */

static inline cudnnHandle_t unwrap_cudnn(CudnnHandle h) {
    return static_cast<cudnnHandle_t>(h.ptr);
}

static inline CudnnHandle wrap_cudnn(cudnnHandle_t h) {
    return {.ptr = h};
}

/* ============================================================================
 * cuDNN Lifecycle
 * ============================================================================ */

extern "C" CudnnHandle cuda_cudnn_create() {
    cudnnHandle_t handle;
    CUDNN_ASSERT(cudnnCreate(&handle));
    return wrap_cudnn(handle);
}

extern "C" void cuda_cudnn_destroy(CudnnHandle handle) {
    CUDNN_ASSERT(cudnnDestroy(unwrap_cudnn(handle)));
}

/* ============================================================================
 * cuDNN Stream Binding
 * ============================================================================ */

extern "C" void cuda_cudnn_set_stream(CudnnHandle handle, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    CUDNN_ASSERT(cudnnSetStream(unwrap_cudnn(handle), cuda_stream));
}

extern "C" void* cuda_cudnn_get_stream(CudnnHandle handle) {
    cudaStream_t stream;
    CUDNN_ASSERT(cudnnGetStream(unwrap_cudnn(handle), &stream));
    return stream;
}

#endif /* __HANDLES_CUDNN_CU__ */
