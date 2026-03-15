/*
 * handles/stream.cu - Standalone stream management
 *
 * For cases where a stream is needed without the full device context.
 */


#ifndef __HANDLES_STREAM_CU__
#define __HANDLES_STREAM_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * Stream Handle Wrapper
 * ============================================================================ */

/* StreamHandle is defined in interop.h */

static inline CUstream unwrap_stream(StreamHandle h) {
    return static_cast<CUstream>(h.ptr);
}

static inline StreamHandle wrap_stream(CUstream s) {
    return {.ptr = s};
}

/* ============================================================================
 * Stream Lifecycle
 * ============================================================================ */

extern "C" StreamHandle cuda_stream_create(unsigned device_id) {
    CUdevice device;
    CUcontext context;

    CUDRIVER_ASSERT(cuInit(0));
    CUDRIVER_ASSERT(cuDeviceGet(&device, device_id));
    // TODO: NULL ctxCreateParams means no execution affinity or CIG (CUDA in Graphics) settings
    CUDRIVER_ASSERT(cuCtxCreate(&context, NULL, 0, device));

    CUstream stream;
    CUDRIVER_ASSERT(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    return wrap_stream(stream);
}

extern "C" void cuda_stream_destroy(StreamHandle handle) {
    CUDRIVER_ASSERT(cuStreamDestroy(unwrap_stream(handle)));
}

/* ============================================================================
 * Stream Operations
 * ============================================================================ */

extern "C" void cuda_stream_sync(StreamHandle handle) {
    CUDRIVER_ASSERT(cuStreamSynchronize(unwrap_stream(handle)));
}

extern "C" bool cuda_stream_query(StreamHandle handle) {
    CUresult result = cuStreamQuery(unwrap_stream(handle));
    if (result == CUDA_SUCCESS) {
        return true;
    } else if (result == CUDA_ERROR_NOT_READY) {
        return false;
    } else {
        CUDRIVER_ASSERT(result);  /* Will abort on other errors */
        return false;
    }
}

/* ============================================================================
 * Event-based Synchronization
 * ============================================================================ */

/* EventHandle is defined in interop.h */

static inline cudaEvent_t unwrap_event(EventHandle h) {
    return static_cast<cudaEvent_t>(h.ptr);
}

static inline EventHandle wrap_event(cudaEvent_t e) {
    return {.ptr = e};
}

extern "C" EventHandle cuda_event_create() {
    cudaEvent_t event;
    CUDA_ASSERT(cudaEventCreate(&event));
    return wrap_event(event);
}

extern "C" void cuda_event_destroy(EventHandle handle) {
    CUDA_ASSERT(cudaEventDestroy(unwrap_event(handle)));
}

extern "C" void cuda_event_record(EventHandle event, StreamHandle stream) {
    CUDA_ASSERT(cudaEventRecord(
        unwrap_event(event),
        static_cast<cudaStream_t>(unwrap_stream(stream))
    ));
}

extern "C" void cuda_stream_wait_event(StreamHandle stream, EventHandle event) {
    CUDA_ASSERT(cudaStreamWaitEvent(
        static_cast<cudaStream_t>(unwrap_stream(stream)),
        unwrap_event(event),
        0
    ));
}

extern "C" bool cuda_event_query(EventHandle handle) {
    cudaError_t result = cudaEventQuery(unwrap_event(handle));
    if (result == cudaSuccess) {
        return true;
    } else if (result == cudaErrorNotReady) {
        return false;
    } else {
        CUDA_ASSERT(result);  /* Will abort on other errors */
        return false;
    }
}

extern "C" void cuda_event_sync(EventHandle handle) {
    CUDA_ASSERT(cudaEventSynchronize(unwrap_event(handle)));
}

/* ============================================================================
 * Device Synchronization
 * ============================================================================ */

extern "C" void cuda_device_sync() {
    CUDA_ASSERT(cudaDeviceSynchronize());
}

#endif /* __HANDLES_STREAM_CU__ */
