/*
 * handles/stream.cpp - Standalone stream management
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

static inline hipStream_t unwrap_stream(StreamHandle h) {
    return static_cast<hipStream_t>(h.ptr);
}

static inline StreamHandle wrap_stream(hipStream_t s) {
    return {.ptr = s};
}

/* ============================================================================
 * Stream Lifecycle
 * ============================================================================ */

extern "C" StreamHandle hip_stream_create(unsigned device_id) {
    HIPDRIVER_ASSERT(hipInit(0));
    HIP_ASSERT(hipSetDevice(device_id));

    hipStream_t stream;
    HIPDRIVER_ASSERT(hipStreamCreateWithFlags(&stream, hipStreamDefault));
    return wrap_stream(stream);
}

extern "C" void hip_stream_destroy(StreamHandle handle) {
    HIPDRIVER_ASSERT(hipStreamDestroy(unwrap_stream(handle)));
}

/* ============================================================================
 * Stream Operations
 * ============================================================================ */

extern "C" void hip_stream_sync(StreamHandle handle) {
    HIPDRIVER_ASSERT(hipStreamSynchronize(unwrap_stream(handle)));
}

extern "C" bool hip_stream_query(StreamHandle handle) {
    hipError_t result = hipStreamQuery(unwrap_stream(handle));
    if (result == hipSuccess) {
        return true;
    } else if (result == hipErrorNotReady) {
        return false;
    } else {
        HIPDRIVER_ASSERT(result);  /* Will abort on other errors */
        return false;
    }
}

/* ============================================================================
 * Event-based Synchronization
 * ============================================================================ */

/* EventHandle is defined in interop.h */

static inline hipEvent_t unwrap_event(EventHandle h) {
    return static_cast<hipEvent_t>(h.ptr);
}

static inline EventHandle wrap_event(hipEvent_t e) {
    return {.ptr = e};
}

extern "C" EventHandle hip_event_create() {
    hipEvent_t event;
    HIP_ASSERT(hipEventCreate(&event));
    return wrap_event(event);
}

extern "C" void hip_event_destroy(EventHandle handle) {
    HIP_ASSERT(hipEventDestroy(unwrap_event(handle)));
}

extern "C" void hip_event_record(EventHandle event, StreamHandle stream) {
    HIP_ASSERT(hipEventRecord(
        unwrap_event(event),
        static_cast<hipStream_t>(unwrap_stream(stream))
    ));
}

extern "C" void hip_stream_wait_event(StreamHandle stream, EventHandle event) {
    HIP_ASSERT(hipStreamWaitEvent(
        static_cast<hipStream_t>(unwrap_stream(stream)),
        unwrap_event(event),
        0
    ));
}

extern "C" bool hip_event_query(EventHandle handle) {
    hipError_t result = hipEventQuery(unwrap_event(handle));
    if (result == hipSuccess) {
        return true;
    } else if (result == hipErrorNotReady) {
        return false;
    } else {
        HIP_ASSERT(result);  /* Will abort on other errors */
        return false;
    }
}

extern "C" void hip_event_sync(EventHandle handle) {
    HIP_ASSERT(hipEventSynchronize(unwrap_event(handle)));
}

/* ============================================================================
 * Device Synchronization
 * ============================================================================ */

extern "C" void hip_device_sync() {
    HIP_ASSERT(hipDeviceSynchronize());
}

#endif /* __HANDLES_STREAM_CU__ */
