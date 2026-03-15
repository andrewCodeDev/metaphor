/*
 * cutensor/backend.cu - cuTENSOR backend handle (stateless)
 *
 * This module provides a thin wrapper around the cuTENSOR handle.
 * Plan caching is NOT handled here - that's a device implementation detail.
 */

#ifndef __CUTENSOR_BACKEND_H__
#define __CUTENSOR_BACKEND_H__

#include "utils.cu"

/* ============================================================================
 * cuTENSOR Backend
 *
 * Owns just the cuTENSOR handle and stream. No plan caching.
 * ============================================================================ */

struct CutensorBackend {
    cudaStream_t stream;
    cutensorHandle_t handle;

    explicit CutensorBackend(cudaStream_t s) : stream(s), handle(nullptr) {
        CUTENSOR_ASSERT(cutensorCreate(&this->handle));
    }

    ~CutensorBackend() {
        if (this->handle) {
            CUTENSOR_ASSERT(cutensorDestroy(this->handle));
        }
    }

    CutensorBackend(const CutensorBackend&) = delete;
    CutensorBackend& operator=(const CutensorBackend&) = delete;
};

/* ============================================================================
 * Handle Wrapper
 * ============================================================================ */

typedef struct { void* ptr; } CutensorHandle;

static inline CutensorBackend* unwrap_cutensor(CutensorHandle h) {
    return static_cast<CutensorBackend*>(h.ptr);
}

static inline CutensorHandle wrap_cutensor(CutensorBackend* b) {
    return {.ptr = b};
}

#endif /* __CUTENSOR_BACKEND_H__ */
