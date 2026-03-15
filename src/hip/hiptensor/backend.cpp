/*
 * hiptensor/backend.cpp - hipTENSOR backend handle (stateless)
 *
 * This module provides a thin wrapper around the hipTENSOR handle.
 * Plan caching is NOT handled here - that's a device implementation detail.
 */

#ifndef __HIPTENSOR_BACKEND_H__
#define __HIPTENSOR_BACKEND_H__

#include "utils.cpp"

/* ============================================================================
 * hipTENSOR Backend
 *
 * Owns just the hipTENSOR handle and stream. No plan caching.
 * ============================================================================ */

struct HiptensorBackend {
    hipStream_t stream;
    hiptensorHandle_t handle;

    explicit HiptensorBackend(hipStream_t s) : stream(s), handle(nullptr) {
        HIPTENSOR_ASSERT(hiptensorCreate(&this->handle));
    }

    ~HiptensorBackend() {
        if (this->handle) {
            HIPTENSOR_ASSERT(hiptensorDestroy(this->handle));
        }
    }

    HiptensorBackend(const HiptensorBackend&) = delete;
    HiptensorBackend& operator=(const HiptensorBackend&) = delete;
};

/* ============================================================================
 * Handle Wrapper
 * ============================================================================ */

typedef struct { void* ptr; } HiptensorHandle;

static inline HiptensorBackend* unwrap_hiptensor(HiptensorHandle h) {
    return static_cast<HiptensorBackend*>(h.ptr);
}

static inline HiptensorHandle wrap_hiptensor(HiptensorBackend* b) {
    return {.ptr = b};
}

#endif /* __HIPTENSOR_BACKEND_H__ */
