/*
 * memory/allocator.cpp - Device memory allocation
 */


#ifndef __MEMORY_ALLOCATOR_CPP__
#define __MEMORY_ALLOCATOR_CPP__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * Allocation / Free
 * ============================================================================ */

extern "C" DevicePtr hip_alloc(HipDeviceHandle device, len_t size) {
    hipStream_t stream = unwrap_device(device)->stream;
    hipDeviceptr_t dptr;
    HIPDRIVER_ASSERT(hipMallocAsync(&dptr, size, static_cast<hipStream_t>(stream)));
    return wrap_device(reinterpret_cast<void*>(dptr));
}

extern "C" void hip_free(HipDeviceHandle device, DevicePtr ptr) {
    hipStream_t stream = unwrap_device(device)->stream;
    hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(unwrap(ptr));
    HIPDRIVER_ASSERT(hipFreeAsync(dptr, static_cast<hipStream_t>(stream)));
}

/* ============================================================================
 * Scratch Buffer Management
 * ============================================================================ */

// Internal helper for hipMallocAsync error handling (must be defined before use)
inline void handleCuresultError(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        const char* msg = NULL;
        hipError_t str_err = hipDrvGetErrorString(err, &msg);
        if (str_err != hipSuccess || msg == NULL) {
            fprintf(stderr, "HIP Driver error at %s:%d: code %d "
                "(could not retrieve error string)\n", file, line, (int)err);
        } else {
            fprintf(stderr, "HIP Driver error at %s:%d: %s\n", file, line, msg);
        }
        abort();
    }
}

template <typename T>
T* alloc_typed(hipStream_t stream, len_t n, const char* file, int line) {
    hipDeviceptr_t dptr;
    hipStream_t cu_stream = static_cast<hipStream_t>(stream);
    handleCuresultError(hipMallocAsync(&dptr, n * sizeof(T), cu_stream), file, line);
    return reinterpret_cast<T*>(dptr);
}

template <typename T>
void free_typed(hipStream_t stream, T* ptr, const char* file, int line) {
    hipStream_t cu_stream = static_cast<hipStream_t>(stream);
    hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(ptr);
    handleCuresultError(hipFreeAsync(dptr, cu_stream), file, line);
}

template <typename T>
void ensure_scratch(
    hipStream_t stream,
    len_t* mem,
    len_t* cap,
    len_t new_cap,
    const char* file,
    int line
) {
    if (new_cap <= *cap) return;

    if (*cap > 0) {
        free_typed<T>(stream, reinterpret_cast<T*>(*mem), file, line);
    }
    *mem = reinterpret_cast<len_t>(alloc_typed<T>(stream, new_cap, file, line));
    *cap = new_cap;
}

void ensure_scratch_dtype(
    Dtype id,
    hipStream_t stream,
    len_t* mem,
    len_t* cap,
    len_t new_cap,
    const char* file,
    int line
) {
    switch (id) {
        case DTYPE_F32: return ensure_scratch<f32>(stream, mem, cap, new_cap, file, line);
        case DTYPE_F64: return ensure_scratch<f64>(stream, mem, cap, new_cap, file, line);
        default: SYSTEM_EXIT("Unsupported datatype");
    }
}

#define ENSURE_SCRATCH(id, stream, mem, cap, new_cap) \
    (ensure_scratch_dtype(id, stream, mem, cap, new_cap, __FILE__, __LINE__))

#endif /* __MEMORY_ALLOCATOR_CPP__ */
