/*
 * memory/allocator.cu - Device memory allocation
 */


#ifndef __MEMORY_ALLOCATOR_CU__
#define __MEMORY_ALLOCATOR_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * Allocation / Free
 * ============================================================================ */

extern "C" DevicePtr cuda_alloc(CudaDeviceHandle device, len_t size) {
    cudaStream_t stream = unwrap_device(device)->stream;
    CUdeviceptr dptr;
    CUDRIVER_ASSERT(cuMemAllocAsync(&dptr, size, static_cast<CUstream>(stream)));
    return wrap_device(reinterpret_cast<void*>(dptr));
}

extern "C" void cuda_free(CudaDeviceHandle device, DevicePtr ptr) {
    cudaStream_t stream = unwrap_device(device)->stream;
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(unwrap(ptr));
    CUDRIVER_ASSERT(cuMemFreeAsync(dptr, static_cast<CUstream>(stream)));
}

/* ============================================================================
 * Scratch Buffer Management
 * ============================================================================ */

// Internal helper for cuMemAllocAsync error handling (must be defined before use)
inline void handleCuresultError(CUresult err, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* msg = NULL;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "CUDA Driver error at %s:%d: %s\n", file, line, msg ? msg : "unknown");
        abort();
    }
}

template <typename T>
T* alloc_typed(cudaStream_t stream, len_t n, const char* file, int line) {
    CUdeviceptr dptr;
    CUstream cu_stream = static_cast<CUstream>(stream);
    handleCuresultError(cuMemAllocAsync(&dptr, n * sizeof(T), cu_stream), file, line);
    return reinterpret_cast<T*>(dptr);
}

template <typename T>
void free_typed(cudaStream_t stream, T* ptr, const char* file, int line) {
    CUstream cu_stream = static_cast<CUstream>(stream);
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    handleCuresultError(cuMemFreeAsync(dptr, cu_stream), file, line);
}

template <typename T>
void ensure_scratch(
    cudaStream_t stream,
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
    cudaStream_t stream,
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

#endif /* __MEMORY_ALLOCATOR_CU__ */
