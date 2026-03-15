/*
 * memory/transfer.cu - Memory transfer operations (HtoD, DtoH, DtoD)
 */


#ifndef __MEMORY_TRANSFER_CU__
#define __MEMORY_TRANSFER_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* Forward declaration to get device stream */
extern "C" void* cuda_device_get_stream(CudaDeviceHandle handle);

/* ============================================================================
 * Host <-> Device Transfers
 *
 * All transfers now use the device's stream for proper synchronization with
 * CUDA graph capture and execution.
 * ============================================================================ */

extern "C" void cuda_memcpy_htod(
    CudaDeviceHandle device,
    DevicePtr dst,
    HostPtr src,
    len_t size
) {
    CUstream stream = static_cast<CUstream>(cuda_device_get_stream(device));
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(unwrap(dst));
    CUDRIVER_ASSERT(cuMemcpyHtoDAsync(dptr, unwrap(src), size, stream));
}

extern "C" void cuda_memcpy_dtoh(
    CudaDeviceHandle device,
    HostPtr dst,
    DevicePtr src,
    len_t size
) {
    CUstream stream = static_cast<CUstream>(cuda_device_get_stream(device));
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(unwrap(src));
    CUDRIVER_ASSERT(cuMemcpyDtoHAsync(unwrap(dst), dptr, size, stream));
}

extern "C" void cuda_memcpy_dtod(
    CudaDeviceHandle device,
    DevicePtr dst,
    DevicePtr src,
    len_t size
) {
    CUstream stream = static_cast<CUstream>(cuda_device_get_stream(device));
    CUdeviceptr dst_dptr = reinterpret_cast<CUdeviceptr>(unwrap(dst));
    CUdeviceptr src_dptr = reinterpret_cast<CUdeviceptr>(unwrap(src));
    CUDRIVER_ASSERT(cuMemcpyDtoDAsync(dst_dptr, src_dptr, size, stream));
}

#endif /* __MEMORY_TRANSFER_CU__ */
