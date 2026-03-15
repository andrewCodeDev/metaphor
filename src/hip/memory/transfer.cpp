/*
 * memory/transfer.cpp - Memory transfer operations (HtoD, DtoH, DtoD)
 */


#ifndef __MEMORY_TRANSFER_CU__
#define __MEMORY_TRANSFER_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* Forward declaration to get device stream */
extern "C" void* hip_device_get_stream(HipDeviceHandle handle);

/* ============================================================================
 * Host <-> Device Transfers
 *
 * All transfers now use the device's stream for proper synchronization with
 * HIP graph capture and execution.
 * ============================================================================ */

extern "C" void hip_memcpy_htod(
    HipDeviceHandle device,
    DevicePtr dst,
    HostPtr src,
    len_t size
) {
    hipStream_t stream = static_cast<hipStream_t>(hip_device_get_stream(device));
    hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(unwrap(dst));
    HIPDRIVER_ASSERT(hipMemcpyHtoDAsync(dptr, unwrap(src), size, stream));
}

extern "C" void hip_memcpy_dtoh(
    HipDeviceHandle device,
    HostPtr dst,
    DevicePtr src,
    len_t size
) {
    hipStream_t stream = static_cast<hipStream_t>(hip_device_get_stream(device));
    hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(unwrap(src));
    HIPDRIVER_ASSERT(hipMemcpyDtoHAsync(unwrap(dst), dptr, size, stream));
}

extern "C" void hip_memcpy_dtod(
    HipDeviceHandle device,
    DevicePtr dst,
    DevicePtr src,
    len_t size
) {
    hipStream_t stream = static_cast<hipStream_t>(hip_device_get_stream(device));
    hipDeviceptr_t dst_dptr = reinterpret_cast<hipDeviceptr_t>(unwrap(dst));
    hipDeviceptr_t src_dptr = reinterpret_cast<hipDeviceptr_t>(unwrap(src));
    HIPDRIVER_ASSERT(hipMemcpyDtoDAsync(dst_dptr, src_dptr, size, stream));
}

#endif /* __MEMORY_TRANSFER_CU__ */
