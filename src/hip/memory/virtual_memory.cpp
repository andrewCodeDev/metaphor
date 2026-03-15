/*
 * memory/virtual_memory.cpp - Virtual address space and memory mapping
 */


#ifndef __MEMORY_VIRTUAL_MEMORY_CU__
#define __MEMORY_VIRTUAL_MEMORY_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"
#include <thread>
#include <vector>

/* ============================================================================
 * Memory Mapping Entry
 * ============================================================================ */

struct MapEntry {
    hipDeviceptr_t dptr;
    len_t size;
    hipMemGenericAllocationHandle_t handle;
};

/* ============================================================================
 * Memory Map Manager
 * ============================================================================ */

struct Memmap {
    hipMemAllocationProp  /* TODO: HIP virtual memory */ prop;
    hipMemAccessDesc  /* TODO: HIP virtual memory */ access_desc;
    std::vector<MapEntry> handles;

    Memmap(unsigned device_id) {
        this->prop = {};
        this->prop.type = hipMemAllocationTypePinned  /* TODO: HIP virtual memory */;
        this->prop.location.type = hipMemLocationTypeDevice  /* TODO: HIP virtual memory */;
        this->prop.location.id = static_cast<int>(device_id);

        this->access_desc = {};
        this->access_desc.location.type = hipMemLocationTypeDevice  /* TODO: HIP virtual memory */;
        this->access_desc.location.id = static_cast<int>(device_id);
        this->access_desc.flags = hipMemAccessFlagsProtReadWrite  /* TODO: HIP virtual memory */;

        this->handles = {};
    }

    void map_alloc(void* address, len_t size) {
        hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(address);

        hipMemGenericAllocationHandle_t handle;
        HIPDRIVER_ASSERT(hipMemCreate  /* TODO: HIP virtual memory */(&handle, size, &this->prop, 0));
        HIPDRIVER_ASSERT(hipMemMap  /* TODO: HIP virtual memory */(dptr, size, 0, handle, 0));
        HIPDRIVER_ASSERT(hipMemSetAccess  /* TODO: HIP virtual memory */(dptr, size, &this->access_desc, 1));

        this->handles.emplace_back(MapEntry{
            .dptr = dptr,
            .size = size,
            .handle = handle
        });
    }

    void reset() {
        for (auto& entry : this->handles) {
            HIPDRIVER_ASSERT(hipMemUnmap  /* TODO: HIP virtual memory */(entry.dptr, entry.size));
            HIPDRIVER_ASSERT(hipMemRelease  /* TODO: HIP virtual memory */(entry.handle));
        }
        this->handles.clear();
    }

    ~Memmap() {
        this->reset();
    }
};

/* ============================================================================
 * Opaque Handle Wrapper
 * ============================================================================ */

typedef struct { void* ptr; } MemmapHandle;

static inline Memmap* unwrap_memmap(MemmapHandle h) {
    return static_cast<Memmap*>(h.ptr);
}

static inline MemmapHandle wrap_memmap(Memmap* m) {
    return {.ptr = m};
}

/* ============================================================================
 * Extern C API
 * ============================================================================ */

extern "C" MemmapHandle hip_memmap_init(unsigned device_id) {
    return wrap_memmap(new Memmap(device_id));
}

extern "C" void hip_memmap_reset(MemmapHandle handle) {
    unwrap_memmap(handle)->reset();
}

extern "C" void hip_memmap_deinit(MemmapHandle handle) {
    delete unwrap_memmap(handle);
}

extern "C" void hip_memmap_alloc(MemmapHandle handle, DevicePtr address, len_t size) {
    unwrap_memmap(handle)->map_alloc(unwrap(address), size);
}

extern "C" len_t hip_mem_page_size(unsigned device_id) {
    hipMemAllocationProp  /* TODO: HIP virtual memory */ prop{};
    prop.type = hipMemAllocationTypePinned  /* TODO: HIP virtual memory */;
    prop.location.type = hipMemLocationTypeDevice  /* TODO: HIP virtual memory */;
    prop.location.id = static_cast<int>(device_id);
    prop.requestedHandleTypes = hipMemHandleTypeNone  /* TODO: HIP virtual memory */;

    len_t page_size;
    HIPDRIVER_ASSERT(hipMemGetAllocationGranularity  /* TODO: HIP virtual memory */(
        &page_size, &prop, hipMemAllocationGranularityMinimum    ));
    return page_size;
}

/* ============================================================================
 * Virtual Address Reservation
 *
 * Note: hipMemAddressReserve requires the correct device context.
 * We spawn a thread to avoid tampering with the caller's context.
 * ============================================================================ */

static inline size_t round_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

extern "C" DevicePtr hip_mem_reserve(unsigned device_id, len_t virtual_size) {
    hipDeviceptr_t base_address = 0;

    std::thread ctx_thread(
        [=](hipDeviceptr_t* dptr_ref) {
            HIP_ASSERT(hipSetDevice(static_cast<int>(device_id)));

            const size_t page = static_cast<size_t>(hip_mem_page_size(device_id));
            const size_t size = round_up(virtual_size ? virtual_size : page, page);
            const size_t align = page;
            HIPDRIVER_ASSERT(hipMemAddressReserve  /* TODO: HIP virtual memory */(dptr_ref, size, align, 0, 0));
        },
        &base_address
    );

    ctx_thread.join();
    return wrap_device(reinterpret_cast<void*>(base_address));
}

extern "C" void hip_mem_release(unsigned device_id, DevicePtr base_address, len_t virtual_size) {
    std::thread ctx_thread([=]() {
        HIP_ASSERT(hipSetDevice(static_cast<int>(device_id)));

        hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(unwrap(base_address));
        const size_t page = static_cast<size_t>(hip_mem_page_size(device_id));
        const size_t size = round_up(virtual_size ? virtual_size : page, page);

        HIPDRIVER_ASSERT(hipMemAddressFree  /* TODO: HIP virtual memory */(dptr, size));
    });

    ctx_thread.join();
}

#endif /* __MEMORY_VIRTUAL_MEMORY_CU__ */
