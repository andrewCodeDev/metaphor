/*
 * memory/virtual_memory.cu - Virtual address space and memory mapping
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
    CUdeviceptr dptr;
    len_t size;
    CUmemGenericAllocationHandle handle;
};

/* ============================================================================
 * Memory Map Manager
 * ============================================================================ */

struct Memmap {
    CUmemAllocationProp prop;
    CUmemAccessDesc access_desc;
    std::vector<MapEntry> handles;

    Memmap(unsigned device_id) {
        this->prop = {};
        this->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        this->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        this->prop.location.id = static_cast<int>(device_id);

        this->access_desc = {};
        this->access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        this->access_desc.location.id = static_cast<int>(device_id);
        this->access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        this->handles = {};
    }

    void map_alloc(void* address, len_t size) {
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(address);

        CUmemGenericAllocationHandle handle;
        CUDRIVER_ASSERT(cuMemCreate(&handle, size, &this->prop, 0));
        CUDRIVER_ASSERT(cuMemMap(dptr, size, 0, handle, 0));
        CUDRIVER_ASSERT(cuMemSetAccess(dptr, size, &this->access_desc, 1));

        this->handles.emplace_back(MapEntry{
            .dptr = dptr,
            .size = size,
            .handle = handle
        });
    }

    void reset() {
        for (auto& entry : this->handles) {
            CUDRIVER_ASSERT(cuMemUnmap(entry.dptr, entry.size));
            CUDRIVER_ASSERT(cuMemRelease(entry.handle));
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

extern "C" MemmapHandle cuda_memmap_init(unsigned device_id) {
    return wrap_memmap(new Memmap(device_id));
}

extern "C" void cuda_memmap_reset(MemmapHandle handle) {
    unwrap_memmap(handle)->reset();
}

extern "C" void cuda_memmap_deinit(MemmapHandle handle) {
    delete unwrap_memmap(handle);
}

extern "C" void cuda_memmap_alloc(MemmapHandle handle, DevicePtr address, len_t size) {
    unwrap_memmap(handle)->map_alloc(unwrap(address), size);
}

extern "C" len_t cuda_mem_page_size(unsigned device_id) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(device_id);
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    len_t page_size;
    CUDRIVER_ASSERT(cuMemGetAllocationGranularity(
        &page_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM
    ));
    return page_size;
}

/* ============================================================================
 * Virtual Address Reservation
 *
 * Note: cuMemAddressReserve requires the correct device context.
 * We spawn a thread to avoid tampering with the caller's context.
 * ============================================================================ */

static inline size_t round_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

extern "C" DevicePtr cuda_mem_reserve(unsigned device_id, len_t virtual_size) {
    CUdeviceptr base_address = 0;

    std::thread ctx_thread(
        [=](CUdeviceptr* dptr_ref) {
            CUcontext ctx;
            CUDRIVER_ASSERT(cuDevicePrimaryCtxRetain(&ctx, static_cast<int>(device_id)));
            CUDRIVER_ASSERT(cuCtxSetCurrent(ctx));

            const size_t page = static_cast<size_t>(cuda_mem_page_size(device_id));
            const size_t size = round_up(virtual_size ? virtual_size : page, page);
            const size_t align = page;
            CUDRIVER_ASSERT(cuMemAddressReserve(dptr_ref, size, align, 0, 0));
        },
        &base_address
    );

    ctx_thread.join();
    return wrap_device(reinterpret_cast<void*>(base_address));
}

extern "C" void cuda_mem_release(unsigned device_id, DevicePtr base_address, len_t virtual_size) {
    std::thread ctx_thread([=]() {
        CUcontext ctx;
        CUDRIVER_ASSERT(cuDevicePrimaryCtxRetain(&ctx, static_cast<int>(device_id)));
        CUDRIVER_ASSERT(cuCtxSetCurrent(ctx));

        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(unwrap(base_address));
        const size_t page = static_cast<size_t>(cuda_mem_page_size(device_id));
        const size_t size = round_up(virtual_size ? virtual_size : page, page);

        CUDRIVER_ASSERT(cuMemAddressFree(dptr, size));
    });

    ctx_thread.join();
}

#endif /* __MEMORY_VIRTUAL_MEMORY_CU__ */
