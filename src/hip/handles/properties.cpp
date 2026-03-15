/*
 * handles/properties.cpp - Device property queries
 */

#ifndef __HANDLES_PROPERTIES_CU__
#define __HANDLES_PROPERTIES_CU__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../interop.h"

/* ============================================================================
 * Device Property Structures
 *
 * These mirror hipDeviceProp_t but use len_t for C3 compatibility.
 * ============================================================================ */

typedef struct {
    len_t major;
    len_t minor;
} ComputeVersion;

typedef struct {
    len_t bus_id;
    len_t device_id;
} PCIEInfo;

typedef struct {
    len_t total_global;
    len_t total_const;
    len_t shared_per_block;
    len_t shared_per_multiprocessor;
} DeviceMemoryInfo;

typedef struct {
    len_t per_block;
    len_t per_multiprocessor;
} DeviceRegisters;

typedef struct {
    len_t x;
    len_t y;
    len_t z;
} Dim3;

struct DeviceProperties {
    len_t multi_processor_count;
    len_t max_threads_per_multi_processor;
    len_t max_threads_per_block;
    len_t warp_size;
    PCIEInfo pcie;
    ComputeVersion compute_version;
    Dim3 max_threads_dim;
    Dim3 max_grid_size;
    DeviceMemoryInfo memory;
    DeviceRegisters registers;

    static DeviceProperties from_hip(const hipDeviceProp_t& prop) {
        return DeviceProperties{
            .multi_processor_count = static_cast<len_t>(prop.multiProcessorCount),
            .max_threads_per_multi_processor = static_cast<len_t>(prop.maxThreadsPerMultiProcessor),
            .max_threads_per_block = static_cast<len_t>(prop.maxThreadsPerBlock),
            .warp_size = static_cast<len_t>(prop.warpSize),
            .pcie = {
                .bus_id = static_cast<len_t>(prop.pciBusID),
                .device_id = static_cast<len_t>(prop.pciDeviceID),
            },
            .compute_version = {
                .major = static_cast<len_t>(prop.major),
                .minor = static_cast<len_t>(prop.minor),
            },
            .max_threads_dim = {
                .x = static_cast<len_t>(prop.maxThreadsDim[0]),
                .y = static_cast<len_t>(prop.maxThreadsDim[1]),
                .z = static_cast<len_t>(prop.maxThreadsDim[2]),
            },
            .max_grid_size = {
                .x = static_cast<len_t>(prop.maxGridSize[0]),
                .y = static_cast<len_t>(prop.maxGridSize[1]),
                .z = static_cast<len_t>(prop.maxGridSize[2]),
            },
            .memory = {
                .total_global = static_cast<len_t>(prop.totalGlobalMem),
                .total_const = static_cast<len_t>(prop.totalConstMem),
                .shared_per_block = static_cast<len_t>(prop.sharedMemPerBlock),
                .shared_per_multiprocessor = static_cast<len_t>(prop.sharedMemPerMultiprocessor),
            },
            .registers = {
                .per_block = static_cast<len_t>(prop.regsPerBlock),
                .per_multiprocessor = static_cast<len_t>(prop.regsPerMultiprocessor),
            },
        };
    }
};

/* ============================================================================
 * Device Property Handle
 * ============================================================================ */

typedef struct { void* ptr; } DevicePropertiesHandle;

static inline DeviceProperties* unwrap_props(DevicePropertiesHandle h) {
    return static_cast<DeviceProperties*>(h.ptr);
}

static inline DevicePropertiesHandle wrap_props(DeviceProperties* p) {
    return {.ptr = p};
}

/* ============================================================================
 * Device Query API
 * ============================================================================ */

extern "C" unsigned hip_device_count() {
    HIPDRIVER_ASSERT(hipInit(0));
    int count;
    HIPDRIVER_ASSERT(hipGetDeviceCount(&count));
    return static_cast<unsigned>(count);
}

extern "C" DevicePropertiesHandle hip_device_properties(unsigned device_id) {
    hipDeviceProp_t prop;
    HIP_ASSERT(hipGetDeviceProperties(&prop, device_id));

    auto* properties = new DeviceProperties(DeviceProperties::from_hip(prop));
    return wrap_props(properties);
}

extern "C" void hip_device_properties_free(DevicePropertiesHandle handle) {
    delete unwrap_props(handle);
}

extern "C" len_t hip_device_total_memory(unsigned device_id) {
    len_t total;
    HIPDRIVER_ASSERT(hipDeviceTotalMem(&total, device_id));
    return total;
}

/* ============================================================================
 * Property Accessors (for C3 to read individual fields)
 * ============================================================================ */

extern "C" len_t hip_props_multi_processor_count(DevicePropertiesHandle h) {
    return unwrap_props(h)->multi_processor_count;
}

extern "C" len_t hip_props_max_threads_per_block(DevicePropertiesHandle h) {
    return unwrap_props(h)->max_threads_per_block;
}

extern "C" len_t hip_props_max_threads_per_mp(DevicePropertiesHandle h) {
    return unwrap_props(h)->max_threads_per_multi_processor;
}

extern "C" len_t hip_props_warp_size(DevicePropertiesHandle h) {
    return unwrap_props(h)->warp_size;
}

extern "C" len_t hip_props_shared_mem_per_block(DevicePropertiesHandle h) {
    return unwrap_props(h)->memory.shared_per_block;
}

extern "C" len_t hip_props_total_global_mem(DevicePropertiesHandle h) {
    return unwrap_props(h)->memory.total_global;
}

extern "C" len_t hip_props_compute_major(DevicePropertiesHandle h) {
    return unwrap_props(h)->compute_version.major;
}

extern "C" len_t hip_props_compute_minor(DevicePropertiesHandle h) {
    return unwrap_props(h)->compute_version.minor;
}

/* ============================================================================
 * Error Checking
 * ============================================================================ */

extern "C" void hip_check_last_error() {
    HIP_ASSERT(hipDeviceSynchronize());
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Error %s: %s\n", hipGetErrorName(err), hipGetErrorString(err));
    }
}

#endif /* __HANDLES_PROPERTIES_CU__ */
