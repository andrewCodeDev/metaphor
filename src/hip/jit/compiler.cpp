/*
 * jit/compiler.cpp - HIPRTC runtime compilation wrapper
 *
 * Compiles HIP source to device code at runtime, loads as hipFunction_t.
 * HIPRTC produces binary device code directly (no intermediate text format).
 */

#ifndef __METAPHOR_JIT_COMPILER_H__
#define __METAPHOR_JIT_COMPILER_H__

#include "../interop.h"
#include "../core/assert.h"
#include "../logging.h"
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdlib>

/* ============================================================================
 * Error Handling
 * ============================================================================ */

#define HIPRTC_ASSERT(call) do { \
    hiprtcResult _err = (call); \
    if (_err != HIPRTC_SUCCESS) { \
        LOG_ERROR("HIPRTC error: %s at %s:%d", hiprtcGetErrorString(_err), __FILE__, __LINE__); \
        return nullptr; \
    } \
} while(0)

#define HIP_DRIVER_ASSERT(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
        const char* errStr = NULL; \
        hipError_t _str_err = hipDrvGetErrorString(_err, &errStr); \
        if (_str_err != hipSuccess || errStr == NULL) { \
            LOG_ERROR("HIP Driver error: code %d at %s:%d (could not retrieve error string)", \
                (int)_err, __FILE__, __LINE__); \
        } else { \
            LOG_ERROR("HIP Driver error: %s at %s:%d", errStr, __FILE__, __LINE__); \
        } \
        return nullptr; \
    } \
} while(0)

/* ============================================================================
 * Compiled Kernel
 * ============================================================================ */

struct CompiledKernel {
    hipModule_t module;
    hipFunction_t function;
    char* code;             /* Owned device code binary (for debugging/caching) */
    size_t code_size;
};

/*
 * Get ROCm include path for HIPRTC.
 * Checks ROCM_PATH, ROCM_HOME environment variables, then falls back to common locations.
 */
static const char* get_hip_include_path() {
    static char include_path[512] = {0};

    if (include_path[0] != '\0') {
        return include_path;
    }

    /* Check environment variables */
    const char* rocm_path = std::getenv("ROCM_PATH");
    if (!rocm_path) rocm_path = std::getenv("ROCM_HOME");

    if (rocm_path) {
        snprintf(include_path, sizeof(include_path), "-I%s/include", rocm_path);
        return include_path;
    }

    /* Check common locations */
    const char* common_paths[] = {
        "/opt/rocm/include",
        "/usr/include/rocm",
    };

    for (const char* path : common_paths) {
        char test_path[512];
        snprintf(test_path, sizeof(test_path), "%s/hip/hip_fp16.h", path);
        FILE* f = fopen(test_path, "r");
        if (f) {
            fclose(f);
            snprintf(include_path, sizeof(include_path), "-I%s", path);
            return include_path;
        }
    }

    /* Last resort */
    snprintf(include_path, sizeof(include_path), "-I/opt/rocm/include");
    return include_path;
}

/*
 * Get GCN architecture string for the current device.
 *
 * HIP uses GCN architecture names like "gfx90a", "gfx1100", etc.
 * This queries the device properties to get the correct arch name.
 */
static const char* get_gcn_arch_name() {
    static char arch_name[64] = {0};
    if (arch_name[0] != '\0') return arch_name;

    hipDeviceProp_t props;
    HIP_ASSERT(hipGetDeviceProperties(&props, 0));
    snprintf(arch_name, sizeof(arch_name), "%s", props.gcnArchName);
    return arch_name;
}

/*
 * Compile HIP source to a kernel function.
 *
 * Returns nullptr on failure, logs error details.
 */
static CompiledKernel* jit_compile(
    const char* source,
    size_t source_len,
    const char* func_name,
    int compute_capability  /* unused — HIP uses gcnArchName instead */
) {
    (void)compute_capability;

    LOG_DEBUG("jit_compile: compiling %zu bytes, func=%s",
              source_len, func_name);

    /* Ensure null-terminated source — hiprtcCreateProgram requires it,
     * but C3 passes (char*, len) pairs that may not be null-terminated */
    char* src_copy = (char*)malloc(source_len + 1);
    memcpy(src_copy, source, source_len);
    src_copy[source_len] = '\0';

    /* Build architecture option using GCN arch name */
    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=%s", get_gcn_arch_name());

    /* Get ROCm include path for half-precision headers */
    const char* hip_include = get_hip_include_path();

    const char* opts[] = {
        arch_opt,
        "--std=c++17",
        "-ffast-math",
        "-mno-cumode",    /* WGP mode — 3x faster than default CU mode on RDNA3 */
        hip_include,
    };
    int num_opts = 5;

    /* Create HIPRTC program */
    hiprtcProgram prog;
    HIPRTC_ASSERT(hiprtcCreateProgram(&prog, src_copy, "kernel.cpp", 0, nullptr, nullptr));

    /* Compile */
    hiprtcResult compile_result = hiprtcCompileProgram(prog, num_opts, opts);

    /* Get compile log (even on success, for warnings) */
    size_t log_size;
    hiprtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size);
        hiprtcGetProgramLog(prog, log);
        if (compile_result != HIPRTC_SUCCESS) {
            LOG_ERROR("HIPRTC compile failed:\n%s", log);
        } else {
            LOG_DEBUG("HIPRTC compile log:\n%s", log);
        }
        free(log);
    }

    /* Source copy no longer needed after compilation */
    free(src_copy);

    if (compile_result != HIPRTC_SUCCESS) {
        hiprtcDestroyProgram(&prog);
        return nullptr;
    }

    /* Get compiled device code (binary) */
    size_t code_size;
    HIPRTC_ASSERT(hiprtcGetCodeSize(prog, &code_size));

    char* code = (char*)malloc(code_size);
    HIPRTC_ASSERT(hiprtcGetCode(prog, code));

    hiprtcDestroyProgram(&prog);

    LOG_DEBUG("jit_compile: device code size=%zu bytes", code_size);

    /* Load module from device code */
    hipModule_t module;
    HIP_DRIVER_ASSERT(hipModuleLoadData(&module, code));

    /* Get function */
    hipFunction_t function;
    HIP_DRIVER_ASSERT(hipModuleGetFunction(&function, module, func_name));

    /* Allocate result */
    CompiledKernel* kernel = (CompiledKernel*)malloc(sizeof(CompiledKernel));
    kernel->module = module;
    kernel->function = function;
    kernel->code = code;
    kernel->code_size = code_size;

    LOG_DEBUG("jit_compile: success, func=%s", func_name);

    return kernel;
}

/*
 * Free a compiled kernel.
 */
static void jit_kernel_free(CompiledKernel* kernel) {
    if (!kernel) return;
    if (kernel->module) {
        HIP_ASSERT(hipModuleUnload(kernel->module));
    }
    if (kernel->code) {
        free(kernel->code);
    }
    free(kernel);
}

/* ============================================================================
 * Kernel Launch
 * ============================================================================ */

/*
 * Launch a compiled kernel.
 *
 * block_size: threads per block (typically 256)
 * n: total number of elements to process
 * args: kernel arguments (void* array)
 * stream: HIP stream
 */
static void jit_launch(
    CompiledKernel* kernel,
    size_t n,
    void** args,
    hipStream_t stream,
    unsigned int block_size = 256
) {
    unsigned int grid_size = (n + block_size - 1) / block_size;

    LOG_DEBUG("jit_launch: n=%zu grid=%u block=%u", n, grid_size, block_size);

    hipError_t err = hipModuleLaunchKernel(
        kernel->function,
        grid_size, 1, 1,      /* grid dims */
        block_size, 1, 1,     /* block dims */
        0,                    /* shared mem */
        stream,
        args,
        nullptr               /* extra */
    );

    if (err != hipSuccess) {
        const char* errStr = NULL;
        hipError_t _str_err = hipDrvGetErrorString(err, &errStr);
        if (_str_err != hipSuccess || errStr == NULL) {
            fprintf(stderr, "hipModuleLaunchKernel failed: code %d at %s:%d\n",
                (int)err, __FILE__, __LINE__);
        } else {
            fprintf(stderr, "hipModuleLaunchKernel failed: %s at %s:%d\n",
                errStr, __FILE__, __LINE__);
        }
        abort();
    }
}

/* ============================================================================
 * Device Query
 * ============================================================================ */

/*
 * Get GCN architecture version for current device.
 * For compatibility with callers expecting a numeric value,
 * this returns e.g. 908 for gfx908, 90 for gfx90a, 1100 for gfx1100.
 */
static int get_compute_capability() {
    static int cached = 0;
    if (cached != 0) return cached;

    hipDevice_t device;
    HIP_ASSERT(hipCtxGetDevice(&device));

    int major, minor;
    HIP_ASSERT(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device));
    HIP_ASSERT(hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device));

    cached = major * 10 + minor;
    return cached;
}

#endif /* __METAPHOR_JIT_COMPILER_H__ */
