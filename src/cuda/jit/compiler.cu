/*
 * jit/compiler.cu - NVRTC runtime compilation wrapper
 *
 * Compiles CUDA source to PTX at runtime, loads as CUfunction.
 */

#ifndef __METAPHOR_JIT_COMPILER_H__
#define __METAPHOR_JIT_COMPILER_H__

#include "../interop.h"
#include "../core/assert.h"
#include "../logging.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cstring>
#include <cstdlib>

/* ============================================================================
 * Error Handling
 * ============================================================================ */

#define NVRTC_ASSERT(call) do { \
    nvrtcResult _err = (call); \
    if (_err != NVRTC_SUCCESS) { \
        LOG_ERROR("NVRTC error: %s at %s:%d", nvrtcGetErrorString(_err), __FILE__, __LINE__); \
        return nullptr; \
    } \
} while(0)

#define CU_ASSERT(call) do { \
    CUresult _err = (call); \
    if (_err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_err, &errStr); \
        LOG_ERROR("CUDA Driver error: %s at %s:%d", errStr, __FILE__, __LINE__); \
        return nullptr; \
    } \
} while(0)

/* ============================================================================
 * Compiled Kernel
 * ============================================================================ */

struct CompiledKernel {
    CUmodule module;
    CUfunction function;
    char* ptx;           /* Owned PTX string (for debugging/caching) */
    size_t ptx_size;
};

/*
 * Get CUDA include path for NVRTC.
 * Checks CUDA_PATH, CUDA_HOME environment variables, then falls back to common locations.
 */
static const char* get_cuda_include_path() {
    static char include_path[512] = {0};

    if (include_path[0] != '\0') {
        return include_path;
    }

    /* Check environment variables */
    const char* cuda_path = std::getenv("CUDA_PATH");
    if (!cuda_path) cuda_path = std::getenv("CUDA_HOME");

    if (cuda_path) {
        snprintf(include_path, sizeof(include_path), "-I%s/include", cuda_path);
        return include_path;
    }

    /* Check common locations - use targets path for CUDA 13+ */
    const char* common_paths[] = {
        "/usr/local/cuda/targets/x86_64-linux/include",
        "/usr/local/cuda/include",
        "/usr/include/cuda",
    };

    for (const char* path : common_paths) {
        char test_path[512];
        snprintf(test_path, sizeof(test_path), "%s/cuda_fp16.h", path);
        FILE* f = fopen(test_path, "r");
        if (f) {
            fclose(f);
            snprintf(include_path, sizeof(include_path), "-I%s", path);
            return include_path;
        }
    }

    /* Last resort - hope it's in the default path */
    snprintf(include_path, sizeof(include_path), "-I/usr/local/cuda/include");
    return include_path;
}

/*
 * Compile CUDA source to a kernel function.
 *
 * Returns nullptr on failure, logs error details.
 */
static CompiledKernel* jit_compile(
    const char* source,
    size_t source_len,
    const char* func_name,
    int compute_capability  /* e.g., 70 for sm_70 */
) {
    LOG_DEBUG("jit_compile: compiling %zu bytes, func=%s, sm_%d",
              source_len, func_name, compute_capability);

    /* Build architecture option */
    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d", compute_capability);

    /* Get CUDA include path for half-precision headers */
    const char* cuda_include = get_cuda_include_path();

    const char* opts[] = {
        arch_opt,
        "--std=c++17",
        "--use_fast_math",
        cuda_include,
    };
    int num_opts = 4;

    /* Create NVRTC program */
    nvrtcProgram prog;
    NVRTC_ASSERT(nvrtcCreateProgram(&prog, source, "kernel.cu", 0, nullptr, nullptr));

    /* Compile */
    nvrtcResult compile_result = nvrtcCompileProgram(prog, num_opts, opts);

    /* Get compile log (even on success, for warnings) */
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size);
        nvrtcGetProgramLog(prog, log);
        if (compile_result != NVRTC_SUCCESS) {
            LOG_ERROR("NVRTC compile failed:\n%s", log);
        } else {
            LOG_DEBUG("NVRTC compile log:\n%s", log);
        }
        free(log);
    }

    if (compile_result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    /* Get PTX */
    size_t ptx_size;
    NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptx_size));

    char* ptx = (char*)malloc(ptx_size);
    NVRTC_ASSERT(nvrtcGetPTX(prog, ptx));

    nvrtcDestroyProgram(&prog);

    LOG_DEBUG("jit_compile: PTX size=%zu bytes", ptx_size);

    /* Load module from PTX */
    CUmodule module;
    CU_ASSERT(cuModuleLoadData(&module, ptx));

    /* Get function */
    CUfunction function;
    CU_ASSERT(cuModuleGetFunction(&function, module, func_name));

    /* Allocate result */
    CompiledKernel* kernel = (CompiledKernel*)malloc(sizeof(CompiledKernel));
    kernel->module = module;
    kernel->function = function;
    kernel->ptx = ptx;
    kernel->ptx_size = ptx_size;

    LOG_DEBUG("jit_compile: success, func=%s", func_name);

    return kernel;
}

/*
 * Free a compiled kernel.
 */
static void jit_kernel_free(CompiledKernel* kernel) {
    if (!kernel) return;
    if (kernel->module) {
        cuModuleUnload(kernel->module);
    }
    if (kernel->ptx) {
        free(kernel->ptx);
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
 * stream: CUDA stream
 */
static void jit_launch(
    CompiledKernel* kernel,
    size_t n,
    void** args,
    CUstream stream,
    unsigned int block_size = 256
) {
    unsigned int grid_size = (n + block_size - 1) / block_size;

    LOG_DEBUG("jit_launch: n=%zu grid=%u block=%u", n, grid_size, block_size);

    CUresult err = cuLaunchKernel(
        kernel->function,
        grid_size, 1, 1,      /* grid dims */
        block_size, 1, 1,     /* block dims */
        0,                    /* shared mem */
        stream,
        args,
        nullptr               /* extra */
    );

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        LOG_ERROR("cuLaunchKernel failed: %s", errStr);
    }
}

/* ============================================================================
 * Device Query
 * ============================================================================ */

/*
 * Get compute capability for current device.
 * Returns e.g. 70 for sm_70, 86 for sm_86.
 */
static int get_compute_capability() {
    static int cached = 0;
    if (cached != 0) return cached;

    CUdevice device;
    cuCtxGetDevice(&device);

    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

    cached = major * 10 + minor;
    return cached;
}

#endif /* __METAPHOR_JIT_COMPILER_H__ */
