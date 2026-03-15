cmake_minimum_required(VERSION 3.21)
project(metaphor LANGUAGES NONE)

# ===========================================================================
# Options (mirror Zig build options)
# ===========================================================================
set(GPU_BACKEND "auto" CACHE STRING "GPU backend: auto, cuda, hip, none")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS auto cuda hip none)

option(ENABLE_SORT   "Enable sorting in SafeTensors serialization"       ON)
option(HOST_ASYNC    "Enable async execution for host streams"           OFF)

set(HOST_BLAS "auto" CACHE STRING "Host BLAS backend: auto, mkl, openblas, accelerate, none")
set_property(CACHE HOST_BLAS PROPERTY STRINGS auto mkl openblas accelerate none)

set(CUDA_LOG_LEVEL "4" CACHE STRING "CUDA log level (0=DEBUG 1=INFO 2=WARN 3=ERROR 4=NONE)")
set(HIP_LOG_LEVEL "4" CACHE STRING "HIP log level (0=DEBUG 1=INFO 2=WARN 3=ERROR 4=NONE)")
set(C3_OPT "O3" CACHE STRING "C3 optimisation level (O0-O5, Os, Oz)")

# ===========================================================================
# Find the C3 compiler
# ===========================================================================
find_program(C3C_EXECUTABLE NAMES c3c REQUIRED)
message(STATUS "Found c3c: ${C3C_EXECUTABLE}")
execute_process(
    COMMAND ${C3C_EXECUTABLE} --version
    OUTPUT_VARIABLE C3C_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "c3c version: ${C3C_VERSION_OUTPUT}")

# ===========================================================================
# Auto-detect BLAS backend when HOST_BLAS=auto
# ===========================================================================
if(HOST_BLAS STREQUAL "auto")
    if(APPLE)
        set(HOST_BLAS "accelerate")
        message(STATUS "Auto-detected BLAS: Accelerate (macOS)")
    elseif(UNIX)
        # Try MKL first, fall back to OpenBLAS
        find_library(_MKL_LIB NAMES mkl_rt
            PATHS ENV MKLROOT /usr/lib/x86_64-linux-gnu /opt/intel/mkl
            PATH_SUFFIXES lib lib/intel64
        )
        if(_MKL_LIB)
            set(HOST_BLAS "mkl")
            message(STATUS "Auto-detected BLAS: MKL (${_MKL_LIB})")
        else()
            find_library(_OPENBLAS_LIB NAMES openblas)
            if(_OPENBLAS_LIB)
                set(HOST_BLAS "openblas")
                message(STATUS "Auto-detected BLAS: OpenBLAS (${_OPENBLAS_LIB})")
            else()
                set(HOST_BLAS "none")
                message(STATUS "No BLAS backend found — set HOST_BLAS manually if needed")
            endif()
        endif()
    else()
        set(HOST_BLAS "mkl")
        message(STATUS "Auto-detected BLAS: MKL (Windows default)")
    endif()
endif()

# ===========================================================================
# Resolve BLAS linked-libraries and search paths
# ===========================================================================
set(_BLAS_LIBS "")
set(_BLAS_PATHS "")

if(HOST_BLAS STREQUAL "mkl")
    list(APPEND _BLAS_LIBS "mkl_rt")
    # Add MKL library path if MKLROOT is set
    if(DEFINED ENV{MKLROOT})
        list(APPEND _BLAS_PATHS "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64")
    endif()
elseif(HOST_BLAS STREQUAL "openblas")
    list(APPEND _BLAS_LIBS "openblas")
elseif(HOST_BLAS STREQUAL "accelerate")
    # Accelerate is a macOS framework — linked via framework, not -l
    list(APPEND _BLAS_LIBS "Accelerate")
endif()

# ===========================================================================
# Auto-detect GPU backend when GPU_BACKEND=auto
# ===========================================================================
set(ENABLE_CUDA OFF)
set(ENABLE_HIP OFF)

if(GPU_BACKEND STREQUAL "cuda")
    set(ENABLE_CUDA ON)
    message(STATUS "GPU backend: CUDA (manual)")
elseif(GPU_BACKEND STREQUAL "hip")
    set(ENABLE_HIP ON)
    message(STATUS "GPU backend: HIP/ROCm (manual)")
elseif(GPU_BACKEND STREQUAL "none")
    message(STATUS "GPU backend: none (manual)")
elseif(GPU_BACKEND STREQUAL "auto")
    # Try CUDA first (check for nvcc)
    find_program(_NVCC NAMES nvcc PATHS ENV CUDA_HOME /usr/local/cuda PATH_SUFFIXES bin)
    if(_NVCC)
        set(ENABLE_CUDA ON)
        message(STATUS "Auto-detected GPU: CUDA (${_NVCC})")
    else()
        # Try HIP/ROCm (check for hipcc)
        find_program(_HIPCC NAMES hipcc PATHS /opt/rocm PATH_SUFFIXES bin)
        if(_HIPCC)
            set(ENABLE_HIP ON)
            message(STATUS "Auto-detected GPU: HIP/ROCm (${_HIPCC})")
        else()
            message(STATUS "No GPU toolkit found — host-only build")
        endif()
    endif()
endif()

# ===========================================================================
# Build feature-flags list
# ===========================================================================
set(_FEATURES "")

if(ENABLE_CUDA)
    list(APPEND _FEATURES "ENABLE_CUDA")
endif()

if(ENABLE_HIP)
    list(APPEND _FEATURES "ENABLE_HIP")
endif()

if(HOST_BLAS STREQUAL "mkl")
    list(APPEND _FEATURES "HOST_BLAS_MKL")
elseif(HOST_BLAS STREQUAL "openblas")
    list(APPEND _FEATURES "HOST_BLAS_OPENBLAS")
elseif(HOST_BLAS STREQUAL "accelerate")
    list(APPEND _FEATURES "HOST_BLAS_ACCELERATE")
endif()

if(ENABLE_SORT)
    list(APPEND _FEATURES "ENABLE_SORT")
endif()

if(HOST_ASYNC)
    list(APPEND _FEATURES "HOST_ASYNC")
endif()


# ===========================================================================
# CUDA support
# ===========================================================================
set(_CUDA_LIBS "")
set(_CUDA_PATHS "")

if(ENABLE_CUDA)
    enable_language(CUDA)

    option(CUDA_DEBUG "Enable CUDA debug mode with device debugging" OFF)

    # CUDA compiler flags
    if(CUDA_DEBUG)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0 --std=c++20 --expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC,-pthread")
        message(STATUS "CUDA debug mode enabled (-g -G -O0)")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --std=c++20 --expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC,-pthread,-O3,-DNDEBUG")
        message(STATUS "CUDA release mode enabled (-O3)")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA_LOG_LEVEL=${CUDA_LOG_LEVEL}")
    # Suppress fmt long double warning (GPUs don't support long double)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20208")

    set(CUDA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

    add_library(metaphor_cuda SHARED ${CUDA_SRC_DIR}/cuda_library.cu)

    target_include_directories(metaphor_cuda PRIVATE
        ${CUDA_SRC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/fmt/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/interop
    )

    set_target_properties(metaphor_cuda PROPERTIES
        OUTPUT_NAME "metaphor_cuda"
        CUDA_ARCHITECTURES native
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )

    if(NOT CUDA_DEBUG)
        target_link_options(metaphor_cuda PRIVATE -s)
    endif()

    target_link_directories(metaphor_cuda PRIVATE /usr/lib/x86_64-linux-gnu/libcutensor/13)

    target_link_libraries(metaphor_cuda PRIVATE
        cuda cudart cublas curand cutensor cudnn nvrtc
    )

    # Add CUDA libs/paths to the C3 project config
    list(APPEND _CUDA_LIBS "metaphor_cuda")
    list(APPEND _CUDA_PATHS "${CMAKE_BINARY_DIR}/lib" "/usr/local/cuda/lib64")
endif()

# ===========================================================================
# HIP/ROCm support
# ===========================================================================
set(_HIP_LIBS "")
set(_HIP_PATHS "")

if(ENABLE_HIP)
    enable_language(HIP)

    option(HIP_DEBUG "Enable HIP debug mode with device debugging" OFF)

    # GPU architecture — gfx1100 for RDNA3 (W7900)
    set(CMAKE_HIP_ARCHITECTURES "gfx1100" CACHE STRING "HIP GPU architectures")

    # ROCm prefix for find_package
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    # HIP compiler flags
    if(HIP_DEBUG)
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0 -std=c++20 -fPIC -pthread")
        message(STATUS "HIP debug mode enabled (-g -O0)")
    else()
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -DNDEBUG -std=c++20 -fPIC -pthread")
        message(STATUS "HIP release mode enabled (-O3)")
    endif()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -DHIP_LOG_LEVEL=${HIP_LOG_LEVEL}")

    set(HIP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/hip)

    add_library(metaphor_hip SHARED ${HIP_SRC_DIR}/hip_library.cpp)

    set_source_files_properties(${HIP_SRC_DIR}/hip_library.cpp PROPERTIES LANGUAGE HIP)

    # Use locally-built hipTensor (develop branch with gfx1100 support)
    set(HIPTENSOR_LOCAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hipTensor)

    target_include_directories(metaphor_hip PRIVATE
        ${HIP_SRC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/fmt/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/interop
        ${HIPTENSOR_LOCAL_DIR}/library/include
        ${HIPTENSOR_LOCAL_DIR}/library/src/include
    )

    set_target_properties(metaphor_hip PROPERTIES
        OUTPUT_NAME "metaphor_hip"
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        BUILD_RPATH "${HIPTENSOR_LOCAL_DIR}/build/lib"
        INSTALL_RPATH "${HIPTENSOR_LOCAL_DIR}/build/lib"
    )

    if(NOT HIP_DEBUG)
        target_link_options(metaphor_hip PRIVATE -s)
    endif()

    # Use RPATH (not RUNPATH) so transitive deps (libhiptensor.so) are resolved
    # via our custom build path rather than the system /opt/rocm/lib
    target_link_options(metaphor_hip PRIVATE -Wl,--disable-new-dtags)

    target_link_libraries(metaphor_hip PRIVATE
        amdhip64 hipblas hiprand
        ${HIPTENSOR_LOCAL_DIR}/build/lib/libhiptensor.so
        MIOpen hiprtc
    )

    target_link_directories(metaphor_hip PRIVATE
        /opt/rocm/lib
    )

    # Copy local hipTensor .so into build/lib so it's found via rpath
    add_custom_command(TARGET metaphor_hip POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${HIPTENSOR_LOCAL_DIR}/build/lib/libhiptensor.so.0.1
            ${CMAKE_BINARY_DIR}/lib/libhiptensor.so.0.1
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libhiptensor.so.0.1 ${CMAKE_BINARY_DIR}/lib/libhiptensor.so.0
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libhiptensor.so.0 ${CMAKE_BINARY_DIR}/lib/libhiptensor.so
    )

    # Add HIP libs/paths to the C3 project config
    list(APPEND _HIP_LIBS "metaphor_hip")
    list(APPEND _HIP_PATHS "${CMAKE_BINARY_DIR}/lib" "/opt/rocm/lib")

    # Generate a wrapper script so local hipTensor is found before system one
    file(WRITE ${CMAKE_BINARY_DIR}/run_tests.sh
"#!/bin/bash\nexport LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:\$LD_LIBRARY_PATH\nexec timeout \${1:-120} ${CMAKE_BINARY_DIR}/testrun\n")
    execute_process(COMMAND chmod +x ${CMAKE_BINARY_DIR}/run_tests.sh)
endif()

# ===========================================================================
# Assemble final linked-libraries and linker-search-paths for project.json
# ===========================================================================
set(_ALL_LIBS ${_BLAS_LIBS} ${_CUDA_LIBS} ${_HIP_LIBS})
set(_ALL_PATHS ${_BLAS_PATHS} ${_CUDA_PATHS} ${_HIP_PATHS})

# Format as JSON arrays: "lib1", "lib2", ...
function(list_to_json_items OUT_VAR INPUT_LIST)
    set(_result "")
    foreach(_item IN LISTS INPUT_LIST)
        if(_result)
            string(APPEND _result ", ")
        endif()
        string(APPEND _result "\"${_item}\"")
    endforeach()
    set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()

# ===========================================================================
# Build source-directory list (conditionally include device backends)
# ===========================================================================
set(_SOURCES
    "src/*"
    "src/cas/**"
    "src/codegen/**"
    "src/device/*"
    "src/device/common/**"
    "src/device/host_device/**"
    "src/graph/**"
    "src/mem/**"
    "src/nn/**"
    "src/tensor/**"
    "src/testing/**"
    "src/tokenizers/**"
    "src/utils/**"
)

if(ENABLE_CUDA)
    list(APPEND _SOURCES "src/device/cuda_device/**")
endif()

if(ENABLE_HIP)
    list(APPEND _SOURCES "src/device/hip_device/**")
endif()

set(_TEST_SOURCES
    "testing/*"
    "testing/cas/**"
    "testing/device/*"
    "testing/device/common/**"
    "testing/device/host_device/**"
    "testing/graph/**"
    "testing/host/**"
    "testing/tensor/**"
    "testing/tokenizers/**"
    "testing/utils/**"
)

if(ENABLE_CUDA)
    list(APPEND _TEST_SOURCES "testing/device/cuda_device/**")
endif()

if(ENABLE_HIP)
    list(APPEND _TEST_SOURCES "testing/device/hip_device/**")
endif()

# Build link-args (embed rpath so shared libs are found at runtime)
# Use --disable-new-dtags to set RPATH (not RUNPATH) so transitive deps
# (e.g. local hipTensor loaded by libmetaphor_hip.so) are resolved correctly.
set(_LINK_ARGS "-Wl,--disable-new-dtags")
foreach(_path IN LISTS _ALL_PATHS)
    list(APPEND _LINK_ARGS "-Wl,-rpath,${_path}")
endforeach()
# Force-link BLAS so the linker doesn't drop it via --as-needed
# (GPU-only executables have no direct BLAS references but the host
# device loads BLAS symbols at runtime via the same linked library)
if(_BLAS_LIBS)
    list(APPEND _LINK_ARGS "-Wl,--no-as-needed")
endif()

list_to_json_items(METAPHOR_SOURCES      "${_SOURCES}")
list_to_json_items(METAPHOR_TEST_SOURCES "${_TEST_SOURCES}")
list_to_json_items(METAPHOR_FEATURES     "${_FEATURES}")
list_to_json_items(METAPHOR_LINKED_LIBS  "${_ALL_LIBS}")
list_to_json_items(METAPHOR_LINKER_PATHS "${_ALL_PATHS}")
list_to_json_items(METAPHOR_LINK_ARGS    "${_LINK_ARGS}")

set(METAPHOR_OPT "${C3_OPT}")

# Debug info: full for O0, line-tables for optimised builds
if(C3_OPT STREQUAL "O0")
    set(METAPHOR_DEBUG_INFO "full")
else()
    set(METAPHOR_DEBUG_INFO "none")
endif()

# ===========================================================================
# Generate project.json from template
# ===========================================================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/project.json.in
    ${CMAKE_CURRENT_SOURCE_DIR}/project.json
    @ONLY
)
message(STATUS "Generated project.json")
message(STATUS "  sources:      [${METAPHOR_SOURCES}]")
message(STATUS "  features:     [${METAPHOR_FEATURES}]")
message(STATUS "  linked-libs:  [${METAPHOR_LINKED_LIBS}]")
message(STATUS "  linker-paths: [${METAPHOR_LINKER_PATHS}]")
message(STATUS "  opt:          ${METAPHOR_OPT}")
message(STATUS "  debug-info:   ${METAPHOR_DEBUG_INFO}")

# ===========================================================================
# Custom targets
# ===========================================================================

# Build the static library
add_custom_target(metaphor ALL
    COMMAND ${C3C_EXECUTABLE} build metaphor
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building metaphor (c3c build)"
)

# Run the test suite
add_custom_target(metaphor_test
    COMMAND ${C3C_EXECUTABLE} test metaphor
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Testing metaphor (c3c test)"
)

# If CUDA is enabled, make the C3 build depend on the CUDA shared library
if(ENABLE_CUDA)
    add_dependencies(metaphor metaphor_cuda)
    add_dependencies(metaphor_test metaphor_cuda)
endif()

# If HIP is enabled, make the C3 build depend on the HIP shared library
if(ENABLE_HIP)
    add_dependencies(metaphor metaphor_hip)
    add_dependencies(metaphor_test metaphor_hip)
endif()

# CTest integration
enable_testing()
add_test(
    NAME metaphor_tests
    COMMAND ${C3C_EXECUTABLE} test metaphor
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
