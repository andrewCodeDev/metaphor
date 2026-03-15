# Metaphor — Base Installation (Host-only)

This guide covers the minimum dependencies needed to build and run Metaphor on CPU. This is the foundation for all backends — if you plan to use CUDA or HIP/ROCm, start here first.

## Prerequisites

| Dependency | Minimum Version | Purpose |
|---|---|---|
| **C3 compiler (`c3c`)** | latest | Compiles all Metaphor source code |
| **CMake** | 3.21+ | Build system, generates `project.json` for c3c |
| **C compiler (gcc/clang)** | gcc 11+ / clang 14+ | Needed for C sources (cJSON) and linking |
| **BLAS library** | — | Required for host tensor operations (GEMM, etc.) |

## Install the C3 Compiler

Download the latest release from [c3-lang.org](https://c3-lang.org) or build from source:

```bash
# Option 1: download a release binary
# See https://github.com/c3lang/c3c/releases

# Option 2: build from source
git clone https://github.com/c3lang/c3c.git
cd c3c
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

Verify it works:

```bash
c3c --version
```

## Install a BLAS Library

You need **one** of the following. The build system auto-detects which is available (MKL is preferred if both are present).

### Option A: Intel MKL (recommended for Intel/AMD CPUs)

```bash
# Ubuntu/Debian
sudo apt install intel-mkl

# Or via Intel's repository (for latest version):
# See https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
```

Set the environment variable if installed to a non-standard location:

```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
```

### Option B: OpenBLAS

```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel

# Arch
sudo pacman -S openblas
```

### Option C: Apple Accelerate (macOS only)

No installation needed — Accelerate is included with macOS. The build system detects it automatically.

## Install CMake

```bash
# Ubuntu/Debian
sudo apt install cmake

# Fedora/RHEL
sudo dnf install cmake

# Arch
sudo pacman -S cmake

# macOS
brew install cmake
```

## Build

```bash
cd metaphor/

# Configure (auto-detects BLAS, no GPU)
cmake -B build -DGPU_BACKEND=none

# Generate project.json and build
cmake --build build -j$(nproc)
```

### Manually selecting a BLAS backend

If auto-detection picks the wrong library, override it:

```bash
cmake -B build -DGPU_BACKEND=none -DHOST_BLAS=openblas
# or
cmake -B build -DGPU_BACKEND=none -DHOST_BLAS=mkl
```

## Verify

A successful build produces `build/metaphor.a`.

## Next Steps

- [CUDA backend](cuda.md) — for NVIDIA GPUs
- [HIP/ROCm backend](hip.md) — for AMD GPUs
