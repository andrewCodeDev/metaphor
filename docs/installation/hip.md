# Metaphor: HIP/ROCm Installation (AMD GPUs)

This guide covers the additional dependencies needed to build Metaphor with HIP/ROCm support. **Complete the [base installation](base.md) first.**

## Prerequisites

Everything from the [base installation](base.md), plus:

| Dependency | Minimum Version | Purpose |
|---|---|---|
| **ROCm** | 5.x+ | HIP runtime, `hipcc` compiler |
| **hipBLAS** | (included with ROCm) | GPU matrix multiplication |
| **hipRAND** | (included with ROCm) | GPU random number generation |
| **MIOpen** | (included with ROCm) | Convolution and normalization ops |
| **hipRTC** | (included with ROCm) | Runtime compilation of JIT kernels |
| **hipTensor** | — | Tensor contractions (bundled in `third_party/`) |

## Install ROCm

Follow AMD's official instructions for your distribution:

### Ubuntu/Debian

```bash
# See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ for current instructions

# Example for Ubuntu 22.04/24.04:
sudo apt install rocm-dev hipblas-dev hiprand-dev miopen-hip-dev
```

### Verify

```bash
hipcc --version
rocminfo    # should list your GPU
```

ROCm is expected at `/opt/rocm`. If installed elsewhere, update `CMAKE_PREFIX_PATH` accordingly.

### Environment

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## Build hipTensor

Metaphor bundles a hipTensor source tree in `third_party/hipTensor` (a development branch with gfx1100/RDNA3 support). You need to build it before the main build:

```bash
cd metaphor/third_party/hipTensor

cmake -B build \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DAMDGPU_TARGETS="gfx1100" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

Adjust `AMDGPU_TARGETS` to match your GPU architecture:
- **gfx1100**: RDNA3 (RX 7900 XTX, W7900, etc.)
- **gfx1030**: RDNA2 (RX 6800/6900 series)
- **gfx90a**: CDNA2 (MI210, MI250)
- **gfx942**: CDNA3 (MI300)

Check your GPU's architecture with:

```bash
rocminfo | grep gfx
```

## Build

```bash
cd metaphor/

# Configure with HIP
cmake -B build -DGPU_BACKEND=hip

# Build the HIP shared library + C3 static library
cmake --build build -j$(nproc)
```

### Build options

```bash
# Target a different GPU architecture
cmake -B build -DGPU_BACKEND=hip -DCMAKE_HIP_ARCHITECTURES="gfx90a"

# Debug mode
cmake -B build -DGPU_BACKEND=hip -DHIP_DEBUG=ON

# Custom HIP log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE)
cmake -B build -DGPU_BACKEND=hip -DHIP_LOG_LEVEL=1
```

## Verify

A successful build produces:
- `build/metaphor.a` (C3 static library)
- `build/lib/libmetaphor_hip.so` (HIP shared library)
- `build/lib/libhiptensor.so` (hipTensor, copied from third_party build)

## Troubleshooting

**`hipcc` not found**: Ensure `/opt/rocm/bin` is on your `PATH`.

**hipTensor link errors**: Make sure you built hipTensor first (see above). The build copies `libhiptensor.so` into `build/lib/` and sets RPATH automatically.

**Wrong GPU architecture**: If you see `hipErrorNoBinaryForGpu`, rebuild with the correct `CMAKE_HIP_ARCHITECTURES` for your GPU. Check `rocminfo | grep gfx`.

**RUNPATH vs RPATH**: The build uses `--disable-new-dtags` to set RPATH (not RUNPATH). This ensures the locally-built hipTensor is found before any system-installed version.

## Next Steps

- [Base installation](base.md)
- [CUDA backend](cuda.md)
