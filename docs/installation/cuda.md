# Metaphor: CUDA Installation (NVIDIA GPUs)

This guide covers the additional dependencies needed to build Metaphor with CUDA support. **Complete the [base installation](base.md) first.**

## Prerequisites

Everything from the [base installation](base.md), plus:

| Dependency | Minimum Version | Purpose |
|---|---|---|
| **CUDA Toolkit** | 11.8+ | `nvcc` compiler, runtime (`cudart`) |
| **cuBLAS** | (included with toolkit) | GPU matrix multiplication |
| **cuRAND** | (included with toolkit) | GPU random number generation |
| **cuDNN** | 8.x+ | Convolution and normalization ops |
| **cuTENSOR** | 1.x+ | Tensor contractions |
| **NVRTC** | (included with toolkit) | Runtime compilation of JIT kernels |

## Install CUDA Toolkit

### Ubuntu/Debian

Follow NVIDIA's official instructions for your distro version:

```bash
# Example for Ubuntu 22.04/24.04; see https://developer.nvidia.com/cuda-downloads for current instructions
sudo apt install nvidia-cuda-toolkit
```

Or install via NVIDIA's `.run` installer or their apt repository for the latest version.

Verify:

```bash
nvcc --version
```

### Environment

Make sure CUDA is on your `PATH`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Install cuDNN

cuDNN is distributed separately from the CUDA toolkit.

```bash
# Ubuntu/Debian (via NVIDIA's apt repo)
sudo apt install libcudnn8-dev

# Or download from https://developer.nvidia.com/cudnn
```

## Install cuTENSOR

```bash
# Ubuntu/Debian (via NVIDIA's apt repo)
sudo apt install libcutensor-dev libcutensor2

# Or download from https://developer.nvidia.com/cutensor
```

The build expects cuTENSOR libraries at `/usr/lib/x86_64-linux-gnu/libcutensor/13`. If your installation is elsewhere, you may need to adjust `LD_LIBRARY_PATH`.

## Build

```bash
cd metaphor/

# Configure with CUDA
cmake -B build -DGPU_BACKEND=cuda

# Build the CUDA shared library + C3 static library
cmake --build build -j$(nproc)
```

### Build options

```bash
# Debug mode (device-side debugging, no optimization)
cmake -B build -DGPU_BACKEND=cuda -DCUDA_DEBUG=ON

# Custom CUDA log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE)
cmake -B build -DGPU_BACKEND=cuda -DCUDA_LOG_LEVEL=1
```

## Verify

A successful build produces:
- `build/metaphor.a` (C3 static library)
- `build/lib/libmetaphor_cuda.so` (CUDA shared library)

## Troubleshooting

**`nvcc` not found**: Ensure `CUDA_HOME` is set and `$CUDA_HOME/bin` is on your `PATH`.

**cuTENSOR link errors**: Check that cuTENSOR `.so` files are installed and discoverable. Try: `ldconfig -p | grep cutensor`

**Architecture mismatch**: The build uses `CUDA_ARCHITECTURES native` by default (compiles for your installed GPU). If cross-compiling, set `CMAKE_CUDA_ARCHITECTURES` explicitly.

## Next Steps

- [Base installation](base.md)
- [HIP/ROCm backend](hip.md)
