# Metaphor

Metaphor is a tensor computation library built around lazy graph construction and deferred execution. Operations are recorded as a symbolic graph and compiled into fused kernels at execution time, enabling automatic operator fusion, memory lifecycle management, and device-agnostic code.

## Design

Tensors are lightweight handles. Operations on them produce graph nodes, not immediate results. When a value is needed, the graph is compiled into an execution sequence that:

- Fuses compatible element-wise operations into single kernel launches
- Manages tensor memory lifetimes via reference-counted liveness analysis
- Supports automatic differentiation through the same graph infrastructure
- Compiles once and re-executes via fingerprint-based change detection

## Backends

- **Host** — JIT-compiled kernels with BLAS acceleration (MKL/OpenBLAS)
- **HIP** — AMD GPUs via ROCm, hipTENSOR, and runtime kernel compilation
- **CUDA** — NVIDIA GPUs via cuTENSOR, cuDNN, and NVRTC

Backend selection is transparent to user code. The same model definition runs on any supported device.

## Autodiff

Backward passes are constructed by composing forward operations — the same mechanism used for forward computation. Gradient kernels are fused and scheduled through the execution graph like any other operation.

## Building

```
c3c build                          # build the static library
cmake --build build -j$(nproc)     # build GPU shared libraries
```

## Testing

```
timeout 60 c3c test metaphor       # run the full test suite
```
