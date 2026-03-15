# Metaphor

Metaphor is a tensor framework for running and training machine learning models, written in C3. Operations are recorded as a lazy symbolic graph and compiled into fused kernels at execution time, enabling automatic operator fusion, memory lifecycle management, and automatic differentiation.

## Design

Tensors are lightweight handles. Operations on them produce graph nodes, not immediate results. When a value is needed, the graph is compiled into an execution sequence that:

- Fuses compatible element-wise operations into single kernel launches
- Manages tensor memory lifetimes via reference-counted liveness analysis
- Constructs backward passes through the same graph infrastructure
- Compiles once and re-executes via fingerprint-based change detection

## Getting Started

Metaphor supports three backends. The same model definition runs on any of them — see the installation guides for dependencies and build instructions:

- [**Host**](docs/installation/base.md) — JIT-compiled kernels with BLAS acceleration (MKL/OpenBLAS)
- [**CUDA**](docs/installation/cuda.md) — NVIDIA GPUs via cuTENSOR, cuDNN, and NVRTC
- [**HIP**](docs/installation/hip.md) — AMD GPUs via ROCm, hipTENSOR, and runtime kernel compilation

## Snippets

Device setup — same code structure for any backend:

```c3
// Host (CPU)
host_device::HostDevice cpu;
cpu.init();
defer cpu.deinit();
DeviceReference dev = cpu.reference();

// CUDA (NVIDIA GPU)
cuda_device::CudaDevice gpu = cuda_device::cuda_device_create({ .device_id = 0 });
defer gpu.deinit();
DeviceReference dev = gpu.reference();

// HIP (AMD GPU)
hip_device::HipDevice gpu = hip_device::hip_device_create({ .device_id = 0 });
defer gpu.deinit();
DeviceReference dev = gpu.reference();
```

Rotary position embeddings — slicing, broadcasting, and concatenation:

```c3
fn Tensor? apply_rope_batched(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(3) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 3)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 3)!;

	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	return tensor::cat({ rot1, rot2 }, 3);
}
```

SSM output with einsum, gating, and optional LoRA:

```c3
Tensor y = c_proj.einsum(h, "bld,bled->ble")!!
	.add(x_silu.einsum(self.d_vec, "ble,e->ble")!!)!!;

Tensor gated = y.mul(z_branch.silu()!!)!!;

return (self.has_out_proj_lora
	? lora_forward(&self.out_proj_lora, gated, self.out_proj, dev)!!
	: functional::linear(gated, self.out_proj)!!).stable();
```
