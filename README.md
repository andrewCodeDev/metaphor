# Metaphor

Metaphor is a tensor computation library built around lazy graph construction and deferred execution. Operations are recorded as a symbolic graph and compiled into fused kernels at execution time, enabling automatic operator fusion, memory lifecycle management, and device-agnostic code.

## Design

Tensors are lightweight handles. Operations on them produce graph nodes, not immediate results. When a value is needed, the graph is compiled into an execution sequence that:

- Fuses compatible element-wise operations into single kernel launches
- Manages tensor memory lifetimes via reference-counted liveness analysis
- Supports automatic differentiation through the same graph infrastructure
- Compiles once and re-executes via fingerprint-based change detection

## Getting Started

See the installation guides for dependencies and build instructions:

- [Base (host-only)](docs/installation/base.md) — C3 compiler, CMake, BLAS
- [CUDA (NVIDIA GPUs)](docs/installation/cuda.md) — CUDA toolkit, cuDNN, cuTENSOR
- [HIP/ROCm (AMD GPUs)](docs/installation/hip.md) — ROCm, hipBLAS, MIOpen, hipTensor

Backend selection is transparent to user code. The same model definition runs on any supported device.

## Autodiff

Backward passes are constructed by composing forward operations — the same mechanism used for forward computation. Gradient kernels are fused and scheduled through the execution graph like any other operation.

## Snippets

```c3
/// Apply rotary position embeddings to q or k (batched prefill).
/// x: [B, S, H, head_dim], cos_pos: [1, S, 1, head_dim/2], sin_pos: [1, S, 1, head_dim/2]
/// Returns [B, S, H, head_dim] with RoPE applied.
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

```c3
	// 10. Output: y = C * h + D * x
Tensor y = c_proj.einsum(h, "bld,bled->ble")!!
	.add(x_silu.einsum(self.d_vec, "ble,e->ble")!!)!!;

// 11. Gate with z_branch
Tensor gated = y.mul(z_branch.silu()!!)!!;

// 12. Output projection with optional LoRA
return self.has_out_proj_lora
	? lora_forward(&self.out_proj_lora, gated, self.out_proj, dev)!!.stable()
	: functional::linear(gated, self.out_proj)!!.stable();
```

```c3
host_device::HostDevice cpu;
cpu.init();
DeviceReference dev = cpu.reference();

graph::@subgraph()
{
	Tensor a = tensor::constant(F32, dev, { 2, 3 }, 2.0)!!;
	Tensor b = tensor::constant(F32, dev, { 2, 3 }, 3.0)!!;
	Tensor c = (a + b).collect();

	dev.sync();

	float[6] result;
	c.get(&result);

	for (usz i = 0; i < 6; i++)
	{
		assert(math::abs(result[i] - 5.0f) < TOL, "add: expected 5.0");
	}
};
```
