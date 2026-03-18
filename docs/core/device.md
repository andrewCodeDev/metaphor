# Device

The device layer provides a uniform interface for memory management and operation dispatch across hardware backends. Every tensor holds a reference to a device, so the same graph can target CPU, NVIDIA GPU, or AMD GPU without changing application code.

## Host Device (CPU)

```c3
HostDevice cpu = host_device::create();
defer cpu.deinit();
DeviceReference dev = cpu.reference();
```

Host memory can be accessed directly. `set()` and `get()` copy bytes in place. The host device JIT-compiles fused kernels to shared libraries at `/tmp/metaphor/`.

## CUDA Device (NVIDIA GPU)

```c3
CudaDevice gpu = cuda_device::create();
defer gpu.deinit();
DeviceReference dev = gpu.reference();
```

Memory resides on the GPU. `set()` and `get()` handle host-to-device and device-to-host transfers automatically. For timing, call `gpu.sync()` to block until all queued work completes:

```c3
Clock step_clock = time::clock::now();
model.loss_tensor.collect()!!;
gpu.sync();
epoch_forward_ns += step_clock.mark();
```

## HIP Device (AMD GPU)

Follows the same pattern as CUDA. Requires ROCm and the `ENABLE_HIP` feature flag.

## DeviceReference

`DeviceReference` is a lightweight wrapper around the underlying device. It adds global memory statistics and compatibility checks. All tensor operations take a `DeviceReference`, not the device directly.

```c3
DeviceReference dev = cpu.reference();
// or
DeviceReference dev = gpu.reference();
```

Pass `dev` to tensor factory functions:

```c3
Tensor x = tensor::empty(Datatype.F32, dev, { 64, 784 })!!;
```

## Weight Initialization

Kaiming He initialization for neural network weights:

```c3
Tensor w = tensor::empty(Datatype.F32, dev, { 784, 128 })!!.enable_grad()
    .fill_random(device_reference::RandType.NORMAL,
        device_reference::kaiming_scale(784));

Tensor b = tensor::empty(Datatype.F32, dev, { 128 })!!.enable_grad();
b.fill(0.0);
```

`kaiming_scale(fan_in)` computes the standard deviation `sqrt(2 / fan_in)`. Two distribution types are available: `RandType.NORMAL` and `RandType.UNIFORM`.

## Device Portability

A model built on one device runs on any other by changing only the device initialization. The graph, tensor operations, and optimizer are all device-agnostic.

From `examples/mnist.c3` (CPU):
```c3
HostDevice cpu = host_device::create();
DeviceReference dev = cpu.reference();
MnistModel model = build_model(dev);
```

From `examples/mnist_cuda.c3` (GPU), same `build_model` with a different device:
```c3
CudaDevice gpu = cuda_device::create();
DeviceReference dev = gpu.reference();
MnistModel model = build_model(dev);
```
