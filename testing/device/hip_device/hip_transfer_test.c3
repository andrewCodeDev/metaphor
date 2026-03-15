//! HIP Transfer Tests
//!
//! Tests Tensor.to() cross-device transfer through the HIP pipeline.
//! Covers GPU→CPU (DTOH), CPU→GPU (HTOD), and no-op (same device).

module metaphor::testing::device::hip_device::hip_transfer_test;
import metaphor::device::device_reference;
import metaphor::device::host_device;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-5;

// ============================================================================
// Transfer: GPU → CPU (DTOH)
// ============================================================================

fn void test_transfer_dtoh(DeviceReference gpu_dev)
{
	host_device::HostDevice cpu;
	cpu.init();
	DeviceReference host_dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor gpu_tensor = tensor::constant(F32, gpu_dev, &dims, 7.0)!!;

		Tensor cpu_tensor = gpu_tensor.to(host_dev)!!;
		cpu_tensor.collect()!!;
		gpu_dev.sync();

		// Result is on host — read directly
		float* host_data = (float*)cpu_tensor.value_bytes().ptr;
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(host_data[i] - 7.0f) < TOL, "dtoh: expected 7.0");
		}
	};
}

// ============================================================================
// Transfer: CPU → GPU (HTOD)
// ============================================================================

fn void test_transfer_htod(DeviceReference gpu_dev)
{
	host_device::HostDevice cpu;
	cpu.init();
	DeviceReference host_dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor cpu_tensor = tensor::constant(F32, host_dev, &dims, 3.0)!!;

		Tensor gpu_tensor = cpu_tensor.to(gpu_dev)!!;
		gpu_tensor.collect()!!;
		gpu_dev.sync();

		// Result is on GPU — read back via tensor.get
		float[6] result;
		gpu_tensor.get(&result);

		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - 3.0f) < TOL, "htod: expected 3.0");
		}
	};
}

// ============================================================================
// Transfer: no-op (same device)
// ============================================================================

fn void test_transfer_noop(DeviceReference gpu_dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, gpu_dev, &dims, 5.0)!!;

		Tensor b = a.to(gpu_dev)!!;

		// Same device — should return the same tensor (no-op)
		assert(a.core() == b.core(), "noop: expected same tensor core");
	};
}

// ============================================================================
// Transfer: GPU → CPU, then use result in CPU computation
// ============================================================================

fn void test_transfer_dtoh_then_compute(DeviceReference gpu_dev)
{
	host_device::HostDevice cpu;
	cpu.init();
	DeviceReference host_dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		// Create [1, 2, 3, 4] on GPU
		float[4] vals = { 1.0, 2.0, 3.0, 4.0 };
		Tensor gpu_tensor = tensor::@from_data(gpu_dev, &dims, &vals)!!;

		// Transfer to host
		Tensor cpu_tensor = gpu_tensor.to(host_dev)!!;
		cpu_tensor.collect()!!;
		gpu_dev.sync();

		float* host_data = (float*)cpu_tensor.value_bytes().ptr;
		for (usz i = 0; i < 4; i++)
		{
			float expected = (float)(i + 1);
			assert(math::abs(host_data[i] - expected) < TOL, "dtoh_compute: value mismatch");
		}
	};
}

fn void run_all(DeviceReference dev)
{
	test_transfer_dtoh(dev);
	test_transfer_htod(dev);
	test_transfer_noop(dev);
	test_transfer_dtoh_then_compute(dev);
}
