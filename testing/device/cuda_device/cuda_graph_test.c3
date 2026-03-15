//! CUDA Graph Capture Tests
//!
//! Tests the @cuda_graph scoped macro for capturing operations into
//! replayable CUDA graphs. Replaces the old close-steal-reopen pattern
//! with a clean macro-based API.
//!
//! Pattern:
//!   CudaGraphCapture graph = gpu.@cuda_graph() { ops... };
//!   graph.replay();
//!   defer graph.deinit();

module metaphor::testing::device::cuda_device::cuda_graph_test;
import metaphor::device::cuda_device;
import metaphor::device::cuda_device::utils;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;
import std::io;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;
alias CudaDevice = cuda_device::CudaDevice;
alias CudaGraphCapture = cuda_device::CudaGraphCapture;

const float TOL = 1e-5;

// ============================================================================
// Basic capture: c = (a + b) * 2
// ============================================================================

fn void test_basic_graph_capture(DeviceReference dev, CudaDevice* gpu)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;
		Tensor two = tensor::constant(F32, dev, &dims, 2.0)!!;

		Tensor c = a.add(b)!!.mul(two)!!;

		// Capture into a CUDA graph
		CudaGraphCapture captured = gpu.@cuda_graph()
		{
			c.collect()!!;
		};
		defer captured.deinit();

		// Replay captured graph and verify result: (2 + 3) * 2 = 10
		captured.replay();
		gpu.sync();

		float[6] result;
		c.get(&result);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - 10.0f) < TOL, "basic capture: expected 10.0");
		}

		// Replay again and verify same result
		captured.replay();
		gpu.sync();

		float[6] replay_result;
		c.get(&replay_result);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(replay_result[i] - 10.0f) < TOL, "basic replay: expected 10.0");
		}
	};
}

// ============================================================================
// Replay loop: capture once, replay 3×
// ============================================================================

fn void test_graph_capture_replay(DeviceReference dev, CudaDevice* gpu)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 4, 4 };

		Tensor a = tensor::constant(F32, dev, &dims, 1.5)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.5)!!;

		Tensor result_t = a.add(b)!!.mul(a)!!;

		// Capture
		CudaGraphCapture captured = gpu.@cuda_graph()
		{
			result_t.collect()!!;
		};
		defer captured.deinit();

		float expected = (1.5f + 2.5f) * 1.5f;  // 6.0

		// Replay 3 times, verify each
		for (int iter = 0; iter < 3; iter++)
		{
			captured.replay();
			gpu.sync();

			float[16] check;
			result_t.get(&check);
			for (usz i = 0; i < 16; i++)
			{
				assert(math::abs(check[i] - expected) < TOL, "replay iter: expected 6.0");
			}
		}
	};
}

// ============================================================================
// Multiple captures: two independent graphs
// ============================================================================

fn void test_multiple_captures(DeviceReference dev, CudaDevice* gpu)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;

		// First computation: sum = a + b = 5.0
		Tensor sum = a.add(b)!!;
		CudaGraphCapture cap_sum = gpu.@cuda_graph()
		{
			sum.collect()!!;
		};
		defer cap_sum.deinit();

		// Second computation: product = a * b = 6.0
		Tensor product = a.mul(b)!!;
		CudaGraphCapture cap_product = gpu.@cuda_graph()
		{
			product.collect()!!;
		};
		defer cap_product.deinit();

		// Replay both, then verify
		cap_sum.replay();
		cap_product.replay();
		gpu.sync();

		float[6] sum_result;
		sum.get(&sum_result);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(sum_result[i] - 5.0f) < TOL, "multi: sum expected 5.0");
		}

		float[6] prod_result;
		product.get(&prod_result);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(prod_result[i] - 6.0f) < TOL, "multi: product expected 6.0");
		}

		// Replay again and verify
		cap_sum.replay();
		cap_product.replay();
		gpu.sync();

		float[6] sum_replay;
		sum.get(&sum_replay);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(sum_replay[i] - 5.0f) < TOL, "multi replay: sum expected 5.0");
		}

		float[6] prod_replay;
		product.get(&prod_replay);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(prod_replay[i] - 6.0f) < TOL, "multi replay: product expected 6.0");
		}
	};
}

// ============================================================================
// Entry point
// ============================================================================

// ============================================================================
// Minimal PTX-during-capture test
// ============================================================================

fn void test_ptx_during_capture(DeviceReference dev, CudaDevice* gpu)
{
	// Trivial kernel: out[i] = 42.0
	char[] source = `extern "C" __global__ void kernel(float* out, unsigned long long n) {
		unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) out[i] = 42.0f;
	}`;

	// Allocate output
	Tensor out_t = tensor::empty(F32, dev, { 6 })!!;
	char[] out_bytes = out_t.value_bytes();
	void* out_ptr = out_bytes.ptr;
	usz n = 6;

	// Pre-compile to PTX outside capture
	usz ptx_size;
	char* ptx = utils::jit_compile_to_ptx(source.ptr, source.len, &ptx_size);
	assert(ptx != null, "PTX compilation failed");

	// Load module + launch inside capture
	void*[2] args = { &out_ptr, &n };
	CudaGraphCapture captured = gpu.@cuda_graph()
	{
		void* handle = utils::load_ptx(ptx, ptx_size);
		assert(handle != null, "PTX load failed");

		utils::launch_kernel(handle, &&args, n, gpu.get_stream());
	};
	utils::jit_free_ptx(ptx);
	defer captured.deinit();

	captured.replay();
	gpu.sync();

	// Verify
	float[6] result;
	out_t.get(&result);
	for (usz i = 0; i < 6; i++)
	{
		assert(math::abs(result[i] - 42.0f) < TOL, "ptx capture: expected 42.0");
	}
}

fn void run_all(DeviceReference dev, CudaDevice* gpu)
{
	test_basic_graph_capture(dev, gpu);
	test_graph_capture_replay(dev, gpu);
	test_multiple_captures(dev, gpu);
	test_ptx_during_capture(dev, gpu);
}
