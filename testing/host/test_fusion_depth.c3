//! Fusion Depth Limit Tests
//!
//! Verifies that large binary operation trees produce correct results
//! and that the compiler handles deep expression trees.
//!
//! For the host compiler, expressions are compiled as-is.
//! For the CUDA compiler, compile_elementwise applies depth-based
//! decomposition (MAX_UNFOLD_DEPTH=2) to prevent monolithic kernels.

module metaphor::testing::host::test_fusion_depth;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-4;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Test: large binary add chain produces correct result
// result = t0 + t1 + t2 + ... + t19  (20 tensors added together)
// ============================================================================

fn void test_large_add_chain() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[1] dims = { 4 };
		const usz N = 20;

		// Create N tensors with values 1.0, 2.0, ..., N
		tensor::Tensor[N] tensors;
		for (usz i = 0; i < N; i++)
		{
			tensors[i] = tensor::constant(F32, dev, &dims, (float)(i + 1))!!;
		}

		// Chain: result = t0 + t1 + t2 + ... + t19
		tensor::Tensor result = tensors[0];
		for (usz i = 1; i < N; i++)
		{
			result = result.add(tensors[i])!!;
		}
		result.collect()!!;

		// Verify correctness: sum of 1..20 = 210
		float* r = (float*)result.value_bytes().ptr;
		float expected = (float)(N * (N + 1) / 2);

		for (usz i = 0; i < 4; i++)
		{
			assert(approx_eq(r[i], expected),
				"large add chain result mismatch");
		}

		// Record execution unit count for diagnostics
		usz num_units = graph::get().exec_seq.len();
		assert(num_units >= 1, "should have at least 1 execution unit");
	};
}

// ============================================================================
// Test: deep nested binary tree produces correct result
// result = ((a+b) * (c+d)) + ((e+f) * (g+h))
// ============================================================================

fn void test_deep_binary_tree() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		tensor::Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		tensor::Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
		tensor::Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;
		tensor::Tensor d = tensor::constant(F32, dev, &dims, 4.0)!!;
		tensor::Tensor e = tensor::constant(F32, dev, &dims, 5.0)!!;
		tensor::Tensor f = tensor::constant(F32, dev, &dims, 6.0)!!;
		tensor::Tensor g = tensor::constant(F32, dev, &dims, 7.0)!!;
		tensor::Tensor h = tensor::constant(F32, dev, &dims, 8.0)!!;

		// Build: ((a+b) * (c+d)) + ((e+f) * (g+h))
		tensor::Tensor ab = a.add(b)!!;
		tensor::Tensor cd = c.add(d)!!;
		tensor::Tensor ef = e.add(f)!!;
		tensor::Tensor gh = g.add(h)!!;

		tensor::Tensor left = ab.mul(cd)!!;
		tensor::Tensor right = ef.mul(gh)!!;

		tensor::Tensor result = left.add(right)!!;
		result.collect()!!;

		// Verify: (1+2)*(3+4) + (5+6)*(7+8) = 3*7 + 11*15 = 21 + 165 = 186
		float* r = (float*)result.value_bytes().ptr;
		float expected = (float)((1+2) * (3+4) + (5+6) * (7+8));

		for (usz i = 0; i < 4; i++)
		{
			assert(approx_eq(r[i], expected),
				"deep binary tree result mismatch");
		}
	};
}

// ============================================================================
// Test: shallow expression stays as single unit
// result = a + b  — depth 1, exactly 1 execution unit
// ============================================================================

fn void test_shallow_single_unit() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		tensor::Tensor a = tensor::constant(F32, dev, &dims, 3.0)!!;
		tensor::Tensor b = tensor::constant(F32, dev, &dims, 7.0)!!;

		tensor::Tensor result = a.add(b)!!;
		result.collect()!!;

		usz num_units = graph::get().exec_seq.len();
		assert(num_units == 1,
			"shallow expression should be a single execution unit");

		float* r = (float*)result.value_bytes().ptr;
		for (usz i = 0; i < 4; i++)
		{
			assert(approx_eq(r[i], 10.0f),
				"shallow add result mismatch");
		}
	};
}

// ============================================================================
// Test: depth 2 expression correctness
// result = (a+b) * (c+d)
// ============================================================================

fn void test_depth_two_correctness() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		tensor::Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		tensor::Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;
		tensor::Tensor c = tensor::constant(F32, dev, &dims, 4.0)!!;
		tensor::Tensor d = tensor::constant(F32, dev, &dims, 5.0)!!;

		// (a+b) * (c+d) = depth 2
		tensor::Tensor ab = a.add(b)!!;
		tensor::Tensor cd = c.add(d)!!;
		tensor::Tensor result = ab.mul(cd)!!;
		result.collect()!!;

		// (2+3) * (4+5) = 5 * 9 = 45
		float* r = (float*)result.value_bytes().ptr;
		for (usz i = 0; i < 4; i++)
		{
			assert(approx_eq(r[i], 45.0f),
				"depth-2 result mismatch");
		}
	};
}
