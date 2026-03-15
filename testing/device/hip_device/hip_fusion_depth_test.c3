//! HIP Fusion Depth Limit Tests
//!
//! Verifies that the HIP compiler's depth-based decomposition
//! (MAX_UNFOLD_DEPTH=2) correctly splits deep expression trees into
//! multiple JIT kernels while maintaining correctness.

module metaphor::testing::device::hip_device::hip_fusion_depth_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-3;

// ============================================================================
// Test: large binary add chain decomposes under depth limit
// result = t0 + t1 + t2 + ... + t19  (20 tensors)
// ============================================================================

fn void test_large_add_chain_decomposition(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };
		const usz N = 20;

		tensor::Tensor[N] tensors;
		for (usz i = 0; i < N; i++)
		{
			tensors[i] = tensor::constant(F32, dev, &dims, (float)(i + 1))!!;
		}

		tensor::Tensor result = tensors[0];
		for (usz i = 1; i < N; i++)
		{
			result = result.add(tensors[i])!!;
		}
		result.collect()!!;
		dev.sync();

		// With 20 binary adds building a depth-19 tree, the compiler
		// MUST decompose into multiple execution units.
		usz num_units = graph::get().exec_seq.len();
		assert(num_units > 1,
			"large add chain should decompose into multiple execution units");

		// Verify correctness: sum of 1..20 = 210
		float[4] r;
		result.get(&r);
		float expected = (float)(N * (N + 1) / 2);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - expected) < TOL,
				"large add chain result mismatch");
		}
	};
}

// ============================================================================
// Test: deep nested binary tree decomposes
// result = ((a+b) * (c+d)) + ((e+f) * (g+h))
// Depth 3 > MAX_UNFOLD_DEPTH=2 → must decompose
// ============================================================================

fn void test_deep_binary_tree_decomposition(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;
		Tensor d = tensor::constant(F32, dev, &dims, 4.0)!!;
		Tensor e = tensor::constant(F32, dev, &dims, 5.0)!!;
		Tensor f = tensor::constant(F32, dev, &dims, 6.0)!!;
		Tensor g = tensor::constant(F32, dev, &dims, 7.0)!!;
		Tensor h = tensor::constant(F32, dev, &dims, 8.0)!!;

		Tensor ab = a.add(b)!!;
		Tensor cd = c.add(d)!!;
		Tensor ef = e.add(f)!!;
		Tensor gh = g.add(h)!!;

		Tensor left = ab.mul(cd)!!;
		Tensor right = ef.mul(gh)!!;

		Tensor result = left.add(right)!!;
		result.collect()!!;
		dev.sync();

		usz num_units = graph::get().exec_seq.len();
		assert(num_units > 1,
			"deep binary tree should decompose into multiple execution units");

		// (1+2)*(3+4) + (5+6)*(7+8) = 3*7 + 11*15 = 21 + 165 = 186
		float[4] r;
		result.get(&r);
		float expected = (float)((1+2) * (3+4) + (5+6) * (7+8));

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - expected) < TOL,
				"deep binary tree result mismatch");
		}
	};
}

// ============================================================================
// Test: shallow expression stays as single unit
// ============================================================================

fn void test_shallow_single_unit(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor a = tensor::constant(F32, dev, &dims, 3.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 7.0)!!;

		Tensor result = a.add(b)!!;
		result.collect()!!;
		dev.sync();

		usz num_units = graph::get().exec_seq.len();
		assert(num_units == 1,
			"shallow expression should be a single execution unit");

		float[4] r;
		result.get(&r);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - 10.0f) < TOL,
				"shallow add result mismatch");
		}
	};
}

// ============================================================================
// Test: depth 2 expression stays as single unit
// (a+b) * (c+d) = depth 2 = MAX_UNFOLD_DEPTH → no decomposition
// ============================================================================

fn void test_depth_two_single_unit(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;
		Tensor c = tensor::constant(F32, dev, &dims, 4.0)!!;
		Tensor d = tensor::constant(F32, dev, &dims, 5.0)!!;

		Tensor ab = a.add(b)!!;
		Tensor cd = c.add(d)!!;
		Tensor result = ab.mul(cd)!!;
		result.collect()!!;
		dev.sync();

		usz num_units = graph::get().exec_seq.len();
		assert(num_units == 1,
			"depth-2 expression should be a single execution unit");

		// (2+3) * (4+5) = 45
		float[4] r;
		result.get(&r);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - 45.0f) < TOL,
				"depth-2 result mismatch");
		}
	};
}

// ============================================================================
// Entry point for HIP test runner
// ============================================================================

fn void run_all(DeviceReference dev)
{
	test_large_add_chain_decomposition(dev);
	test_deep_binary_tree_decomposition(dev);
	test_shallow_single_unit(dev);
	test_depth_two_single_unit(dev);
}
