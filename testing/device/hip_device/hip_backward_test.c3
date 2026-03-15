//! HIP Backward Pass Tests
//!
//! Tests gradient computation through various operations:
//! simple grad, chain grad, slice grad, cat grad, stack grad.
//!

module metaphor::testing::device::hip_device::hip_backward_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_ops;
import metaphor::device::common::operations;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-4;

// ============================================================================
// y = x * 2 → dy/dx = 2
// ============================================================================

fn void test_simple_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();
		Tensor two = tensor::constant(F32, dev, &dims, 2.0)!!;

		Tensor y = x.mul(two)!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - 2.0f) < TOL, "simple grad: expected 2.0");
		}
	};
}

// ============================================================================
// z = (x * 2) + 3 → dz/dx = 2
// ============================================================================

fn void test_chain_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();
		Tensor two = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor three = tensor::constant(F32, dev, &dims, 3.0)!!;

		Tensor z = x.mul(two)!!.add(three)!!;
		z.collect()!!;
		z.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - 2.0f) < TOL, "chain grad: expected 2.0");
		}
	};
}

// ============================================================================
// Slice gradient: elements in slice get gradient, others get 0
// x[1:3, 1:3] * 2 → grad_x has 2.0 in slice region, 0 elsewhere
// ============================================================================

fn void test_slice_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims_two = { 2, 2 };

		Tensor x = tensor::constant(F32, dev, { 4, 4 }, 1.0)!!.enable_grad();
		Tensor two = tensor::constant(F32, dev, &dims_two, 2.0)!!;

		operations::SliceRange[2] ranges = {
			{ .start = 1, .end = 3, .step = 1 },
			{ .start = 1, .end = 3, .step = 1 },
		};
		Tensor sliced = x.slice(&ranges)!!;
		Tensor z = sliced.mul(two)!!;
		z.collect()!!;
		z.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[16] g;
		grad_x.get(&g);

		float[16] expected = {
			0, 0, 0, 0,
			0, 2, 2, 0,
			0, 2, 2, 0,
			0, 0, 0, 0,
		};

		for (usz i = 0; i < 16; i++)
		{
			assert(math::abs(g[i] - expected[i]) < TOL, "slice grad mismatch");
		}
	};
}

// ============================================================================
// Cat gradient: grad flows back to inputs
// cat([a, b], dim=0) * 2 → grad_a = grad_b = 2.0
// ============================================================================

fn void test_cat_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims_ab = { 2, 3 };
		ulong[2] dims_two = { 4, 3 };

		Tensor a = tensor::constant(F32, dev, &dims_ab, 1.0)!!.enable_grad();
		Tensor b = tensor::constant(F32, dev, &dims_ab, 2.0)!!.enable_grad();
		Tensor two = tensor::constant(F32, dev, &dims_two, 2.0)!!;

		Tensor c = tensor_ops::cat({ a, b }, 0)!!;
		Tensor z = c.mul(two)!!;
		z.collect()!!;
		z.backward();

		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[6] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(ga[i] - 2.0f) < TOL, "cat grad_a: expected 2.0");
		}

		float[6] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(gb[i] - 2.0f) < TOL, "cat grad_b: expected 2.0");
		}
	};
}

// ============================================================================
// Stack gradient: grad flows back to inputs
// stack([a, b], dim=0) * 2 → grad_a = grad_b = 2.0
// ============================================================================

fn void test_stack_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims_ab = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims_ab, 1.0)!!.enable_grad();
		Tensor b = tensor::constant(F32, dev, &dims_ab, 2.0)!!.enable_grad();
		Tensor two = tensor::constant(F32, dev, { 2, 2, 3 }, 2.0)!!;

		Tensor c = tensor_ops::stack({ a, b }, 0)!!;
		Tensor z = c.mul(two)!!;
		z.collect()!!;
		z.backward();

		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[6] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(ga[i] - 2.0f) < TOL, "stack grad_a: expected 2.0");
		}

		float[6] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(gb[i] - 2.0f) < TOL, "stack grad_b: expected 2.0");
		}
	};
}

fn void run_all(DeviceReference dev)
{
	test_simple_grad(dev);
	test_chain_grad(dev);
	test_slice_grad(dev);
	test_cat_grad(dev);
	test_stack_grad(dev);
}
