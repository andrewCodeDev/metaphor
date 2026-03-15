//! Cat & Stack Gradient Tests (CUDA device)
//!
//! Tests that backward pass through cat/stack only produces non-zero
//! gradients for inputs that have requires_grad enabled.
//! Runs on the CUDA device (GPU required).

module metaphor::testing::device::cuda_device::cuda_catstack_grad_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_ops;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-5;

// ============================================================================
// Cat: requires_grad on first input only
// ============================================================================

/// a (requires_grad) = [1, 2, 3], b (no grad) = [4, 5]
/// c = cat([a, b], dim=0) -> [1, 2, 3, 4, 5]
/// loss = sum(c) = 15
/// grad(a) should be [1, 1, 1], grad(b) should be [0, 0]
fn void test_cat_grad_partial(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[3] a_data = { 1, 2, 3 };
		float[2] b_data = { 4, 5 };
		ulong[1] a_dims = { 3 };
		ulong[1] b_dims = { 2 };

		Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!.enable_grad();
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor c = tensor_ops::cat({ a, b }, 0)!!;
		Tensor loss = c.reduce("i->")!!;
		loss.collect()!!;
		loss.backward();

		// a has requires_grad — should get gradient of all 1s
		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		float[3] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 3; i++)
		{
			assert(math::abs(ga[i] - 1.0f) < TOL, "cat grad_a should be 1.0");
		}

		// b has no requires_grad — gradient should be all zeros
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		float[2] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(gb[i]) < TOL, "cat grad_b should be 0.0");
		}
	};
}

// ============================================================================
// Cat 2D: requires_grad on second input only
// ============================================================================

/// a (no grad) = [[1, 2], [3, 4]], b (requires_grad) = [[5], [6]]
/// c = cat([a, b], dim=1) -> [[1, 2, 5], [3, 4, 6]]
/// loss = sum(c) = 21
/// grad(a) should be all zeros, grad(b) should be [1, 1]
fn void test_cat_2d_grad_partial(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[4] a_data = { 1, 2, 3, 4 };
		float[2] b_data = { 5, 6 };

		Tensor a = tensor::@from_data(dev, { 2, 2 }, &a_data)!!;
		Tensor b = tensor::@from_data(dev, { 2, 1 }, &b_data)!!.enable_grad();

		Tensor c = tensor_ops::cat({ a, b }, 1)!!;
		Tensor loss = c.reduce("ij->")!!;
		loss.collect()!!;
		loss.backward();

		// a has no requires_grad — gradient should be all zeros
		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		float[4] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(ga[i]) < TOL, "cat_2d grad_a should be 0.0");
		}

		// b has requires_grad — should get gradient of all 1s
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		float[2] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(gb[i] - 1.0f) < TOL, "cat_2d grad_b should be 1.0");
		}
	};
}

// ============================================================================
// Stack: requires_grad on first input only
// ============================================================================

/// a (requires_grad) = [1, 2], b (no grad) = [3, 4]
/// c = stack([a, b], dim=0) -> [[1, 2], [3, 4]]
/// loss = sum(c) = 10
/// grad(a) should be [1, 1], grad(b) should be [0, 0]
fn void test_stack_grad_partial(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[2] a_data = { 1, 2 };
		float[2] b_data = { 3, 4 };
		ulong[1] dims = { 2 };

		Tensor a = tensor::@from_data(dev, &dims, &a_data)!!.enable_grad();
		Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

		Tensor c = tensor_ops::stack({ a, b }, 0)!!;
		Tensor loss = c.reduce("ij->")!!;
		loss.collect()!!;
		loss.backward();

		// a has requires_grad — should get gradient of all 1s
		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		float[2] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(ga[i] - 1.0f) < TOL, "stack grad_a should be 1.0");
		}

		// b has no requires_grad — gradient should be all zeros
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		float[2] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(gb[i]) < TOL, "stack grad_b should be 0.0");
		}
	};
}

// ============================================================================
// Public entry point
// ============================================================================

fn void run_all(DeviceReference dev)
{
	test_cat_grad_partial(dev);
	test_cat_2d_grad_partial(dev);
	test_stack_grad_partial(dev);
}
