//! Gradient Tests (CUDA device)
//!
//! Tests that the backward pass correctly builds gradient computation
//! chains and produces correct gradient values for various operations.
//! Runs on the CUDA device (GPU required).

module metaphor::testing::device::cuda_device::cuda_grad_test;
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
// Test: Reduce Gradient
// ============================================================================

/// loss = sum(x), where x = [1, 2, 3, 4]
/// dloss/dx_i = 1 for all i
/// Expected: grad(x) = [1, 1, 1, 1]
fn void test_reduce_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[4] data = { 1, 2, 3, 4 };
		ulong[1] dims = { 4 };

		Tensor x = tensor::@from_data(dev, &dims, &data)!!.enable_grad();

		Tensor loss = x.reduce("i->")!!.stable();

		loss.collect()!!;
		loss.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - 1.0f) < TOL, "reduce grad should be 1.0");
		}
	};
}

// ============================================================================
// Test: Add Gradient
// ============================================================================

/// c = a + b, loss = sum(c)
/// dc/da = 1, dc/db = 1
/// Expected: grad(a) = [1,1,1,1], grad(b) = [1,1,1,1]
fn void test_add_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[4] a_data = { 1, 2, 3, 4 };
		float[4] b_data = { 5, 6, 7, 8 };
		ulong[1] dims = { 4 };

		Tensor a = tensor::@from_data(dev, &dims, &a_data)!!.enable_grad();
		Tensor b = tensor::@from_data(dev, &dims, &b_data)!!.enable_grad();

		Tensor c = a.add(b)!!;
		Tensor loss = c.reduce("i->")!!.stable();

		loss.collect()!!;
		loss.backward();

		Tensor grad_a = a.grad();
		grad_a.collect()!!;

		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[4] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(ga[i] - 1.0f) < TOL, "add grad_a should be 1.0");
		}

		float[4] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(gb[i] - 1.0f) < TOL, "add grad_b should be 1.0");
		}
	};
}

// ============================================================================
// Test: ReLU Gradient
// ============================================================================

/// y = relu(x), loss = sum(y)
/// For x = [-1, 2, -3, 4]: grad(x) = [0, 1, 0, 1]
fn void test_relu_grad(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[4] data = { -1.0, 2.0, -3.0, 4.0 };
		ulong[1] dims = { 4 };

		Tensor x = tensor::@from_data(dev, &dims, &data)!!.enable_grad();

		Tensor y = x.relu()!!;
		Tensor loss = y.reduce("i->")!!.stable();

		loss.collect()!!;
		loss.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] expected = { 0.0, 1.0, 0.0, 1.0 };
		float[4] g;
		grad_x.get(&g);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected[i]) < TOL, "relu grad mismatch");
		}
	};
}

// ============================================================================
// Test: Cat Gradient (partial requires_grad)
// ============================================================================

/// a (requires_grad) = [1, 2, 3], b (no grad) = [4, 5]
/// c = cat([a, b], dim=0), loss = sum(c)
/// grad(a) = [1, 1, 1], grad(b) = [0, 0]
fn void test_cat_grad_partial(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[3] a_data = { 1, 2, 3 };
		float[2] b_data = { 4, 5 };

		Tensor a = tensor::@from_data(dev, { 3 }, &a_data)!!.enable_grad();
		Tensor b = tensor::@from_data(dev, { 2 }, &b_data)!!;

		Tensor c = tensor_ops::cat({ a, b }, 0)!!;
		Tensor loss = c.reduce("i->")!!.stable();

		loss.collect()!!;
		loss.backward();

		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[3] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 3; i++)
		{
			assert(math::abs(ga[i] - 1.0f) < TOL, "cat grad_a should be 1.0");
		}

		float[2] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(gb[i]) < TOL, "cat grad_b should be 0.0");
		}
	};
}

// ============================================================================
// Test: Stack Gradient (partial requires_grad)
// ============================================================================

/// a (requires_grad) = [1, 2], b (no grad) = [3, 4]
/// c = stack([a, b], dim=0), loss = sum(c)
/// grad(a) = [1, 1], grad(b) = [0, 0]
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
		Tensor loss = c.reduce("ij->")!!.stable();

		loss.collect()!!;
		loss.backward();

		Tensor grad_a = a.grad();
		grad_a.collect()!!;
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[2] ga;
		grad_a.get(&ga);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(ga[i] - 1.0f) < TOL, "stack grad_a should be 1.0");
		}

		float[2] gb;
		grad_b.get(&gb);
		for (usz i = 0; i < 2; i++)
		{
			assert(math::abs(gb[i]) < TOL, "stack grad_b should be 0.0");
		}
	};
}

fn void run_all(DeviceReference dev)
{
	test_reduce_grad(dev);
	test_add_grad(dev);
	test_relu_grad(dev);
	test_cat_grad_partial(dev);
	test_stack_grad_partial(dev);
}
