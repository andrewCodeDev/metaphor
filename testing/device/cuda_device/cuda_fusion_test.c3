//! CUDA Fusion Pass Verification Tests
//!
//! Tests that CUDA elementwise fusion:
//! 1. Produces correct forward/backward results for fused chains
//! 2. Respects backward requirements (sigmoid/tanh/sqrt only last)
//! 3. Respects input count limits
//!

module metaphor::testing::device::cuda_device::cuda_fusion_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-3;

// ============================================================================
// Long unary chain: y = relu(exp(neg(sin(cos(x)))))
// ============================================================================

fn void test_long_unary_chain(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 0.5)!!.enable_grad();

		Tensor y = x.cos()!!.sin()!!.neg()!!.exp()!!.relu()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		double x_val = 0.5;
		double v1 = math::cos(x_val);
		double v2 = math::sin(v1);
		double v3 = -v2;
		double v4 = math::exp(v3);

		double dy_dv4 = 1.0;
		double dv4_dv3 = v4;
		double dv3_dv2 = -1.0;
		double dv2_dv1 = math::cos(v1);
		double dv1_dx = -math::sin(x_val);

		float expected = (float)(dy_dv4 * dv4_dv3 * dv3_dv2 * dv2_dv1 * dv1_dx);

		const float CHAIN_TOL = 5e-2;
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < CHAIN_TOL, "long unary chain grad mismatch");
		}
	};
}

// ============================================================================
// Binary chain limit: x + a + b + c + d + e + f + g (8 binary ops)
// Should split due to input limit. grad = 1.0
// ============================================================================

fn void test_binary_chain_limit(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();
		Tensor a = tensor::constant(F32, dev, &dims, 0.1)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 0.2)!!;
		Tensor c = tensor::constant(F32, dev, &dims, 0.3)!!;
		Tensor d = tensor::constant(F32, dev, &dims, 0.4)!!;
		Tensor e = tensor::constant(F32, dev, &dims, 0.5)!!;
		Tensor f = tensor::constant(F32, dev, &dims, 0.6)!!;
		Tensor g_t = tensor::constant(F32, dev, &dims, 0.7)!!;

		Tensor y = x.add(a)!!.add(b)!!.add(c)!!.add(d)!!.add(e)!!.add(f)!!.add(g_t)!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - 1.0f) < TOL, "binary chain grad: expected 1.0");
		}
	};
}

// ============================================================================
// Sigmoid last OK: y = sigmoid(-x)
// ============================================================================

fn void test_sigmoid_last_ok(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 2.0)!!.enable_grad();

		Tensor y = x.neg()!!.sigmoid()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		float x_val = 2.0f;
		float sig = 1.0f / (1.0f + math::exp(x_val));
		float expected = -sig * (1.0f - sig);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < TOL, "sigmoid last grad mismatch");
		}
	};
}

// ============================================================================
// Sigmoid NOT last: y = -sigmoid(x) — must not fuse together
// ============================================================================

fn void test_sigmoid_not_last(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

		Tensor y = x.sigmoid()!!.neg()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		float x_val = 1.0f;
		float sig = 1.0f / (1.0f + math::exp(-x_val));
		float expected = -sig * (1.0f - sig);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < TOL, "sigmoid not-last grad mismatch");
		}
	};
}

// ============================================================================
// Tanh last OK: y = tanh(-x)
// ============================================================================

fn void test_tanh_last_ok(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 0.5)!!.enable_grad();

		Tensor y = x.neg()!!.tanh()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		float x_val = 0.5f;
		float tanh_val = (float)math::tanh(-x_val);
		float expected = -(1.0f - tanh_val * tanh_val);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < TOL, "tanh last grad mismatch");
		}
	};
}

// ============================================================================
// Sqrt last OK: y = sqrt(exp(x))
// ============================================================================

fn void test_sqrt_last_ok(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

		Tensor y = x.exp()!!.sqrt()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		float x_val = 1.0f;
		float expected = 0.5f * math::exp(x_val / 2.0f);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < TOL, "sqrt last grad mismatch");
		}
	};
}

// ============================================================================
// ReLU middle OK: y = -exp(relu(x))
// ============================================================================

fn void test_relu_middle_ok(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 0.5)!!.enable_grad();

		Tensor y = x.relu()!!.exp()!!.neg()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		float x_val = 0.5f;
		float expected = -math::exp(x_val);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < TOL, "relu middle grad mismatch");
		}
	};
}

// ============================================================================
// Mixed chain: y = sigmoid(-x + a)
// ============================================================================

fn void test_mixed_chain(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();
		Tensor a = tensor::constant(F32, dev, &dims, 0.5)!!;

		Tensor y = x.neg()!!.add(a)!!.sigmoid()!!;
		y.collect()!!;
		y.backward();

		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[4] g;
		grad_x.get(&g);

		float x_val = 1.0f;
		float a_val = 0.5f;
		float sig = 1.0f / (1.0f + math::exp(-(-x_val + a_val)));
		float expected = -sig * (1.0f - sig);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(g[i] - expected) < TOL, "mixed chain grad mismatch");
		}
	};
}

fn void run_all(DeviceReference dev)
{
	test_long_unary_chain(dev);
	test_binary_chain_limit(dev);
	test_sigmoid_last_ok(dev);
	test_sigmoid_not_last(dev);
	test_tanh_last_ok(dev);
	test_sqrt_last_ok(dev);
	test_relu_middle_ok(dev);
	test_mixed_chain(dev);
}
