//! CUDA Basic Operations Tests
//!
//! Tests elementwise operations, contractions, chained ops,
//! graph caching/replay, and structural ops (slice, cat, stack, gather)
//! through the full CUDA pipeline.
//!

module metaphor::testing::device::cuda_device::cuda_basic_ops_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-5;

// ============================================================================
// Elementwise: add
// ============================================================================

fn void test_elementwise_add(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;

		Tensor c = a.add(b)!!;
		c.collect()!!;
		dev.sync();

		float[6] result;
		c.get(&result);

		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - 5.0f) < TOL, "add: expected 5.0");
		}
	};
}

// ============================================================================
// Elementwise: mul
// ============================================================================

fn void test_elementwise_mul(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 4.0)!!;

		Tensor c = a.mul(b)!!;
		c.collect()!!;
		dev.sync();

		float[6] result;
		c.get(&result);

		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - 8.0f) < TOL, "mul: expected 8.0");
		}
	};
}

// ============================================================================
// Elementwise: neg
// ============================================================================

fn void test_elementwise_neg(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 5.0)!!;

		Tensor b = a.neg()!!;
		b.collect()!!;
		dev.sync();

		float[6] result;
		b.get(&result);

		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - (-5.0f)) < TOL, "neg: expected -5.0");
		}
	};
}

// ============================================================================
// Matmul
// ============================================================================

fn void test_matmul(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims_b = { 3, 4 };

		Tensor a = tensor::constant(F32, dev, { 2, 3 }, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims_b, 1.0)!!;

		Tensor c = a.matmul(b)!!;
		c.collect()!!;
		dev.sync();

		float[8] result;
		c.get(&result);

		// Each element = dot of row of ones (len 3) · col of ones (len 3) = 3.0
		for (usz i = 0; i < 8; i++)
		{
			assert(math::abs(result[i] - 3.0f) < TOL, "matmul: expected 3.0");
		}
	};
}

// ============================================================================
// Chained ops: (1 + 2) * 3 = 9
// ============================================================================

fn void test_chained_ops(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

		Tensor result_t = a.add(b)!!.mul(c)!!;
		result_t.collect()!!;
		dev.sync();

		float[6] result;
		result_t.get(&result);

		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - 9.0f) < TOL, "chained: expected 9.0");
		}
	};
}

// ============================================================================
// Linear pattern: matmul + bias = 3.0 + 0.5 = 3.5
// ============================================================================

fn void test_linear_pattern(DeviceReference dev)
{
	graph::@subgraph()
	{

		Tensor x = tensor::constant(F32, dev, { 2, 3 }, 1.0)!!;
		Tensor w = tensor::constant(F32, dev, { 3, 4 }, 1.0)!!;
		Tensor bias = tensor::constant(F32, dev, { 2, 4 }, 0.5)!!;

		Tensor output = x.matmul(w)!!.add(bias)!!;
		output.collect()!!;
		dev.sync();

		float[8] result;
		output.get(&result);

		for (usz i = 0; i < 8; i++)
		{
			assert(math::abs(result[i] - 3.5f) < TOL, "linear: expected 3.5");
		}
	};
}

// ============================================================================
// Graph replay: collect twice, same result
// ============================================================================

fn void test_graph_replay(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;

		Tensor c = a.add(b)!!;
		// First collect — compiles graph
		c.collect()!!;
		// Second collect — replays graph
		c.collect()!!;
		dev.sync();

		float[6] result;
		c.get(&result);

		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(result[i] - 5.0f) < TOL, "replay: expected 5.0");
		}
	};
}

// ============================================================================
// Multiple observables
// ============================================================================

fn void test_multiple_observables(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };

		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

		Tensor c = a.add(b)!!;
		c.collect()!!;

		Tensor d = a.mul(b)!!;
		d.collect()!!;
		dev.sync();

		float[6] c_result;
		c.get(&c_result);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(c_result[i] - 3.0f) < TOL, "multi-obs add: expected 3.0");
		}

		float[6] d_result;
		d.get(&d_result);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(d_result[i] - 2.0f) < TOL, "multi-obs mul: expected 2.0");
		}
	};
}

fn void run_all(DeviceReference dev)
{
	test_elementwise_add(dev);
	test_elementwise_mul(dev);
	test_elementwise_neg(dev);
	test_matmul(dev);
	test_chained_ops(dev);
	test_linear_pattern(dev);
	test_graph_replay(dev);
	test_multiple_observables(dev);
}
