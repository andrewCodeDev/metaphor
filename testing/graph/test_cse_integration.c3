//! CSE (Common Subexpression Elimination) Integration Tests
//!
//! Tests that the operation cache correctly deduplicates tensor operations
//! through the full tensor API when op_caching is enabled.
//!

module metaphor::testing::graph::test_cse_integration;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// Flags with CSE enabled (all other defaults preserved).
const graph::Flags CSE_FLAGS = {
	.eager_teardown = true,
	.track_ops = true,
	.allow_recompute = true,
	.symbolic_recompute = true,
	.op_caching = true,
};

// ============================================================================
// Core CSE behavior
// ============================================================================

fn void test_same_add_returns_same_tensor() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

		Tensor c1 = a.add(b)!!;
		Tensor c2 = a.add(b)!!;

		assert(c1.core() == c2.core(), "same add should return same tensor (CSE)");
	};
}

fn void test_same_mul_returns_same_tensor() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 3.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 4.0)!!;

		Tensor c1 = a.mul(b)!!;
		Tensor c2 = a.mul(b)!!;

		assert(c1.core() == c2.core(), "same mul should return same tensor (CSE)");
	};
}

fn void test_different_ops_return_different_tensors() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

		Tensor add_result = a.add(b)!!;
		Tensor mul_result = a.mul(b)!!;

		assert(add_result.core() != mul_result.core(),
			"different ops should return different tensors");
	};
}

// ============================================================================
// CSE with chained operations
// ============================================================================

fn void test_chain_with_cse() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 3.0)!!;

		// First chain: c = a + b, d = c * a
		Tensor c = a.add(b)!!;
		Tensor d = c.mul(a)!!;

		// Second chain: same operations, should hit cache
		Tensor e = a.add(b)!!;
		Tensor f = e.mul(a)!!;

		assert(c.core() == e.core(), "a+b should be cached");
		assert(d.core() == f.core(), "c*a should be cached (c == e from CSE)");
	};
}

// ============================================================================
// Unary map CSE
// ============================================================================

fn void test_same_unary_returns_same_tensor() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;

		Tensor n1 = a.neg()!!;
		Tensor n2 = a.neg()!!;

		assert(n1.core() == n2.core(), "same neg should return same tensor (CSE)");
	};
}

fn void test_different_unary_returns_different_tensors() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;

		Tensor neg_result = a.neg()!!;
		Tensor exp_result = a.exp()!!;

		assert(neg_result.core() != exp_result.core(),
			"different unary ops should return different tensors");
	};
}

// ============================================================================
// Correctness: cached tensors compute correct values
// ============================================================================

fn void test_cached_tensor_correct_values() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 3.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 4.0)!!;

		Tensor c1 = a.add(b)!!;
		Tensor c2 = a.add(b)!!;

		// c1 and c2 should be the same tensor core
		assert(c1.core() == c2.core(), "CSE should return same tensor");

		// Collect and verify values
		c1.collect()!!;

		char[] bytes = c1.value_bytes();
		assert(bytes.len >= 4 * float.sizeof, "should have data");
		float* result = (float*)bytes.ptr;

		assert(approx_eq(result[0], 7.0), "add[0] should be 7");
		assert(approx_eq(result[1], 7.0), "add[1] should be 7");
		assert(approx_eq(result[2], 7.0), "add[2] should be 7");
		assert(approx_eq(result[3], 7.0), "add[3] should be 7");
	};
}

// ============================================================================
// Statistics tracking through the graph
// ============================================================================

fn void test_cache_stats_tracking() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph(CSE_FLAGS)
	{
		graph::Graph* g = graph::get();
		ulong initial_hits = g.cse_cache.hits;

		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

		a.add(b)!!;  // miss + register
		a.add(b)!!;  // hit

		assert(g.cse_cache.hits > initial_hits, "expected cache hit on second add");
	};
}

// ============================================================================
// CSE disabled by default
// ============================================================================

fn void test_cse_disabled_returns_different_tensors() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Default flags: op_caching = false
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

		Tensor c1 = a.add(b)!!;
		Tensor c2 = a.add(b)!!;

		assert(c1.core() != c2.core(),
			"with op_caching=false, same add should return different tensors");
	};
}
