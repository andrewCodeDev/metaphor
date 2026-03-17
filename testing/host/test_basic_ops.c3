//! Basic Operations — Forward Pass Integration Tests
//!
//! Tests element-wise operations through the full pipeline:
//! tensor creation → symbolic graph → CAS mapper → JIT compile → execute.
//!

module metaphor::testing::host::test_basic_ops;
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

// ============================================================================
// Add
// ============================================================================

fn void test_add_basic() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] a_data = { 1, 2, 3, 4 };
	float[4] b_data = { 10, 20, 30, 40 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = a.add(b)!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	assert(bytes.len >= 4 * float.sizeof, "should have data");
	float* result = (float*)bytes.ptr;

	assert(approx_eq(result[0], 11), "add[0]");
	assert(approx_eq(result[1], 22), "add[1]");
	assert(approx_eq(result[2], 33), "add[2]");
	assert(approx_eq(result[3], 44), "add[3]");
}

// ============================================================================
// Sub
// ============================================================================

fn void test_sub_basic() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] a_data = { 10, 20, 30, 40 };
	float[4] b_data = { 1, 2, 3, 4 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = a.sub(b)!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	float* result = (float*)bytes.ptr;

	assert(approx_eq(result[0], 9), "sub[0]");
	assert(approx_eq(result[1], 18), "sub[1]");
	assert(approx_eq(result[2], 27), "sub[2]");
	assert(approx_eq(result[3], 36), "sub[3]");
}

// ============================================================================
// Mul
// ============================================================================

fn void test_mul_basic() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] a_data = { 2, 3, 4, 5 };
	float[4] b_data = { 10, 10, 10, 10 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = a.mul(b)!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	float* result = (float*)bytes.ptr;

	assert(approx_eq(result[0], 20), "mul[0]");
	assert(approx_eq(result[1], 30), "mul[1]");
	assert(approx_eq(result[2], 40), "mul[2]");
	assert(approx_eq(result[3], 50), "mul[3]");
}

// ============================================================================
// Div
// ============================================================================

fn void test_div_basic() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] a_data = { 10, 20, 30, 40 };
	float[4] b_data = { 2, 4, 5, 8 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = a.div(b)!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	float* result = (float*)bytes.ptr;

	assert(approx_eq(result[0], 5), "div[0]");
	assert(approx_eq(result[1], 5), "div[1]");
	assert(approx_eq(result[2], 6), "div[2]");
	assert(approx_eq(result[3], 5), "div[3]");
}

// ============================================================================
// Chained Operations
// ============================================================================

fn void test_chain_add_mul_neg() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] a_data = { 1, 2, 3, 4 };
	float[4] b_data = { 10, 10, 10, 10 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	// (a + b) * a  then neg
	// (1+10)*1=11, (2+10)*2=24, (3+10)*3=39, (4+10)*4=56
	// neg: -11, -24, -39, -56
	Tensor sum = a.add(b)!!;
	Tensor prod = sum.mul(a)!!;
	Tensor result = prod.neg()!!;
	result.collect()!!;

	char[] bytes = result.value_bytes();
	float* data = (float*)bytes.ptr;

	assert(approx_eq(data[0], -11), "chain[0]");
	assert(approx_eq(data[1], -24), "chain[1]");
	assert(approx_eq(data[2], -39), "chain[2]");
	assert(approx_eq(data[3], -56), "chain[3]");
}

// ============================================================================
// Unary Ops
// ============================================================================

fn void test_neg() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] data = { 1, -2, 3, -4 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &data)!!;
	Tensor result = a.neg()!!;
	result.collect()!!;

	char[] bytes = result.value_bytes();
	float* r = (float*)bytes.ptr;

	assert(approx_eq(r[0], -1), "neg[0]");
	assert(approx_eq(r[1], 2), "neg[1]");
	assert(approx_eq(r[2], -3), "neg[2]");
	assert(approx_eq(r[3], 4), "neg[3]");
}

fn void test_exp() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[3] data = { 0, 1, 2 };
	ulong[1] dims = { 3 };

	Tensor a = tensor::@from_data(dev, &dims, &data)!!;
	Tensor result = a.exp()!!;
	result.collect()!!;

	char[] bytes = result.value_bytes();
	float* r = (float*)bytes.ptr;

	assert(approx_eq(r[0], 1.0), "exp(0)");
	assert(math::abs(r[1] - 2.71828) < 0.001, "exp(1)");
	assert(math::abs(r[2] - 7.38906) < 0.001, "exp(2)");
}

fn void test_abs() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] data = { -3, 2, -1, 4 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &data)!!;
	Tensor result = a.abs()!!;
	result.collect()!!;

	char[] bytes = result.value_bytes();
	float* r = (float*)bytes.ptr;

	assert(approx_eq(r[0], 3), "abs[0]");
	assert(approx_eq(r[1], 2), "abs[1]");
	assert(approx_eq(r[2], 1), "abs[2]");
	assert(approx_eq(r[3], 4), "abs[3]");
}

// ============================================================================
// 2D Operations
// ============================================================================

fn void test_add_2d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[6] a_data = { 1, 2, 3, 4, 5, 6 };
	float[6] b_data = { 10, 20, 30, 40, 50, 60 };
	ulong[2] dims = { 2, 3 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = a.add(b)!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	float* result = (float*)bytes.ptr;

	float[6] expected = { 11, 22, 33, 44, 55, 66 };
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(result[i], expected[i]), "add_2d");
	}
}

// ============================================================================
// Constant tensor operations
// ============================================================================

fn void test_constant_ops() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	ulong[1] dims = { 4 };
	Tensor a = tensor::constant(F32, dev, &dims, 3.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

	Tensor c = a.mul(b)!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	float* result = (float*)bytes.ptr;

	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(result[i], 6.0), "constant_ops: 3*2 should be 6");
	}
}
