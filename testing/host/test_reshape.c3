//! Reshape — Forward, Stability, and Gradient Tests
//!
//! Tests reshape operation through the full pipeline:
//! tensor creation → symbolic graph → boundary op → execute.

module metaphor::testing::host::test_reshape;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Forward: basic reshape preserves data
// ============================================================================

fn void test_reshape_1d_to_2d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 1, 2, 3, 4, 5, 6 };
	ulong[1] dims = { 6 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor y = x.reshape({ 2, 3 })!!;
	y.collect()!!;

	// Shape should be [2, 3]
	assert(y.shape().len == 2, "reshape dims");
	assert(y.shape().get(0) == 2, "reshape dim0");
	assert(y.shape().get(1) == 3, "reshape dim1");

	// Data should be unchanged
	float* r = (float*)y.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "reshape data mismatch");
	}
}

fn void test_reshape_2d_to_3d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor y = x.reshape({ 2, 2, 3 })!!;
	y.collect()!!;

	assert(y.shape().len == 3, "reshape dims");
	assert(y.shape().get(0) == 2, "reshape dim0");
	assert(y.shape().get(1) == 2, "reshape dim1");
	assert(y.shape().get(2) == 3, "reshape dim2");

	float* r = (float*)y.value_bytes().ptr;
	for (usz i = 0; i < 12; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "reshape data mismatch");
	}
}

fn void test_reshape_3d_to_1d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[24] x_data;
	for (usz i = 0; i < 24; i++) { x_data[i] = (float)i; }
	ulong[3] dims = { 2, 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor y = x.reshape({ 24 })!!;
	y.collect()!!;

	assert(y.shape().len == 1, "reshape dims");
	assert(y.shape().get(0) == 24, "reshape dim0");

	float* r = (float*)y.value_bytes().ptr;
	for (usz i = 0; i < 24; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "reshape data mismatch");
	}
}

// ============================================================================
// Forward: reshape to same shape (identity)
// ============================================================================

fn void test_reshape_identity() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 10, 20, 30, 40, 50, 60 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor y = x.reshape({ 2, 3 })!!;
	y.collect()!!;

	assert(y.shape().get(0) == 2, "identity dim0");
	assert(y.shape().get(1) == 3, "identity dim1");

	float* r = (float*)y.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "identity data mismatch");
	}
}

// ============================================================================
// Stability: reshape output used as .stable() survives liveness
// ============================================================================

fn void test_reshape_stable() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 1, 2, 3, 4, 5, 6 };
	ulong[1] dims = { 6 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// Reshape and mark stable — data should survive past normal liveness
	Tensor y = x.reshape({ 2, 3 })!!.stable();

	// Do some unrelated work to push liveness
	float[4] dummy_data = { 0, 0, 0, 0 };
	Tensor d = tensor::@from_data(dev, { 4 }, &dummy_data)!!;
	Tensor d2 = d.add(d)!!;
	d2.collect()!!;

	// Now collect the stable reshape — should still have valid data
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "stable reshape data mismatch");
	}
}

// ============================================================================
// Chained reshapes
// ============================================================================

fn void test_reshape_chain() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data;
	for (usz i = 0; i < 12; i++) { x_data[i] = (float)(i + 1); }
	ulong[1] dims = { 12 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// [12] -> [3, 4] -> [2, 6] -> [4, 3]
	Tensor y = x.reshape({ 3, 4 })!!.reshape({ 2, 6 })!!.reshape({ 4, 3 })!!;
	y.collect()!!;

	assert(y.shape().get(0) == 4, "chain dim0");
	assert(y.shape().get(1) == 3, "chain dim1");

	float* r = (float*)y.value_bytes().ptr;
	for (usz i = 0; i < 12; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "chain data mismatch");
	}
}

// ============================================================================
// Gradient: reshape backward restores input shape
// ============================================================================

fn void test_reshape_gradient_basic() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	// reshape [2,3] -> [6], then reduce to scalar
	Tensor y = x.reshape({ 6 })!!;
	Tensor loss = y.reduce("i->")!!;
	loss.collect()!!;
	loss.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	// grad should be all 1s with shape [2, 3]
	assert(grad_x.shape().len == 2, "grad dims");
	assert(grad_x.shape().get(0) == 2, "grad dim0");
	assert(grad_x.shape().get(1) == 3, "grad dim1");

	float* g = (float*)grad_x.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(g[i], 1.0), "grad should be 1.0");
	}
}

fn void test_reshape_gradient_chain() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data;
	for (usz i = 0; i < 12; i++) { x_data[i] = (float)(i + 1); }
	ulong[3] dims = { 2, 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	// [2,2,3] -> [4,3] -> [12] -> reduce
	Tensor y = x.reshape({ 4, 3 })!!.reshape({ 12 })!!;
	Tensor loss = y.reduce("i->")!!;
	loss.collect()!!;
	loss.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	// grad shape must match original [2, 2, 3]
	assert(grad_x.shape().len == 3, "chain grad dims");
	assert(grad_x.shape().get(0) == 2, "chain grad dim0");
	assert(grad_x.shape().get(1) == 2, "chain grad dim1");
	assert(grad_x.shape().get(2) == 3, "chain grad dim2");

	float* g = (float*)grad_x.value_bytes().ptr;
	for (usz i = 0; i < 12; i++)
	{
		assert(approx_eq(g[i], 1.0), "chain grad should be 1.0");
	}
}

// ============================================================================
// Gradient: reshape + arithmetic
// ============================================================================

fn void test_reshape_gradient_with_multiply() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 1, 2, 3, 4, 5, 6 };
	float[6] w_data = { 2, 2, 2, 2, 2, 2 };

	Tensor x = tensor::@from_data(dev, { 2, 3 }, &x_data)!!.enable_grad();
	Tensor w = tensor::@from_data(dev, { 6 }, &w_data)!!.enable_grad();

	// reshape x [2,3] -> [6], multiply with w [6], reduce
	Tensor x_flat = x.reshape({ 6 })!!;
	Tensor y = x_flat.mul(w)!!;
	Tensor loss = y.reduce("i->")!!;
	loss.collect()!!;
	loss.backward();

	// grad_x = w reshaped back to [2,3], so all 2s
	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	assert(grad_x.shape().len == 2, "mul grad dims");
	assert(grad_x.shape().get(0) == 2, "mul grad dim0");
	assert(grad_x.shape().get(1) == 3, "mul grad dim1");

	float* gx = (float*)grad_x.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(gx[i], 2.0), "grad_x should be 2.0");
	}

	// grad_w = x_flat values = [1,2,3,4,5,6]
	Tensor grad_w = w.grad();
	grad_w.collect()!!;

	assert(grad_w.shape().len == 1, "grad_w dims");
	assert(grad_w.shape().get(0) == 6, "grad_w dim0");

	float* gw = (float*)grad_w.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(gw[i], x_data[i]), "grad_w should match x");
	}
}
