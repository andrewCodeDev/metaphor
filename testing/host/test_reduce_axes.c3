//! Reduce Axis Tests — Forward Pass
//!
//! Tests reduction operations along specific axes and full reductions.
//! Also tests mean via reduce + numel + div pattern.
//!
//! Mirrors forward-pass portions of:

module metaphor::testing::host::test_reduce_axes;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-4;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Sum all elements: [2,3] -> scalar
// ============================================================================

fn void test_sum_all() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3], [4, 5, 6]]  sum = 21
	float[6] data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("ij->")!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	assert(approx_eq(r[0], 21.0), "sum_all=21");
}

// ============================================================================
// Sum axis 0 (collapse rows): [2,3] -> [3]
// ============================================================================

fn void test_sum_axis0() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3], [4, 5, 6]]
	// Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
	float[6] data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("ij->j")!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	assert(approx_eq(r[0], 5), "sum_axis0[0]=5");
	assert(approx_eq(r[1], 7), "sum_axis0[1]=7");
	assert(approx_eq(r[2], 9), "sum_axis0[2]=9");
}

// ============================================================================
// Sum axis 1 (collapse columns): [2,3] -> [2]
// ============================================================================

fn void test_sum_axis1() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3], [4, 5, 6]]
	// Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
	float[6] data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("ij->i")!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	assert(approx_eq(r[0], 6), "sum_axis1[0]=6");
	assert(approx_eq(r[1], 15), "sum_axis1[1]=15");
}

// ============================================================================
// Mean via reduce + numel + div: [2,3] -> scalar
// ============================================================================

fn void test_mean_all() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3], [4, 5, 6]]  mean = 21/6 = 3.5
	float[6] data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor sum = x.reduce("ij->")!!;
	Tensor count = x.numel()!!;
	Tensor mean = sum.div(count)!!;
	mean.collect()!!;

	float* r = (float*)mean.value_bytes().ptr;
	assert(approx_eq(r[0], 3.5), "mean_all=3.5");
}

// ============================================================================
// Sum of larger tensor: [3,4] -> scalar, axis0, axis1
// ============================================================================

fn void test_sum_3x4_all() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
	// sum = 78
	float[12] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("ij->")!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	assert(approx_eq(r[0], 78.0), "sum_3x4=78");
}

fn void test_sum_3x4_axis0() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
	// Sum axis 0: [15, 18, 21, 24]
	float[12] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("ij->j")!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	assert(approx_eq(r[0], 15), "sum_axis0[0]=15");
	assert(approx_eq(r[1], 18), "sum_axis0[1]=18");
	assert(approx_eq(r[2], 21), "sum_axis0[2]=21");
	assert(approx_eq(r[3], 24), "sum_axis0[3]=24");
}

fn void test_sum_3x4_axis1() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
	// Sum axis 1: [10, 26, 42]
	float[12] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("ij->i")!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;
	assert(approx_eq(r[0], 10), "sum_axis1[0]=10");
	assert(approx_eq(r[1], 26), "sum_axis1[1]=26");
	assert(approx_eq(r[2], 42), "sum_axis1[2]=42");
}

// ============================================================================
// Mean via reduce + div pattern for [3,4]
// ============================================================================

fn void test_mean_3x4() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[12] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	Tensor x = tensor::@from_data(dev, { 3, 4 }, &data)!!;

	Tensor sum = x.reduce("ij->")!!;
	Tensor count = x.numel()!!;
	Tensor mean = sum.div(count)!!;
	mean.collect()!!;

	float* r = (float*)mean.value_bytes().ptr;
	// 78 / 12 = 6.5
	assert(approx_eq(r[0], 6.5), "mean_3x4=6.5");
}

// ============================================================================
// Ellipsis reduce: [2,3,4] -> [2,3] via "...i->"
// Verifies batch dims are prepended (not appended) in walk_reduce
// ============================================================================

fn void test_ellipsis_reduce_last_axis() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[24] data;
	for (usz i = 0; i < 24; i++) data[i] = (float)(i + 1);
	ulong[3] dims = { 2, 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!;

	Tensor y = x.reduce("...i->...")!!;
	y.collect()!!;

	assert(y.shape().len == 2, "ellipsis output should be rank 2");
	assert(y.shape().get(0) == 2, "ellipsis output dim0 should be 2");
	assert(y.shape().get(1) == 3, "ellipsis output dim1 should be 3");

	float[6] r;
	y.get(&r);
	assert(approx_eq(r[0], 10.0), "ellipsis_reduce [0,0]=10");
	assert(approx_eq(r[1], 26.0), "ellipsis_reduce [0,1]=26");
	assert(approx_eq(r[2], 42.0), "ellipsis_reduce [0,2]=42");
	assert(approx_eq(r[3], 58.0), "ellipsis_reduce [1,0]=58");
	assert(approx_eq(r[4], 74.0), "ellipsis_reduce [1,1]=74");
	assert(approx_eq(r[5], 90.0), "ellipsis_reduce [1,2]=90");
}

fn void test_ellipsis_reduce_backward() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[24] data;
	for (usz i = 0; i < 24; i++) data[i] = (float)(i + 1);
	ulong[3] dims = { 2, 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &data)!!.enable_grad();

	// Ellipsis reduce + backward: gradient should expand back to input shape
	Tensor reduced = x.reduce("...i->...")!!;
	Tensor loss = reduced.reduce("ij->")!!;
	loss.collect()!!;

	float[1] loss_val;
	loss.get(&loss_val);
	assert(approx_eq(loss_val[0], 300.0), "ellipsis_reduce_bwd loss=300");

	loss.backward();

	Tensor g = x.grad();
	g.collect()!!;

	float[24] grad;
	g.get(&grad);
	for (usz i = 0; i < 24; i++)
	{
		assert(approx_eq(grad[i], 1.0), "ellipsis_reduce_bwd grad=1");
	}
}
