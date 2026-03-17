//! Slice, Cat, Stack, Transpose & Squeeze — Forward Pass Tests
//!
//! Tests shape-manipulation operations through the full pipeline:
//! tensor creation → symbolic graph → CAS mapper → JIT compile → execute.
//!

module metaphor::testing::host::test_slice_cat_stack;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::device::common::operations;
import metaphor::tensor::tensor_ops;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Slice Forward
// ============================================================================

fn void test_slice_first_half() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// x = [0, 1, 2, 3, 4, 5, 6, 7]
	float[8] x_data;
	for (usz i = 0; i < 8; i++) { x_data[i] = (float)i; }
	ulong[1] dims = { 8 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// slice [0:4]
	operations::SliceRange[1] ranges = { { .start = 0, .end = 4, .step = 1 } };
	Tensor sliced = x.slice(&ranges)!!;
	sliced.collect()!!;

	float* r = (float*)sliced.value_bytes().ptr;
	assert(approx_eq(r[0], 0), "slice[0]");
	assert(approx_eq(r[1], 1), "slice[1]");
	assert(approx_eq(r[2], 2), "slice[2]");
	assert(approx_eq(r[3], 3), "slice[3]");
}

fn void test_slice_second_half() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[8] x_data;
	for (usz i = 0; i < 8; i++) { x_data[i] = (float)i; }
	ulong[1] dims = { 8 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// slice [4:8]
	operations::SliceRange[1] ranges = { { .start = 4, .end = 8, .step = 1 } };
	Tensor sliced = x.slice(&ranges)!!;
	sliced.collect()!!;

	float* r = (float*)sliced.value_bytes().ptr;
	assert(approx_eq(r[0], 4), "slice[0]");
	assert(approx_eq(r[1], 5), "slice[1]");
	assert(approx_eq(r[2], 6), "slice[2]");
	assert(approx_eq(r[3], 7), "slice[3]");
}

fn void test_slice_with_step() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[8] x_data;
	for (usz i = 0; i < 8; i++) { x_data[i] = (float)i; }
	ulong[1] dims = { 8 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// slice [0:8:2] -> every other element
	operations::SliceRange[1] ranges = { { .start = 0, .end = 8, .step = 2 } };
	Tensor sliced = x.slice(&ranges)!!;
	sliced.collect()!!;

	float* r = (float*)sliced.value_bytes().ptr;
	assert(approx_eq(r[0], 0), "step[0]");
	assert(approx_eq(r[1], 2), "step[1]");
	assert(approx_eq(r[2], 4), "step[2]");
	assert(approx_eq(r[3], 6), "step[3]");
}

fn void test_slice_2d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// 4x4 matrix: 0..15
	float[16] x_data;
	for (usz i = 0; i < 16; i++) { x_data[i] = (float)i; }
	ulong[2] dims = { 4, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// slice [1:3, 1:3] -> [[5, 6], [9, 10]]
	operations::SliceRange[2] ranges = {
		{ .start = 1, .end = 3, .step = 1 },
		{ .start = 1, .end = 3, .step = 1 },
	};
	Tensor sliced = x.slice(&ranges)!!;
	sliced.collect()!!;

	float* r = (float*)sliced.value_bytes().ptr;
	assert(approx_eq(r[0], 5), "2d_slice[0,0]");
	assert(approx_eq(r[1], 6), "2d_slice[0,1]");
	assert(approx_eq(r[2], 9), "2d_slice[1,0]");
	assert(approx_eq(r[3], 10), "2d_slice[1,1]");
}

fn void test_slice_then_compute() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[8] x_data;
	for (usz i = 0; i < 8; i++) { x_data[i] = (float)i; }

	Tensor x = tensor::@from_data(dev, { 8 }, &x_data)!!;

	operations::SliceRange[1] ranges = { { .start = 0, .end = 4, .step = 1 } };
	Tensor sliced = x.slice(&ranges)!!;

	Tensor ten = tensor::constant(F32, dev, { 4 }, 10.0)!!;
	Tensor result = sliced.add(ten)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 10), "slice+add[0]");
	assert(approx_eq(r[1], 11), "slice+add[1]");
	assert(approx_eq(r[2], 12), "slice+add[2]");
	assert(approx_eq(r[3], 13), "slice+add[3]");
}

// ============================================================================
// Transpose (Permute) Forward
// ============================================================================

fn void test_transpose_2d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// [[0, 1, 2], [3, 4, 5]] -> [[0, 3], [1, 4], [2, 5]]
	float[6] x_data;
	for (usz i = 0; i < 6; i++) { x_data[i] = (float)i; }
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor transposed = x.transpose(0, 1)!!;
	transposed.collect()!!;

	float* r = (float*)transposed.value_bytes().ptr;
	float[6] expected = { 0, 3, 1, 4, 2, 5 };
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], expected[i]), "transpose_2d");
	}
}

// ============================================================================
// Cat Forward
// ============================================================================

fn void test_cat_1d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[3] a_data = { 1, 2, 3 };
	float[2] b_data = { 4, 5 };
	ulong[1] a_dims = { 3 };
	ulong[1] b_dims = { 2 };

	Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

	Tensor c = tensor_ops::cat({ a, b }, 0)!!;
	c.collect()!!;

	float* r = (float*)c.value_bytes().ptr;
	float[5] expected = { 1, 2, 3, 4, 5 };
	for (usz i = 0; i < 5; i++)
	{
		assert(approx_eq(r[i], expected[i]), "cat_1d");
	}
}

fn void test_cat_2d_dim0() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// a = [[1, 2], [3, 4]], b = [[5, 6]]
	float[4] a_data = { 1, 2, 3, 4 };
	float[2] b_data = { 5, 6 };
	ulong[2] a_dims = { 2, 2 };
	ulong[2] b_dims = { 1, 2 };

	Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

	Tensor c = tensor_ops::cat({ a, b }, 0)!!;
	c.collect()!!;

	// Shape should be [3, 2]
	assert(c.shape().get(0) == 3, "cat_2d_dim0: rows=3");
	assert(c.shape().get(1) == 2, "cat_2d_dim0: cols=2");

	float* r = (float*)c.value_bytes().ptr;
	float[6] expected = { 1, 2, 3, 4, 5, 6 };
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], expected[i]), "cat_2d_dim0");
	}
}

fn void test_cat_2d_dim1() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// a = [[1, 2], [3, 4]], b = [[5], [6]]
	float[4] a_data = { 1, 2, 3, 4 };
	float[2] b_data = { 5, 6 };
	ulong[2] a_dims = { 2, 2 };
	ulong[2] b_dims = { 2, 1 };

	Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

	Tensor c = tensor_ops::cat({ a, b }, 1)!!;
	c.collect()!!;

	// Shape should be [2, 3]
	assert(c.shape().get(0) == 2, "cat_2d_dim1: rows=2");
	assert(c.shape().get(1) == 3, "cat_2d_dim1: cols=3");

	float* r = (float*)c.value_bytes().ptr;
	// [[1, 2, 5], [3, 4, 6]] row-major
	float[6] expected = { 1, 2, 5, 3, 4, 6 };
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], expected[i]), "cat_2d_dim1");
	}
}

fn void test_cat_three_tensors() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[2] a_data = { 1, 2 };
	float[3] b_data = { 3, 4, 5 };
	float[2] c_data = { 6, 7 };

	Tensor a = tensor::@from_data(dev, { 2 }, &a_data)!!;
	Tensor b = tensor::@from_data(dev, { 3 }, &b_data)!!;
	Tensor c = tensor::@from_data(dev, { 2 }, &c_data)!!;

	Tensor result = tensor_ops::cat({ a, b, c }, 0)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	float[7] expected = { 1, 2, 3, 4, 5, 6, 7 };
	for (usz i = 0; i < 7; i++)
	{
		assert(approx_eq(r[i], expected[i]), "cat_three");
	}
}

// ============================================================================
// Stack Forward
// ============================================================================

fn void test_stack_1d_dim0() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[2] a_data = { 1, 2 };
	float[2] b_data = { 3, 4 };
	ulong[1] dims = { 2 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = tensor_ops::stack({ a, b }, 0)!!;
	c.collect()!!;

	// Shape should be [2, 2]
	assert(c.shape().len == 2, "stack_dim0: ndim=2");
	assert(c.shape().get(0) == 2, "stack_dim0: dim0=2");
	assert(c.shape().get(1) == 2, "stack_dim0: dim1=2");

	float* r = (float*)c.value_bytes().ptr;
	float[4] expected = { 1, 2, 3, 4 };
	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(r[i], expected[i]), "stack_dim0");
	}
}

fn void test_stack_1d_dim1() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[2] a_data = { 1, 2 };
	float[2] b_data = { 3, 4 };
	ulong[1] dims = { 2 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = tensor_ops::stack({ a, b }, 1)!!;
	c.collect()!!;

	float* r = (float*)c.value_bytes().ptr;
	// [[1, 3], [2, 4]] -> row-major: [1, 3, 2, 4]
	float[4] expected = { 1, 3, 2, 4 };
	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(r[i], expected[i]), "stack_dim1");
	}
}

fn void test_stack_2d_dim0() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] a_data;
	float[6] b_data;
	for (usz i = 0; i < 6; i++) { a_data[i] = (float)i; }
	for (usz i = 0; i < 6; i++) { b_data[i] = (float)(i + 6); }
	ulong[2] dims = { 2, 3 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = tensor_ops::stack({ a, b }, 0)!!;
	c.collect()!!;

	// Shape: [2, 2, 3]
	assert(c.shape().len == 3, "stack_2d: ndim=3");
	assert(c.shape().get(0) == 2, "stack_2d: dim0=2");
	assert(c.shape().get(1) == 2, "stack_2d: dim1=2");
	assert(c.shape().get(2) == 3, "stack_2d: dim2=3");

	float* r = (float*)c.value_bytes().ptr;
	float[12] expected = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
	for (usz i = 0; i < 12; i++)
	{
		assert(approx_eq(r[i], expected[i]), "stack_2d");
	}
}

fn void test_stack_three_tensors() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[2] a_data = { 1, 2 };
	float[2] b_data = { 3, 4 };
	float[2] c_data = { 5, 6 };
	ulong[1] dims = { 2 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;
	Tensor c = tensor::@from_data(dev, &dims, &c_data)!!;

	Tensor result = tensor_ops::stack({ a, b, c }, 0)!!;
	result.collect()!!;

	// Shape: [3, 2]
	assert(result.shape().len == 2, "stack_three: ndim=2");
	assert(result.shape().get(0) == 3, "stack_three: dim0=3");
	assert(result.shape().get(1) == 2, "stack_three: dim1=2");

	float* r = (float*)result.value_bytes().ptr;
	float[6] expected = { 1, 2, 3, 4, 5, 6 };
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], expected[i]), "stack_three");
	}
}

// ============================================================================
// Squeeze / Unsqueeze
// ============================================================================

fn void test_unsqueeze() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// Unsqueeze at dim 0: [2, 3] -> [1, 2, 3]
	Tensor unsq = x.unsqueeze(0)!!;
	assert(unsq.shape().len == 3, "unsqueeze: ndim=3");
	assert(unsq.shape().get(0) == 1, "unsqueeze: dim0=1");
	assert(unsq.shape().get(1) == 2, "unsqueeze: dim1=2");
	assert(unsq.shape().get(2) == 3, "unsqueeze: dim2=3");

	unsq.collect()!!;

	float* r = (float*)unsq.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "unsqueeze data");
	}
}

fn void test_squeeze() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] x_data = { 1, 2, 3, 4, 5, 6 };

	Tensor x = tensor::@from_data(dev, { 1, 2, 3 }, &x_data)!!;

	// Squeeze dim 0: [1, 2, 3] -> [2, 3]
	Tensor sq = x.squeeze(0)!!;
	assert(sq.shape().len == 2, "squeeze: ndim=2");
	assert(sq.shape().get(0) == 2, "squeeze: dim0=2");
	assert(sq.shape().get(1) == 3, "squeeze: dim1=3");

	sq.collect()!!;

	float* r = (float*)sq.value_bytes().ptr;
	for (usz i = 0; i < 6; i++)
	{
		assert(approx_eq(r[i], x_data[i]), "squeeze data");
	}
}

// ============================================================================
// Gather
// ============================================================================

fn void test_gather_1d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// input = [10, 20, 30, 40, 50]
	float[5] x_data = { 10, 20, 30, 40, 50 };

	// indices = [0, 2, 4] -> gather elements at positions 0, 2, 4

	ulong[3] idx_data = { 0, 2, 4 };
	Tensor x = tensor::@from_data(dev, { 5 }, &x_data)!!;
	Tensor idx = tensor::@from_data(dev, { 3 }, &idx_data)!!;

	Tensor result = x.gather(0, idx)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 10), "gather[0]");
	assert(approx_eq(r[1], 30), "gather[1]");
	assert(approx_eq(r[2], 50), "gather[2]");
}
