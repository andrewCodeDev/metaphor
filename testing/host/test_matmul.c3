//! Matmul & Einsum Tests — Forward Pass
//!
//! Tests matrix multiplication and einsum operations through
//! the full pipeline: tensor creation → CAS → JIT → execute → verify.
//!

module metaphor::testing::host::test_matmul;
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
// Basic Matmul: [2,3] @ [3,4] -> [2,4]
// ============================================================================

fn void test_matmul_2x3_3x4() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// A = [[1, 2, 3],
	//      [4, 5, 6]]
	float[6] a_data = { 1, 2, 3, 4, 5, 6 };

	// B = [[1, 2, 3, 4],
	//      [5, 6, 7, 8],
	//      [9, 10, 11, 12]]
	float[12] b_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	Tensor a = tensor::@from_data(dev, { 2, 3 }, &a_data)!!;
	Tensor b = tensor::@from_data(dev, { 3, 4 }, &b_data)!!;

	// C = A @ B  via einsum "ij,jk->ik"
	Tensor c = a.einsum(b, "ij,jk->ik")!!;
	c.collect()!!;

	char[] bytes = c.value_bytes();
	assert(bytes.len >= 8 * float.sizeof, "matmul should produce [2,4]");
	float* r = (float*)bytes.ptr;

	// Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
	//       = [38, 44, 50, 56]
	assert(approx_eq(r[0], 38), "C[0,0]=38");
	assert(approx_eq(r[1], 44), "C[0,1]=44");
	assert(approx_eq(r[2], 50), "C[0,2]=50");
	assert(approx_eq(r[3], 56), "C[0,3]=56");

	// Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
	//       = [83, 98, 113, 128]
	assert(approx_eq(r[4], 83), "C[1,0]=83");
	assert(approx_eq(r[5], 98), "C[1,1]=98");
	assert(approx_eq(r[6], 113), "C[1,2]=113");
	assert(approx_eq(r[7], 128), "C[1,3]=128");
}

// ============================================================================
// Square Matmul: [2,2] @ [2,2] -> [2,2]
// ============================================================================

fn void test_matmul_square() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// A = [[1, 2], [3, 4]]
	float[4] a_data = { 1, 2, 3, 4 };
	ulong[2] dims = { 2, 2 };

	// B = [[5, 6], [7, 8]]
	float[4] b_data = { 5, 6, 7, 8 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	Tensor c = a.einsum(b, "ij,jk->ik")!!;
	c.collect()!!;

	float* r = (float*)c.value_bytes().ptr;

	// [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
	assert(approx_eq(r[0], 19), "C[0,0]=19");
	assert(approx_eq(r[1], 22), "C[0,1]=22");
	assert(approx_eq(r[2], 43), "C[1,0]=43");
	assert(approx_eq(r[3], 50), "C[1,1]=50");
}

// ============================================================================
// Identity matmul: A @ I = A
// ============================================================================

fn void test_matmul_identity() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// A = [[1, 2], [3, 4]]
	float[4] a_data = { 1, 2, 3, 4 };
	ulong[2] dims = { 2, 2 };

	// I = [[1, 0], [0, 1]]
	float[4] i_data = { 1, 0, 0, 1 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor i = tensor::@from_data(dev, &dims, &i_data)!!;

	Tensor c = a.einsum(i, "ij,jk->ik")!!;
	c.collect()!!;

	float* r = (float*)c.value_bytes().ptr;

	assert(approx_eq(r[0], 1), "A@I[0,0]=1");
	assert(approx_eq(r[1], 2), "A@I[0,1]=2");
	assert(approx_eq(r[2], 3), "A@I[1,0]=3");
	assert(approx_eq(r[3], 4), "A@I[1,1]=4");
}
