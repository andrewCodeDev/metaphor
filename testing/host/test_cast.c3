//! Cast Kernel Tests
//!
//! Tests type casting between dtypes: f32, f64, u8, u64.

module metaphor::testing::host::test_cast;
import metaphor::device::host_device::kernels::cast;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-5;

// ============================================================================
// Same-type (memcpy) path
// ============================================================================

fn void test_cast_same_type_f32() @test
{
	float[4] src = { 1.5, 2.5, 3.5, 4.5 };
	float[4] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 4, F32, F32);
	for (usz i = 0; i < 4; i++)
	{
		assert(src[i] == dst[i], "same-type f32 copy failed");
	}
}

// ============================================================================
// Float -> Double
// ============================================================================

fn void test_cast_f32_to_f64() @test
{
	float[3] src = { 1.5, -2.5, 3.0 };
	double[3] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 3, F32, F64);

	assert(math::abs(dst[0] - 1.5) < TOLERANCE, "f32->f64 [0]");
	assert(math::abs(dst[1] - (-2.5)) < TOLERANCE, "f32->f64 [1]");
	assert(math::abs(dst[2] - 3.0) < TOLERANCE, "f32->f64 [2]");
}

// ============================================================================
// Double -> Float
// ============================================================================

fn void test_cast_f64_to_f32() @test
{
	double[3] src = { 1.5, -2.5, 3.0 };
	float[3] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 3, F64, F32);

	assert(math::abs(dst[0] - 1.5) < TOLERANCE, "f64->f32 [0]");
	assert(math::abs(dst[1] - (-2.5)) < TOLERANCE, "f64->f32 [1]");
	assert(math::abs(dst[2] - 3.0) < TOLERANCE, "f64->f32 [2]");
}

// ============================================================================
// Float -> U64 and back
// ============================================================================

fn void test_cast_f32_to_u64() @test
{
	float[3] src = { 1.0, 100.0, 255.0 };
	ulong[3] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 3, F32, U64);

	assert(dst[0] == 1, "f32->u64 [0]");
	assert(dst[1] == 100, "f32->u64 [1]");
	assert(dst[2] == 255, "f32->u64 [2]");
}

fn void test_cast_u64_to_f32() @test
{
	ulong[3] src = { 1, 100, 65535 };
	float[3] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 3, U64, F32);

	assert(math::abs(dst[0] - 1.0) < TOLERANCE, "u64->f32 [0]");
	assert(math::abs(dst[1] - 100.0) < TOLERANCE, "u64->f32 [1]");
	assert(math::abs(dst[2] - 65535.0) < TOLERANCE, "u64->f32 [2]");
}

// ============================================================================
// U8 conversions
// ============================================================================

fn void test_cast_u8_to_f32() @test
{
	char[4] src = { 0, 128, 255, 42 };
	float[4] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 4, U8, F32);

	assert(math::abs(dst[0] - 0.0) < TOLERANCE, "u8->f32 [0]");
	assert(math::abs(dst[1] - 128.0) < TOLERANCE, "u8->f32 [1]");
	assert(math::abs(dst[2] - 255.0) < TOLERANCE, "u8->f32 [2]");
	assert(math::abs(dst[3] - 42.0) < TOLERANCE, "u8->f32 [3]");
}

fn void test_cast_u8_to_f64() @test
{
	char[3] src = { 1, 127, 200 };
	double[3] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 3, U8, F64);

	assert(math::abs(dst[0] - 1.0) < TOLERANCE, "u8->f64 [0]");
	assert(math::abs(dst[1] - 127.0) < TOLERANCE, "u8->f64 [1]");
	assert(math::abs(dst[2] - 200.0) < TOLERANCE, "u8->f64 [2]");
}

fn void test_cast_u64_to_f64() @test
{
	ulong[3] src = { 0, 1000, 123456 };
	double[3] dst;
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 3, U64, F64);

	assert(math::abs(dst[0] - 0.0) < TOLERANCE, "u64->f64 [0]");
	assert(math::abs(dst[1] - 1000.0) < TOLERANCE, "u64->f64 [1]");
	assert(math::abs(dst[2] - 123456.0) < TOLERANCE, "u64->f64 [2]");
}

// ============================================================================
// Empty input
// ============================================================================

fn void test_cast_empty() @test
{
	float[1] src = { 42 };
	double[1] dst = { 0 };
	cast::cast_elements((void*)&src[0], (void*)&dst[0], 0, F32, F64);
	// dst should be unchanged
	assert(dst[0] == 0, "empty cast should not modify dst");
}
