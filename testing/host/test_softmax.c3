//! Softmax Kernel Tests
//!
//! Tests softmax, log-softmax (forward and backward), rowwise and columnwise.
//! Pure kernel function tests — no tensor pipeline needed.
//!

module metaphor::testing::host::test_softmax;
import metaphor::device::host_device::kernels::softmax;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Softmax Forward
// ============================================================================

fn void test_softmax_1d_uniform() @test
{
	// Softmax of uniform values should give 1/n for each
	float[4] x = { 1, 1, 1, 1 };
	float[4] y;
	softmax::softmax_1d_float(&x[0], &y[0], 4);

	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(y[i], 0.25), "softmax uniform: expected 0.25");
	}
}

fn void test_softmax_1d_sums_to_one() @test
{
	float[5] x = { 1, 2, 3, 4, 5 };
	float[5] y;
	softmax::softmax_1d_float(&x[0], &y[0], 5);

	float sum = 0;
	for (usz i = 0; i < 5; i++) { sum += y[i]; }
	assert(approx_eq(sum, 1), "softmax should sum to 1");
}

fn void test_softmax_1d_monotone() @test
{
	// Larger input should produce larger softmax output
	float[3] x = { 1, 2, 3 };
	float[3] y;
	softmax::softmax_1d_float(&x[0], &y[0], 3);

	assert(y[0] < y[1] && y[1] < y[2], "softmax should be monotonically increasing");
	assert(y[0] > 0 && y[2] < 1, "softmax values should be in (0, 1)");
}

fn void test_softmax_1d_large_values() @test
{
	// Numerical stability: large values should not overflow
	float[3] x = { 1000, 1001, 1002 };
	float[3] y;
	softmax::softmax_1d_float(&x[0], &y[0], 3);

	float sum = 0;
	for (usz i = 0; i < 3; i++) { sum += y[i]; }
	assert(approx_eq(sum, 1), "softmax with large values should still sum to 1");
}

fn void test_softmax_rows() @test
{
	// 2x3 matrix: each row independently normalized
	float[6] x = { 1, 1, 1, 0, 0, 0 };
	float[6] y;
	softmax::softmax_rows_float(&x[0], &y[0], 2, 3);

	// Row 0: uniform -> 1/3 each
	for (usz i = 0; i < 3; i++)
	{
		assert(approx_eq(y[i], 1.0/3.0), "softmax row 0: expected 1/3");
	}

	// Row 1: uniform -> 1/3 each
	for (usz i = 3; i < 6; i++)
	{
		assert(approx_eq(y[i], 1.0/3.0), "softmax row 1: expected 1/3");
	}
}

fn void test_softmax_cols() @test
{
	// 3x2 matrix: each column independently normalized
	float[6] x = { 1, 0, 1, 0, 1, 0 };
	float[6] y;
	softmax::softmax_cols_float(&x[0], &y[0], 3, 2);

	// Column 0: all 1's -> each = 1/3
	assert(approx_eq(y[0], 1.0/3.0), "softmax col 0 row 0");
	assert(approx_eq(y[2], 1.0/3.0), "softmax col 0 row 1");
	assert(approx_eq(y[4], 1.0/3.0), "softmax col 0 row 2");

	// Column 1: all 0's -> each = 1/3
	assert(approx_eq(y[1], 1.0/3.0), "softmax col 1 row 0");
	assert(approx_eq(y[3], 1.0/3.0), "softmax col 1 row 1");
	assert(approx_eq(y[5], 1.0/3.0), "softmax col 1 row 2");
}

// ============================================================================
// Log-Softmax Forward
// ============================================================================

fn void test_log_softmax_1d() @test
{
	// log(softmax(x)) should equal x - max - log(sum(exp(x - max)))
	float[3] x = { 1, 2, 3 };
	float[3] y_log;
	float[3] y_softmax;

	softmax::log_softmax_1d_float(&x[0], &y_log[0], 3);
	softmax::softmax_1d_float(&x[0], &y_softmax[0], 3);

	for (usz i = 0; i < 3; i++)
	{
		float expected = math::ln(y_softmax[i]);
		assert(approx_eq(y_log[i], expected), "log_softmax should equal ln(softmax)");
	}
}

fn void test_log_softmax_sums_correctly() @test
{
	// exp(log_softmax(x)) should sum to 1
	float[4] x = { 0.5, 1.5, 2.5, 3.5 };
	float[4] y;
	softmax::log_softmax_1d_float(&x[0], &y[0], 4);

	float sum = 0;
	for (usz i = 0; i < 4; i++) { sum += math::exp(y[i]); }
	assert(approx_eq(sum, 1), "exp(log_softmax) should sum to 1");
}

fn void test_log_softmax_rows() @test
{
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[6] y;
	softmax::log_softmax_rows_float(&x[0], &y[0], 2, 3);

	// Each row's exp(log_softmax) should sum to 1
	float sum1 = 0;
	float sum2 = 0;
	for (usz i = 0; i < 3; i++) { sum1 += math::exp(y[i]); }
	for (usz i = 3; i < 6; i++) { sum2 += math::exp(y[i]); }
	assert(approx_eq(sum1, 1), "log_softmax row 0 should sum to 1");
	assert(approx_eq(sum2, 1), "log_softmax row 1 should sum to 1");
}

// ============================================================================
// Softmax Backward
// ============================================================================

fn void test_softmax_backward_rows() @test
{
	// Forward: compute softmax
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[6] y;
	softmax::softmax_rows_float(&x[0], &y[0], 2, 3);

	// Backward with grad_y = 1.0 everywhere
	float[6] grad_y = { 1, 1, 1, 1, 1, 1 };
	float[6] grad_x;
	softmax::softmax_rows_backward_float(&grad_y[0], &y[0], &grad_x[0], 2, 3);

	// When grad_y is uniform, grad_x should be all zeros
	// (since softmax is shift-invariant and sum(y * (1 - sum(y*1))) = 0)
	for (usz i = 0; i < 6; i++)
	{
		assert(math::abs(grad_x[i]) < 1e-5, "softmax backward with uniform grad should be ~0");
	}
}

fn void test_softmax_backward_preserves_shape() @test
{
	// Verify backward produces output of same shape as input
	float[4] x = { 0, 1, 2, 3 };
	float[4] y;
	softmax::softmax_1d_float(&x[0], &y[0], 4);

	// Non-uniform gradient
	float[4] grad_y = { 1, 0, 0, 0 };
	float[4] grad_x;
	softmax::softmax_rows_backward_float(&grad_y[0], &y[0], &grad_x[0], 1, 4);

	// grad_x[0] should be positive (we want more of element 0)
	// grad_x[1..3] should be negative (we want less of elements 1-3)
	assert(grad_x[0] > 0, "grad_x[0] should be positive");
	assert(grad_x[1] < 0 && grad_x[2] < 0 && grad_x[3] < 0, "grad_x[1:] should be negative");
}

// ============================================================================
// Log-Softmax Backward
// ============================================================================

fn void test_log_softmax_backward_rows() @test
{
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[6] y;
	softmax::log_softmax_rows_float(&x[0], &y[0], 2, 3);

	float[6] grad_y = { 1, 0, 0, 0, 0, 1 };
	float[6] grad_x;
	softmax::log_softmax_rows_backward_float(&grad_y[0], &y[0], &grad_x[0], 2, 3);

	// Verify grad_x is not all zeros (non-trivial gradient)
	float sum_abs = 0;
	for (usz i = 0; i < 6; i++) { sum_abs += math::abs(grad_x[i]); }
	assert(sum_abs > 0.01, "log_softmax backward should produce non-zero gradients");
}

// ============================================================================
// SIMD Path (large arrays)
// ============================================================================

fn void test_softmax_large_array() @test
{
	float[256] x;
	float[256] y;
	for (usz i = 0; i < 256; i++) { x[i] = (float)i * 0.01; }

	softmax::softmax_1d_float(&x[0], &y[0], 256);

	float sum = 0;
	for (usz i = 0; i < 256; i++) { sum += y[i]; }
	assert(approx_eq(sum, 1), "large array softmax should sum to 1");

	// Should be monotonically increasing
	for (usz i = 1; i < 256; i++)
	{
		assert(y[i] >= y[i - 1], "large array softmax should be monotone");
	}
}
