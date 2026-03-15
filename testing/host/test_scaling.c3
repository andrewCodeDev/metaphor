//! Scaling Kernel Tests
//!
//! Tests for min-max, z-score, L2/L1 normalization, max-abs scaling,
//! and clamping. Includes flat, row-wise, and SIMD-exercising variants.
//!

module metaphor::testing::host::test_scaling;
import metaphor::device::host_device::kernels::scaling;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

fn bool approx_eq_arr(float* a, float* b, usz len)
{
	for (usz i = 0; i < len; i++)
	{
		if (!approx_eq(a[i], b[i])) return false;
	}
	return true;
}

// ============================================================================
// Min-Max Tests
// ============================================================================

fn void test_min_max_basic() @test
{
	float[5] x = { 1, 5, 3, 9, 2 };
	scaling::MinMaxResult result = scaling::min_max_float(&x[0], 5);
	assert(result.min_val == 1 && result.max_val == 9, "min_max basic failed");
}

fn void test_min_max_scale_basic() @test
{
	float[3] x = { 0, 50, 100 };
	float[3] y;
	scaling::min_max_scale_auto_float(&x[0], &y[0], 3);
	float[3] expected = { 0, 0.5, 1 };
	assert(approx_eq_arr(&y[0], &expected[0], 3), "min_max_scale basic failed");
}

fn void test_min_max_scale_negative() @test
{
	float[3] x = { -10, 0, 10 };
	float[3] y;
	scaling::min_max_scale_auto_float(&x[0], &y[0], 3);
	float[3] expected = { 0, 0.5, 1 };
	assert(approx_eq_arr(&y[0], &expected[0], 3), "min_max_scale negative failed");
}

fn void test_min_max_scale_constant() @test
{
	float[3] x = { 5, 5, 5 };
	float[3] y;
	scaling::min_max_scale_auto_float(&x[0], &y[0], 3);
	float[3] expected = { 0, 0, 0 };
	assert(approx_eq_arr(&y[0], &expected[0], 3), "min_max_scale constant failed");
}

fn void test_min_max_scale_range() @test
{
	float[3] x = { 0, 50, 100 };
	float[3] y;
	scaling::min_max_scale_range_auto_float(&x[0], &y[0], 3, -1, 1);
	float[3] expected = { -1, 0, 1 };
	assert(approx_eq_arr(&y[0], &expected[0], 3), "min_max_scale range failed");
}

// ============================================================================
// Z-Score Tests
// ============================================================================

fn void test_mean_basic() @test
{
	float[5] x = { 1, 2, 3, 4, 5 };
	float result = scaling::mean_float(&x[0], 5);
	assert(approx_eq(result, 3), "mean basic failed");
}

fn void test_variance_basic() @test
{
	float[8] x = { 2, 4, 4, 4, 5, 5, 7, 9 };
	float result = scaling::variance_float(&x[0], 8);
	assert(approx_eq(result, 4), "variance basic failed");
}

fn void test_std_dev_basic() @test
{
	float[8] x = { 2, 4, 4, 4, 5, 5, 7, 9 };
	float result = scaling::std_dev_float(&x[0], 8);
	assert(approx_eq(result, 2), "std_dev basic failed");
}

fn void test_zscore_basic() @test
{
	float[3] x = { 0, 1, 2 };
	float[3] y;
	scaling::zscore_auto_float(&x[0], &y[0], 3);

	float result_mean = scaling::mean_float(&y[0], 3);
	float result_std = scaling::std_dev_float(&y[0], 3);

	assert(approx_eq(result_mean, 0) && approx_eq(result_std, 1), "zscore basic failed");
}

fn void test_zscore_constant() @test
{
	float[3] x = { 5, 5, 5 };
	float[3] y;
	scaling::zscore_auto_float(&x[0], &y[0], 3);
	float[3] expected = { 0, 0, 0 };
	assert(approx_eq_arr(&y[0], &expected[0], 3), "zscore constant failed");
}

// ============================================================================
// L2 Normalization Tests
// ============================================================================

fn void test_l2_norm_basic() @test
{
	float[2] x = { 3, 4 };
	float result = scaling::l2_norm_compute(&x[0], 2);
	assert(approx_eq(result, 5), "l2_norm basic failed");
}

fn void test_l2_normalize_basic() @test
{
	float[2] x = { 3, 4 };
	float[2] y;
	scaling::l2_normalize_float(&x[0], &y[0], 2, 1e-8);
	float[2] expected = { 0.6, 0.8 };
	assert(approx_eq_arr(&y[0], &expected[0], 2), "l2_normalize basic failed");
}

fn void test_l2_normalize_unit_norm() @test
{
	float[5] x = { 1, 2, 3, 4, 5 };
	float[5] y;
	scaling::l2_normalize_float(&x[0], &y[0], 5, 1e-8);
	float result_norm = scaling::l2_norm_compute(&y[0], 5);
	assert(approx_eq(result_norm, 1), "l2_normalize unit norm failed");
}

fn void test_l2_normalize_zero() @test
{
	float[3] x = { 0, 0, 0 };
	float[3] y;
	scaling::l2_normalize_float(&x[0], &y[0], 3, 1e-8);
	// With epsilon, result won't be exactly zero but very small
	for (usz i = 0; i < 3; i++)
	{
		assert(math::abs(y[i]) < 1e-4, "l2_normalize zero should produce near-zero");
	}
}

// ============================================================================
// L1 Normalization Tests
// ============================================================================

fn void test_l1_norm_basic() @test
{
	float[4] x = { 1, -2, 3, -4 };
	float result = scaling::l1_norm_compute(&x[0], 4);
	assert(approx_eq(result, 10), "l1_norm basic failed");
}

fn void test_l1_normalize_basic() @test
{
	float[4] x = { 1, 2, 3, 4 };
	float[4] y;
	scaling::l1_normalize_float(&x[0], &y[0], 4, 1e-8);
	float[4] expected = { 0.1, 0.2, 0.3, 0.4 };
	assert(approx_eq_arr(&y[0], &expected[0], 4), "l1_normalize basic failed");
}

fn void test_l1_normalize_sums_to_one() @test
{
	float[5] x = { 1, 2, 3, 4, 5 };
	float[5] y;
	scaling::l1_normalize_float(&x[0], &y[0], 5, 1e-8);

	float sum = 0;
	for (usz i = 0; i < 5; i++) { sum += y[i]; }
	assert(approx_eq(sum, 1), "l1_normalize sums_to_one failed");
}

// ============================================================================
// Max Abs Scaling Tests
// ============================================================================

fn void test_max_abs_basic() @test
{
	float[4] x = { -5, 3, 2, -1 };
	float result = scaling::max_abs_float(&x[0], 4);
	assert(approx_eq(result, 5), "max_abs basic failed");
}

fn void test_max_abs_scale_basic() @test
{
	float[4] x = { -10, 5, 0, 2 };
	float[4] y;
	scaling::max_abs_scale_float(&x[0], &y[0], 4);
	float[4] expected = { -1, 0.5, 0, 0.2 };
	assert(approx_eq_arr(&y[0], &expected[0], 4), "max_abs_scale basic failed");
}

fn void test_max_abs_scale_range() @test
{
	float[7] x = { -100, -50, 0, 25, 50, 75, 100 };
	float[7] y;
	scaling::max_abs_scale_float(&x[0], &y[0], 7);

	for (usz i = 0; i < 7; i++)
	{
		assert(y[i] >= -1 && y[i] <= 1, "max_abs_scale range: value out of [-1,1]");
	}
	assert(approx_eq(y[0], -1) && approx_eq(y[6], 1), "max_abs_scale range endpoints failed");
}

// ============================================================================
// Clamp Tests
// ============================================================================

fn void test_clamp_basic() @test
{
	float[5] x = { -5, 0, 5, 10, 15 };
	float[5] y;
	scaling::clamp_float(&x[0], &y[0], 5, 0, 10);
	float[5] expected = { 0, 0, 5, 10, 10 };
	assert(approx_eq_arr(&y[0], &expected[0], 5), "clamp basic failed");
}

fn void test_clamp_symmetric() @test
{
	float[5] x = { -10, -5, 0, 5, 10 };
	float[5] y;
	scaling::clamp_float(&x[0], &y[0], 5, -3, 3);
	float[5] expected = { -3, -3, 0, 3, 3 };
	assert(approx_eq_arr(&y[0], &expected[0], 5), "clamp symmetric failed");
}

// ============================================================================
// Row-wise Tests
// ============================================================================

fn void test_min_max_scale_rows() @test
{
	float[6] x = { 0, 50, 100, 10, 20, 30 };
	float[6] y;
	scaling::min_max_scale_rows_float(&x[0], &y[0], 2, 3);
	float[6] expected = { 0, 0.5, 1, 0, 0.5, 1 };
	assert(approx_eq_arr(&y[0], &expected[0], 6), "min_max_scale_rows failed");
}

fn void test_l2_normalize_rows() @test
{
	float[6] x = { 3, 4, 0, 0, 3, 4 };
	float[6] y;
	scaling::l2_normalize_rows_float(&x[0], &y[0], 2, 3, 1e-8);
	float norm1 = scaling::l2_norm_compute(&y[0], 3);
	float norm2 = scaling::l2_norm_compute(&y[3], 3);
	assert(approx_eq(norm1, 1) && approx_eq(norm2, 1), "l2_normalize_rows failed");
}

fn void test_zscore_rows() @test
{
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[6] y;
	scaling::zscore_rows_auto_float(&x[0], &y[0], 2, 3);
	float mean1 = scaling::mean_float(&y[0], 3);
	float std1 = scaling::std_dev_float(&y[0], 3);
	float mean2 = scaling::mean_float(&y[3], 3);
	float std2 = scaling::std_dev_float(&y[3], 3);
	assert(approx_eq(mean1, 0) && approx_eq(std1, 1) &&
	       approx_eq(mean2, 0) && approx_eq(std2, 1), "zscore_rows failed");
}

fn void test_l1_normalize_rows() @test
{
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[6] y;
	scaling::l1_normalize_rows_float(&x[0], &y[0], 2, 3, 1e-8);
	float sum1 = 0;
	float sum2 = 0;
	for (usz i = 0; i < 3; i++) { sum1 += y[i]; }
	for (usz i = 3; i < 6; i++) { sum2 += y[i]; }
	assert(approx_eq(sum1, 1) && approx_eq(sum2, 1), "l1_normalize_rows failed");
}

// ============================================================================
// Large Array Tests (SIMD verification)
// ============================================================================

fn void test_large_array_min_max_scale() @test
{
	float[1024] x;
	float[1024] y;
	for (usz i = 0; i < 1024; i++) { x[i] = (float)i; }

	scaling::min_max_scale_auto_float(&x[0], &y[0], 1024);

	assert(approx_eq(y[0], 0) && approx_eq(y[1023], 1),
		"large array min_max_scale endpoints failed");

	float mid = y[512];
	assert(math::abs(mid - 512.0f / 1023.0f) < 0.001f,
		"large array min_max_scale midpoint failed");
}

fn void test_large_array_l2_normalize() @test
{
	float[1024] x;
	float[1024] y;
	for (usz i = 0; i < 1024; i++) { x[i] = (float)(i + 1); }

	scaling::l2_normalize_float(&x[0], &y[0], 1024, 1e-8);

	float result_norm = scaling::l2_norm_compute(&y[0], 1024);
	assert(approx_eq(result_norm, 1), "large array l2_normalize norm should be 1");
}
