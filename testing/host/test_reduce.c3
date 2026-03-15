//! Reduce Kernel Tests
//!
//! Tests for scaled copy, flat reduce, fold rows/cols, and
//! max/min backward passes.
//!
//! Pure kernel function tests — no tensor pipeline needed.

module metaphor::testing::host::test_reduce;
import metaphor::device::host_device::kernels::reduce;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Scaled Copy
// ============================================================================

fn void test_scaled_copy_identity() @test
{
	float[4] x = { 1, 2, 3, 4 };
	float[4] y = { 0, 0, 0, 0 };
	reduce::scaled_copy_float(&x[0], &y[0], 4, 1.0, 0.0);
	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(y[i], x[i]), "scaled_copy identity failed");
	}
}

fn void test_scaled_copy_alpha_beta() @test
{
	float[3] x = { 1, 2, 3 };
	float[3] y = { 10, 20, 30 };
	reduce::scaled_copy_float(&x[0], &y[0], 3, 2.0, 0.5);
	// y = 2*x + 0.5*y_old = [2+5, 4+10, 6+15] = [7, 14, 21]
	assert(approx_eq(y[0], 7), "scaled_copy alpha_beta [0]");
	assert(approx_eq(y[1], 14), "scaled_copy alpha_beta [1]");
	assert(approx_eq(y[2], 21), "scaled_copy alpha_beta [2]");
}

// ============================================================================
// Flat Reduce
// ============================================================================

fn void test_flat_reduce_sum() @test
{
	float[5] x = { 1, 2, 3, 4, 5 };
	float[1] y = { 0 };
	reduce::flat_reduce_float(&x[0], &y[0], 5, 1.0, 0.0);
	assert(approx_eq(y[0], 15), "flat_reduce sum = 15");
}

fn void test_flat_reduce_with_beta() @test
{
	float[3] x = { 2, 3, 5 };
	float[1] y = { 100 };
	reduce::flat_reduce_float(&x[0], &y[0], 3, 1.0, 0.5);
	// y = 1.0 * sum(x) + 0.5 * 100 = 10 + 50 = 60
	assert(approx_eq(y[0], 60), "flat_reduce with beta");
}

fn void test_flat_reduce_large_simd() @test
{
	// Exercise SIMD path
	float[128] x;
	for (usz i = 0; i < 128; i++) { x[i] = 1.0; }
	float[1] y = { 0 };
	reduce::flat_reduce_float(&x[0], &y[0], 128, 1.0, 0.0);
	assert(approx_eq(y[0], 128), "flat_reduce SIMD: sum of 128 ones");
}

// ============================================================================
// Fold Rows: (M,N) -> (1,N)
// ============================================================================

fn void test_fold_rows() @test
{
	// 3x2 matrix: sum columns
	// [[1, 2], [3, 4], [5, 6]] -> [9, 12]
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[2] y = { 0, 0 };
	reduce::fold_rows_float(&x[0], &y[0], 3, 2, 1.0, 0.0);
	assert(approx_eq(y[0], 9), "fold_rows [0] = 1+3+5 = 9");
	assert(approx_eq(y[1], 12), "fold_rows [1] = 2+4+6 = 12");
}

fn void test_fold_rows_with_alpha() @test
{
	float[4] x = { 1, 2, 3, 4 };
	float[2] y = { 0, 0 };
	reduce::fold_rows_float(&x[0], &y[0], 2, 2, 2.0, 0.0);
	// alpha=2: y = 2 * [4, 6] = [8, 12]
	assert(approx_eq(y[0], 8), "fold_rows alpha [0]");
	assert(approx_eq(y[1], 12), "fold_rows alpha [1]");
}

// ============================================================================
// Fold Cols: (M,N) -> (M,1)
// ============================================================================

fn void test_fold_cols() @test
{
	// 2x3 matrix: sum rows
	// [[1, 2, 3], [4, 5, 6]] -> [6, 15]
	float[6] x = { 1, 2, 3, 4, 5, 6 };
	float[2] y = { 0, 0 };
	reduce::fold_cols_float(&x[0], &y[0], 2, 3, 1.0, 0.0);
	assert(approx_eq(y[0], 6), "fold_cols [0] = 1+2+3 = 6");
	assert(approx_eq(y[1], 15), "fold_cols [1] = 4+5+6 = 15");
}

fn void test_fold_cols_with_beta() @test
{
	float[4] x = { 1, 2, 3, 4 };
	float[2] y = { 10, 20 };
	reduce::fold_cols_float(&x[0], &y[0], 2, 2, 1.0, 0.5);
	// y = 1.0 * [3, 7] + 0.5 * [10, 20] = [3+5, 7+10] = [8, 17]
	assert(approx_eq(y[0], 8), "fold_cols beta [0]");
	assert(approx_eq(y[1], 17), "fold_cols beta [1]");
}

// ============================================================================
// Reduce Max/Min Backward
// ============================================================================

fn void test_reduce_max_cols_backward() @test
{
	// 2x3 matrix: [[1, 5, 3], [4, 2, 6]]
	// Max per row (cols reduce): row0=5, row1=6
	// Gradient passes to max position only
	float[6] original = { 1, 5, 3, 4, 2, 6 };
	float[2] upstream = { 1, 1 };
	float[6] grad;
	reduce::reduce_max_cols_backward_float(&original[0], &upstream[0], &grad[0], 2, 3);

	// Row 0: max at index 1 -> [0, 1, 0]
	assert(approx_eq(grad[0], 0), "max_cols_bwd row0[0]");
	assert(approx_eq(grad[1], 1), "max_cols_bwd row0[1]");
	assert(approx_eq(grad[2], 0), "max_cols_bwd row0[2]");

	// Row 1: max at index 2 -> [0, 0, 1]
	assert(approx_eq(grad[3], 0), "max_cols_bwd row1[0]");
	assert(approx_eq(grad[4], 0), "max_cols_bwd row1[1]");
	assert(approx_eq(grad[5], 1), "max_cols_bwd row1[2]");
}

fn void test_reduce_min_cols_backward() @test
{
	float[6] original = { 1, 5, 3, 4, 2, 6 };
	float[2] upstream = { 1, 1 };
	float[6] grad;
	reduce::reduce_min_cols_backward_float(&original[0], &upstream[0], &grad[0], 2, 3);

	// Row 0: min at index 0 -> [1, 0, 0]
	assert(approx_eq(grad[0], 1), "min_cols_bwd row0[0]");
	assert(approx_eq(grad[1], 0), "min_cols_bwd row0[1]");

	// Row 1: min at index 1 -> [0, 1, 0]
	assert(approx_eq(grad[3], 0), "min_cols_bwd row1[0]");
	assert(approx_eq(grad[4], 1), "min_cols_bwd row1[1]");
}

fn void test_reduce_max_flat_backward() @test
{
	float[5] original = { 3, 1, 4, 1, 5 };
	float[5] grad;
	reduce::reduce_max_flat_backward_float(&original[0], 2.0, &grad[0], 5);

	// Max is 5 at index 4
	assert(approx_eq(grad[0], 0), "max_flat_bwd [0]");
	assert(approx_eq(grad[4], 2), "max_flat_bwd [4] = upstream");
}

fn void test_reduce_min_flat_backward() @test
{
	float[5] original = { 3, 1, 4, 1, 5 };
	float[5] grad;
	reduce::reduce_min_flat_backward_float(&original[0], 2.0, &grad[0], 5);

	// Min is 1 at index 1 (and 3, but first occurrence)
	// Both positions with min value get the gradient
	assert(approx_eq(grad[0], 0), "min_flat_bwd [0]");
	assert(approx_eq(grad[1], 2), "min_flat_bwd [1] = upstream");
	assert(approx_eq(grad[2], 0), "min_flat_bwd [2]");
	assert(approx_eq(grad[3], 2), "min_flat_bwd [3] = upstream (also min)");
	assert(approx_eq(grad[4], 0), "min_flat_bwd [4]");
}
