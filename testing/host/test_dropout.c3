//! Dropout Operation Tests
//!
//! Tests dropout forward (inverted dropout), backward, and mask variability.
//! Verifies:
//! - ~p fraction of elements are zeroed
//! - Non-zero elements are scaled by 1/(1-p)
//! - Different seeds produce different masks
//! - Backward passes gradient through the same mask
//! - Rebuilding the graph produces a different mask (seed changes)

module metaphor::testing::host::test_dropout;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 0.05;  // statistical tests need wider tolerance
const usz N = 10000;           // large enough for statistical assertions

fn bool approx_eq(float a, float b, float tol = TOLERANCE)
{
	return math::abs(a - b) < tol;
}

// ============================================================================
// Forward: statistical properties
// ============================================================================

fn void test_dropout_zeros_fraction() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Create input of all 1.0s
	float[N] x_data;
	for (usz i = 0; i < N; i++) { x_data[i] = 1.0; }
	ulong[1] dims = { N };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	float p = 0.3;
	Tensor result = functional::dropout(x, p)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;

	// Count zeros and non-zeros
	usz zero_count = 0;
	for (usz i = 0; i < N; i++)
	{
		if (r[i] == 0.0) { zero_count += 1; }
	}

	// Check ~30% are zeroed (within tolerance)
	float actual_drop_rate = (float)zero_count / (float)N;
	assert(approx_eq(actual_drop_rate, p, 0.05),
		"dropout should zero ~p fraction of elements");
}

fn void test_dropout_inverted_scaling() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[N] x_data;
	for (usz i = 0; i < N; i++) { x_data[i] = 1.0; }
	ulong[1] dims = { N };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	float p = 0.5;
	Tensor result = functional::dropout(x, p)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	float expected_scale = 1.0 / (1.0 - p);  // 2.0

	// All non-zero elements should be scaled by 1/(1-p)
	for (usz i = 0; i < N; i++)
	{
		if (r[i] != 0.0)
		{
			assert(approx_eq(r[i], expected_scale, 1e-4),
				"non-zero elements should be scaled by 1/(1-p)");
		}
	}
}

fn void test_dropout_preserves_expected_value() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[N] x_data;
	for (usz i = 0; i < N; i++) { x_data[i] = 3.0; }
	ulong[1] dims = { N };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	float p = 0.4;
	Tensor result = functional::dropout(x, p)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;

	// Mean of output should be close to input value (inverted dropout property)
	float sum = 0;
	for (usz i = 0; i < N; i++) { sum += r[i]; }
	float mean = sum / (float)N;

	// Expected mean = 3.0 (inverted dropout preserves expected value)
	assert(approx_eq(mean, 3.0, 0.2),
		"inverted dropout should preserve expected value");
}

fn void test_dropout_p_zero_is_identity() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[8] x_data = { 1, 2, 3, 4, 5, 6, 7, 8 };
	ulong[1] dims = { 8 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// p=0 means no dropout — identity
	Tensor result = functional::dropout(x, 0.0)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	for (usz i = 0; i < 8; i++)
	{
		assert(approx_eq(r[i], x_data[i], 1e-6),
			"dropout with p=0 should be identity");
	}
}

// ============================================================================
// Mask variability: different seeds → different masks
// ============================================================================

fn void test_dropout_different_calls_different_masks() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[N] x_data;
	for (usz i = 0; i < N; i++) { x_data[i] = 1.0; }
	ulong[1] dims = { N };

	Tensor x1 = tensor::@from_data(dev, &dims, &x_data)!!;
	Tensor x2 = tensor::@from_data(dev, &dims, &x_data)!!;

	float p = 0.5;
	// Two separate dropout calls get different seeds from the global counter
	Tensor r1 = functional::dropout(x1, p)!!;
	Tensor r2 = functional::dropout(x2, p)!!;
	r1.collect()!!;
	r2.collect()!!;

	float* d1 = (float*)r1.value_bytes().ptr;
	float* d2 = (float*)r2.value_bytes().ptr;

	// Count how many elements differ between the two masks
	usz differ_count = 0;
	for (usz i = 0; i < N; i++)
	{
		if ((d1[i] == 0.0) != (d2[i] == 0.0)) { differ_count += 1; }
	}

	// With p=0.5 and independent masks, ~50% of positions should differ
	// (P(one zero, other not) = 2 * 0.5 * 0.5 = 0.5)
	float differ_rate = (float)differ_count / (float)N;
	assert(differ_rate > 0.2,
		"different dropout calls should produce different masks");
}

fn void test_dropout_rebuilding_graph_changes_mask() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[N] x_data;
	for (usz i = 0; i < N; i++) { x_data[i] = 1.0; }
	ulong[1] dims = { N };

	// Build graph, execute, collect — simulating training step 1
	Tensor x1 = tensor::@from_data(dev, &dims, &x_data)!!;
	Tensor r1 = functional::dropout(x1, 0.5)!!;
	r1.collect()!!;
	float* d1 = (float*)r1.value_bytes().ptr;

	// Save first result
	float[N] saved;
	for (usz i = 0; i < N; i++) { saved[i] = d1[i]; }

	// Build graph again — simulating training step 2
	// This calls functional::dropout again, which increments the seed counter
	Tensor x2 = tensor::@from_data(dev, &dims, &x_data)!!;
	Tensor r2 = functional::dropout(x2, 0.5)!!;
	r2.collect()!!;
	float* d2 = (float*)r2.value_bytes().ptr;

	// Masks should differ between steps
	usz differ_count = 0;
	for (usz i = 0; i < N; i++)
	{
		if ((saved[i] == 0.0) != (d2[i] == 0.0)) { differ_count += 1; }
	}

	float differ_rate = (float)differ_count / (float)N;
	assert(differ_rate > 0.2,
		"rebuilding graph should produce different dropout mask (seed changes)");
}

// ============================================================================
// Backward: gradient passes through same mask
// ============================================================================

fn void test_dropout_backward_same_mask() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	usz n = 1000;
	float[1000] x_data;
	for (usz i = 0; i < n; i++) { x_data[i] = (float)(i + 1); }
	ulong[1] dims = { n };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	float p = 0.5;
	Tensor y = functional::dropout(x, p)!!;

	// Collect forward first to materialize the dropout output
	y.collect()!!;

	// Save forward result for comparison
	float* fwd_ptr = (float*)y.value_bytes().ptr;
	float[1000] fwd_saved;
	for (usz i = 0; i < n; i++) { fwd_saved[i] = fwd_ptr[i]; }

	// Now compute sum for backward
	Tensor loss = y.reduce("i->")!!;
	loss.collect()!!;
	loss.backward();

	// Get the gradient
	Tensor x_grad = x.grad();
	x_grad.collect()!!;

	float* grad = (float*)x_grad.value_bytes().ptr;
	float scale = 1.0 / (1.0 - p);

	// For each element: if forward zeroed it, gradient should also be zero
	// If forward kept it (scaled by 1/(1-p)), gradient should be 1/(1-p)
	for (usz i = 0; i < n; i++)
	{
		if (fwd_saved[i] == 0.0)
		{
			assert(grad[i] == 0.0,
				"gradient should be zero where forward was zero");
		}
		else
		{
			assert(approx_eq(grad[i], scale, 1e-4),
				"gradient should be 1/(1-p) where forward was non-zero");
		}
	}
}
