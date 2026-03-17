//! Neural Network Module Tests
//!
//! Tests nn/functional (softmax, normalization, linear, mean) and
//! nn/loss (MSE, cross-entropy, softmax cross-entropy) through the
//! full pipeline.
//!

module metaphor::testing::host::test_nn;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::nn::loss;
import metaphor::rtti;
import std::math;

const float TOLERANCE = 1e-4;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// Softmax (Pipeline)
// ============================================================================

fn void test_softmax_pipeline_uniform() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Softmax of uniform values -> 1/n for each
	float[4] x_data = { 1, 1, 1, 1 };
	ulong[2] dims = { 1, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// Softmax along dim 1 (last dim)
	Tensor result = functional::softmax(x, 1)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(r[i], 0.25), "softmax uniform");
	}
}

fn void test_softmax_pipeline_sums_to_one() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[5] x_data = { 1, 2, 3, 4, 5 };
	ulong[2] dims = { 1, 5 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::softmax(x, 1)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	float sum = 0;
	for (usz i = 0; i < 5; i++) { sum += r[i]; }
	assert(approx_eq(sum, 1.0), "softmax sums to 1");
}

fn void test_softmax_pipeline_monotone() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[3] x_data = { 1, 2, 3 };
	ulong[2] dims = { 1, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::softmax(x, 1)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(r[0] < r[1] && r[1] < r[2], "softmax monotone");
}

fn void test_softmax_pipeline_numerical_stability() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Large values shouldn't cause overflow
	float[3] x_data = { 1000, 1001, 1002 };
	ulong[2] dims = { 1, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::softmax(x, 1)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	float sum = 0;
	for (usz i = 0; i < 3; i++) { sum += r[i]; }
	assert(approx_eq(sum, 1.0), "softmax large values sums to 1");
}

// ============================================================================
// Log-Softmax (Pipeline)
// ============================================================================

fn void test_log_softmax_pipeline() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[3] x_data = { 1, 2, 3 };
	ulong[2] dims = { 1, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::log_softmax(x, 1)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;

	// exp(log_softmax(x)) should sum to 1
	float sum = 0;
	for (usz i = 0; i < 3; i++) { sum += math::exp(r[i]); }
	assert(approx_eq(sum, 1.0), "exp(log_softmax) sums to 1");

	// All values should be negative (log of probability)
	for (usz i = 0; i < 3; i++)
	{
		assert(r[i] < 0, "log_softmax should be negative");
	}
}

// ============================================================================
// Mean / Sum (Pipeline)
// ============================================================================

fn void test_mean_pipeline() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[4] x_data = { 2, 4, 6, 8 };
	ulong[1] dims = { 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::mean(x)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 5.0), "mean of [2,4,6,8] = 5");
}

fn void test_sum_all_pipeline() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[4] x_data = { 1, 2, 3, 4 };
	ulong[1] dims = { 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::sum_all(x)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 10.0), "sum of [1,2,3,4] = 10");
}

fn void test_mean_2d() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// [[1, 2, 3], [4, 5, 6]] -> mean = 3.5
	float[6] x_data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] dims = { 2, 3 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::mean(x)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 3.5), "mean [[1..6]] = 3.5");
}

// ============================================================================
// Linear Layer (Pipeline)
// ============================================================================

fn void test_linear_no_bias() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// x = [1, 2] (1x2 input)
	// w = [[1, 2], [3, 4], [5, 6]] (3x2 weight, out=3, in=2)
	// y = x @ w^T = [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
	float[2] x_data = { 1, 2 };
	ulong[2] x_dims = { 1, 2 };

	float[6] w_data = { 1, 2, 3, 4, 5, 6 };
	ulong[2] w_dims = { 3, 2 };

	Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!;
	Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!;

	Tensor result = functional::linear(x, w)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 5), "linear[0]=5");
	assert(approx_eq(r[1], 11), "linear[1]=11");
	assert(approx_eq(r[2], 17), "linear[2]=17");
}

fn void test_linear_with_bias() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[2] x_data = { 1, 2 };

	float[6] w_data = { 1, 2, 3, 4, 5, 6 };

	float[3] b_data = { 10, 20, 30 };

	Tensor x = tensor::@from_data(dev, { 1, 2 }, &x_data)!!;
	Tensor w = tensor::@from_data(dev, { 3, 2 }, &w_data)!!;
	Tensor b = tensor::@from_data(dev, { 3 }, &b_data)!!;

	Tensor result = functional::linear(x, w, b)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	// [5+10, 11+20, 17+30] = [15, 31, 47]
	assert(approx_eq(r[0], 15), "linear+bias[0]=15");
	assert(approx_eq(r[1], 31), "linear+bias[1]=31");
	assert(approx_eq(r[2], 47), "linear+bias[2]=47");
}

// ============================================================================
// MSE Loss
// ============================================================================

fn void test_mse_loss_zero() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// MSE of identical tensors should be 0
	float[4] data = { 1, 2, 3, 4 };
	ulong[1] dims = { 4 };

	Tensor pred = tensor::@from_data(dev, &dims, &data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &data)!!;

	Tensor l = loss::mse_loss(pred, target)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	assert(approx_eq(r[0], 0.0), "MSE of identical = 0");
}

fn void test_mse_loss_nonzero() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// pred = [0, 0, 0, 0], target = [1, 1, 1, 1]
	// MSE = mean((0-1)^2) = mean(1,1,1,1) = 1
	float[4] p_data = { 0, 0, 0, 0 };
	float[4] t_data = { 1, 1, 1, 1 };
	ulong[1] dims = { 4 };

	Tensor pred = tensor::@from_data(dev, &dims, &p_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &t_data)!!;

	Tensor l = loss::mse_loss(pred, target)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	assert(approx_eq(r[0], 1.0), "MSE [0,0,0,0] vs [1,1,1,1] = 1");
}

fn void test_mse_loss_values() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// pred = [1, 2, 3], target = [4, 5, 6]
	// MSE = mean(9, 9, 9) = 9
	float[3] p_data = { 1, 2, 3 };
	float[3] t_data = { 4, 5, 6 };
	ulong[1] dims = { 3 };

	Tensor pred = tensor::@from_data(dev, &dims, &p_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &t_data)!!;

	Tensor l = loss::mse_loss(pred, target)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	assert(approx_eq(r[0], 9.0), "MSE [1,2,3] vs [4,5,6] = 9");
}

// ============================================================================
// Softmax Cross-Entropy Loss
// ============================================================================

fn void test_softmax_cross_entropy() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// logits = [[2, 1, 0.1]] (1 sample, 3 classes)
	// target = [[1, 0, 0]]  (class 0 is correct)
	float[3] logit_data = { 2, 1, 0.1 };
	float[3] target_data = { 1, 0, 0 };
	ulong[2] dims = { 1, 3 };

	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

	Tensor l = loss::softmax_cross_entropy(logits, target)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	// Loss should be positive and relatively small since logits[0] is highest
	assert(r[0] > 0.0, "cross-entropy > 0");
	assert(r[0] < 2.0, "cross-entropy reasonable magnitude");
}

fn void test_softmax_cross_entropy_perfect() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// When logits strongly favor the correct class, loss should be near zero
	float[3] logit_data = { 100, 0, 0 };
	float[3] target_data = { 1, 0, 0 };
	ulong[2] dims = { 1, 3 };

	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

	Tensor l = loss::softmax_cross_entropy(logits, target)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	assert(r[0] < 0.01, "perfect prediction: loss ~0");
}

fn void test_softmax_cross_entropy_wrong() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// When logits strongly favor wrong class, loss should be large
	float[3] logit_data = { 0, 0, 100 };
	float[3] target_data = { 1, 0, 0 };
	ulong[2] dims = { 1, 3 };

	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

	Tensor l = loss::softmax_cross_entropy(logits, target)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	assert(r[0] > 50, "wrong prediction: loss large");
}

// ============================================================================
// Weighted Softmax Cross-Entropy
// ============================================================================

fn void test_weighted_softmax_cross_entropy() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// 4 samples, 3 classes
	// Weights: [1, 0, 1, 0] — only samples 0 and 2 contribute
	float[12] logit_data = {
		2.0, 1.0, 0.1,   // sample 0: correct=class0 (high logit)
		0.0, 0.0, 100.0, // sample 1: correct=class0 (wrong, but weight=0)
		1.0, 2.0, 0.5,   // sample 2: correct=class1 (high logit)
		100.0, 0.0, 0.0,  // sample 3: correct=class2 (wrong, but weight=0)
	};
	float[12] target_data = {
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	};
	float[4] weight_data = { 1.0, 0.0, 1.0, 0.0 };

	ulong[2] dims = { 4, 3 };
	ulong[1] wdims = { 4 };

	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;
	Tensor weight = tensor::@from_data(dev, &wdims, &weight_data)!!;

	Tensor l = loss::softmax_cross_entropy(logits, target, weight)!!;
	l.collect()!!;

	float* r = (float*)l.value_bytes().ptr;
	// Only samples 0 and 2 matter — both have correct class with highest logit
	// Loss should be small and positive
	assert(r[0] > 0.0, "weighted loss > 0");
	assert(r[0] < 2.0, "weighted loss small (correct predictions)");
}

fn void test_weighted_softmax_cross_entropy_zero_weight_ignores_wrong() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// 2 samples, 3 classes
	// Sample 0: perfect prediction, weight=0 (ignored)
	// Sample 1: terrible prediction, weight=1
	float[6] logit_data = {
		100.0, 0.0, 0.0,  // sample 0: correct=class0, perfect
		100.0, 0.0, 0.0,  // sample 1: correct=class2, terrible
	};
	float[6] target_data = {
		1, 0, 0,  // class 0
		0, 0, 1,  // class 2
	};

	ulong[2] dims = { 2, 3 };
	ulong[1] wdims = { 2 };

	// weight=[0, 1] — only terrible sample counts
	float[2] w2 = { 0.0, 1.0 };
	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;
	Tensor weight2 = tensor::@from_data(dev, &wdims, &w2)!!;

	Tensor l2 = loss::softmax_cross_entropy(logits, target, weight2)!!;
	l2.collect()!!;
	float loss_wrong = *((float*)l2.value_bytes().ptr);

	// Wrong prediction: loss should be huge (~100)
	assert(loss_wrong > 50.0, "wrong-only weighted loss large");
}

fn void test_weighted_softmax_cross_entropy_zero_weight_ignores_perfect() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Same setup: 2 samples, 3 classes
	float[6] logit_data = {
		100.0, 0.0, 0.0,  // sample 0: correct=class0, perfect
		100.0, 0.0, 0.0,  // sample 1: correct=class2, terrible
	};
	float[6] target_data = {
		1, 0, 0,  // class 0
		0, 0, 1,  // class 2
	};

	ulong[2] dims = { 2, 3 };
	ulong[1] wdims = { 2 };

	// weight=[1, 0] — only perfect sample counts
	float[2] w1 = { 1.0, 0.0 };
	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;
	Tensor weight1 = tensor::@from_data(dev, &wdims, &w1)!!;

	Tensor l1 = loss::softmax_cross_entropy(logits, target, weight1)!!;
	l1.collect()!!;
	float loss_perfect = *((float*)l1.value_bytes().ptr);

	// Perfect prediction: loss should be near 0
	assert(loss_perfect < 0.01, "perfect-only weighted loss ~0");
}

fn void test_weighted_softmax_ce_matches_unweighted_with_uniform() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// With uniform weights, weighted loss should equal unweighted loss
	float[6] logit_data = { 2.0, 1.0, 0.1, 0.5, 1.5, 0.3 };
	float[6] target_data = { 1, 0, 0, 0, 1, 0 };
	float[2] weight_data = { 1.0, 1.0 };

	ulong[2] dims = { 2, 3 };

	Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
	Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;
	Tensor weight = tensor::@from_data(dev, { 2 }, &weight_data)!!;

	Tensor l_weighted = loss::softmax_cross_entropy(logits, target, weight)!!;
	l_weighted.collect()!!;
	float val_weighted = *((float*)l_weighted.value_bytes().ptr);

	Tensor l_unweighted = loss::softmax_cross_entropy(logits, target)!!;
	l_unweighted.collect()!!;
	float val_unweighted = *((float*)l_unweighted.value_bytes().ptr);

	float diff = val_weighted - val_unweighted;
	if (diff < 0) { diff = -diff; }
	assert(diff < 1e-5, "uniform weights should match unweighted");
}

// ============================================================================
// Normalization (Pipeline)
// ============================================================================

fn void test_l2_normalize() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// [3, 4] -> L2 norm = 5 -> [3/5, 4/5] = [0.6, 0.8]
	float[2] x_data = { 3, 4 };
	ulong[2] dims = { 1, 2 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor result = functional::l2_normalize(x, 1, 1e-8)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	assert(approx_eq(r[0], 0.6), "l2_norm[0]=0.6");
	assert(approx_eq(r[1], 0.8), "l2_norm[1]=0.8");
}

fn void test_zscore_normalize() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// [1, 2, 3] -> mean=2, std=sqrt(2/3)
	// zscore = [-1, 0, 1] / std
	float[3] x_data = { 1, 2, 3 };

	Tensor x = tensor::@from_data(dev, { 1, 3 }, &x_data)!!;

	Tensor result = functional::zscore(x, 1, 1e-8)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;
	// After zscore, mean should be ~0 and std should be ~1
	float sum = 0;
	for (usz i = 0; i < 3; i++) { sum += r[i]; }
	assert(math::abs(sum) < 0.01, "zscore mean ~0");

	// Element 1 (middle) should be near 0
	assert(math::abs(r[1]) < 0.01, "zscore center ~0");
}
