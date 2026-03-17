//! Chained Operation Tests — Forward Pass
//!
//! Tests multi-step operation chains through the full pipeline.
//! Exercises kernel fusion and composition of unary/binary ops.
//!

module metaphor::testing::host::test_chain_ops;
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
// Long unary chain: cos → sin → neg → exp → relu
// ============================================================================

fn void test_unary_chain_cos_sin_neg_exp_relu() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	// x = [0.5, 0.5, 0.5, 0.5]
	float[4] x_data = { 0.5, 0.5, 0.5, 0.5 };
	ulong[1] dims = { 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	// Chain: cos → sin → neg → exp → relu
	Tensor y = x.cos()!!;
	y = y.sin()!!;
	y = y.neg()!!;
	y = y.exp()!!;
	y = y.relu()!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;

	// Analytically:
	// v1 = cos(0.5) ≈ 0.87758
	// v2 = sin(0.87758) ≈ 0.76842
	// v3 = -0.76842
	// v4 = exp(-0.76842) ≈ 0.46370
	// y  = relu(0.46370) = 0.46370
	float expected = math::exp(-math::sin(math::cos(0.5f)));

	for (usz i = 0; i < 4; i++)
	{
		assert(approx_eq(r[i], expected), "chain result mismatch");
	}
}

// ============================================================================
// Binary chain: (a + b) * (a - b)  [difference of squares]
// ============================================================================

fn void test_binary_diff_of_squares() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] a_data = { 5, 4, 3, 2 };
	float[4] b_data = { 1, 2, 1, 1 };
	ulong[1] dims = { 4 };

	Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
	Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

	// (a+b) * (a-b) = a^2 - b^2
	Tensor sum = a.add(b)!!;
	Tensor diff = a.sub(b)!!;
	Tensor result = sum.mul(diff)!!;
	result.collect()!!;

	float* r = (float*)result.value_bytes().ptr;

	// Expected: [25-1, 16-4, 9-1, 4-1] = [24, 12, 8, 3]
	assert(approx_eq(r[0], 24), "diff_sq[0]=24");
	assert(approx_eq(r[1], 12), "diff_sq[1]=12");
	assert(approx_eq(r[2], 8), "diff_sq[2]=8");
	assert(approx_eq(r[3], 3), "diff_sq[3]=3");
}

// ============================================================================
// Mixed chain: exp(neg(x)) — softmax building block
// ============================================================================

fn void test_exp_neg() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] x_data = { 0, 1, 2, 3 };
	ulong[1] dims = { 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!;

	Tensor y = x.neg()!!;
	y = y.exp()!!;
	y.collect()!!;

	float* r = (float*)y.value_bytes().ptr;

	assert(approx_eq(r[0], 1.0), "exp(-0)=1");
	assert(approx_eq(r[1], math::exp(-1.0f)), "exp(-1)");
	assert(approx_eq(r[2], math::exp(-2.0f)), "exp(-2)");
	assert(approx_eq(r[3], math::exp(-3.0f)), "exp(-3)");
}

// ============================================================================
// Sigmoid via ops: 1 / (1 + exp(-x))  using recip(exp(neg(x)) + 1)
// Not a direct sigmoid call — tests composition.
// ============================================================================

fn void test_sigmoid_composition() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();


	float[4] x_data = { -2, -1, 0, 1 };

	Tensor x = tensor::@from_data(dev, { 4 }, &x_data)!!;

	// Direct sigmoid call for reference
	Tensor sig = x.sigmoid()!!;
	sig.collect()!!;

	float* r = (float*)sig.value_bytes().ptr;

	// sigmoid(x) = 1 / (1 + exp(-x))
	for (usz i = 0; i < 4; i++)
	{
		float expected = 1.0f / (1.0f + math::exp(-x_data[i]));
		assert(approx_eq(r[i], expected), "sigmoid mismatch");
	}
}
