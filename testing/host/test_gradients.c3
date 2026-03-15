//! Gradient Comparison Tests
//!
//! Tests backward pass gradients against PyTorch-generated test vectors.
//! Each test performs forward computation, verifies the output, runs backward,
//! and checks gradients against expected values.
//!

module metaphor::testing::host::test_gradients;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

const float RTOL = 1e-4;
const float ATOL = 1e-5;

macro void @check_values(float* actual, float[] expected, $msg)
{
	for (usz i = 0; i < expected.len; i++)
	{
		float diff = math::abs(actual[i] - expected[i]);
		float threshold = ATOL + RTOL * math::abs(expected[i]);
		assert(diff <= threshold, $msg);
	}
}

// ============================================================================
// Matmul: [2,3] @ [3,4] -> [2,4]
// ============================================================================

fn void test_matmul_basic() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] a_data = { 0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328 };
	ulong[2] a_dims = { 2, 3 };
	float[12] b_data = { 2.208201, -0.637997, 0.461657, 0.267351,
	                     0.534905, 0.809357, 1.110290, -1.689799,
	                    -0.988960, 0.957972, 1.322135, 0.817190 };
	ulong[2] b_dims = { 3, 4 };

	Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!.enable_grad();
	Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!.enable_grad();

	// Forward: c = a @ b
	Tensor c = a.einsum(b, "ij,jk->ik")!!;
	c.collect()!!;

	float[8] expected_c = { 0.580507, 0.114054, 0.608442, 0.063953,
	                        0.092272, -1.234241, -1.386713, 1.806716 };
	@check_values((float*)c.value_bytes().ptr, &expected_c, "matmul_basic: forward");

	// Backward: loss = sum(c)
	Tensor loss = c.reduce("ik->")!!;
	loss.collect()!!;
	loss.backward();

	Tensor grad_a = a.grad();
	grad_a.collect()!!;
	Tensor grad_b = b.grad();
	grad_b.collect()!!;

	float[6] expected_grad_a = { 2.299212, 0.764753, 2.108337,
	                             2.299212, 0.764753, 2.108337 };
	@check_values((float*)grad_a.value_bytes().ptr, &expected_grad_a, "matmul_basic: grad_a");

	float[12] expected_grad_b = { 0.567023, 0.567023, 0.567023, 0.567023,
	                             -0.994047, -0.994047, -0.994047, -0.994047,
	                              0.048134, 0.048134, 0.048134, 0.048134 };
	@check_values((float*)grad_b.value_bytes().ptr, &expected_grad_b, "matmul_basic: grad_b");
}

// ============================================================================
// Matmul: [4,4] @ [4,4] -> [4,4]
// ============================================================================

fn void test_matmul_square() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[16] a_data = { -0.016604, -0.466795, 2.090898, 0.614903,
	                     -0.562096, -0.414251, -1.300092, -0.101196,
	                     -1.099946, -0.468654, -0.840039, 0.686832,
	                     -0.067765, -0.088595, -0.312354, -0.355219 };
	ulong[2] a_dims = { 4, 4 };
	float[16] b_data = { 0.714247, 0.419830, -1.767226, -0.499593,
	                     0.308268, -0.294669, -0.766227, -0.996158,
	                     0.778119, -0.685558, -0.732556, 0.300430,
	                    -0.362998, 1.582230, -0.442951, 1.846213 };
	ulong[2] b_dims = { 4, 4 };

	Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!.enable_grad();
	Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!.enable_grad();

	Tensor c = a.einsum(b, "ij,jk->ik")!!;
	c.collect()!!;

	float[16] expected_c = { 1.248001, -0.329934, -1.417059, 2.236706,
	                        -1.504067, 0.617256, 2.307976, 0.116064,
	                        -1.833073, 1.338929, 2.614091, 2.032045,
	                        -0.189817, -0.350245, 0.573802, -0.627541 };
	@check_values((float*)c.value_bytes().ptr, &expected_c, "matmul_square: forward");

	Tensor loss = c.reduce("ik->")!!;
	loss.collect()!!;
	loss.backward();

	Tensor grad_a = a.grad();
	grad_a.collect()!!;
	Tensor grad_b = b.grad();
	grad_b.collect()!!;

	// grad.a: each row is sum of B columns = [1.437636, -0.748516, -3.708959, 0.651071] repeated
	// Actually from safetensors: row sums of B
	float[16] expected_grad_a = { 1.437636, -0.748516, -3.708959, 0.651071, 1.437636, -0.748516, -3.708959, 0.651071, 1.437636, -0.748516, -3.708959, 0.651071, 1.437636, -0.748516, -3.708959, 0.651071 };
	// Recompute: B row 0: 0.714247+0.419830+(-1.767226)+(-0.499593) = -1.132742
	// Wait, grad.a[i,j] = sum_k(b[j,k]). Let me recalculate...
	// b row 0 sum: 0.714247+0.419830+(-1.767226)+(-0.499593) = -1.132742
	// b row 1 sum: 0.308268+(-0.294669)+(-0.766227)+(-0.996158) = -1.748786
	// b row 2 sum: 0.778119+(-0.685558)+(-0.732556)+0.300430 = -0.339565
	// b row 3 sum: -0.362998+1.582230+(-0.442951)+1.846213 = 2.622494

	// Each row of grad_a is the same (each row of A sees same B)
	float[16] expected_ga = { -1.132742, -1.748786, -0.339565, 2.622494,
	                          -1.132742, -1.748786, -0.339565, 2.622494,
	                          -1.132742, -1.748786, -0.339565, 2.622494,
	                          -1.132742, -1.748786, -0.339565, 2.622494 };
	@check_values((float*)grad_a.value_bytes().ptr, &expected_ga, "matmul_square: grad_a");

	// grad.b[j,k] = sum_i(a[i,j]). Each column of B gets the column sum of A.
	// a col 0 sum: -0.016604+(-0.562096)+(-1.099946)+(-0.067765) = -1.746411
	// a col 1 sum: -0.466795+(-0.414251)+(-0.468654)+(-0.088595) = -1.438295
	// a col 2 sum: 2.090898+(-1.300092)+(-0.840039)+(-0.312354) = -0.361587
	// a col 3 sum: 0.614903+(-0.101196)+0.686832+(-0.355219) = 0.845320
	float[16] expected_gb = { -1.746411, -1.746411, -1.746411, -1.746411,
	                          -1.438295, -1.438295, -1.438295, -1.438295,
	                          -0.361587, -0.361587, -0.361587, -0.361587,
	                           0.845320,  0.845320,  0.845320,  0.845320 };
	@check_values((float*)grad_b.value_bytes().ptr, &expected_gb, "matmul_square: grad_b");
}

// ============================================================================
// ReLU activation
// ============================================================================

fn void test_relu() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { -1.603511, 0.201137, -0.798855, 1.571311,
	                      0.354188, -0.362733, -0.323012, 0.625703,
	                     -0.237012, -1.934958, -0.034083, -0.587502 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	Tensor y = x.relu()!!;
	y.collect()!!;

	float[12] expected_y = { 0.0, 0.201137, 0.0, 1.571311,
	                         0.354188, 0.0, 0.0, 0.625703,
	                         0.0, 0.0, 0.0, 0.0 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "relu: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	float[12] expected_grad = { 0.0, 1.0, 0.0, 1.0,
	                            1.0, 0.0, 0.0, 1.0,
	                            0.0, 0.0, 0.0, 0.0 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "relu: grad");
}

// ============================================================================
// Sigmoid activation
// ============================================================================

fn void test_sigmoid() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { 1.567321, 0.599645, -0.386021, -2.288830,
	                     0.852644, 0.711739, 0.497929, -0.393447,
	                    -0.653978, 1.342442, -0.531888, 1.484346 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	Tensor y = x.sigmoid()!!;
	y.collect()!!;

	float[12] expected_y = { 0.827401, 0.645575, 0.404676, 0.092052,
	                         0.701122, 0.670785, 0.621973, 0.402888,
	                         0.342094, 0.792891, 0.370077, 0.815228 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "sigmoid: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	float[12] expected_grad = { 0.142808, 0.228808, 0.240913, 0.083579,
	                            0.209550, 0.220832, 0.235123, 0.240569,
	                            0.225066, 0.164215, 0.233120, 0.150631 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "sigmoid: grad");
}

// ============================================================================
// Tanh activation
// ============================================================================

fn void test_tanh() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { -1.000113, -0.725041, -0.355963, -0.270573,
	                      2.150344, 0.477857, -2.955660, -0.356691,
	                     -0.376601, -0.920792, 1.724084, 0.413296 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	Tensor y = x.tanh()!!;
	y.collect()!!;

	float[12] expected_y = { -0.761642, -0.620022, -0.341653, -0.264158,
	                          0.973244, 0.444526, -0.994597, -0.342296,
	                         -0.359752, -0.726272, 0.938353, 0.391268 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "tanh: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	float[12] expected_grad = { 0.419902, 0.615573, 0.883273, 0.930221,
	                            0.052796, 0.802397, 0.010776, 0.882834,
	                            0.870579, 0.472529, 0.119494, 0.846910 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "tanh: grad");
}

// ============================================================================
// Reduce: sum all (ij->)
// ============================================================================

fn void test_sum_all() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { 0.732041, -0.658706, -0.326079, -0.849282,
	                     -0.074263, -0.640351, -1.653356, 0.358214,
	                     -1.611705, 0.768713, -1.515824, -0.632555 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	Tensor y = x.reduce("ij->")!!;
	y.collect()!!;

	float[1] expected_y = { -6.103152 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "sum_all: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	// d(sum)/dx_i = 1 for all i
	float[12] expected_grad = { 1.0, 1.0, 1.0, 1.0,
	                            1.0, 1.0, 1.0, 1.0,
	                            1.0, 1.0, 1.0, 1.0 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "sum_all: grad");
}

// ============================================================================
// Reduce: sum axis 0 (ij->j)
// ============================================================================

fn void test_sum_axis0() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { 0.278770, -0.206994, 1.180174, 0.014751,
	                     1.311184, 1.357639, -0.761117, 1.295596,
	                    -2.272615, 1.964259, -1.044282, 0.976557 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	Tensor y = x.reduce("ij->j")!!;
	y.collect()!!;

	float[4] expected_y = { -0.682661, 3.114904, -0.625226, 2.286905 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "sum_axis0: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	float[12] expected_grad = { 1.0, 1.0, 1.0, 1.0,
	                            1.0, 1.0, 1.0, 1.0,
	                            1.0, 1.0, 1.0, 1.0 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "sum_axis0: grad");
}

// ============================================================================
// Reduce: sum axis 1 (ij->i)
// ============================================================================

fn void test_sum_axis1() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { -0.293001, -0.475929, -0.029462, -1.060958,
	                      0.380990, 0.765482, -0.087926, -2.950995,
	                     -1.220826, 0.196006, -1.020344, -0.331097 };
	ulong[2] dims = { 3, 4 };

	Tensor x = tensor::@from_data(dev, &dims, &x_data)!!.enable_grad();

	Tensor y = x.reduce("ij->i")!!;
	y.collect()!!;

	float[3] expected_y = { -1.859350, -1.892448, -2.376262 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "sum_axis1: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	float[12] expected_grad = { 1.0, 1.0, 1.0, 1.0,
	                            1.0, 1.0, 1.0, 1.0,
	                            1.0, 1.0, 1.0, 1.0 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "sum_axis1: grad");
}

// ============================================================================
// Reduce: mean all  (sum / numel)
// ============================================================================

fn void test_mean_all() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] x_data = { -0.890008, 0.468439, 0.830839, 0.982824,
	                     -1.095016, 0.379780, 1.151549, 0.260939,
	                     -0.492978, -0.619114, -1.169711, -0.249104 };

	Tensor x = tensor::@from_data(dev, { 3, 4 }, &x_data)!!.enable_grad();

	// mean = sum / count
	Tensor sum = x.reduce("ij->")!!;
	Tensor count = x.numel()!!;
	Tensor y = sum.div(count)!!;
	y.collect()!!;

	float[1] expected_y = { -0.036797 };
	@check_values((float*)y.value_bytes().ptr, &expected_y, "mean_all: forward");

	y.backward();

	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	// d(mean)/dx_i = 1/N = 1/12
	float[12] expected_grad = { 0.083333, 0.083333, 0.083333, 0.083333,
	                            0.083333, 0.083333, 0.083333, 0.083333,
	                            0.083333, 0.083333, 0.083333, 0.083333 };
	@check_values((float*)grad_x.value_bytes().ptr, &expected_grad, "mean_all: grad");
}

// ============================================================================
// Broadcast add: [3,4] + [4] -> [3,4]
// ============================================================================

fn void test_broadcast_add_row() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[12] a_data = { 0.753373, 1.854116, 0.794341, -0.763790,
	                     0.777561, 1.162870, 1.440082, 0.386561,
	                     1.321123, -0.853067, 1.914510, 0.463615 };
	ulong[2] a_dims = { 3, 4 };
	float[4] b_data = { 1.272071, -0.089187, -1.180051, 0.926489 };
	ulong[1] b_dims = { 4 };

	Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!.enable_grad();
	Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!.enable_grad();

	// c = a + b (broadcast add)
	Tensor c = a.add(b)!!;
	c.collect()!!;

	float[12] expected_c = { 2.025443, 1.764929, -0.385709, 0.162699,
	                         2.049631, 1.073683, 0.260032, 1.313049,
	                         2.593194, -0.942253, 0.734460, 1.390103 };
	@check_values((float*)c.value_bytes().ptr, &expected_c, "broadcast_add_row: forward");

	c.backward();

	Tensor grad_a = a.grad();
	grad_a.collect()!!;
	Tensor grad_b = b.grad();
	grad_b.collect()!!;

	// d(c)/d(a) = 1 everywhere
	float[12] expected_grad_a = { 1.0, 1.0, 1.0, 1.0,
	                              1.0, 1.0, 1.0, 1.0,
	                              1.0, 1.0, 1.0, 1.0 };
	@check_values((float*)grad_a.value_bytes().ptr, &expected_grad_a, "broadcast_add_row: grad_a");

	// d(c)/d(b): b appears in each row, so grad sums over rows = 3.0 each
	float[4] expected_grad_b = { 3.0, 3.0, 3.0, 3.0 };
	@check_values((float*)grad_b.value_bytes().ptr, &expected_grad_b, "broadcast_add_row: grad_b");
}

// ============================================================================
// Broadcast mul: [2,3] * scalar -> [2,3]
// ============================================================================

fn void test_broadcast_mul_scalar() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	float[6] a_data = { -1.184542, 0.216623, -1.208973, -0.510929, -0.181538, 0.095374 };
	float[1] b_data = { 2.500000 };
	ulong[] b_dims;  // scalar: 0-dimensional

	Tensor a = tensor::@from_data(dev, { 2, 3 }, &a_data)!!.enable_grad();
	Tensor b = tensor::@from_data(dev, b_dims, &b_data)!!.enable_grad();

	// c = a * b (broadcast scalar mul)
	Tensor c = a.mul(b)!!;
	c.collect()!!;

	float[6] expected_c = { -2.961355, 0.541558, -3.022432, -1.277322, -0.453846, 0.238436 };
	@check_values((float*)c.value_bytes().ptr, &expected_c, "broadcast_mul_scalar: forward");

	c.backward();

	Tensor grad_a = a.grad();
	grad_a.collect()!!;
	Tensor grad_b = b.grad();
	grad_b.collect()!!;

	// d(c)/d(a) = b = 2.5 everywhere
	float[6] expected_grad_a = { 2.5, 2.5, 2.5, 2.5, 2.5, 2.5 };
	@check_values((float*)grad_a.value_bytes().ptr, &expected_grad_a, "broadcast_mul_scalar: grad_a");

	// d(c)/d(b) = sum(a) = -2.773984
	float[1] expected_grad_b = { -2.773984 };
	@check_values((float*)grad_b.value_bytes().ptr, &expected_grad_b, "broadcast_mul_scalar: grad_b");
}
