//! Optimizer Unit Tests
//!
//! Tests basic correctness of SGD and Adam optimizers:
//! - Parameter updates go in the right direction
//! - State is properly maintained across steps
//! - Multiple parameters handled correctly
//! - LR scheduler computes correct values
//!

module metaphor::testing::host::test_optimizer;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::tensor;
import metaphor::rtti;
import metaphor::nn::optimizer;
import std::math;

const float TOLERANCE = 1e-5;

fn bool approx_eq(float a, float b)
{
	return math::abs(a - b) < TOLERANCE;
}

// ============================================================================
// SGD Tests
// ============================================================================

/// SGD basic update: param = 1.0, grad = 0.1, lr = 0.5
/// Expected: 1.0 - 0.5 * 0.1 = 0.95
fn void test_sgd_basic_update() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 4 };
	Tensor param = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

	// Materialize gradient and fill with 0.1
	Tensor grad = param.grad();
	grad.fill(0.0);  // Allocate data
	grad.fill(0.1);  // Set gradient = 0.1

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.5 });
	defer opt.deinit();
	opt.sgd(&params, optimizer::DEFAULT_SGD_OPTIONS);

	// Step with run_grad = false (we manually filled gradients)
	opt.step(run_grad: false);

	// Check: param = 1.0 - 0.5 * 0.1 = 0.95
	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	assert(approx_eq(result[0], 0.95), "sgd basic: expected 0.95");
	assert(approx_eq(result[1], 0.95), "sgd basic: expected 0.95 [1]");
}

/// SGD multiple steps with different gradients.
/// Step 1: param = 1.0, grad = 1.0, lr = 0.1 → param = 0.9
/// Step 2: grad = 0.5 → param = 0.9 - 0.1 * 0.5 = 0.85
fn void test_sgd_multiple_steps() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 2 };
	Tensor param = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.1 });
	defer opt.deinit();
	opt.sgd(&params, optimizer::DEFAULT_SGD_OPTIONS);

	Tensor grad = param.grad();
	grad.fill(0.0);  // Allocate

	// Step 1: grad = 1.0, param: 1.0 - 0.1 * 1.0 = 0.9
	grad.fill(1.0);
	opt.step(run_grad: false);

	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	assert(approx_eq(result[0], 0.9), "sgd step1: expected 0.9");

	// Step 2: grad = 0.5, param: 0.9 - 0.1 * 0.5 = 0.85
	grad.fill(0.5);
	opt.step(run_grad: false);

	bytes = param.value_bytes();
	result = (float*)bytes.ptr;
	assert(approx_eq(result[0], 0.85), "sgd step2: expected 0.85");
}

/// SGD with LR step decay scheduler.
fn void test_sgd_with_scheduler() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 2 };
	Tensor param = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({
		.lr = 1.0,
		.scheduler = optimizer::scheduler_step_decay(0.1, 2),
	});
	defer opt.deinit();
	opt.sgd(&params, optimizer::DEFAULT_SGD_OPTIONS);

	Tensor grad = param.grad();
	grad.fill(0.0);
	grad.fill(1.0);

	// Step 0: lr = base * 0.1^(0/2) = 1.0 * 0.1^0 = 1.0
	opt.step(run_grad: false);
	assert(approx_eq(opt.get_lr(), 1.0), "scheduler step0 lr");

	// Step 1: lr = 1.0 * 0.1^(1/2) = 1.0 * 0.1^0 = 1.0
	opt.step(run_grad: false);
	assert(approx_eq(opt.get_lr(), 1.0), "scheduler step1 lr");

	// Step 2: lr = 1.0 * 0.1^(2/2) = 1.0 * 0.1 = 0.1
	opt.step(run_grad: false);
	assert(approx_eq(opt.get_lr(), 0.1), "scheduler step2 lr");
}

// ============================================================================
// Adam Tests
// ============================================================================

/// Adam basic update: param should decrease when given positive gradient.
fn void test_adam_basic_update() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 2 };
	Tensor param = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.1 });
	defer opt.deinit();
	opt.adam(&params, optimizer::DEFAULT_ADAM_OPTIONS);

	Tensor grad = param.grad();
	grad.fill(0.0);
	grad.fill(1.0);

	opt.step(run_grad: false);

	// Param should have decreased
	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	assert(result[0] < 1.0, "adam basic: param should decrease");
}

/// Adam bias correction: after many steps with constant gradient,
/// the update magnitude should grow (bias correction effect).
fn void test_adam_bias_correction() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 1 };
	Tensor param = tensor::constant(F32, dev, &dims, 0.0)!!.enable_grad();

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.1 });
	defer opt.deinit();
	opt.adam(&params, optimizer::DEFAULT_ADAM_OPTIONS);

	Tensor grad = param.grad();
	grad.fill(0.0);
	grad.fill(1.0);

	opt.step(run_grad: false);

	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	float first_update = math::abs(result[0]);

	// Run 100 more steps with same gradient
	for (usz i = 0; i < 100; i++)
	{
		opt.step(run_grad: false);
	}

	bytes = param.value_bytes();
	result = (float*)bytes.ptr;
	float final_magnitude = math::abs(result[0]);

	// After bias correction kicks in, total movement should be much larger
	assert(final_magnitude > first_update * 10.0,
		"adam bias correction: magnitude should grow significantly");
}

/// Adam numerical stability: random-ish gradients shouldn't produce NaN.
fn void test_adam_numerical_stability() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 1 };
	Tensor param = tensor::constant(F32, dev, &dims, 0.0)!!.enable_grad();

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.001 });
	defer opt.deinit();
	opt.adam(&params, optimizer::DEFAULT_ADAM_OPTIONS);

	Tensor grad = param.grad();
	grad.fill(0.0);

	// Alternate positive/negative gradients
	float[8] grad_vals = { 0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, -0.6 };
	for (usz i = 0; i < 50; i++)
	{
		grad.fill(grad_vals[i % 8]);
		opt.step(run_grad: false);
	}

	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	float val = result[0];

	// Check for NaN/Inf
	assert(!math::is_nan(val), "adam stability: value should not be NaN");
	assert(!math::is_inf(val), "adam stability: value should not be Inf");
}

// ============================================================================
// Multi-parameter Tests
// ============================================================================

/// Multiple parameters with different shapes should all be updated.
fn void test_optimizer_multiple_params() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	Tensor param1 = tensor::constant(F32, dev, { 2, 2 }, 1.0)!!.enable_grad();
	Tensor param2 = tensor::constant(F32, dev, { 3 }, 2.0)!!.enable_grad();

	Tensor[2] params = { param1, param2 };
	Optimizer opt = optimizer::init({ .lr = 0.1 });
	defer opt.deinit();
	opt.adam(&params, optimizer::DEFAULT_ADAM_OPTIONS);

	// Fill gradients
	Tensor grad1 = param1.grad();
	Tensor grad2 = param2.grad();
	grad1.fill(0.5);
	grad2.fill(1.0);

	opt.step(run_grad: false);

	// Both params should have decreased
	char[] bytes1 = param1.value_bytes();
	float* r1 = (float*)bytes1.ptr;
	assert(r1[0] < 1.0, "multi param1 should decrease");

	char[] bytes2 = param2.value_bytes();
	float* r2 = (float*)bytes2.ptr;
	assert(r2[0] < 2.0, "multi param2 should decrease");
}

// ============================================================================
// LR Scheduler Tests
// ============================================================================

/// Test all scheduler types compute correct values.
fn void test_lr_schedulers() @test
{
	// None
	Scheduler s_none = optimizer::scheduler_none();
	assert(approx_eq(s_none.compute(0.1, 0), 0.1), "none: step 0");
	assert(approx_eq(s_none.compute(0.1, 100), 0.1), "none: step 100");

	// Step decay: lr = 0.1 * 0.5^(step/10)
	Scheduler s_step = optimizer::scheduler_step_decay(0.5, 10);
	assert(approx_eq(s_step.compute(0.1, 0), 0.1), "step_decay: step 0");    // 0.1 * 0.5^0
	assert(approx_eq(s_step.compute(0.1, 10), 0.05), "step_decay: step 10"); // 0.1 * 0.5^1

	// Exponential: lr = 0.1 * 0.99^step
	Scheduler s_exp = optimizer::scheduler_exponential(0.99);
	assert(approx_eq(s_exp.compute(0.1, 0), 0.1), "exp: step 0");
	float expected_exp = 0.1f * math::pow(0.99f, 10.0f);
	assert(approx_eq(s_exp.compute(0.1, 10), expected_exp), "exp: step 10");

	// Warmup: linearly increases to base_lr over warmup_steps
	Scheduler s_warmup = optimizer::scheduler_warmup(100);
	assert(approx_eq(s_warmup.compute(0.1, 0), 0.001), "warmup: step 0");  // 0.1 * 1/100
	assert(approx_eq(s_warmup.compute(0.1, 49), 0.05), "warmup: step 49"); // 0.1 * 50/100
	assert(approx_eq(s_warmup.compute(0.1, 100), 0.1), "warmup: step 100"); // past warmup
}

// ============================================================================
// Global Gradient Clipping
// ============================================================================

/// Global gradient clipping should scale gradients when total norm exceeds max.
fn void test_global_gradient_clipping() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	Tensor param = tensor::constant(F32, dev, { 4 }, 1.0)!!.enable_grad();

	Tensor[1] params = { param };

	// Set a very small global clip norm so clipping definitely triggers
	Optimizer opt = optimizer::init({
		.lr = 1.0,
		.global_clip_norm = 0.1,  // very tight
	});
	defer opt.deinit();
	opt.sgd(&params, optimizer::DEFAULT_SGD_OPTIONS);

	// Fill gradient with large values
	Tensor grad = param.grad();
	grad.fill(10.0);

	opt.step(run_grad: false);

	// The update should be much smaller than lr * grad = 1.0 * 10.0 = 10.0
	// With clipping to 0.1 norm, the effective grad magnitude is ~0.05
	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	float update = 1.0 - result[0];
	assert(update < 1.0, "global clip: update should be much smaller than unclipped");
	assert(update > 0.0, "global clip: update should still be positive");
}

// ============================================================================
// Gradient Accumulation Tests
// ============================================================================

/// SGD gradient accumulation: 4 micro-batches each with grad = 0.4.
/// Averaged gradient = 0.4 (each scaled by 1/4 and summed = 4 * 0.25 * 0.4 = 0.4).
/// Expected: param unchanged after 3 steps, then param = 1.0 - 0.5 * 0.4 = 0.8 after 4th.
fn void test_sgd_grad_accumulation() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 4 };
	Tensor param = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

	Tensor grad = param.grad();
	grad.fill(0.0);

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.5, .grad_accum_interval = 4 });
	defer opt.deinit();
	opt.sgd(&params, optimizer::DEFAULT_SGD_OPTIONS);

	grad.fill(0.4);

	// Steps 1-3: accumulate only, no param update
	for (int i = 0; i < 3; i++)
	{
		opt.step(run_grad: false);
	}
	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	assert(approx_eq(result[0], 1.0), "sgd accum: param should be unchanged after 3 steps");

	// Step 4: cycle complete, param updates
	opt.step(run_grad: false);
	bytes = param.value_bytes();
	result = (float*)bytes.ptr;
	assert(approx_eq(result[0], 0.8), "sgd accum: expected 0.8 after 4 steps");
	assert(approx_eq(result[1], 0.8), "sgd accum: expected 0.8 [1]");
}

/// Adam gradient accumulation: 4 micro-batches with constant gradient.
/// The accumulated gradient should cause the param to decrease after 4 steps.
fn void test_adam_grad_accumulation() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 2 };
	Tensor param = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();

	Tensor[1] params = { param };
	Optimizer opt = optimizer::init({ .lr = 0.1, .grad_accum_interval = 4 });
	defer opt.deinit();
	opt.adam(&params, optimizer::DEFAULT_ADAM_OPTIONS);

	Tensor grad = param.grad();
	grad.fill(1.0);

	// 4 steps to complete one cycle
	for (int i = 0; i < 4; i++)
	{
		opt.step(run_grad: false);
	}

	// Param should have decreased
	char[] bytes = param.value_bytes();
	float* result = (float*)bytes.ptr;
	assert(result[0] < 1.0, "adam accum: param should decrease");
}

/// Verify grad_accum_interval=0 produces identical results to default.
fn void test_grad_accum_disabled_unchanged() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();



	ulong[1] dims = { 4 };

	// Path A: standard step (no accum_interval)
	Tensor param_a = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();
	Tensor grad_a = param_a.grad();
	grad_a.fill(0.1);

	Tensor[1] params_a = { param_a };
	Optimizer opt_a = optimizer::init({ .lr = 0.5 });
	defer opt_a.deinit();
	opt_a.sgd(&params_a, optimizer::DEFAULT_SGD_OPTIONS);
	opt_a.step(run_grad: false);

	// Path B: explicit grad_accum_interval=0
	Tensor param_b = tensor::constant(F32, dev, &dims, 1.0)!!.enable_grad();
	Tensor grad_b = param_b.grad();
	grad_b.fill(0.1);

	Tensor[1] params_b = { param_b };
	Optimizer opt_b = optimizer::init({ .lr = 0.5, .grad_accum_interval = 0 });
	defer opt_b.deinit();
	opt_b.sgd(&params_b, optimizer::DEFAULT_SGD_OPTIONS);
	opt_b.step(run_grad: false);

	// Both should be identical
	char[] bytes_a = param_a.value_bytes();
	char[] bytes_b = param_b.value_bytes();
	float* ra = (float*)bytes_a.ptr;
	float* rb = (float*)bytes_b.ptr;
	assert(approx_eq(ra[0], rb[0]), "disabled_unchanged: results should match");
	assert(approx_eq(ra[1], rb[1]), "disabled_unchanged: results should match [1]");
}
