//! Neural Network Optimizer
//!
//! Optimizer container with device-specific update backends.
//! Supports SGD and Adam with optional LR scheduling and gradient clipping.
//!
//! Usage:
//!   Optimizer opt = optimizer::init({ .lr = 0.001 });
//!   opt.sgd(params, {});
//!   opt.adam(biases, { .beta1 = 0.9 });
//!   opt.step();
//!   opt.deinit();
//!

module metaphor::nn::optimizer;
import metaphor::tensor;
import metaphor::rtti;
import metaphor::device::host_device::blas::blas_callbacks;
import std::math;
import std::io;

// Learning Rate Scheduler

enum SchedulerType : char
{
	/// No scheduling — constant LR.
	NONE,
	/// Step decay: lr = base_lr * gamma^(step / step_size)
	STEP_DECAY,
	/// Exponential decay: lr = base_lr * gamma^step
	EXPONENTIAL,
	/// Cosine annealing: lr oscillates between max and min.
	COSINE,
	/// Linear warmup then constant.
	WARMUP,
	/// Linear warmup then cosine decay.
	WARMUP_COSINE,
	/// Linear warmup then linear decay.
	WARMUP_LINEAR,
}

struct Scheduler
{
	SchedulerType type;

	// Step decay / exponential params
	float gamma;
	uint step_size;

	// Cosine params
	float min_lr;
	uint period;

	// Warmup params
	uint warmup_steps;
	uint total_steps;
}

fn Scheduler scheduler_none()
{
	return { .type = SchedulerType.NONE };
}

fn Scheduler scheduler_step_decay(float gamma, uint step_size)
{
	return { .type = SchedulerType.STEP_DECAY, .gamma = gamma, .step_size = step_size };
}

fn Scheduler scheduler_exponential(float gamma)
{
	return { .type = SchedulerType.EXPONENTIAL, .gamma = gamma };
}

fn Scheduler scheduler_cosine(float min_lr, uint period)
{
	return { .type = SchedulerType.COSINE, .min_lr = min_lr, .period = period };
}

fn Scheduler scheduler_warmup(uint warmup_steps)
{
	return { .type = SchedulerType.WARMUP, .warmup_steps = warmup_steps };
}

fn Scheduler scheduler_warmup_cosine(uint warmup_steps, uint total_steps, float min_lr)
{
	return {
		.type = SchedulerType.WARMUP_COSINE,
		.warmup_steps = warmup_steps,
		.total_steps = total_steps,
		.min_lr = min_lr,
	};
}

fn Scheduler scheduler_warmup_linear(uint warmup_steps, uint total_steps, float min_lr)
{
	return {
		.type = SchedulerType.WARMUP_LINEAR,
		.warmup_steps = warmup_steps,
		.total_steps = total_steps,
		.min_lr = min_lr,
	};
}

/// Compute LR for current step.
fn float Scheduler.compute(&self, float base_lr, uint step_num)
{
	switch (self.type)
	{
		case NONE:
			return base_lr;

		case STEP_DECAY:
			float exp_sd = (float)(step_num / self.step_size);
			return base_lr * math::pow(self.gamma, exp_sd);

		case EXPONENTIAL:
			float exp_e = (float)step_num;
			return base_lr * math::pow(self.gamma, exp_e);

		case COSINE:
			float t_c = (float)(step_num % self.period);
			float per = (float)self.period;
			return self.min_lr + 0.5f * (base_lr - self.min_lr) *
				(1.0f + (float)math::cos(math::PI * (double)t_c / (double)per));

		case WARMUP:
			if (step_num < self.warmup_steps)
			{
				return base_lr * (float)(step_num + 1) / (float)self.warmup_steps;
			}
			return base_lr;

		case WARMUP_COSINE:
			if (step_num < self.warmup_steps)
			{
				return base_lr * (float)(step_num + 1) / (float)self.warmup_steps;
			}
			if (step_num >= self.total_steps)
			{
				return self.min_lr;
			}
			float t_wc = (float)(step_num - self.warmup_steps);
			float total = (float)(self.total_steps - self.warmup_steps);
			return self.min_lr + 0.5f * (base_lr - self.min_lr) *
				(1.0f + (float)math::cos(math::PI * (double)t_wc / (double)total));

		case WARMUP_LINEAR:
			if (step_num < self.warmup_steps)
			{
				return base_lr * (float)(step_num + 1) / (float)self.warmup_steps;
			}
			if (step_num >= self.total_steps)
			{
				return self.min_lr;
			}
			float t_wl = (float)(step_num - self.warmup_steps);
			float total_wl = (float)(self.total_steps - self.warmup_steps);
			return base_lr - (base_lr - self.min_lr) * t_wl / total_wl;
	}
}

// Optimizer Unit

/// Function pointer types for device-specific operations.
alias ExecuteFn = fn void(void* ctx, float lr);
alias DeinitFn = fn void(void* ctx);
alias AccumFn = fn void(void* ctx, float scale);
alias ZeroAccumFn = fn void(void* ctx);

/// A single optimizer entry — holds param and device-specific update logic.
/// Note: param is stored here AND in the device-specific context (e.g. HostSgdContext).
/// This is intentional — Tensor is a lightweight handle (pointer), and the unit needs
/// param for grad collection while the context needs it for the update kernel.
struct OptimizerUnit
{
	Tensor param;
	ExecuteFn execute_fn;
	void* ctx;
	DeinitFn deinit_fn;
	AccumFn accum_fn;
	ZeroAccumFn zero_accum_fn;
}

// Optimizer

struct OptimizerOptions
{
	float lr;
	Scheduler scheduler;
	/// Global gradient clipping max norm. If set, clips the total gradient
	/// norm across all parameters.
	float global_clip_norm;
	/// Gradient accumulation interval. 0 = disabled (default).
	/// When set to N > 0, gradients are accumulated over N calls to step(),
	/// and parameters are only updated every Nth call.
	uint grad_accum_interval;
}

const OptimizerOptions DEFAULT_OPTIONS = {
	.lr = 0.001,
	.scheduler = { .type = SchedulerType.NONE },
	.global_clip_norm = 0.0,
	.grad_accum_interval = 0,
};


/// SGD options per parameter group.
struct SgdOptions
{
	float clip_grad; // 0 = no clipping
}

const SgdOptions DEFAULT_SGD_OPTIONS = {
	.clip_grad = 0.0,
};

/// Adam options per parameter group.
struct AdamOptions
{
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	float clip_grad; // 0 = no clipping
}

const AdamOptions DEFAULT_ADAM_OPTIONS = {
	.beta1 = 0.9,
	.beta2 = 0.999,
	.eps = 1e-8,
	.weight_decay = 0.0,
	.clip_grad = 0.0,
};

/// Lightweight optimizer container.
///
/// Holds a dynamic list of OptimizerUnit entries, each with a device-specific
/// update function. Call step() to collect gradients and update all params.
///
/// When grad_accum_interval > 0, gradients are accumulated over N step() calls.
/// Parameters are only updated every Nth call. The LR scheduler sees only
/// actual parameter updates, not micro-batch steps.
struct Optimizer
{
	float base_lr;
	float lr;
	Scheduler scheduler;
	uint step_count;
	float global_clip_norm; // 0 = disabled
	float last_grad_norm;   // pre-clip gradient norm (for diagnostics)
	uint grad_accum_interval; // 0 = disabled
	uint accum_step;          // current position in accumulation cycle

	// Dynamic array of units (heap-allocated)
	OptimizerUnit* units;
	usz unit_count;
	usz unit_capacity;
}

fn Optimizer init(OptimizerOptions opts)
{
	return {
		.base_lr = opts.lr,
		.lr = opts.scheduler.compute(opts.lr, 0),
		.scheduler = opts.scheduler,
		.step_count = 0,
		.global_clip_norm = opts.global_clip_norm,
		.grad_accum_interval = opts.grad_accum_interval,
		.accum_step = 0,
		.units = null,
		.unit_count = 0,
		.unit_capacity = 0,
	};
}

fn void Optimizer.deinit(&self)
{
	// Deinit each unit's context
	for (usz i = 0; i < self.unit_count; i++)
	{
		if (self.units[i].deinit_fn != null)
		{
			self.units[i].deinit_fn(self.units[i].ctx);
		}
	}
	if (self.units != null)
	{
		mem::free(self.units);
		self.units = null;
	}
	self.unit_count = 0;
	self.unit_capacity = 0;
}

// --- Internal: append a unit ---
// Uses mem::malloc/realloc because the optimizer outlives the graph allocator
// and manages its own memory lifecycle independently of graph teardown.

fn void Optimizer.append_unit(&self, OptimizerUnit unit)
{
	if (self.unit_count >= self.unit_capacity)
	{
		usz new_cap = self.unit_capacity == 0 ? 8 : self.unit_capacity * 2;
		void* new_ptr = mem::realloc(self.units, new_cap * OptimizerUnit.sizeof);
		assert(new_ptr != null, "Optimizer: allocation failed");
		self.units = (OptimizerUnit*)new_ptr;
		self.unit_capacity = new_cap;
	}
	self.units[self.unit_count] = unit;
	self.unit_count++;
}

// SGD Registration

/// Register parameters for SGD optimization.
fn void Optimizer.sgd(&self, Tensor[] params, SgdOptions opts)
{
	foreach (&p : params)
	{
		self.append_unit(create_sgd_unit(*p, opts));
	}
}

macro OptimizerUnit create_sgd_unit(Tensor param, SgdOptions opts) {
	$if $defined(cuda_sgd_create):
		if (param.device().name() == "cuda") { return cuda_sgd_create(param, opts); }
	$endif
	$if $defined(hip_sgd_create):
		if (param.device().name() == "hip") { return hip_sgd_create(param, opts); }
	$endif
	if (param.device().name() == "host") { return host_sgd_create(param, opts); }
	unreachable("create_sgd_unit: no optimizer for device '%s'", param.device().name());
}

// Adam Registration

/// Register parameters for Adam optimization.
fn void Optimizer.adam(&self, Tensor[] params, AdamOptions opts)
{
	foreach (&p : params)
	{
		self.append_unit(create_adam_unit(*p, opts));
	}
}

macro OptimizerUnit create_adam_unit(Tensor param, AdamOptions opts) {
	$if $defined(cuda_adam_create):
		if (param.device().name() == "cuda") { return cuda_adam_create(param, opts); }
	$endif
	$if $defined(hip_adam_create):
		if (param.device().name() == "hip") { return hip_adam_create(param, opts); }
	$endif
	if (param.device().name() == "host") { return host_adam_create(param, opts); }
	unreachable("create_adam_unit: no optimizer for device '%s'", param.device().name());
}

// Step

/// Step all registered optimizer units.
///
/// When grad_accum_interval == 0 (default): standard behavior.
///   Collects gradients, applies clipping, updates parameters.
///
/// When grad_accum_interval > 0: counter-based gradient accumulation.
///   Each call collects and accumulates gradients (scaled by 1/N).
///   Parameters are only updated every Nth call.
///   The external training loop calls step() once per micro-batch.
fn void Optimizer.step(&self, bool run_grad = true)
{
	if (self.unit_count == 0) return;

	if (self.grad_accum_interval == 0)
	{
		// Standard path: collect, clip, update
		self.lr = self.scheduler.compute(self.base_lr, self.step_count);

		if (run_grad)
		{
			for (usz i = 0; i < self.unit_count; i++)
			{
				self.units[i].param.grad().collect()!!;
			}
		}

		if (self.global_clip_norm > 0.0f)
		{
			self.apply_global_clip(self.global_clip_norm);
		}

		for (usz i = 0; i < self.unit_count; i++)
		{
			self.units[i].execute_fn(self.units[i].ctx, self.lr);
		}

		self.step_count++;
		return;
	}

	// --- Gradient accumulation path ---
	float scale = 1.0f / (float)self.grad_accum_interval;

	// At the start of a new cycle, zero the accum buffers
	if (self.accum_step == 0)
	{
		for (usz i = 0; i < self.unit_count; i++)
		{
			self.units[i].zero_accum_fn(self.units[i].ctx);
		}
	}

	// Collect gradients for this micro-batch
	if (run_grad)
	{
		for (usz i = 0; i < self.unit_count; i++)
		{
			self.units[i].param.grad().collect()!!;
		}
	}

	// Accumulate: accum += (1/N) * grad
	for (usz i = 0; i < self.unit_count; i++)
	{
		self.units[i].accum_fn(self.units[i].ctx, scale);
	}

	self.accum_step++;

	// Cycle complete — update params from accumulated gradient, reset
	if (self.accum_step >= self.grad_accum_interval)
	{
		self.lr = self.scheduler.compute(self.base_lr, self.step_count);

		// NOTE: global_clip_norm is not applied in the accum path because
		// apply_global_clip reads param.grad() (last micro-batch only), not
		// the accumulated gradient buffer. The 1/N scaling of accumulated
		// gradients provides natural regularization. TODO: add accum-aware
		// global clip if needed.

		for (usz i = 0; i < self.unit_count; i++)
		{
			self.units[i].execute_fn(self.units[i].ctx, self.lr);
		}

		self.accum_step = 0;
		self.step_count++;
	}
}

/// Get current learning rate.
fn float Optimizer.get_lr(&self)
{
	return self.lr;
}

/// Get current step count.
fn uint Optimizer.get_step_count(&self)
{
	return self.step_count;
}

// Global Gradient Clipping

fn void Optimizer.apply_global_clip(&self, float max_norm)
{
	// Compute total squared norm across all gradients
	float total_sq = 0.0f;
	for (usz i = 0; i < self.unit_count; i++)
	{
		total_sq += device_grad_norm_sq(self.units[i].param.grad());
	}

	float total_norm = (float)math::sqrt((double)total_sq);
	self.last_grad_norm = total_norm;

	if (total_norm <= max_norm || total_norm == 0.0f) return;

	float scale = max_norm / total_norm;

	// Scale all gradients
	for (usz i = 0; i < self.unit_count; i++)
	{
		device_scale_grad(self.units[i].param.grad(), scale);
	}
}

macro float device_grad_norm_sq(Tensor grad) {
	$if $defined(cuda_grad_norm_sq):
		if (grad.device().name() == "cuda") { return cuda_grad_norm_sq(grad); }
	$endif
	$if $defined(hip_grad_norm_sq):
		if (grad.device().name() == "hip") { return hip_grad_norm_sq(grad); }
	$endif
	if (grad.device().name() == "host") { return host_grad_norm_sq(grad); }
	unreachable("device_grad_norm_sq: no impl for device '%s'", grad.device().name());
}

macro void device_scale_grad(Tensor grad, float scale) {
	$if $defined(cuda_scale_grad):
		if (grad.device().name() == "cuda") { cuda_scale_grad(grad, scale); return; }
	$endif
	$if $defined(hip_scale_grad):
		if (grad.device().name() == "hip") { hip_scale_grad(grad, scale); return; }
	$endif
	if (grad.device().name() == "host") { host_scale_grad(grad, scale); return; }
	unreachable("device_scale_grad: no impl for device '%s'", grad.device().name());
}

// Host SGD Implementation

struct HostSgdContext
{
	Tensor param;
	char[] param_bytes;
	rtti::Datatype dtype;
	usz numel;
	float clip_grad;
	char[] accum;
	bool use_accum;
}

fn OptimizerUnit host_sgd_create(Tensor param, SgdOptions opts)
{
	HostSgdContext* ctx = (HostSgdContext*)mem::malloc(HostSgdContext.sizeof);
	assert(ctx != null, "host_sgd_create: allocation failed");
	ctx.param = param;
	ctx.param_bytes = param.value_bytes();
	ctx.dtype = param.dtype();
	ctx.numel = param.shape().size();
	ctx.clip_grad = opts.clip_grad;
	ctx.accum = {};
	ctx.use_accum = false;

	return {
		.param = param,
		.execute_fn = &host_sgd_execute,
		.ctx = ctx,
		.deinit_fn = &host_sgd_deinit,
		.accum_fn = &host_grad_accum,
		.zero_accum_fn = &host_zero_accum,
	};
}

fn void host_sgd_execute(void* ptr, float lr)
{
	HostSgdContext* ctx = (HostSgdContext*)ptr;
	usz n = ctx.numel;

	// Resolve gradient source
	char[] grad_bytes;
	bool grad_is_f32;
	if (ctx.use_accum)
	{
		assert(ctx.accum.len > 0, "host_sgd_execute: accum buffer not allocated");
		grad_bytes = ctx.accum;
		grad_is_f32 = true;
		if (ctx.clip_grad > 0.0f) { host_scale_grad_clip_bytes(grad_bytes, n, ctx.clip_grad); }
	}
	else
	{
		Tensor grad = ctx.param.grad();
		grad_bytes = grad.value_bytes();
		assert(grad_bytes.len > 0, "host_sgd_execute: grad has no data");
		grad_is_f32 = (ctx.dtype == F32);
		if (ctx.clip_grad > 0.0f) { host_scale_grad_clip(grad, ctx.clip_grad); }
	}

	// SGD: param -= lr * grad
	switch (ctx.dtype)
	{
		case BF16:
			bfloat[] p = ((bfloat*)ctx.param_bytes.ptr)[:n];
			if (grad_is_f32)
			{
				float[] g = ((float*)grad_bytes.ptr)[:n];
				for (usz i = 0; i < n; i++) { p[i] = (bfloat)((float)p[i] - lr * g[i]); }
			}
			else
			{
				bfloat[] g = ((bfloat*)grad_bytes.ptr)[:n];
				for (usz i = 0; i < n; i++) { p[i] = (bfloat)((float)p[i] - lr * (float)g[i]); }
			}
		case F16:
			float16[] p16 = ((float16*)ctx.param_bytes.ptr)[:n];
			if (grad_is_f32)
			{
				float[] g = ((float*)grad_bytes.ptr)[:n];
				for (usz i = 0; i < n; i++) { p16[i] = (float16)((float)p16[i] - lr * g[i]); }
			}
			else
			{
				float16[] g = ((float16*)grad_bytes.ptr)[:n];
				for (usz i = 0; i < n; i++) { p16[i] = (float16)((float)p16[i] - lr * (float)g[i]); }
			}
		case F32:
			float[] p32 = ((float*)ctx.param_bytes.ptr)[:n];
			float[] g32 = ((float*)grad_bytes.ptr)[:n];
			for (usz i = 0; i < n; i++) { p32[i] -= lr * g32[i]; }
		case F64:
			double[] p64 = ((double*)ctx.param_bytes.ptr)[:n];
			if (grad_is_f32)
			{
				float[] g = ((float*)grad_bytes.ptr)[:n];
				for (usz i = 0; i < n; i++) { p64[i] -= (double)lr * (double)g[i]; }
			}
			else
			{
				double[] g64 = ((double*)grad_bytes.ptr)[:n];
				for (usz i = 0; i < n; i++) { p64[i] -= (double)lr * g64[i]; }
			}
		default:
			unreachable("host SGD: unsupported dtype");
	}
	ctx.use_accum = false;
}

fn void host_scale_grad_clip_bytes(char[] buf, usz n, float max_norm)
{
	float* g = ((float*)buf.ptr);
	float norm_sq = 0.0f;
	for (usz i = 0; i < n; i++) { norm_sq += g[i] * g[i]; }
	float norm = (float)math::sqrt((double)norm_sq);
	if (norm > max_norm && norm > 0.0f)
	{
		float s = max_norm / norm;
		for (usz i = 0; i < n; i++) { g[i] *= s; }
	}
}

fn void host_sgd_deinit(void* ptr)
{
	HostSgdContext* ctx = (HostSgdContext*)ptr;
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_free(ctx.accum);
	}
	mem::free(ptr);
}

// Host Adam Implementation

struct HostAdamContext
{
	Tensor param;
	char[] param_bytes;
	rtti::Datatype dtype;
	usz numel;
	char[] m;     // First moment (always f32)
	char[] v;     // Second moment (always f32)
	char[] accum; // Gradient accumulation buffer (always f32, lazy-allocated)
	bool use_accum;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	float clip_grad;
	uint t;
}

fn OptimizerUnit host_adam_create(Tensor param, AdamOptions opts)
{
	usz numel = param.shape().size();

	// Allocate moments through the device (always f32)
	char[] m = param.device().mem_alloc(F32, numel);
	char[] v = param.device().mem_alloc(F32, numel);
	param.device().mem_fill(F32, m, 0.0);
	param.device().mem_fill(F32, v, 0.0);

	HostAdamContext* ctx = (HostAdamContext*)mem::malloc(HostAdamContext.sizeof);
	assert(ctx != null, "host_adam_create: allocation failed");
	ctx.param = param;
	ctx.param_bytes = param.value_bytes();
	ctx.dtype = param.dtype();
	ctx.numel = numel;
	ctx.m = m;
	ctx.v = v;
	ctx.accum = {};
	ctx.use_accum = false;
	ctx.beta1 = opts.beta1;
	ctx.beta2 = opts.beta2;
	ctx.eps = opts.eps;
	ctx.weight_decay = opts.weight_decay;
	ctx.clip_grad = opts.clip_grad;
	ctx.t = 0;

	return {
		.param = param,
		.execute_fn = &host_adam_execute,
		.ctx = ctx,
		.deinit_fn = &host_adam_deinit,
		.accum_fn = &host_adam_grad_accum,
		.zero_accum_fn = &host_adam_zero_accum,
	};
}

fn void host_adam_execute(void* ptr, float lr)
{
	HostAdamContext* ctx = (HostAdamContext*)ptr;
	ctx.t++;
	usz n = ctx.numel;

	float bc1 = 1.0f - math::pow(ctx.beta1, (float)ctx.t);
	float bc2 = 1.0f - math::pow(ctx.beta2, (float)ctx.t);

	// Resolve gradient source
	char[] grad_bytes;
	bool grad_is_f32;
	if (ctx.use_accum)
	{
		assert(ctx.accum.len > 0, "host_adam_execute: accum buffer not allocated");
		grad_bytes = ctx.accum;
		grad_is_f32 = true;
		if (ctx.clip_grad > 0.0f) { host_scale_grad_clip_bytes(grad_bytes, n, ctx.clip_grad); }
	}
	else
	{
		Tensor grad = ctx.param.grad();
		grad_bytes = grad.value_bytes();
		assert(grad_bytes.len > 0, "host_adam_execute: grad has no data");
		grad_is_f32 = (ctx.dtype == F32);
		if (ctx.clip_grad > 0.0f) { host_scale_grad_clip(grad, ctx.clip_grad); }
	}

	float[] m = ((float*)ctx.m.ptr)[:n];
	float[] v = ((float*)ctx.v.ptr)[:n];

	// Adam update — one dtype switch, grad cast inside the loop
	switch (ctx.dtype)
	{
		case BF16:
			bfloat[] p = ((bfloat*)ctx.param_bytes.ptr)[:n];
			for (usz i = 0; i < n; i++)
			{
				float gi = grad_is_f32 ? ((float*)grad_bytes.ptr)[i] : (float)((bfloat*)grad_bytes.ptr)[i];
				float pi = (float)p[i];
				if (ctx.weight_decay > 0.0f) { pi -= lr * ctx.weight_decay * pi; }
				m[i] = ctx.beta1 * m[i] + (1.0f - ctx.beta1) * gi;
				v[i] = ctx.beta2 * v[i] + (1.0f - ctx.beta2) * gi * gi;
				pi -= lr * (m[i] / bc1) / ((float)math::sqrt((double)(v[i] / bc2)) + ctx.eps);
				p[i] = (bfloat)pi;
			}
		case F16:
			float16[] p16 = ((float16*)ctx.param_bytes.ptr)[:n];
			for (usz i = 0; i < n; i++)
			{
				float gi = grad_is_f32 ? ((float*)grad_bytes.ptr)[i] : (float)((float16*)grad_bytes.ptr)[i];
				float pi = (float)p16[i];
				if (ctx.weight_decay > 0.0f) { pi -= lr * ctx.weight_decay * pi; }
				m[i] = ctx.beta1 * m[i] + (1.0f - ctx.beta1) * gi;
				v[i] = ctx.beta2 * v[i] + (1.0f - ctx.beta2) * gi * gi;
				pi -= lr * (m[i] / bc1) / ((float)math::sqrt((double)(v[i] / bc2)) + ctx.eps);
				p16[i] = (float16)pi;
			}
		case F32:
			float[] p32 = ((float*)ctx.param_bytes.ptr)[:n];
			float[] g32 = ((float*)grad_bytes.ptr)[:n];
			for (usz i = 0; i < n; i++)
			{
				float gi = g32[i];
				if (ctx.weight_decay > 0.0f) { p32[i] -= lr * ctx.weight_decay * p32[i]; }
				m[i] = ctx.beta1 * m[i] + (1.0f - ctx.beta1) * gi;
				v[i] = ctx.beta2 * v[i] + (1.0f - ctx.beta2) * gi * gi;
				p32[i] -= lr * (m[i] / bc1) / ((float)math::sqrt((double)(v[i] / bc2)) + ctx.eps);
			}
		case F64:
			double[] p64 = ((double*)ctx.param_bytes.ptr)[:n];
			double lr64 = (double)lr;
			for (usz i = 0; i < n; i++)
			{
				float gi = grad_is_f32 ? ((float*)grad_bytes.ptr)[i] : (float)((double*)grad_bytes.ptr)[i];
				if (ctx.weight_decay > 0.0f) { p64[i] -= lr64 * (double)ctx.weight_decay * p64[i]; }
				m[i] = ctx.beta1 * m[i] + (1.0f - ctx.beta1) * gi;
				v[i] = ctx.beta2 * v[i] + (1.0f - ctx.beta2) * gi * gi;
				p64[i] -= lr64 * (double)(m[i] / bc1) / (math::sqrt((double)(v[i] / bc2)) + (double)ctx.eps);
			}
		default:
			unreachable("host Adam: unsupported dtype");
	}
	ctx.use_accum = false;
}

fn void host_adam_deinit(void* ptr)
{
	HostAdamContext* ctx = (HostAdamContext*)ptr;
	ctx.param.device().mem_free(ctx.m);
	ctx.param.device().mem_free(ctx.v);
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_free(ctx.accum);
	}
	mem::free(ctx);
}

// Host Gradient Accumulation Helpers

/// Accumulate gradient into a buffer: accum += scale * grad.
/// Lazy-allocates the accum buffer on first use.
fn void host_accum_grad_into(Tensor param, char[]* accum, bool* use_accum, float scale)
{
	Tensor grad = param.grad();
	char[] grad_bytes = grad.value_bytes();
	assert(grad_bytes.len > 0, "host_accum_grad_into: grad has no data");

	usz numel = param.shape().size();

	// Lazy-allocate accum buffer (always f32)
	if (accum.len == 0)
	{
		*accum = param.device().mem_alloc(F32, numel);
		param.device().mem_fill(F32, *accum, 0.0);
	}

	float* dst = (float*)accum.ptr;

	switch (param.dtype())
	{
		case BF16:
			bfloat* gb = (bfloat*)grad_bytes.ptr;
			for (usz i = 0; i < numel; i++) { dst[i] += scale * (float)gb[i]; }
		case F32:
			blas_callbacks::cblas_saxpy((int)numel, scale, (float*)grad_bytes.ptr, 1, dst, 1);
		case F64:
			double* gd = (double*)grad_bytes.ptr;
			for (usz i = 0; i < numel; i++) { dst[i] += scale * (float)gd[i]; }
		default:
			unreachable("host_accum_grad_into: unsupported dtype");
	}

	*use_accum = true;
}

fn void host_zero_accum_buf(char[] accum, Tensor param)
{
	if (accum.len > 0)
	{
		param.device().mem_fill(F32, accum, 0.0);
	}
}

fn void host_grad_accum(void* ptr, float scale)
{
	HostSgdContext* ctx = (HostSgdContext*)ptr;
	host_accum_grad_into(ctx.param, &ctx.accum, &ctx.use_accum, scale);
}

fn void host_zero_accum(void* ptr)
{
	HostSgdContext* ctx = (HostSgdContext*)ptr;
	host_zero_accum_buf(ctx.accum, ctx.param);
}

fn void host_adam_grad_accum(void* ptr, float scale)
{
	HostAdamContext* ctx = (HostAdamContext*)ptr;
	host_accum_grad_into(ctx.param, &ctx.accum, &ctx.use_accum, scale);
}

fn void host_adam_zero_accum(void* ptr)
{
	HostAdamContext* ctx = (HostAdamContext*)ptr;
	host_zero_accum_buf(ctx.accum, ctx.param);
}

// Host Gradient Clipping Helpers

/// Compute sum of squared gradient values.
fn float host_grad_norm_sq(Tensor grad)
{
	char[] bytes = grad.value_bytes();
	if (bytes.len == 0) return 0.0f;

	float sum_sq = 0.0f;
	switch (grad.dtype())
	{
		case BF16:
			bfloat* vb = (bfloat*)bytes.ptr;
			usz nb = bytes.len / bfloat.sizeof;
			for (usz i = 0; i < nb; i++)
			{
				float f = (float)vb[i];
				sum_sq += f * f;
			}

		case F32:
			float* vals = (float*)bytes.ptr;
			usz n = bytes.len / float.sizeof;
			for (usz i = 0; i < n; i++)
			{
				sum_sq += vals[i] * vals[i];
			}

		case F64:
			double* vd = (double*)bytes.ptr;
			usz nd = bytes.len / double.sizeof;
			for (usz i = 0; i < nd; i++)
			{
				float f = (float)vd[i];
				sum_sq += f * f;
			}

		default:
			unreachable("host_grad_norm_sq: unsupported dtype");
	}
	return sum_sq;
}

/// Scale all gradient values by a factor.
fn void host_scale_grad(Tensor grad, float scale)
{
	char[] bytes = grad.value_bytes();
	if (bytes.len == 0) return;

	switch (grad.dtype())
	{
		case BF16:
			bfloat* vb = (bfloat*)bytes.ptr;
			usz nb = bytes.len / bfloat.sizeof;
			for (usz i = 0; i < nb; i++)
			{
				vb[i] = (bfloat)((float)vb[i] * scale);
			}

		case F32:
			float* vals = (float*)bytes.ptr;
			usz n = bytes.len / float.sizeof;
			for (usz i = 0; i < n; i++)
			{
				vals[i] *= scale;
			}

		case F64:
			double* vd = (double*)bytes.ptr;
			usz nd = bytes.len / double.sizeof;
			for (usz i = 0; i < nd; i++)
			{
				vd[i] *= (double)scale;
			}

		default:
			unreachable("host_scale_grad: unsupported dtype");
	}
}

/// Clip gradient by L2 norm (per-parameter).
fn void host_scale_grad_clip(Tensor grad, float max_norm)
{
	float norm_sq = host_grad_norm_sq(grad);
	float norm = (float)math::sqrt((double)norm_sq);
	if (norm > max_norm && norm > 0.0f)
	{
		host_scale_grad(grad, max_norm / norm);
	}
}

