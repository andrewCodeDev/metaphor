//! HIP Optimizer Backends
//!
//! Device-specific SGD and Adam implementations for HIP.
//! Called from metaphor::nn::optimizer via device dispatch.
//!
//! For BF16/F16 parameters, uses master weights in F32 for numerical stability.
//! The visible weights stay in half precision, but optimizer math runs in F32.

module metaphor::nn::optimizer;
import metaphor::tensor;
import metaphor::rtti;
import metaphor::device::hip_device;
import metaphor::device::hip_device::kernels;
import metaphor::device::hip_device::utils;
import std::math;
import std::io;


// HIP SGD Implementation

struct HipSgdContext
{
	Tensor param;
	float clip_grad;
	hip_device::HipDevice* device;
	char[] master;  // F32 master weights for BF16/F16 (GPU)
	char[] accum;   // Gradient accumulation buffer (f32, GPU, lazy-allocated)
	bool use_accum;
	bool use_master;
	usz numel;
}

fn OptimizerUnit hip_sgd_create(Tensor param, SgdOptions opts)
{
	hip_device::HipDevice* device = (hip_device::HipDevice*)param.device().device;
	usz numel = param.shape().size();
	bool use_master = (param.dtype() == BF16 || param.dtype() == F16);

	char[] master;
	if (use_master)
	{
		master = param.device().mem_alloc(F32, numel);
		utils::StreamHandle stream = device.get_stream();
		int vis_dtype = utils::dtype_from_datatype(param.dtype());
		kernels::hip_optim_init_master(vis_dtype, stream, master.ptr, param.value_bytes().ptr, (ulong)numel);
	}

	HipSgdContext* ctx = (HipSgdContext*)mem::malloc(HipSgdContext.sizeof);
	assert(ctx != null, "hip_sgd_create: allocation failed");
	ctx.param = param;
	ctx.clip_grad = opts.clip_grad;
	ctx.device = device;
	ctx.master = master;
	ctx.accum = {};
	ctx.use_accum = false;
	ctx.use_master = use_master;
	ctx.numel = numel;

	return {
		.param = param,
		.execute_fn = &hip_sgd_execute,
		.ctx = ctx,
		.deinit_fn = &hip_sgd_deinit,
		.accum_fn = &hip_sgd_grad_accum,
		.zero_accum_fn = &hip_sgd_zero_accum,
	};
}

fn void hip_sgd_execute(void* ptr, float lr)
{
	HipSgdContext* ctx = (HipSgdContext*)ptr;
	Tensor param = ctx.param;

	char[] param_bytes = param.value_bytes();
	assert(param_bytes.len > 0, "hip_sgd_execute: param has no data");

	utils::StreamHandle stream = ctx.device.get_stream();

	if (ctx.use_accum && ctx.use_master)
	{
		// BF16/F16 param + F32 accumulated gradient:
		// 1. SGD update on F32 master: master -= lr * accum
		int f32_dtype = utils::dtype_from_datatype(F32);
		kernels::hip_optim_sgd(f32_dtype, stream, ctx.master.ptr, ctx.accum.ptr, (ulong)ctx.numel, (double)lr);
		// 2. Copy master → visible
		int vis_dtype = utils::dtype_from_datatype(param.dtype());
		kernels::hip_optim_copy_master_to_visible(vis_dtype, stream, param_bytes.ptr, ctx.master.ptr, (ulong)ctx.numel);
	}
	else if (ctx.use_accum)
	{
		// F32 param + F32 accumulated gradient
		int f32_dtype = utils::dtype_from_datatype(F32);
		kernels::hip_optim_sgd(f32_dtype, stream, param_bytes.ptr, ctx.accum.ptr, (ulong)ctx.numel, (double)lr);
	}
	else
	{
		// Standard path: use compute graph gradient
		Tensor grad = param.grad();
		char[] grad_bytes = grad.value_bytes();
		assert(grad_bytes.len > 0, "hip_sgd_execute: grad has no data");
		int dtype = utils::dtype_from_datatype(param.dtype());

		// Clip per-param gradients if enabled
		if (ctx.clip_grad > 0.0f)
		{
			float norm_sq = kernels::hip_optim_grad_norm_sq(dtype, stream, grad_bytes.ptr, (ulong)ctx.numel);
			float norm = (float)math::sqrt((double)norm_sq);
			if (norm > ctx.clip_grad && norm > 0.0f)
			{
				kernels::hip_optim_scale_grad(dtype, stream, grad_bytes.ptr, (ulong)ctx.numel, ctx.clip_grad / norm);
			}
		}

		if (ctx.use_master)
		{
			kernels::hip_optim_sgd_master(dtype, stream, param_bytes.ptr, ctx.master.ptr, grad_bytes.ptr, (ulong)ctx.numel, (double)lr);
		}
		else
		{
			kernels::hip_optim_sgd(dtype, stream, param_bytes.ptr, grad_bytes.ptr, (ulong)ctx.numel, (double)lr);
		}
	}
}

fn void hip_sgd_deinit(void* ptr)
{
	HipSgdContext* ctx = (HipSgdContext*)ptr;
	if (ctx.master.len > 0) { ctx.param.device().mem_free(ctx.master); }
	if (ctx.accum.len > 0) { ctx.param.device().mem_free(ctx.accum); }
	mem::free(ptr);
}

// HIP Adam Implementation

struct HipAdamContext
{
	Tensor param;
	hip_device::HipDevice* device;
	char[] m;       // First moment (f32, GPU)
	char[] v;       // Second moment (f32, GPU)
	char[] master;  // F32 master weights for BF16/F16 (GPU)
	char[] accum;   // Gradient accumulation buffer (f32, GPU, lazy-allocated)
	char[] norm_sq_buf;  // Device-side float for fused clip (1 element)
	bool use_accum;
	bool use_master;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	float clip_grad;
	uint t;
	usz numel;
}

fn OptimizerUnit hip_adam_create(Tensor param, AdamOptions opts)
{
	usz numel = param.shape().size();
	hip_device::HipDevice* device = (hip_device::HipDevice*)param.device().device;
	bool use_master = (param.dtype() == BF16 || param.dtype() == F16);

	// Allocate moments on GPU (always f32)
	char[] m = param.device().mem_alloc(F32, numel);
	char[] v = param.device().mem_alloc(F32, numel);
	param.device().mem_fill(F32, m, 0.0);
	param.device().mem_fill(F32, v, 0.0);

	// Master weights for half precision
	char[] master;
	if (use_master)
	{
		master = param.device().mem_alloc(F32, numel);
		utils::StreamHandle stream = device.get_stream();
		int vis_dtype = utils::dtype_from_datatype(param.dtype());
		kernels::hip_optim_init_master(vis_dtype, stream, master.ptr, param.value_bytes().ptr, (ulong)numel);
	}

	// Device-side norm² buffer for fused clipping
	char[] norm_sq_buf;
	if (opts.clip_grad > 0.0f)
	{
		norm_sq_buf = param.device().mem_alloc(F32, 1);
	}

	HipAdamContext* ctx = (HipAdamContext*)mem::malloc(HipAdamContext.sizeof);
	assert(ctx != null, "hip_adam_create: allocation failed");
	ctx.param = param;
	ctx.device = device;
	ctx.m = m;
	ctx.v = v;
	ctx.master = master;
	ctx.accum = {};
	ctx.norm_sq_buf = norm_sq_buf;
	ctx.use_accum = false;
	ctx.use_master = use_master;
	ctx.beta1 = opts.beta1;
	ctx.beta2 = opts.beta2;
	ctx.eps = opts.eps;
	ctx.weight_decay = opts.weight_decay;
	ctx.clip_grad = opts.clip_grad;
	ctx.t = 0;
	ctx.numel = numel;

	return {
		.param = param,
		.execute_fn = &hip_adam_execute,
		.ctx = ctx,
		.deinit_fn = &hip_adam_deinit,
		.accum_fn = &hip_adam_grad_accum,
		.zero_accum_fn = &hip_adam_zero_accum,
	};
}

fn void hip_adam_execute(void* ptr, float lr)
{
	HipAdamContext* ctx = (HipAdamContext*)ptr;
	ctx.t++;

	Tensor param = ctx.param;

	char[] param_bytes = param.value_bytes();
	assert(param_bytes.len > 0, "hip_adam_execute: param has no data");

	utils::StreamHandle stream = ctx.device.get_stream();
	int f32_dtype = utils::dtype_from_datatype(F32);

	if (ctx.use_accum && ctx.use_master)
	{
		// BF16/F16 param + F32 accumulated gradient:
		// 1. Adam update on F32 master weights using F32 accum
		kernels::hip_optim_adam(
			f32_dtype, stream, ctx.master.ptr, ctx.accum.ptr,
			ctx.m.ptr, ctx.v.ptr, (ulong)ctx.numel,
			(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps,
			(ulong)ctx.t, (double)ctx.weight_decay);
		// 2. Copy master → visible
		int vis_dtype = utils::dtype_from_datatype(param.dtype());
		kernels::hip_optim_copy_master_to_visible(vis_dtype, stream, param_bytes.ptr, ctx.master.ptr, (ulong)ctx.numel);
	}
	else if (ctx.use_accum)
	{
		// F32 param + F32 accumulated gradient
		kernels::hip_optim_adam(
			f32_dtype, stream, param_bytes.ptr, ctx.accum.ptr,
			ctx.m.ptr, ctx.v.ptr, (ulong)ctx.numel,
			(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps,
			(ulong)ctx.t, (double)ctx.weight_decay);
	}
	else
	{
		// Standard path: use compute graph gradient
		Tensor grad = param.grad();
		char[] grad_bytes = grad.value_bytes();
		assert(grad_bytes.len > 0, "hip_adam_execute: grad has no data");
		int dtype = utils::dtype_from_datatype(param.dtype());

		if (ctx.clip_grad > 0.0f)
		{
			// Fused path: no DtoH sync
			ctx.param.device().mem_fill(F32, ctx.norm_sq_buf, 0.0);

			kernels::hip_optim_grad_norm_sq_device(
				dtype, stream, grad_bytes.ptr, ctx.norm_sq_buf.ptr, (ulong)ctx.numel);

			kernels::hip_optim_adam_clipped(
				dtype, stream, param_bytes.ptr, grad_bytes.ptr,
				ctx.m.ptr, ctx.v.ptr, ctx.norm_sq_buf.ptr, (ulong)ctx.numel,
				(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps,
				(ulong)ctx.t, (double)ctx.clip_grad);
		}
		else if (ctx.use_master)
		{
			kernels::hip_optim_adam_master(
				dtype, stream, param_bytes.ptr, ctx.master.ptr, grad_bytes.ptr,
				ctx.m.ptr, ctx.v.ptr, (ulong)ctx.numel,
				(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps,
				(ulong)ctx.t, (double)ctx.weight_decay);
		}
		else
		{
			kernels::hip_optim_adam(
				dtype, stream, param_bytes.ptr, grad_bytes.ptr,
				ctx.m.ptr, ctx.v.ptr, (ulong)ctx.numel,
				(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps,
				(ulong)ctx.t, (double)ctx.weight_decay);
		}
	}
}

fn void hip_adam_deinit(void* ptr)
{
	HipAdamContext* ctx = (HipAdamContext*)ptr;
	ctx.param.device().mem_free(ctx.m);
	ctx.param.device().mem_free(ctx.v);
	if (ctx.master.len > 0) { ctx.param.device().mem_free(ctx.master); }
	if (ctx.accum.len > 0) { ctx.param.device().mem_free(ctx.accum); }
	if (ctx.norm_sq_buf.len > 0) { ctx.param.device().mem_free(ctx.norm_sq_buf); }
	mem::free(ctx);
}

// HIP Gradient Accumulation Helpers

fn void hip_sgd_grad_accum(void* ptr, float scale)
{
	HipSgdContext* ctx = (HipSgdContext*)ptr;
	Tensor grad = ctx.param.grad();
	char[] grad_bytes = grad.value_bytes();
	assert(grad_bytes.len > 0, "hip_sgd_grad_accum: grad has no data");

	if (ctx.accum.len == 0)
	{
		ctx.accum = ctx.param.device().mem_alloc(F32, ctx.numel);
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}

	int grad_dtype = utils::dtype_from_datatype(ctx.param.dtype());
	utils::StreamHandle stream = ctx.device.get_stream();

	// accum += scale * (f32)grad — handles any grad dtype → F32 accum
	kernels::hip_optim_grad_accum(grad_dtype, stream, ctx.accum.ptr, grad_bytes.ptr, (ulong)ctx.numel, (double)scale);
	ctx.use_accum = true;
}

fn void hip_sgd_zero_accum(void* ptr)
{
	HipSgdContext* ctx = (HipSgdContext*)ptr;
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}
}

fn void hip_adam_grad_accum(void* ptr, float scale)
{
	HipAdamContext* ctx = (HipAdamContext*)ptr;
	Tensor grad = ctx.param.grad();
	char[] grad_bytes = grad.value_bytes();
	assert(grad_bytes.len > 0, "hip_adam_grad_accum: grad has no data");

	if (ctx.accum.len == 0)
	{
		ctx.accum = ctx.param.device().mem_alloc(F32, ctx.numel);
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}

	int grad_dtype = utils::dtype_from_datatype(ctx.param.dtype());
	utils::StreamHandle stream = ctx.device.get_stream();

	// accum += scale * (f32)grad — handles any grad dtype → F32 accum
	kernels::hip_optim_grad_accum(grad_dtype, stream, ctx.accum.ptr, grad_bytes.ptr, (ulong)ctx.numel, (double)scale);
	ctx.use_accum = true;
}

fn void hip_adam_zero_accum(void* ptr)
{
	HipAdamContext* ctx = (HipAdamContext*)ptr;
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}
}

// HIP Global Gradient Clipping Helpers

fn float hip_grad_norm_sq(Tensor grad)
{
	hip_device::HipDevice* device = (hip_device::HipDevice*)grad.device().device;
	int dtype = utils::dtype_from_datatype(grad.dtype());
	utils::StreamHandle stream = device.get_stream();
	usz numel = grad.shape().size();
	return kernels::hip_optim_grad_norm_sq(dtype, stream, grad.value_bytes().ptr, (ulong)numel);
}

fn void hip_scale_grad(Tensor grad, float scale)
{
	hip_device::HipDevice* device = (hip_device::HipDevice*)grad.device().device;
	int dtype = utils::dtype_from_datatype(grad.dtype());
	utils::StreamHandle stream = device.get_stream();
	usz numel = grad.shape().size();
	kernels::hip_optim_scale_grad(dtype, stream, grad.value_bytes().ptr, (ulong)numel, scale);
}
