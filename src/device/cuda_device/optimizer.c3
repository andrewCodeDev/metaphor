//! CUDA Optimizer Backends
//!
//! Device-specific SGD and Adam implementations for CUDA.
//! Called from metaphor::nn::optimizer via device dispatch.

module metaphor::nn::optimizer;
import metaphor::tensor;
import metaphor::rtti;
import metaphor::device::cuda_device;
import metaphor::device::cuda_device::kernels;
import metaphor::device::cuda_device::utils;
import std::math;


// CUDA SGD Implementation

struct CudaSgdContext
{
	Tensor param;
	float clip_grad;
	cuda_device::CudaDevice* device;
	char[] accum;   // Gradient accumulation buffer (f32, GPU, lazy-allocated)
	bool use_accum;
}

fn OptimizerUnit cuda_sgd_create(Tensor param, SgdOptions opts)
{
	CudaSgdContext* ctx = (CudaSgdContext*)mem::malloc(CudaSgdContext.sizeof);
	assert(ctx != null, "cuda_sgd_create: allocation failed");
	ctx.param = param;
	ctx.clip_grad = opts.clip_grad;
	ctx.device = (cuda_device::CudaDevice*)param.device().device;
	ctx.accum = {};
	ctx.use_accum = false;

	return {
		.param = param,
		.execute_fn = &cuda_sgd_execute,
		.ctx = ctx,
		.deinit_fn = &cuda_sgd_deinit,
		.accum_fn = &cuda_sgd_grad_accum,
		.zero_accum_fn = &cuda_sgd_zero_accum,
	};
}

fn void cuda_sgd_execute(void* ptr, float lr)
{
	CudaSgdContext* ctx = (CudaSgdContext*)ptr;
	Tensor param = ctx.param;
	Tensor grad = param.grad();

	char[] param_bytes = param.value_bytes();
	assert(param_bytes.len > 0, "cuda_sgd_execute: param has no data");
	char[] grad_bytes = grad.value_bytes();
	assert(grad_bytes.len > 0, "cuda_sgd_execute: grad has no data");

	usz numel = param_bytes.len / param.dtype().size_of();
	int dtype = utils::dtype_from_datatype(param.dtype());
	utils::StreamHandle stream = ctx.device.get_stream();

	// Clip per-param gradients if enabled
	if (ctx.clip_grad > 0.0f)
	{
		float norm_sq = kernels::cuda_optim_grad_norm_sq(dtype, stream, grad_bytes.ptr, (ulong)numel);
		float norm = (float)math::sqrt((double)norm_sq);
		if (norm > ctx.clip_grad && norm > 0.0f)
		{
			kernels::cuda_optim_scale_grad(dtype, stream, grad_bytes.ptr, (ulong)numel, ctx.clip_grad / norm);
		}
	}

	kernels::cuda_optim_sgd(dtype, stream, param_bytes.ptr, grad_bytes.ptr, (ulong)numel, (double)lr);
}

fn void cuda_sgd_deinit(void* ptr)
{
	CudaSgdContext* ctx = (CudaSgdContext*)ptr;
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_free(ctx.accum);
	}
	mem::free(ptr);
}

// CUDA Adam Implementation

struct CudaAdamContext
{
	Tensor param;
	cuda_device::CudaDevice* device;
	char[] m;     // First moment (f32, GPU memory)
	char[] v;     // Second moment (f32, GPU memory)
	char[] accum; // Gradient accumulation buffer (f32, GPU, lazy-allocated)
	char[] norm_sq_buf;  // Device-side float for fused clip (1 element)
	bool use_accum;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	float clip_grad;
	uint t;
	usz numel;
}

fn OptimizerUnit cuda_adam_create(Tensor param, AdamOptions opts)
{
	usz numel = param.shape().size();
	cuda_device::CudaDevice* device = (cuda_device::CudaDevice*)param.device().device;

	// Allocate moments on GPU (always f32)
	char[] m = param.device().mem_alloc(F32, numel);
	char[] v = param.device().mem_alloc(F32, numel);
	// Zero-initialize
	param.device().mem_fill(F32, m, 0.0);
	param.device().mem_fill(F32, v, 0.0);

	// Allocate device-side norm² buffer (1 float) for fused clipping
	char[] norm_sq_buf;
	if (opts.clip_grad > 0.0f)
	{
		norm_sq_buf = param.device().mem_alloc(F32, 1);
	}

	CudaAdamContext* ctx = (CudaAdamContext*)mem::malloc(CudaAdamContext.sizeof);
	assert(ctx != null, "cuda_adam_create: allocation failed");
	ctx.param = param;
	ctx.device = device;
	ctx.m = m;
	ctx.v = v;
	ctx.accum = {};
	ctx.norm_sq_buf = norm_sq_buf;
	ctx.use_accum = false;
	ctx.beta1 = opts.beta1;
	ctx.beta2 = opts.beta2;
	ctx.eps = opts.eps;
	ctx.weight_decay = opts.weight_decay;
	ctx.clip_grad = opts.clip_grad;
	ctx.t = 0;
	ctx.numel = numel;

	return {
		.param = param,
		.execute_fn = &cuda_adam_execute,
		.ctx = ctx,
		.deinit_fn = &cuda_adam_deinit,
		.accum_fn = &cuda_adam_grad_accum,
		.zero_accum_fn = &cuda_adam_zero_accum,
	};
}

fn void cuda_adam_execute(void* ptr, float lr)
{
	CudaAdamContext* ctx = (CudaAdamContext*)ptr;
	ctx.t++;

	Tensor param = ctx.param;
	Tensor grad = param.grad();

	char[] param_bytes = param.value_bytes();
	char[] grad_bytes = grad.value_bytes();

	assert(param_bytes.len > 0, "cuda_adam_execute: param has no data");
	assert(grad_bytes.len > 0, "cuda_adam_execute: grad has no data");

	int dtype = utils::dtype_from_datatype(param.dtype());
	utils::StreamHandle stream = ctx.device.get_stream();

	if (ctx.clip_grad > 0.0f)
	{
		// Fused path: no DtoH sync
		// 1. Zero the device-side norm² buffer
		ctx.param.device().mem_fill(F32, ctx.norm_sq_buf, 0.0);

		// 2. Reduce grad² to device buffer
		kernels::cuda_optim_grad_norm_sq_device(
			dtype, stream, grad_bytes.ptr, ctx.norm_sq_buf.ptr, (ulong)ctx.numel);

		// 3. Clipped Adam update (reads device norm inline)
		kernels::cuda_optim_adam_clipped(
			dtype, stream, param_bytes.ptr, grad_bytes.ptr,
			ctx.m.ptr, ctx.v.ptr, ctx.norm_sq_buf.ptr, (ulong)ctx.numel,
			(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps,
			(ulong)ctx.t, (double)ctx.clip_grad);
	}
	else
	{
		// No clipping — use standard Adam
		kernels::cuda_optim_adam(dtype, stream, param_bytes.ptr, grad_bytes.ptr,
			ctx.m.ptr, ctx.v.ptr, (ulong)ctx.numel,
			(double)lr, (double)ctx.beta1, (double)ctx.beta2, (double)ctx.eps, (ulong)ctx.t);
	}
}

fn void cuda_adam_deinit(void* ptr)
{
	CudaAdamContext* ctx = (CudaAdamContext*)ptr;
	ctx.param.device().mem_free(ctx.m);
	ctx.param.device().mem_free(ctx.v);
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_free(ctx.accum);
	}
	if (ctx.norm_sq_buf.len > 0)
	{
		ctx.param.device().mem_free(ctx.norm_sq_buf);
	}
	mem::free(ctx);
}

// CUDA Gradient Accumulation Helpers

/// Accumulate gradient into GPU buffer: accum += scale * grad.
/// Uses cuda_optim_sgd kernel to perform the scaled add (axpy).
fn void cuda_sgd_grad_accum(void* ptr, float scale)
{
	CudaSgdContext* ctx = (CudaSgdContext*)ptr;
	Tensor grad = ctx.param.grad();
	char[] grad_bytes = grad.value_bytes();
	assert(grad_bytes.len > 0, "cuda_sgd_grad_accum: grad has no data");

	usz numel = ctx.param.shape().size();

	// Lazy-allocate accum buffer on GPU (always f32)
	if (ctx.accum.len == 0)
	{
		ctx.accum = ctx.param.device().mem_alloc(F32, numel);
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}

	int dtype = utils::dtype_from_datatype(ctx.param.dtype());
	utils::StreamHandle stream = ctx.device.get_stream();

	// accum += scale * grad  →  use SGD kernel: accum = accum - (-scale) * grad
	kernels::cuda_optim_sgd(dtype, stream, ctx.accum.ptr, grad_bytes.ptr, (ulong)numel, (double)-scale);

	ctx.use_accum = true;
}

fn void cuda_sgd_zero_accum(void* ptr)
{
	CudaSgdContext* ctx = (CudaSgdContext*)ptr;
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}
}

fn void cuda_adam_grad_accum(void* ptr, float scale)
{
	CudaAdamContext* ctx = (CudaAdamContext*)ptr;
	Tensor grad = ctx.param.grad();
	char[] grad_bytes = grad.value_bytes();
	assert(grad_bytes.len > 0, "cuda_adam_grad_accum: grad has no data");

	usz numel = ctx.param.shape().size();

	if (ctx.accum.len == 0)
	{
		ctx.accum = ctx.param.device().mem_alloc(F32, numel);
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}

	int dtype = utils::dtype_from_datatype(ctx.param.dtype());
	utils::StreamHandle stream = ctx.device.get_stream();

	kernels::cuda_optim_sgd(dtype, stream, ctx.accum.ptr, grad_bytes.ptr, (ulong)numel, (double)-scale);

	ctx.use_accum = true;
}

fn void cuda_adam_zero_accum(void* ptr)
{
	CudaAdamContext* ctx = (CudaAdamContext*)ptr;
	if (ctx.accum.len > 0)
	{
		ctx.param.device().mem_fill(F32, ctx.accum, 0.0);
	}
}

// CUDA Global Gradient Clipping Helpers

fn float cuda_grad_norm_sq(Tensor grad)
{
	cuda_device::CudaDevice* device = (cuda_device::CudaDevice*)grad.device().device;
	int dtype = utils::dtype_from_datatype(grad.dtype());
	utils::StreamHandle stream = device.get_stream();
	usz numel = grad.shape().size();
	return kernels::cuda_optim_grad_norm_sq(dtype, stream, grad.value_bytes().ptr, (ulong)numel);
}

fn void cuda_scale_grad(Tensor grad, float scale)
{
	cuda_device::CudaDevice* device = (cuda_device::CudaDevice*)grad.device().device;
	int dtype = utils::dtype_from_datatype(grad.dtype());
	utils::StreamHandle stream = device.get_stream();
	usz numel = grad.shape().size();
	kernels::cuda_optim_scale_grad(dtype, stream, grad.value_bytes().ptr, (ulong)numel, scale);
}
