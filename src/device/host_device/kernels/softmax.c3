//! Softmax Kernel
//!
//! Fast CPU softmax with SIMD vectorization.
//! Softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
//!
//! Includes forward and backward passes for rowwise and columnwise variants.
//!

module metaphor::device::host_device::kernels::softmax;
import std::math;

const usz VEC_WIDTH = 8; // float[<8>] = 256-bit SIMD

/// Element-wise max of two SIMD vectors.
macro float[<VEC_WIDTH>] vec_max(float[<VEC_WIDTH>] a, float[<VEC_WIDTH>] b)
{
	bool[<VEC_WIDTH>] mask = a.comp_gt(b);
	float[<VEC_WIDTH>] result;
	for (usz k = 0; k < VEC_WIDTH; k++)
	{
		result[k] = mask[k] ? a[k] : b[k];
	}
	return result;
}

// Forward: Softmax

/// Softmax over a single 1D vector. Numerically stable.
/// x and y may overlap (in-place safe).
fn void softmax_1d_float(float* x, float* y, usz n)
{
	assert(n > 0);

	// Step 1: Find max for numerical stability
	float max_val = x[0];
	usz i = 1;

	if (n >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] max_vec = ((float[<VEC_WIDTH>]*)x)[0];
		i = VEC_WIDTH;
		while (i + VEC_WIDTH <= n)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			max_vec = vec_max(max_vec, v);
			i += VEC_WIDTH;
		}
		max_val = max_vec.max();
	}

	while (i < n)
	{
		if (x[i] > max_val) { max_val = x[i]; }
		i++;
	}

	// Step 2: Compute exp(x - max) and sum
	float sum = 0;
	i = 0;

	if (n >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] max_splat = (float[<VEC_WIDTH>])max_val;
		float[<VEC_WIDTH>] sum_vec = (float[<VEC_WIDTH>])0;

		while (i + VEC_WIDTH <= n)
		{
			float[<VEC_WIDTH>] x_vec = ((float[<VEC_WIDTH>]*)(x + i))[0];
			float[<VEC_WIDTH>] shifted = x_vec - max_splat;
			float[<VEC_WIDTH>] exp_vec = math::exp(shifted);
			((float[<VEC_WIDTH>]*)(y + i))[0] = exp_vec;
			sum_vec += exp_vec;
			i += VEC_WIDTH;
		}
		sum = sum_vec.sum();
	}

	// Scalar remainder
	while (i < n)
	{
		float exp_val = math::exp(x[i] - max_val);
		y[i] = exp_val;
		sum += exp_val;
		i++;
	}

	// Step 3: Normalize
	float inv_sum = 1.0f / sum;
	i = 0;

	if (n >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] inv_splat = (float[<VEC_WIDTH>])inv_sum;
		while (i + VEC_WIDTH <= n)
		{
			float[<VEC_WIDTH>] y_vec = ((float[<VEC_WIDTH>]*)(y + i))[0];
			((float[<VEC_WIDTH>]*)(y + i))[0] = y_vec * inv_splat;
			i += VEC_WIDTH;
		}
	}

	while (i < n)
	{
		y[i] *= inv_sum;
		i++;
	}
}

/// Softmax over rows (batch dimension). Each row is independently normalized.
/// x, y: [batch * dim] flattened.
fn void softmax_rows_float(float* x, float* y, usz batch, usz dim)
{
	for (usz b = 0; b < batch; b++)
	{
		softmax_1d_float(x + b * dim, y + b * dim, dim);
	}
}

/// Softmax over columns. Each column is independently normalized across batch.
/// x, y: [batch * dim] flattened.
fn void softmax_cols_float(float* x, float* y, usz batch, usz dim)
{
	for (usz d = 0; d < dim; d++)
	{
		// Step 1: Find max for this column
		float max_val = x[d];
		for (usz b = 1; b < batch; b++)
		{
			float v = x[b * dim + d];
			if (v > max_val) { max_val = v; }
		}

		// Step 2: Compute exp(x - max) and sum
		float sum = 0;
		for (usz b = 0; b < batch; b++)
		{
			usz idx = b * dim + d;
			float exp_val = math::exp(x[idx] - max_val);
			y[idx] = exp_val;
			sum += exp_val;
		}

		// Step 3: Normalize
		float inv_sum = 1.0f / sum;
		for (usz b = 0; b < batch; b++)
		{
			y[b * dim + d] *= inv_sum;
		}
	}
}

// Forward: Log-Softmax

/// Log-softmax over a single 1D vector.
/// log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
fn void log_softmax_1d_float(float* x, float* y, usz n)
{
	assert(n > 0);

	// Step 1: Find max
	float max_val = x[0];
	usz i = 1;
	if (n >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] max_vec = ((float[<VEC_WIDTH>]*)x)[0];
		i = VEC_WIDTH;
		while (i + VEC_WIDTH <= n)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			max_vec = vec_max(max_vec, v);
			i += VEC_WIDTH;
		}
		max_val = max_vec.max();
	}
	while (i < n)
	{
		if (x[i] > max_val) { max_val = x[i]; }
		i++;
	}

	// Step 2: Compute sum of exp(x - max)
	float sum = 0;
	i = 0;
	if (n >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] max_splat = (float[<VEC_WIDTH>])max_val;
		float[<VEC_WIDTH>] sum_vec = (float[<VEC_WIDTH>])0;
		while (i + VEC_WIDTH <= n)
		{
			float[<VEC_WIDTH>] x_vec = ((float[<VEC_WIDTH>]*)(x + i))[0];
			sum_vec += math::exp(x_vec - max_splat);
			i += VEC_WIDTH;
		}
		sum = sum_vec.sum();
	}
	while (i < n)
	{
		sum += math::exp(x[i] - max_val);
		i++;
	}

	// Step 3: y = x - max - log(sum)
	float offset = max_val + math::ln(sum);
	i = 0;
	if (n >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] offset_splat = (float[<VEC_WIDTH>])offset;
		while (i + VEC_WIDTH <= n)
		{
			float[<VEC_WIDTH>] x_vec = ((float[<VEC_WIDTH>]*)(x + i))[0];
			((float[<VEC_WIDTH>]*)(y + i))[0] = x_vec - offset_splat;
			i += VEC_WIDTH;
		}
	}
	while (i < n)
	{
		y[i] = x[i] - offset;
		i++;
	}
}

/// Log-softmax over rows.
fn void log_softmax_rows_float(float* x, float* y, usz batch, usz dim)
{
	for (usz b = 0; b < batch; b++)
	{
		log_softmax_1d_float(x + b * dim, y + b * dim, dim);
	}
}

/// Log-softmax over columns.
fn void log_softmax_cols_float(float* x, float* y, usz batch, usz dim)
{
	for (usz d = 0; d < dim; d++)
	{
		float max_val = x[d];
		for (usz b = 1; b < batch; b++)
		{
			float v = x[b * dim + d];
			if (v > max_val) { max_val = v; }
		}

		float sum = 0;
		for (usz b = 0; b < batch; b++)
		{
			sum += math::exp(x[b * dim + d] - max_val);
		}

		float offset = max_val + math::ln(sum);
		for (usz b = 0; b < batch; b++)
		{
			usz idx = b * dim + d;
			y[idx] = x[idx] - offset;
		}
	}
}

// Backward: Softmax

/// Softmax backward (rowwise): grad_x = y * (grad_y - sum(y * grad_y))
fn void softmax_rows_backward_float(float* grad_y, float* y, float* grad_x, usz batch, usz dim)
{
	for (usz b = 0; b < batch; b++)
	{
		softmax_1d_backward_float(grad_y + b * dim, y + b * dim, grad_x + b * dim, dim);
	}
}

/// Softmax backward (columnwise): grad_x = y * (grad_y - sum(y * grad_y))
fn void softmax_cols_backward_float(float* grad_y, float* y, float* grad_x, usz batch, usz dim)
{
	for (usz d = 0; d < dim; d++)
	{
		// Compute dot = sum(y * grad_y) for this column
		float dot = 0;
		for (usz b = 0; b < batch; b++)
		{
			usz idx = b * dim + d;
			dot += y[idx] * grad_y[idx];
		}

		// grad_x = y * (grad_y - dot)
		for (usz b = 0; b < batch; b++)
		{
			usz idx = b * dim + d;
			grad_x[idx] = y[idx] * (grad_y[idx] - dot);
		}
	}
}

/// Softmax 1D backward helper.
fn void softmax_1d_backward_float(float* grad_y, float* y, float* grad_x, usz n) @private
{
	// Compute dot = sum(y * grad_y)
	float dot = 0;
	for (usz i = 0; i < n; i++)
	{
		dot += y[i] * grad_y[i];
	}

	// grad_x = y * (grad_y - dot)
	for (usz i = 0; i < n; i++)
	{
		grad_x[i] = y[i] * (grad_y[i] - dot);
	}
}

// Backward: Log-Softmax

/// Log-softmax backward (rowwise): grad_x = grad_y - exp(y) * sum(grad_y)
fn void log_softmax_rows_backward_float(float* grad_y, float* y, float* grad_x, usz batch, usz dim)
{
	for (usz b = 0; b < batch; b++)
	{
		log_softmax_1d_backward_float(grad_y + b * dim, y + b * dim, grad_x + b * dim, dim);
	}
}

/// Log-softmax backward (columnwise).
fn void log_softmax_cols_backward_float(float* grad_y, float* y, float* grad_x, usz batch, usz dim)
{
	for (usz d = 0; d < dim; d++)
	{
		float sum_grad = 0;
		for (usz b = 0; b < batch; b++)
		{
			sum_grad += grad_y[b * dim + d];
		}

		for (usz b = 0; b < batch; b++)
		{
			usz idx = b * dim + d;
			grad_x[idx] = grad_y[idx] - math::exp(y[idx]) * sum_grad;
		}
	}
}

/// Log-softmax 1D backward helper.
fn void log_softmax_1d_backward_float(float* grad_y, float* y, float* grad_x, usz n) @private
{
	float sum_grad = 0;
	for (usz i = 0; i < n; i++)
	{
		sum_grad += grad_y[i];
	}

	for (usz i = 0; i < n; i++)
	{
		grad_x[i] = grad_y[i] - math::exp(y[i]) * sum_grad;
	}
}
