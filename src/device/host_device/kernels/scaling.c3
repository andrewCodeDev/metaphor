//! Scaling Kernels
//!
//! Feature scaling operations: min-max, z-score, normalization (L1, L2),
//! and clamping. Each has rowwise and flat variants.
//! SIMD-vectorized where beneficial.
//!

module metaphor::device::host_device::kernels::scaling;
import std::math;

const usz VEC_WIDTH = 8; // float[<8>] = 256-bit SIMD

// Statistical Helpers

struct MinMaxResult
{
	float min_val;
	float max_val;
}

/// Find min and max values in an array.
fn MinMaxResult min_max_float(float* x, usz len)
{
	assert(len > 0, "min_max_float: empty input");
	float rmin = x[0];
	float rmax = x[0];
	for (usz i = 1; i < len; i++)
	{
		if (x[i] < rmin) { rmin = x[i]; }
		if (x[i] > rmax) { rmax = x[i]; }
	}
	return { .min_val = rmin, .max_val = rmax };
}

/// Compute arithmetic mean.
fn float mean_float(float* x, usz len)
{
	assert(len > 0, "mean_float: empty input");
	float sum = 0;
	for (usz i = 0; i < len; i++) { sum += x[i]; }
	return sum / (float)len;
}

/// Compute population variance.
fn float variance_float(float* x, usz len)
{
	assert(len > 0, "variance_float: empty input");
	float m = mean_float(x, len);
	float sum_sq = 0;
	for (usz i = 0; i < len; i++)
	{
		float diff = x[i] - m;
		sum_sq += diff * diff;
	}
	return sum_sq / (float)len;
}

/// Compute standard deviation (population).
fn float std_dev_float(float* x, usz len)
{
	return math::sqrt(variance_float(x, len));
}

/// Compute L2 norm (Euclidean length).
fn float l2_norm_compute(float* x, usz len)
{
	assert(len > 0, "l2_norm_compute: empty input");
	float sum_sq = 0;
	for (usz i = 0; i < len; i++) { sum_sq += x[i] * x[i]; }
	return math::sqrt(sum_sq);
}

/// Compute L1 norm (sum of absolute values).
fn float l1_norm_compute(float* x, usz len)
{
	assert(len > 0, "l1_norm_compute: empty input");
	float sum_abs = 0;
	for (usz i = 0; i < len; i++) { sum_abs += math::abs(x[i]); }
	return sum_abs;
}

/// Find maximum absolute value.
fn float max_abs_float(float* x, usz len)
{
	assert(len > 0, "max_abs_float: empty input");
	float max_val = math::abs(x[0]);
	for (usz i = 1; i < len; i++)
	{
		float a = math::abs(x[i]);
		if (a > max_val) { max_val = a; }
	}
	return max_val;
}

/// Max absolute scaling: y = x / max(|x|)
fn void max_abs_scale_float(float* x, float* y, usz len)
{
	float ma = max_abs_float(x, len);
	if (ma < 1e-8f)
	{
		for (usz i = 0; i < len; i++) { y[i] = 0; }
		return;
	}
	float inv = 1.0f / ma;
	for (usz i = 0; i < len; i++) { y[i] = x[i] * inv; }
}

// Auto-wrapper Functions (compute stats then delegate)

/// Min-max scale with auto-detected min/max.
fn void min_max_scale_auto_float(float* x, float* y, usz len)
{
	MinMaxResult mm = min_max_float(x, len);
	min_max_scale_float(x, y, len, mm.min_val, mm.max_val);
}

/// Min-max scale to [new_min, new_max] with auto-detected min/max.
fn void min_max_scale_range_auto_float(float* x, float* y, usz len, float new_min, float new_max)
{
	MinMaxResult mm = min_max_float(x, len);
	min_max_scale_range_float(x, y, len, mm.min_val, mm.max_val, new_min, new_max);
}

/// Z-score normalization with auto-computed mean/std.
fn void zscore_auto_float(float* x, float* y, usz len)
{
	float m = mean_float(x, len);
	float s = std_dev_float(x, len);
	if (s < 1e-8f)
	{
		for (usz i = 0; i < len; i++) { y[i] = 0; }
		return;
	}
	zscore_float(x, y, len, m, s, 0);
}

/// Z-score normalization per row with auto-computed stats.
fn void zscore_rows_auto_float(float* x, float* y, usz batch, usz dim)
{
	for (usz b = 0; b < batch; b++)
	{
		zscore_auto_float(x + b * dim, y + b * dim, dim);
	}
}

// Min-Max Scaling

/// Flat min-max scaling: y = (x - min) / (max - min)
fn void min_max_scale_float(float* x, float* y, usz len, float min_val, float max_val)
{
	float range = max_val - min_val;
	if (range == 0) { range = 1.0f; }
	float inv_range = 1.0f / range;

	for (usz i = 0; i < len; i++)
	{
		y[i] = (x[i] - min_val) * inv_range;
	}
}

/// Min-max scaling with target range: y = (x - min) / (max - min) * (new_max - new_min) + new_min
fn void min_max_scale_range_float(
	float* x, float* y, usz len,
	float min_val, float max_val,
	float new_min, float new_max)
{
	float range = max_val - min_val;
	if (range == 0) { range = 1.0f; }
	float new_range = new_max - new_min;
	float scale = new_range / range;

	for (usz i = 0; i < len; i++)
	{
		y[i] = (x[i] - min_val) * scale + new_min;
	}
}

/// Min-max scaling per row: each row [1 x dim] is independently normalized.
fn void min_max_scale_rows_float(float* x, float* y, usz batch, usz dim)
{
	for (usz b = 0; b < batch; b++)
	{
		float* x_row = x + b * dim;
		float* y_row = y + b * dim;

		// Find min and max
		float rmin = x_row[0];
		float rmax = x_row[0];
		for (usz i = 1; i < dim; i++)
		{
			if (x_row[i] < rmin) { rmin = x_row[i]; }
			if (x_row[i] > rmax) { rmax = x_row[i]; }
		}

		min_max_scale_float(x_row, y_row, dim, rmin, rmax);
	}
}

// Z-Score Normalization

/// Flat z-score normalization: y = (x - mean) / (std + epsilon)
fn void zscore_float(float* x, float* y, usz len, float mean, float std_val, float epsilon)
{
	float inv_std = 1.0f / (std_val + epsilon);

	usz i = 0;
	if (len >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] mean_splat = (float[<VEC_WIDTH>])mean;
		float[<VEC_WIDTH>] inv_splat = (float[<VEC_WIDTH>])inv_std;
		while (i + VEC_WIDTH <= len)
		{
			float[<VEC_WIDTH>] x_vec = ((float[<VEC_WIDTH>]*)(x + i))[0];
			((float[<VEC_WIDTH>]*)(y + i))[0] = (x_vec - mean_splat) * inv_splat;
			i += VEC_WIDTH;
		}
	}

	while (i < len)
	{
		y[i] = (x[i] - mean) * inv_std;
		i++;
	}
}

/// Per-row z-score: compute mean and std per row.
fn void zscore_rows_float(float* x, float* y, usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		float* x_row = x + b * dim;
		float* y_row = y + b * dim;

		// Compute mean
		float mean = 0;
		usz i = 0;
		if (dim >= VEC_WIDTH)
		{
			float[<VEC_WIDTH>] sum_vec = (float[<VEC_WIDTH>])0;
			while (i + VEC_WIDTH <= dim)
			{
				float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x_row + i))[0];
				sum_vec += v;
				i += VEC_WIDTH;
			}
			mean = sum_vec.sum();
		}
		while (i < dim) { mean += x_row[i]; i++; }
		mean /= (float)dim;

		// Compute variance
		float variance = 0;
		i = 0;
		while (i < dim)
		{
			float diff = x_row[i] - mean;
			variance += diff * diff;
			i++;
		}
		variance /= (float)dim;
		float std_val = math::sqrt(variance);

		zscore_float(x_row, y_row, dim, mean, std_val, epsilon);
	}
}

// RMS Normalization (Root Mean Square)

/// Flat RMS normalization: y = x / rms(x), where rms = sqrt(mean(x²) + eps).
fn void rms_norm_float(float* x, float* y, usz len, float epsilon)
{
	// Compute mean of squares
	float sum_sq = 0;
	usz i = 0;

	if (len >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] sum_sq_vec = (float[<VEC_WIDTH>])0;
		while (i + VEC_WIDTH <= len)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			sum_sq_vec += v * v;
			i += VEC_WIDTH;
		}
		sum_sq = sum_sq_vec.sum();
	}
	while (i < len)
	{
		sum_sq += x[i] * x[i];
		i++;
	}

	float rms = math::sqrt(sum_sq / (float)len + epsilon);
	float inv_rms = 1.0f / rms;

	// Normalize
	i = 0;
	if (len >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] inv_splat = (float[<VEC_WIDTH>])inv_rms;
		while (i + VEC_WIDTH <= len)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			((float[<VEC_WIDTH>]*)(y + i))[0] = v * inv_splat;
			i += VEC_WIDTH;
		}
	}
	while (i < len)
	{
		y[i] = x[i] * inv_rms;
		i++;
	}
}

/// RMS normalization per row.
fn void rms_norm_rows_float(float* x, float* y, usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		rms_norm_float(x + b * dim, y + b * dim, dim, epsilon);
	}
}

/// RMS normalization per row with learnable weight.
///   y_i = (x_i / rms) * weight_i, where rms = sqrt(mean(x²) + eps)
fn void rms_norm_weighted_rows_float(
	float* x, float* weight, float* y,
	usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		float* x_row = x + b * dim;
		float* y_row = y + b * dim;

		// Compute RMS
		float sum_sq = 0;
		for (usz i = 0; i < dim; i++) { sum_sq += x_row[i] * x_row[i]; }
		float rms = math::sqrt(sum_sq / (float)dim + epsilon);
		float inv_rms = 1.0f / rms;

		// y = (x / rms) * weight
		for (usz i = 0; i < dim; i++)
		{
			y_row[i] = x_row[i] * inv_rms * weight[i];
		}
	}
}

/// RMS norm backward w.r.t. input x, per row.
///   rms = sqrt(mean(x²) + eps)
///   c = sum(grad_y_i * weight_i * x_i) / (dim * rms²)
///   grad_x_j = (grad_y_j * weight_j - x_j * c) / rms
fn void rms_norm_backward_x_rows_float(
	float* grad_y, float* x, float* weight, float* grad_x,
	usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		float* gy_row = grad_y + b * dim;
		float* x_row = x + b * dim;
		float* gx_row = grad_x + b * dim;

		// Compute RMS
		float sum_sq = 0;
		for (usz i = 0; i < dim; i++) { sum_sq += x_row[i] * x_row[i]; }
		float rms_sq = sum_sq / (float)dim + epsilon;
		float rms = math::sqrt(rms_sq);
		float inv_rms = 1.0f / rms;

		// c = sum(grad_y * weight * x) / (dim * rms²)
		float dot = 0;
		for (usz i = 0; i < dim; i++)
		{
			dot += gy_row[i] * weight[i] * x_row[i];
		}
		float c = dot / ((float)dim * rms_sq);

		// grad_x = (grad_y * weight - x * c) / rms
		for (usz i = 0; i < dim; i++)
		{
			gx_row[i] = (gy_row[i] * weight[i] - x_row[i] * c) * inv_rms;
		}
	}
}

/// RMS norm backward w.r.t. weight, accumulated across batch rows.
///   grad_w_j += sum_b(grad_y[b,j] * x[b,j] / rms[b])
/// grad_w must be zero-initialized before calling.
fn void rms_norm_backward_weight_rows_float(
	float* grad_y, float* x, float* grad_w,
	usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		float* gy_row = grad_y + b * dim;
		float* x_row = x + b * dim;

		// Compute RMS for this row
		float sum_sq = 0;
		for (usz i = 0; i < dim; i++) { sum_sq += x_row[i] * x_row[i]; }
		float rms = math::sqrt(sum_sq / (float)dim + epsilon);
		float inv_rms = 1.0f / rms;

		// Accumulate grad_w
		for (usz i = 0; i < dim; i++)
		{
			grad_w[i] += gy_row[i] * x_row[i] * inv_rms;
		}
	}
}

// L2 Normalization

/// Flat L2 normalization: y = x / (||x||_2 + epsilon)
fn void l2_normalize_float(float* x, float* y, usz len, float epsilon)
{
	float sum_sq = 0;
	usz i = 0;

	if (len >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] sum_sq_vec = (float[<VEC_WIDTH>])0;
		while (i + VEC_WIDTH <= len)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			sum_sq_vec += v * v;
			i += VEC_WIDTH;
		}
		sum_sq = sum_sq_vec.sum();
	}
	while (i < len)
	{
		sum_sq += x[i] * x[i];
		i++;
	}

	float norm = math::sqrt(sum_sq) + epsilon;
	float inv_norm = 1.0f / norm;

	i = 0;
	if (len >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] inv_splat = (float[<VEC_WIDTH>])inv_norm;
		while (i + VEC_WIDTH <= len)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			((float[<VEC_WIDTH>]*)(y + i))[0] = v * inv_splat;
			i += VEC_WIDTH;
		}
	}
	while (i < len)
	{
		y[i] = x[i] * inv_norm;
		i++;
	}
}

/// L2 normalization per row.
fn void l2_normalize_rows_float(float* x, float* y, usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		l2_normalize_float(x + b * dim, y + b * dim, dim, epsilon);
	}
}

// L1 Normalization

/// Flat L1 normalization: y = x / (||x||_1 + epsilon)
fn void l1_normalize_float(float* x, float* y, usz len, float epsilon)
{
	float sum_abs = 0;
	for (usz i = 0; i < len; i++)
	{
		sum_abs += math::abs(x[i]);
	}

	float inv_norm = 1.0f / (sum_abs + epsilon);

	for (usz i = 0; i < len; i++)
	{
		y[i] = x[i] * inv_norm;
	}
}

/// L1 normalization per row.
fn void l1_normalize_rows_float(float* x, float* y, usz batch, usz dim, float epsilon)
{
	for (usz b = 0; b < batch; b++)
	{
		l1_normalize_float(x + b * dim, y + b * dim, dim, epsilon);
	}
}

// Clamping

/// Element-wise clamp: y = clamp(x, min, max)
fn void clamp_float(float* x, float* y, usz len, float min_val, float max_val)
{
	for (usz i = 0; i < len; i++)
	{
		float v = x[i];
		if (v < min_val) { v = min_val; }
		if (v > max_val) { v = max_val; }
		y[i] = v;
	}
}

/// Clamp per row.
fn void clamp_rows_float(
	float* x, float* y, usz batch, usz dim,
	float min_val, float max_val)
{
	for (usz b = 0; b < batch; b++)
	{
		clamp_float(x + b * dim, y + b * dim, dim, min_val, max_val);
	}
}
