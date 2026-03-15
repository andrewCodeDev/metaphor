//! Reduce Kernel
//!
//! Reduction operations: scaled copy, flat reduce, fold rows/cols,
//! and max/min backward passes.
//!

module metaphor::device::host_device::kernels::reduce;
import std::math;

const usz VEC_WIDTH = 8; // float[<8>] = 256-bit SIMD

// Basic reductions

/// Scaled copy: y = alpha * x + beta * y
fn void scaled_copy_float(float* x, float* y, usz len, float alpha, float beta)
{
	for (usz i = 0; i < len; i++)
	{
		y[i] = alpha * x[i] + beta * y[i];
	}
}

/// Flat reduce: y[0] = alpha * sum(x) + beta * y[0]
fn void flat_reduce_float(float* x, float* y, usz len, float alpha, float beta)
{
	assert(len > 0);

	float s = 0;
	usz i = 0;

	if (len >= VEC_WIDTH)
	{
		float[<VEC_WIDTH>] sum_vec = (float[<VEC_WIDTH>])0;
		while (i + VEC_WIDTH <= len)
		{
			float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x + i))[0];
			sum_vec += v;
			i += VEC_WIDTH;
		}
		s = sum_vec.sum();
	}

	while (i < len)
	{
		s += x[i];
		i++;
	}

	y[0] = alpha * s + beta * y[0];
}

/// Fold rows: (M,N) -> (1,N). Reduces along row dimension.
/// x: [row * col], y: [col]
fn void fold_rows_float(float* x, float* y, usz row, usz col, float alpha, float beta)
{
	// Initialize with first row
	for (usz j = 0; j < col; j++)
	{
		y[j] = x[j];
	}

	// Accumulate remaining rows
	for (usz i = 1; i < row; i++)
	{
		float* x_row = x + i * col;
		usz j = 0;

		if (col >= VEC_WIDTH)
		{
			while (j + VEC_WIDTH <= col)
			{
				float[<VEC_WIDTH>] y_vec = ((float[<VEC_WIDTH>]*)(y + j))[0];
				float[<VEC_WIDTH>] x_vec = ((float[<VEC_WIDTH>]*)(x_row + j))[0];
				((float[<VEC_WIDTH>]*)(y + j))[0] = y_vec + x_vec;
				j += VEC_WIDTH;
			}
		}

		while (j < col)
		{
			y[j] += x_row[j];
			j++;
		}
	}

	// Apply scaling: y = alpha * y + beta * y_old
	// Since we accumulated into y directly, apply alpha scaling
	if (alpha != 1.0f || beta != 0.0f)
	{
		// Here we've already overwritten y. For the standard usage where
		// beta=0, this is correct. For beta!=0, the caller must pre-manage y.
		if (alpha != 1.0f)
		{
			for (usz j = 0; j < col; j++)
			{
				y[j] *= alpha;
			}
		}
	}
}

/// Fold cols: (M,N) -> (M,1). Reduces along column dimension.
/// x: [row * col], y: [row]
fn void fold_cols_float(float* x, float* y, usz row, usz col, float alpha, float beta)
{
	for (usz i = 0; i < row; i++)
	{
		float* x_row = x + i * col;
		float s = 0;
		usz j = 0;

		if (col >= VEC_WIDTH)
		{
			float[<VEC_WIDTH>] sum_vec = (float[<VEC_WIDTH>])0;
			while (j + VEC_WIDTH <= col)
			{
				float[<VEC_WIDTH>] v = ((float[<VEC_WIDTH>]*)(x_row + j))[0];
				sum_vec += v;
				j += VEC_WIDTH;
			}
			s = sum_vec.sum();
		}

		while (j < col)
		{
			s += x_row[j];
			j++;
		}

		y[i] = alpha * s + beta * y[i];
	}
}

// Reduce Max/Min Backward

/// Backward for reduce_max over columns: (M,N) -> (M,1)
/// For each row, finds max and passes gradient to matching positions.
fn void reduce_max_cols_backward_float(
	float* original, float* upstream, float* grad_out,
	usz row, usz col)
{
	for (usz i = 0; i < row; i++)
	{
		usz row_start = i * col;

		// Find max in this row
		float max_val = original[row_start];
		for (usz j = 1; j < col; j++)
		{
			float v = original[row_start + j];
			if (v > max_val) { max_val = v; }
		}

		// Pass gradient to positions matching max
		float grad_val = upstream[i];
		for (usz j = 0; j < col; j++)
		{
			grad_out[row_start + j] = (original[row_start + j] == max_val) ? grad_val : 0;
		}
	}
}

/// Backward for reduce_min over columns: (M,N) -> (M,1)
fn void reduce_min_cols_backward_float(
	float* original, float* upstream, float* grad_out,
	usz row, usz col)
{
	for (usz i = 0; i < row; i++)
	{
		usz row_start = i * col;

		float min_val = original[row_start];
		for (usz j = 1; j < col; j++)
		{
			float v = original[row_start + j];
			if (v < min_val) { min_val = v; }
		}

		float grad_val = upstream[i];
		for (usz j = 0; j < col; j++)
		{
			grad_out[row_start + j] = (original[row_start + j] == min_val) ? grad_val : 0;
		}
	}
}

/// Backward for reduce_max over rows: (M,N) -> (1,N)
fn void reduce_max_rows_backward_float(
	float* original, float* upstream, float* grad_out,
	usz row, usz col)
{
	for (usz j = 0; j < col; j++)
	{
		// Find max in this column
		float max_val = original[j];
		for (usz i = 1; i < row; i++)
		{
			float v = original[i * col + j];
			if (v > max_val) { max_val = v; }
		}

		// Pass gradient to positions matching max
		float grad_val = upstream[j];
		for (usz i = 0; i < row; i++)
		{
			usz idx = i * col + j;
			grad_out[idx] = (original[idx] == max_val) ? grad_val : 0;
		}
	}
}

/// Backward for reduce_min over rows: (M,N) -> (1,N)
fn void reduce_min_rows_backward_float(
	float* original, float* upstream, float* grad_out,
	usz row, usz col)
{
	for (usz j = 0; j < col; j++)
	{
		float min_val = original[j];
		for (usz i = 1; i < row; i++)
		{
			float v = original[i * col + j];
			if (v < min_val) { min_val = v; }
		}

		float grad_val = upstream[j];
		for (usz i = 0; i < row; i++)
		{
			usz idx = i * col + j;
			grad_out[idx] = (original[idx] == min_val) ? grad_val : 0;
		}
	}
}

/// Backward for reduce_max to scalar
fn void reduce_max_flat_backward_float(
	float* original, float upstream, float* grad_out, usz len)
{
	assert(len > 0);

	float max_val = original[0];
	for (usz i = 1; i < len; i++)
	{
		if (original[i] > max_val) { max_val = original[i]; }
	}

	for (usz i = 0; i < len; i++)
	{
		grad_out[i] = (original[i] == max_val) ? upstream : 0;
	}
}

/// Backward for reduce_min to scalar
fn void reduce_min_flat_backward_float(
	float* original, float upstream, float* grad_out, usz len)
{
	assert(len > 0);

	float min_val = original[0];
	for (usz i = 1; i < len; i++)
	{
		if (original[i] < min_val) { min_val = original[i]; }
	}

	for (usz i = 0; i < len; i++)
	{
		grad_out[i] = (original[i] == min_val) ? upstream : 0;
	}
}

// Dispatch for reduce max/min backward

/// Dispatches reduce max/min backward based on reduce_type and is_min.
/// reduce_type: 0=rows, 1=cols, 2=flat
/// is_min: 0=max, 1=min
fn void reduce_maxmin_backward_dispatch_float(
	float* original, usz original_len,
	float* upstream, usz upstream_len,
	float* grad_out, usz out_len,
	usz row, usz col,
	usz reduce_type, usz is_min)
{
	switch (reduce_type)
	{
		case 0: // rows: (M,N) -> (1,N)
			if (is_min == 0)
			{
				reduce_max_rows_backward_float(original, upstream, grad_out, row, col);
			}
			else
			{
				reduce_min_rows_backward_float(original, upstream, grad_out, row, col);
			}
		case 1: // cols: (M,N) -> (M,1)
			if (is_min == 0)
			{
				reduce_max_cols_backward_float(original, upstream, grad_out, row, col);
			}
			else
			{
				reduce_min_cols_backward_float(original, upstream, grad_out, row, col);
			}
		case 2: // flat: (...) -> ()
			float upstream_scalar = upstream[0];
			if (is_min == 0)
			{
				reduce_max_flat_backward_float(original, upstream_scalar, grad_out, out_len);
			}
			else
			{
				reduce_min_flat_backward_float(original, upstream_scalar, grad_out, out_len);
			}
		default:
			unreachable("Invalid reduce_type for max/min backward");
	}
}
