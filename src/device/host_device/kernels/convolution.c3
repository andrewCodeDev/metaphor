//! Convolution Kernels
//!
//! Static 2D convolution implementations for forward, transpose (input grad),
//! and weight gradient passes. All operate on raw float data with explicit
//! dimensions and strides.
//!
//! Dimension packing convention:
//!   [0] N (batch)     [1] C_in    [2] H_in   [3] W_in
//!   [4] C_out         [5] kH      [6] kW
//!   [7] stride_h      [8] stride_w
//!   [9] dilation_h    [10] dilation_w
//!   [11] groups       [12] H_out  [13] W_out
//!

module metaphor::device::host_device::kernels::convolution;

// Dimension index constants
const usz DIM_N = 0;
const usz DIM_C_IN = 1;
const usz DIM_H_IN = 2;
const usz DIM_W_IN = 3;
const usz DIM_C_OUT = 4;
const usz DIM_KH = 5;
const usz DIM_KW = 6;
const usz DIM_STRIDE_H = 7;
const usz DIM_STRIDE_W = 8;
const usz DIM_DILATION_H = 9;
const usz DIM_DILATION_W = 10;
const usz DIM_GROUPS = 11;
const usz DIM_H_OUT = 12;
const usz DIM_W_OUT = 13;
const usz NUM_DIMS = 14;

// Forward 2D Convolution

/// Forward 2D convolution.
/// Input layout: [N, C_in, H, W]
/// Weight layout: [C_out, C_in/groups, kH, kW]
/// Output layout: [N, C_out, H_out, W_out]
fn void conv2d_float(
	float* input,  ulong[4] input_strides,
	float* weight, ulong[4] weight_strides,
	float* output, ulong[4] out_strides,
	ulong* dims,
	long pad_h, long pad_w)
{
	usz nn = (usz)dims[DIM_N];
	usz c_in = (usz)dims[DIM_C_IN];
	usz h_in = (usz)dims[DIM_H_IN];
	usz w_in = (usz)dims[DIM_W_IN];
	usz c_out = (usz)dims[DIM_C_OUT];
	usz kh = (usz)dims[DIM_KH];
	usz kw = (usz)dims[DIM_KW];
	usz stride_h = (usz)dims[DIM_STRIDE_H];
	usz stride_w = (usz)dims[DIM_STRIDE_W];
	usz dilation_h = (usz)dims[DIM_DILATION_H];
	usz dilation_w = (usz)dims[DIM_DILATION_W];
	usz groups = (usz)dims[DIM_GROUPS];
	usz h_out = (usz)dims[DIM_H_OUT];
	usz w_out = (usz)dims[DIM_W_OUT];

	usz c_in_per_group = c_in / groups;
	usz c_out_per_group = c_out / groups;

	for (usz n = 0; n < nn; n++)
	{
		for (usz oc = 0; oc < c_out; oc++)
		{
			usz g = oc / c_out_per_group;
			usz c_in_start = g * c_in_per_group;

			for (usz oh = 0; oh < h_out; oh++)
			{
				for (usz ow = 0; ow < w_out; ow++)
				{
					float acc = 0;

					for (usz ic_rel = 0; ic_rel < c_in_per_group; ic_rel++)
					{
						usz ic = c_in_start + ic_rel;
						for (usz kh_i = 0; kh_i < kh; kh_i++)
						{
							for (usz kw_i = 0; kw_i < kw; kw_i++)
							{
								long ih_signed = (long)(oh * stride_h + kh_i * dilation_h) - pad_h;
								long iw_signed = (long)(ow * stride_w + kw_i * dilation_w) - pad_w;

								if (ih_signed >= 0 && ih_signed < (long)h_in &&
									iw_signed >= 0 && iw_signed < (long)w_in)
								{
									usz ih = (usz)ih_signed;
									usz iw = (usz)iw_signed;
									usz input_idx = n * (usz)input_strides[0] + ic * (usz)input_strides[1] +
										ih * (usz)input_strides[2] + iw * (usz)input_strides[3];
									usz weight_idx = oc * (usz)weight_strides[0] + ic_rel * (usz)weight_strides[1] +
										kh_i * (usz)weight_strides[2] + kw_i * (usz)weight_strides[3];
									acc += input[input_idx] * weight[weight_idx];
								}
							}
						}
					}

					usz out_idx = n * (usz)out_strides[0] + oc * (usz)out_strides[1] +
						oh * (usz)out_strides[2] + ow * (usz)out_strides[3];
					output[out_idx] = acc;
				}
			}
		}
	}
}

// Transposed 2D Convolution (input gradient backward)

/// Computes dL/dInput from dL/dOutput and weights.
fn void conv_transpose2d_float(
	float* grad_out,    ulong[4] grad_out_strides,
	float* weight,      ulong[4] weight_strides,
	float* grad_input,  ulong[4] grad_in_strides,
	ulong* dims,
	long pad_h, long pad_w)
{
	usz nn = (usz)dims[DIM_N];
	usz c_in = (usz)dims[DIM_C_IN];
	usz h_in = (usz)dims[DIM_H_IN];
	usz w_in = (usz)dims[DIM_W_IN];
	usz c_out = (usz)dims[DIM_C_OUT];
	usz kh = (usz)dims[DIM_KH];
	usz kw = (usz)dims[DIM_KW];
	usz stride_h = (usz)dims[DIM_STRIDE_H];
	usz stride_w = (usz)dims[DIM_STRIDE_W];
	usz dilation_h = (usz)dims[DIM_DILATION_H];
	usz dilation_w = (usz)dims[DIM_DILATION_W];
	usz groups = (usz)dims[DIM_GROUPS];
	usz h_out = (usz)dims[DIM_H_OUT];
	usz w_out = (usz)dims[DIM_W_OUT];

	usz c_in_per_group = c_in / groups;
	usz c_out_per_group = c_out / groups;

	for (usz n = 0; n < nn; n++)
	{
		for (usz ic = 0; ic < c_in; ic++)
		{
			usz g = ic / c_in_per_group;
			usz ic_rel = ic - g * c_in_per_group;
			usz c_out_start = g * c_out_per_group;

			for (usz ih = 0; ih < h_in; ih++)
			{
				for (usz iw = 0; iw < w_in; iw++)
				{
					float acc = 0;

					for (usz oc_rel = 0; oc_rel < c_out_per_group; oc_rel++)
					{
						usz oc = c_out_start + oc_rel;
						for (usz kh_i = 0; kh_i < kh; kh_i++)
						{
							for (usz kw_i = 0; kw_i < kw; kw_i++)
							{
								long oh_test = (long)ih + pad_h - (long)(kh_i * dilation_h);
								long ow_test = (long)iw + pad_w - (long)(kw_i * dilation_w);

								if (oh_test >= 0 && (oh_test % (long)stride_h) == 0 &&
									ow_test >= 0 && (ow_test % (long)stride_w) == 0)
								{
									usz oh = (usz)(oh_test / (long)stride_h);
									usz ow = (usz)(ow_test / (long)stride_w);
									if (oh < h_out && ow < w_out)
									{
										usz go_idx = n * (usz)grad_out_strides[0] + oc * (usz)grad_out_strides[1] +
											oh * (usz)grad_out_strides[2] + ow * (usz)grad_out_strides[3];
										usz w_idx = oc * (usz)weight_strides[0] + ic_rel * (usz)weight_strides[1] +
											kh_i * (usz)weight_strides[2] + kw_i * (usz)weight_strides[3];
										acc += grad_out[go_idx] * weight[w_idx];
									}
								}
							}
						}
					}

					usz gi_idx = n * (usz)grad_in_strides[0] + ic * (usz)grad_in_strides[1] +
						ih * (usz)grad_in_strides[2] + iw * (usz)grad_in_strides[3];
					grad_input[gi_idx] = acc;
				}
			}
		}
	}
}

// Weight Gradient

/// Computes dL/dWeight from input and dL/dOutput.
fn void conv_weight_grad_float(
	float* input,      ulong[4] input_strides,
	float* grad_out,   ulong[4] grad_out_strides,
	float* grad_w,     ulong[4] grad_w_strides,
	ulong* dims,
	long pad_h, long pad_w)
{
	usz nn = (usz)dims[DIM_N];
	usz c_in = (usz)dims[DIM_C_IN];
	usz h_in = (usz)dims[DIM_H_IN];
	usz w_in = (usz)dims[DIM_W_IN];
	usz c_out = (usz)dims[DIM_C_OUT];
	usz kh = (usz)dims[DIM_KH];
	usz kw = (usz)dims[DIM_KW];
	usz stride_h = (usz)dims[DIM_STRIDE_H];
	usz stride_w = (usz)dims[DIM_STRIDE_W];
	usz dilation_h = (usz)dims[DIM_DILATION_H];
	usz dilation_w = (usz)dims[DIM_DILATION_W];
	usz groups = (usz)dims[DIM_GROUPS];
	usz h_out = (usz)dims[DIM_H_OUT];
	usz w_out = (usz)dims[DIM_W_OUT];

	usz c_in_per_group = c_in / groups;
	usz c_out_per_group = c_out / groups;

	for (usz oc = 0; oc < c_out; oc++)
	{
		usz g = oc / c_out_per_group;
		usz c_in_start = g * c_in_per_group;

		for (usz ic_rel = 0; ic_rel < c_in_per_group; ic_rel++)
		{
			usz ic = c_in_start + ic_rel;
			for (usz kh_i = 0; kh_i < kh; kh_i++)
			{
				for (usz kw_i = 0; kw_i < kw; kw_i++)
				{
					float acc = 0;

					for (usz n = 0; n < nn; n++)
					{
						for (usz oh = 0; oh < h_out; oh++)
						{
							for (usz ow = 0; ow < w_out; ow++)
							{
								long ih_signed = (long)(oh * stride_h + kh_i * dilation_h) - pad_h;
								long iw_signed = (long)(ow * stride_w + kw_i * dilation_w) - pad_w;

								if (ih_signed >= 0 && ih_signed < (long)h_in &&
									iw_signed >= 0 && iw_signed < (long)w_in)
								{
									usz ih = (usz)ih_signed;
									usz iw = (usz)iw_signed;
									usz i_idx = n * (usz)input_strides[0] + ic * (usz)input_strides[1] +
										ih * (usz)input_strides[2] + iw * (usz)input_strides[3];
									usz go_idx = n * (usz)grad_out_strides[0] + oc * (usz)grad_out_strides[1] +
										oh * (usz)grad_out_strides[2] + ow * (usz)grad_out_strides[3];
									acc += input[i_idx] * grad_out[go_idx];
								}
							}
						}
					}

					usz gw_idx = oc * (usz)grad_w_strides[0] + ic_rel * (usz)grad_w_strides[1] +
						kh_i * (usz)grad_w_strides[2] + kw_i * (usz)grad_w_strides[3];
					grad_w[gw_idx] = acc;
				}
			}
		}
	}
}
