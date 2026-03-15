/*
 * nn/conv2d.cpp - 2D convolution using MIOpen
 *
 * Provides convolution forward and backward passes.
 * Uses MIOpen Find API for algorithm selection.
 *
 * MIOpen notes:
 *   - No separate FilterDescriptor — weights use TensorDescriptor
 *   - miopenSet4dTensorDescriptor has no format parameter (NCHW assumed)
 *   - Mandatory Find algorithm step before execution
 *   - BackwardFilter → BackwardWeights
 *   - alpha must be 1.0, beta must be 0.0 for 2D convolutions
 */

#ifndef __NN_CONV2D_H__
#define __NN_CONV2D_H__

#include "utils.cpp"

/* ============================================================================
 * Convolution Parameters Struct
 *
 * Passed from C3 as a packed struct for efficient parameter transfer.
 * ============================================================================
 */

struct ConvParams {
  // Input shape [N, C, H, W]
  int N;
  int C_in;
  int H_in;
  int W_in;

  // Weight shape [C_out, C_in/groups, kH, kW]
  int C_out;
  int kH;
  int kW;

  // Convolution parameters
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;

  // Output shape [N, C_out, H_out, W_out]
  int H_out;
  int W_out;
};

/* ============================================================================
 * Helper: Create tensor descriptor (replaces both tensor and filter descriptors)
 *
 * MIOpen has no separate FilterDescriptor — weights use TensorDescriptor.
 * MIOpen's Set4dTensorDescriptor takes no format parameter (NCHW assumed).
 * ============================================================================
 */

static inline miopenTensorDescriptor_t
create_tensor_desc_4d(miopenDataType_t dtype, int n, int c, int h, int w) {
  miopenTensorDescriptor_t desc;
  MIOPEN_ASSERT(miopenCreateTensorDescriptor(&desc));
  MIOPEN_ASSERT(miopenSet4dTensorDescriptor(desc, dtype, n, c, h, w));
  return desc;
}

/* ============================================================================
 * Helper: Create convolution descriptor
 *
 * MIOpen uses miopenConvolution mode for cross-correlation (not a separate enum).
 * ============================================================================
 */

static inline miopenConvolutionDescriptor_t
create_conv_desc(const ConvParams &p, miopenDataType_t compute_type) {
  miopenConvolutionDescriptor_t desc;
  MIOPEN_ASSERT(miopenCreateConvolutionDescriptor(&desc));
  MIOPEN_ASSERT(miopenInitConvolutionDescriptor(
      desc, miopenConvolution, p.pad_h, p.pad_w,
      p.stride_h, p.stride_w, p.dilation_h, p.dilation_w));
  if (p.groups > 1) {
    MIOPEN_ASSERT(miopenSetConvolutionGroupCount(desc, p.groups));
  }
  (void)compute_type;  /* MIOpen infers compute type from tensor dtypes */
  return desc;
}

/* ============================================================================
 * Forward Convolution
 *
 * MIOpen requires Find algorithm before execution. We use exhaustiveSearch=false
 * for faster heuristic-based selection.
 * ============================================================================
 */

extern "C" void hip_nn_conv2d_fwd(
    Dtype dtype, MiopenHandle miopen_handle, const void *input,
    const void *weight, void *output, void *workspace, size_t workspace_size,
    int N, int C_in, int H_in, int W_in, int C_out, int kH, int kW, int pad_h,
    int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
    int groups, int H_out, int W_out) {
  miopenHandle_t handle = cast_miopen(miopen_handle);
  miopenDataType_t data_type = miopen_dtype(dtype);

  // Create descriptors (MIOpen: no separate filter descriptor)
  miopenTensorDescriptor_t input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  miopenTensorDescriptor_t output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  miopenTensorDescriptor_t filter_desc =
      create_tensor_desc_4d(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  miopenConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Find best algorithm (MIOpen requires this before execution)
  miopenConvAlgoPerf_t perf_results[4];
  int returned_algo_count;
  MIOPEN_ASSERT(miopenFindConvolutionForwardAlgorithm(
      handle, input_desc, input, filter_desc, weight, conv_desc,
      output_desc, output, 4, &returned_algo_count, perf_results,
      workspace, workspace_size, false));

  // Perform convolution (MIOpen: alpha=1.0, beta=0.0 required for 2D)
  const float alpha = 1.0f;
  const float beta = 0.0f;
  MIOPEN_ASSERT(miopenConvolutionForward(
      handle, &alpha, input_desc, input, filter_desc, weight, conv_desc,
      perf_results[0].fwd_algo, &beta, output_desc, output,
      workspace, workspace_size));

  // Cleanup
  miopenDestroyTensorDescriptor(input_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(filter_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);

  HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Backward Data (Input Gradient)
 *
 * Computes dL/dInput from dL/dOutput and weights.
 * ============================================================================
 */

extern "C" void hip_nn_conv2d_bwd_data(
    Dtype dtype, MiopenHandle miopen_handle, const void *weight,
    const void *grad_output, void *grad_input, void *workspace,
    size_t workspace_size, int N, int C_in, int H_in, int W_in, int C_out,
    int kH, int kW, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out) {
  miopenHandle_t handle = cast_miopen(miopen_handle);
  miopenDataType_t data_type = miopen_dtype(dtype);

  // Create descriptors
  miopenTensorDescriptor_t grad_input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  miopenTensorDescriptor_t grad_output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  miopenTensorDescriptor_t filter_desc =
      create_tensor_desc_4d(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  miopenConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Find best algorithm
  miopenConvAlgoPerf_t perf_results[4];
  int returned_algo_count;
  MIOPEN_ASSERT(miopenFindConvolutionBackwardDataAlgorithm(
      handle, grad_output_desc, grad_output, filter_desc, weight, conv_desc,
      grad_input_desc, grad_input, 4, &returned_algo_count, perf_results,
      workspace, workspace_size, false));

  // Perform backward data
  const float alpha = 1.0f;
  const float beta = 0.0f;
  MIOPEN_ASSERT(miopenConvolutionBackwardData(
      handle, &alpha, grad_output_desc, grad_output, filter_desc, weight,
      conv_desc, perf_results[0].bwd_data_algo, &beta, grad_input_desc,
      grad_input, workspace, workspace_size));

  // Cleanup
  miopenDestroyTensorDescriptor(grad_input_desc);
  miopenDestroyTensorDescriptor(grad_output_desc);
  miopenDestroyTensorDescriptor(filter_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);

  HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Backward Weights (Weight Gradient)
 *
 * Computes dL/dWeight from input and dL/dOutput.
 * Note: MIOpen calls this "BackwardWeights" (the weight gradient computation).
 * ============================================================================
 */

extern "C" void hip_nn_conv2d_bwd_filter(
    Dtype dtype, MiopenHandle miopen_handle, const void *input,
    const void *grad_output, void *grad_weight, void *workspace,
    size_t workspace_size, int N, int C_in, int H_in, int W_in, int C_out,
    int kH, int kW, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out) {
  miopenHandle_t handle = cast_miopen(miopen_handle);
  miopenDataType_t data_type = miopen_dtype(dtype);

  // Create descriptors
  miopenTensorDescriptor_t input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  miopenTensorDescriptor_t grad_output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  miopenTensorDescriptor_t grad_weight_desc =
      create_tensor_desc_4d(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  miopenConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Find best algorithm
  miopenConvAlgoPerf_t perf_results[4];
  int returned_algo_count;
  MIOPEN_ASSERT(miopenFindConvolutionBackwardWeightsAlgorithm(
      handle, grad_output_desc, grad_output, input_desc, input, conv_desc,
      grad_weight_desc, grad_weight, 4, &returned_algo_count, perf_results,
      workspace, workspace_size, false));

  // Perform backward weights
  const float alpha = 1.0f;
  const float beta = 0.0f;
  MIOPEN_ASSERT(miopenConvolutionBackwardWeights(
      handle, &alpha, grad_output_desc, grad_output, input_desc, input,
      conv_desc, perf_results[0].bwd_weights_algo, &beta, grad_weight_desc,
      grad_weight, workspace, workspace_size));

  // Cleanup
  miopenDestroyTensorDescriptor(input_desc);
  miopenDestroyTensorDescriptor(grad_output_desc);
  miopenDestroyTensorDescriptor(grad_weight_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);

  HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Get Workspace Size
 *
 * Returns required workspace size for the given convolution parameters.
 * Should be called once during compile time to pre-allocate workspace.
 *
 * MIOpen workspace queries return the max across all algorithms.
 * ============================================================================
 */

extern "C" size_t hip_nn_conv2d_get_workspace_size(
    Dtype dtype, MiopenHandle miopen_handle, int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out) {
  miopenHandle_t handle = cast_miopen(miopen_handle);
  miopenDataType_t data_type = miopen_dtype(dtype);

  // Create descriptors
  miopenTensorDescriptor_t input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  miopenTensorDescriptor_t output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  miopenTensorDescriptor_t filter_desc =
      create_tensor_desc_4d(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  miopenConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Get workspace sizes for all three operations
  size_t fwd_size = 0, bwd_data_size = 0, bwd_weights_size = 0;

  miopenConvolutionForwardGetWorkSpaceSize(
      handle, filter_desc, input_desc, conv_desc, output_desc, &fwd_size);
  miopenConvolutionBackwardDataGetWorkSpaceSize(
      handle, output_desc, filter_desc, conv_desc, input_desc, &bwd_data_size);
  miopenConvolutionBackwardWeightsGetWorkSpaceSize(
      handle, output_desc, input_desc, conv_desc, filter_desc, &bwd_weights_size);

  // Cleanup
  miopenDestroyTensorDescriptor(input_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(filter_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);

  // Return max of all workspace sizes
  size_t max_size = fwd_size;
  if (bwd_data_size > max_size)
    max_size = bwd_data_size;
  if (bwd_weights_size > max_size)
    max_size = bwd_weights_size;

  return max_size;
}

#endif /* __NN_CONV2D_H__ */
