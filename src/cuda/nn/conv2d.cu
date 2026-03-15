/*
 * nn/conv2d.cu - 2D convolution using cuDNN
 *
 * Provides convolution forward and backward passes.
 * Graph-capture safe: uses pre-computed workspace sizes.
 */

#ifndef __NN_CONV2D_H__
#define __NN_CONV2D_H__

#include "utils.cu"

/* ============================================================================
 * Convolution Parameters Struct
 *
 * Passed from Zig as a packed struct for efficient parameter transfer.
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
 * Helper: Create tensor descriptor
 * ============================================================================
 */

static inline cudnnTensorDescriptor_t
create_tensor_desc_4d(cudnnDataType_t dtype, int n, int c, int h, int w) {
  cudnnTensorDescriptor_t desc;
  CUDNN_ASSERT(cudnnCreateTensorDescriptor(&desc));
  CUDNN_ASSERT(
      cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dtype, n, c, h, w));
  return desc;
}

/* ============================================================================
 * Helper: Create filter descriptor
 * ============================================================================
 */

static inline cudnnFilterDescriptor_t
create_filter_desc(cudnnDataType_t dtype, int k, int c, int h, int w) {
  cudnnFilterDescriptor_t desc;
  CUDNN_ASSERT(cudnnCreateFilterDescriptor(&desc));
  CUDNN_ASSERT(
      cudnnSetFilter4dDescriptor(desc, dtype, CUDNN_TENSOR_NCHW, k, c, h, w));
  return desc;
}

/* ============================================================================
 * Helper: Create convolution descriptor
 * ============================================================================
 */

static inline cudnnConvolutionDescriptor_t
create_conv_desc(const ConvParams &p, cudnnDataType_t compute_type) {
  cudnnConvolutionDescriptor_t desc;
  CUDNN_ASSERT(cudnnCreateConvolutionDescriptor(&desc));
  CUDNN_ASSERT(cudnnSetConvolution2dDescriptor(
      desc, p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h,
      p.dilation_w, CUDNN_CROSS_CORRELATION, compute_type));
  CUDNN_ASSERT(cudnnSetConvolutionGroupCount(desc, p.groups));
  return desc;
}

/* ============================================================================
 * Forward Convolution
 * ============================================================================
 */

extern "C" void cuda_nn_conv2d_fwd(
    Dtype dtype, CudnnHandle cudnn_handle, const void *input,
    const void *weight, void *output, void *workspace, size_t workspace_size,
    // Params packed as: N, C_in, H_in, W_in, C_out, kH, kW, pad_h, pad_w,
    //                   stride_h, stride_w, dilation_h, dilation_w, groups,
    //                   H_out, W_out
    int N, int C_in, int H_in, int W_in, int C_out, int kH, int kW, int pad_h,
    int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
    int groups, int H_out, int W_out) {
  cudnnHandle_t handle = cast_cudnn(cudnn_handle);
  cudnnDataType_t data_type = cudnn_dtype(dtype);

  // Create descriptors
  cudnnTensorDescriptor_t input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  cudnnTensorDescriptor_t output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  cudnnFilterDescriptor_t filter_desc =
      create_filter_desc(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Choose algorithm (use implicit GEMM for graph capture compatibility)
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  // Perform convolution
  if (dtype == DTYPE_F32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_ASSERT(cudnnConvolutionForward(
        handle, &alpha, input_desc, input, filter_desc, weight, conv_desc, algo,
        workspace, workspace_size, &beta, output_desc, output));
  } else {
    const double alpha = 1.0;
    const double beta = 0.0;
    CUDNN_ASSERT(cudnnConvolutionForward(
        handle, &alpha, input_desc, input, filter_desc, weight, conv_desc, algo,
        workspace, workspace_size, &beta, output_desc, output));
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);

  CUDA_ASSERT(cudaPeekAtLastError());
}

/* ============================================================================
 * Backward Data (Input Gradient)
 *
 * Computes dL/dInput from dL/dOutput and weights.
 * ============================================================================
 */

extern "C" void cuda_nn_conv2d_bwd_data(
    Dtype dtype, CudnnHandle cudnn_handle, const void *weight,
    const void *grad_output, void *grad_input, void *workspace,
    size_t workspace_size, int N, int C_in, int H_in, int W_in, int C_out,
    int kH, int kW, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out) {
  cudnnHandle_t handle = cast_cudnn(cudnn_handle);
  cudnnDataType_t data_type = cudnn_dtype(dtype);

  // Create descriptors
  cudnnTensorDescriptor_t grad_input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  cudnnTensorDescriptor_t grad_output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  cudnnFilterDescriptor_t filter_desc =
      create_filter_desc(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Choose algorithm
  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  // Perform backward data
  if (dtype == DTYPE_F32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_ASSERT(cudnnConvolutionBackwardData(
        handle, &alpha, filter_desc, weight, grad_output_desc, grad_output,
        conv_desc, algo, workspace, workspace_size, &beta, grad_input_desc,
        grad_input));
  } else {
    const double alpha = 1.0;
    const double beta = 0.0;
    CUDNN_ASSERT(cudnnConvolutionBackwardData(
        handle, &alpha, filter_desc, weight, grad_output_desc, grad_output,
        conv_desc, algo, workspace, workspace_size, &beta, grad_input_desc,
        grad_input));
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(grad_input_desc);
  cudnnDestroyTensorDescriptor(grad_output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);

  CUDA_ASSERT(cudaPeekAtLastError());
}

/* ============================================================================
 * Backward Filter (Weight Gradient)
 *
 * Computes dL/dWeight from input and dL/dOutput.
 * ============================================================================
 */

extern "C" void cuda_nn_conv2d_bwd_filter(
    Dtype dtype, CudnnHandle cudnn_handle, const void *input,
    const void *grad_output, void *grad_weight, void *workspace,
    size_t workspace_size, int N, int C_in, int H_in, int W_in, int C_out,
    int kH, int kW, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out) {
  cudnnHandle_t handle = cast_cudnn(cudnn_handle);
  cudnnDataType_t data_type = cudnn_dtype(dtype);

  // Create descriptors
  cudnnTensorDescriptor_t input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  cudnnTensorDescriptor_t grad_output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  cudnnFilterDescriptor_t grad_filter_desc =
      create_filter_desc(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Choose algorithm
  cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  // Perform backward filter
  if (dtype == DTYPE_F32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_ASSERT(cudnnConvolutionBackwardFilter(
        handle, &alpha, input_desc, input, grad_output_desc, grad_output,
        conv_desc, algo, workspace, workspace_size, &beta, grad_filter_desc,
        grad_weight));
  } else {
    const double alpha = 1.0;
    const double beta = 0.0;
    CUDNN_ASSERT(cudnnConvolutionBackwardFilter(
        handle, &alpha, input_desc, input, grad_output_desc, grad_output,
        conv_desc, algo, workspace, workspace_size, &beta, grad_filter_desc,
        grad_weight));
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(grad_output_desc);
  cudnnDestroyFilterDescriptor(grad_filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);

  CUDA_ASSERT(cudaPeekAtLastError());
}

/* ============================================================================
 * Get Workspace Size
 *
 * Returns required workspace size for the given convolution parameters.
 * Should be called once during compile time to pre-allocate workspace.
 * ============================================================================
 */

extern "C" size_t cuda_nn_conv2d_get_workspace_size(
    Dtype dtype, CudnnHandle cudnn_handle, int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out) {
  cudnnHandle_t handle = cast_cudnn(cudnn_handle);
  cudnnDataType_t data_type = cudnn_dtype(dtype);

  // Create descriptors
  cudnnTensorDescriptor_t input_desc =
      create_tensor_desc_4d(data_type, N, C_in, H_in, W_in);
  cudnnTensorDescriptor_t output_desc =
      create_tensor_desc_4d(data_type, N, C_out, H_out, W_out);
  cudnnFilterDescriptor_t filter_desc =
      create_filter_desc(data_type, C_out, C_in / groups, kH, kW);

  ConvParams p = {N,          C_in,   H_in,  W_in,     C_out,    kH,
                  kW,         pad_h,  pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, groups, H_out, W_out};
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(p, data_type);

  // Get workspace sizes for all three operations
  size_t fwd_size = 0, bwd_data_size = 0, bwd_filter_size = 0;

  cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc,
                                          conv_desc, output_desc, fwd_algo,
                                          &fwd_size);
  cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, output_desc,
                                               conv_desc, input_desc,
                                               bwd_data_algo, &bwd_data_size);
  cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc, output_desc, conv_desc, filter_desc, bwd_filter_algo,
      &bwd_filter_size);

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);

  // Return max of all workspace sizes
  size_t max_size = fwd_size;
  if (bwd_data_size > max_size)
    max_size = bwd_data_size;
  if (bwd_filter_size > max_size)
    max_size = bwd_filter_size;

  return max_size;
}

#endif /* __NN_CONV2D_H__ */
