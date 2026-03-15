/*
 * nn/smax_2D_row.cu - 2D row-wise softmax forward and backward
 */

#ifndef __NN_SMAX_2D_ROW_H__
#define __NN_SMAX_2D_ROW_H__

#include "utils.cu"

/*
 * NOTE: For F16/BF16 softmax, cuDNN uses FP32 alpha/beta scaling factors.
 */
static void smax_2d_row_fwd_impl(
    Dtype id,
    CudnnHandle w,
    SoftmaxType smax_type,
    const void* x,
    void* y,
    len_t m,
    len_t n
) {
    const int _m = static_cast<int>(m);
    const int _n = static_cast<int>(n);

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnn_dtype(id), _m, 1, 1, _n);

    const cudnnSoftmaxAlgorithm_t algo = cudnn_softmax_algo(smax_type);

    switch (id) {
        case DTYPE_F32:
        case DTYPE_F16:
        case DTYPE_BF16: {
            /* F16/BF16 use FP32 alpha/beta like F32 */
            const f32 alpha = 1.0f;
            const f32 beta = 0.0f;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                cast_cudnn(w),
                algo,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, x,
                &beta, desc, y
            ));
            break;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 0.0;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                cast_cudnn(w),
                algo,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, x,
                &beta, desc, y
            ));
            break;
        }
        default:
            cudnnDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_2d_row_fwd");
    }

    cudnnDestroyTensorDescriptor(desc);
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_nn_smax_2d_row_fwd(
    Dtype id,
    CudnnHandle w,
    const void* x,
    void* y,
    len_t m,
    len_t n
) {
    smax_2d_row_fwd_impl(id, w, SMAX_MAX, x, y, m, n);
}

extern "C" void cuda_nn_log_smax_2d_row_fwd(
    Dtype id,
    CudnnHandle w,
    const void* x,
    void* y,
    len_t m,
    len_t n
) {
    smax_2d_row_fwd_impl(id, w, SMAX_LOG, x, y, m, n);
}

static void smax_2d_row_bwd_impl(
    Dtype id,
    CudnnHandle w,
    SoftmaxType smax_type,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t m,
    len_t n
) {
    const int _m = static_cast<int>(m);
    const int _n = static_cast<int>(n);

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnn_dtype(id), _m, 1, 1, _n);

    const cudnnSoftmaxAlgorithm_t algo = cudnn_softmax_algo(smax_type);

    switch (id) {
        case DTYPE_F32:
        case DTYPE_F16:
        case DTYPE_BF16: {
            /* F16/BF16 use FP32 alpha/beta like F32 */
            const f32 alpha = 1.0f;
            const f32 beta = 0.0f;
            CUDNN_ASSERT(cudnnSoftmaxBackward(
                cast_cudnn(w),
                algo,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd
            ));
            break;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 0.0;
            CUDNN_ASSERT(cudnnSoftmaxBackward(
                cast_cudnn(w),
                algo,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd
            ));
            break;
        }
        default:
            cudnnDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_2d_row_bwd");
    }

    cudnnDestroyTensorDescriptor(desc);
    CUDA_ASSERT(cudaPeekAtLastError());
}

extern "C" void cuda_nn_smax_2d_row_bwd(
    Dtype id,
    CudnnHandle w,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t m,
    len_t n
) {
    smax_2d_row_bwd_impl(id, w, SMAX_MAX, y_val, y_grd, x_grd, m, n);
}

extern "C" void cuda_nn_log_smax_2d_row_bwd(
    Dtype id,
    CudnnHandle w,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t m,
    len_t n
) {
    smax_2d_row_bwd_impl(id, w, SMAX_LOG, y_val, y_grd, x_grd, m, n);
}

#endif /* __NN_SMAX_2D_ROW_H__ */
