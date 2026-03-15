/*
 * nn/smax_vec.cu - Vector softmax forward and backward
 */

#ifndef __NN_SMAX_VEC_H__
#define __NN_SMAX_VEC_H__

#include "utils.cu"

extern "C" void cuda_nn_smax_vec_fwd(
    Dtype id,
    CudnnHandle w,
    const void* x,
    void* y,
    len_t n,
    SoftmaxType op
) {
    const int _n = static_cast<int>(n);

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnn_dtype(id), 1, 1, 1, _n);

    switch (id) {
        case DTYPE_F32: {
            const f32 alpha = 1.0f;
            const f32 beta = 0.0f;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                cast_cudnn(w),
                cudnn_softmax_algo(op),
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, x,
                &beta, desc, y
            ));
            cudnnDestroyTensorDescriptor(desc);
            return;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 0.0;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                cast_cudnn(w),
                cudnn_softmax_algo(op),
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, x,
                &beta, desc, y
            ));
            cudnnDestroyTensorDescriptor(desc);
            return;
        }
        default:
            cudnnDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_vec_fwd");
    }
}

extern "C" void cuda_nn_smax_vec_bwd(
    Dtype id,
    CudnnHandle w,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t n
) {
    const int _n = static_cast<int>(n);

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnn_dtype(id), 1, 1, 1, _n);

    switch (id) {
        case DTYPE_F32: {
            const f32 alpha = 1.0f;
            const f32 beta = 1.0f;
            CUDNN_ASSERT(cudnnSoftmaxBackward(
                cast_cudnn(w),
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd
            ));
            cudnnDestroyTensorDescriptor(desc);
            return;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 1.0;
            CUDNN_ASSERT(cudnnSoftmaxBackward(
                cast_cudnn(w),
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd
            ));
            cudnnDestroyTensorDescriptor(desc);
            return;
        }
        default:
            cudnnDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_vec_bwd");
    }
}

#endif /* __NN_SMAX_VEC_H__ */
