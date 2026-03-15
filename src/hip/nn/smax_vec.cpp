/*
 * nn/smax_vec.cpp - Vector softmax forward and backward (MIOpen)
 *
 * Uses miopenSoftmaxForward_V2/Backward_V2 for algorithm+mode control.
 * MIOpen notes:
 *   - No format param in Set4dTensorDescriptor (NCHW assumed)
 *   - algo+mode go at END of arg list (not after handle)
 */

#ifndef __NN_SMAX_VEC_H__
#define __NN_SMAX_VEC_H__

#include "utils.cpp"

extern "C" void hip_nn_smax_vec_fwd(
    Dtype id,
    MiopenHandle w,
    const void* x,
    void* y,
    len_t n,
    SoftmaxType op
) {
    const int _n = static_cast<int>(n);

    miopenTensorDescriptor_t desc;
    miopenCreateTensorDescriptor(&desc);
    miopenSet4dTensorDescriptor(desc, miopen_dtype(id), 1, 1, 1, _n);

    switch (id) {
        case DTYPE_F32: {
            const f32 alpha = 1.0f;
            const f32 beta = 0.0f;
            MIOPEN_ASSERT(miopenSoftmaxForward_V2(
                cast_miopen(w),
                &alpha, desc, x,
                &beta, desc, y,
                miopen_softmax_algo(op),
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            miopenDestroyTensorDescriptor(desc);
            return;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 0.0;
            MIOPEN_ASSERT(miopenSoftmaxForward_V2(
                cast_miopen(w),
                &alpha, desc, x,
                &beta, desc, y,
                miopen_softmax_algo(op),
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            miopenDestroyTensorDescriptor(desc);
            return;
        }
        default:
            miopenDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_vec_fwd");
    }
}

extern "C" void hip_nn_smax_vec_bwd(
    Dtype id,
    MiopenHandle w,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t n
) {
    const int _n = static_cast<int>(n);

    miopenTensorDescriptor_t desc;
    miopenCreateTensorDescriptor(&desc);
    miopenSet4dTensorDescriptor(desc, miopen_dtype(id), 1, 1, 1, _n);

    switch (id) {
        case DTYPE_F32: {
            const f32 alpha = 1.0f;
            const f32 beta = 1.0f;
            MIOPEN_ASSERT(miopenSoftmaxBackward_V2(
                cast_miopen(w),
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd,
                MIOPEN_SOFTMAX_ACCURATE,
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            miopenDestroyTensorDescriptor(desc);
            return;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 1.0;
            MIOPEN_ASSERT(miopenSoftmaxBackward_V2(
                cast_miopen(w),
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd,
                MIOPEN_SOFTMAX_ACCURATE,
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            miopenDestroyTensorDescriptor(desc);
            return;
        }
        default:
            miopenDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_vec_bwd");
    }
}

#endif /* __NN_SMAX_VEC_H__ */
