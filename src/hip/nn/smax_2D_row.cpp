/*
 * nn/smax_2D_row.cpp - 2D row-wise softmax forward and backward
 *
 * Uses MIOpen for f32/f64 (where it works reliably).
 * Uses custom kernels for f16/bf16 (MIOpen returns miopenStatusNotImplemented
 * for bf16 on RDNA3/gfx1100).
 *
 * Custom kernel attribution: see softmax_kernels.h (BSD-3-Clause, PyTorch).
 */

#ifndef __NN_SMAX_2D_ROW_H__
#define __NN_SMAX_2D_ROW_H__

#include "utils.cpp"
#include "softmax_kernels.h"

/*
 * NOTE: For F16/BF16 softmax, MIOpen uses FP32 alpha/beta scaling factors.
 */
static void smax_2d_row_fwd_impl(
    Dtype id,
    MiopenHandle w,
    SoftmaxType smax_type,
    const void* x,
    void* y,
    len_t m,
    len_t n
) {
    const bool is_log = (smax_type == SMAX_LOG);

    /* F16/BF16: custom kernels (MIOpen returns NotImplemented on RDNA3) */
    if (id == DTYPE_F16) {
        hipStream_t stream = stream_from_miopen(w);
        custom_softmax_forward<f16>(
            static_cast<const f16*>(x), static_cast<f16*>(y),
            static_cast<int>(m), static_cast<int>(n), is_log, stream);
        HIP_ASSERT(hipPeekAtLastError());
        return;
    }
    if (id == DTYPE_BF16) {
        hipStream_t stream = stream_from_miopen(w);
        custom_softmax_forward<bf16>(
            static_cast<const bf16*>(x), static_cast<bf16*>(y),
            static_cast<int>(m), static_cast<int>(n), is_log, stream);
        HIP_ASSERT(hipPeekAtLastError());
        return;
    }

    /* F32/F64: MIOpen path */
    const int _m = static_cast<int>(m);
    const int _n = static_cast<int>(n);

    miopenTensorDescriptor_t desc;
    miopenCreateTensorDescriptor(&desc);
    miopenSet4dTensorDescriptor(desc, miopen_dtype(id), _m, 1, 1, _n);

    const miopenSoftmaxAlgorithm_t algo = miopen_softmax_algo(smax_type);

    switch (id) {
        case DTYPE_F32: {
            const f32 alpha = 1.0f;
            const f32 beta = 0.0f;
            MIOPEN_ASSERT(miopenSoftmaxForward_V2(
                cast_miopen(w),
                &alpha, desc, x,
                &beta, desc, y,
                algo,
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            break;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 0.0;
            MIOPEN_ASSERT(miopenSoftmaxForward_V2(
                cast_miopen(w),
                &alpha, desc, x,
                &beta, desc, y,
                algo,
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            break;
        }
        default:
            miopenDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_2d_row_fwd");
    }

    miopenDestroyTensorDescriptor(desc);
    HIP_ASSERT(hipPeekAtLastError());
}

extern "C" void hip_nn_smax_2d_row_fwd(
    Dtype id,
    MiopenHandle w,
    const void* x,
    void* y,
    len_t m,
    len_t n
) {
    smax_2d_row_fwd_impl(id, w, SMAX_MAX, x, y, m, n);
}

extern "C" void hip_nn_log_smax_2d_row_fwd(
    Dtype id,
    MiopenHandle w,
    const void* x,
    void* y,
    len_t m,
    len_t n
) {
    smax_2d_row_fwd_impl(id, w, SMAX_LOG, x, y, m, n);
}

static void smax_2d_row_bwd_impl(
    Dtype id,
    MiopenHandle w,
    SoftmaxType smax_type,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t m,
    len_t n
) {
    const bool is_log = (smax_type == SMAX_LOG);

    /* F16/BF16: custom kernels */
    if (id == DTYPE_F16) {
        hipStream_t stream = stream_from_miopen(w);
        custom_softmax_backward<f16>(
            static_cast<const f16*>(y_val), static_cast<const f16*>(y_grd),
            static_cast<f16*>(x_grd),
            static_cast<int>(m), static_cast<int>(n), is_log, stream);
        HIP_ASSERT(hipPeekAtLastError());
        return;
    }
    if (id == DTYPE_BF16) {
        hipStream_t stream = stream_from_miopen(w);
        custom_softmax_backward<bf16>(
            static_cast<const bf16*>(y_val), static_cast<const bf16*>(y_grd),
            static_cast<bf16*>(x_grd),
            static_cast<int>(m), static_cast<int>(n), is_log, stream);
        HIP_ASSERT(hipPeekAtLastError());
        return;
    }

    /* F32/F64: MIOpen path */
    const int _m = static_cast<int>(m);
    const int _n = static_cast<int>(n);

    miopenTensorDescriptor_t desc;
    miopenCreateTensorDescriptor(&desc);
    miopenSet4dTensorDescriptor(desc, miopen_dtype(id), _m, 1, 1, _n);

    const miopenSoftmaxAlgorithm_t algo = miopen_softmax_algo(smax_type);

    switch (id) {
        case DTYPE_F32: {
            const f32 alpha = 1.0f;
            const f32 beta = 0.0f;
            MIOPEN_ASSERT(miopenSoftmaxBackward_V2(
                cast_miopen(w),
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd,
                algo,
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            break;
        }
        case DTYPE_F64: {
            const f64 alpha = 1.0;
            const f64 beta = 0.0;
            MIOPEN_ASSERT(miopenSoftmaxBackward_V2(
                cast_miopen(w),
                &alpha, desc, y_val, desc, y_grd,
                &beta, desc, x_grd,
                algo,
                MIOPEN_SOFTMAX_MODE_INSTANCE
            ));
            break;
        }
        default:
            miopenDestroyTensorDescriptor(desc);
            SYSTEM_EXIT("Unsupported dtype for smax_2d_row_bwd");
    }

    miopenDestroyTensorDescriptor(desc);
    HIP_ASSERT(hipPeekAtLastError());
}

extern "C" void hip_nn_smax_2d_row_bwd(
    Dtype id,
    MiopenHandle w,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t m,
    len_t n
) {
    smax_2d_row_bwd_impl(id, w, SMAX_MAX, y_val, y_grd, x_grd, m, n);
}

extern "C" void hip_nn_log_smax_2d_row_bwd(
    Dtype id,
    MiopenHandle w,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t m,
    len_t n
) {
    smax_2d_row_bwd_impl(id, w, SMAX_LOG, y_val, y_grd, x_grd, m, n);
}

#endif /* __NN_SMAX_2D_ROW_H__ */
