/*
 * nn/nll_loss_1d_index.cpp - Negative log likelihood loss with index targets
 */

#ifndef __NN_NLL_INDEX_H__
#define __NN_NLL_INDEX_H__

#include "utils.cpp"
#include "smax_vec.cpp"

template <typename T>
__global__ void nll_loss_1d_index_reduce_kernel(
    const T* src,
    len_t trg,
    T* dst,
    len_t n,
    ReductionType reduxop
) {
    if (reduxop == REDUX_SUM) {
        *dst = -src[trg];
    } else {
        *dst = -src[trg] / T(n);
    }
}

template <typename T>
static void nll_loss_1d_index_reduce(
    MiopenHandle w,
    const void* x,
    len_t trg,
    void* y,
    len_t n,
    ReductionType reduxop
) {
    hipStream_t stream = stream_from_miopen(w);
    const T* _src = static_cast<const T*>(x);
    T* _dst = static_cast<T*>(y);
    nll_loss_1d_index_reduce_kernel<T><<<1, 1, 0, stream>>>(_src, trg, _dst, n, reduxop);
}

extern "C" void hip_nn_nll_loss_1d_index_fwd(
    Dtype id,
    MiopenHandle w,
    void* src,
    len_t trg,
    void* dst,
    len_t n,
    bool inplace_smax,
    ReductionType reduxop
) {
    if (inplace_smax) {
        hip_nn_smax_vec_fwd(id, w, src, src, n, SMAX_LOG);
    }

    if (id == DTYPE_F32) {
        nll_loss_1d_index_reduce<f32>(w, src, trg, dst, n, reduxop);
    } else if (id == DTYPE_F64) {
        nll_loss_1d_index_reduce<f64>(w, src, trg, dst, n, reduxop);
    } else {
        SYSTEM_EXIT("Unsupported dtype for nll_loss_1d_index_fwd");
    }

    HIP_ASSERT(hipPeekAtLastError());
}

template <typename T>
struct NLLIndexBwdFunctor {
    len_t trg;
    T denom;

    __device__ T operator()(T src_val, len_t idx) const {
        const T tmp = (idx == this->trg) ? T(1) : T(0);
        return (src_val - tmp) / this->denom;
    }
};

template <typename T>
static void nll_loss_1d_index_bwd_impl(
    MiopenHandle w,
    const void* x_val,
    void* x_grd,
    len_t trg,
    len_t n,
    ReductionType reduxop
) {
    hipStream_t stream = stream_from_miopen(w);
    auto idx_iter = thrust::make_counting_iterator(0UL);
    const T* _x_val_iter = static_cast<const T*>(x_val);
    T* _x_grd_iter = static_cast<T*>(x_grd);
    const T denom = (reduxop == REDUX_SUM) ? T(1) : static_cast<T>(n);

    thrust::transform(
        thrust::hip::par.on(stream),
        _x_val_iter, _x_val_iter + n,
        idx_iter,
        _x_grd_iter,
        NLLIndexBwdFunctor<T>{.trg = trg, .denom = denom}
    );
}

extern "C" void hip_nn_nll_loss_1d_index_bwd(
    Dtype id,
    MiopenHandle w,
    const void* x_val,
    void* x_grd,
    len_t trg,
    len_t n,
    ReductionType reduxop
) {
    if (id == DTYPE_F32) {
        nll_loss_1d_index_bwd_impl<f32>(w, x_val, x_grd, trg, n, reduxop);
    } else if (id == DTYPE_F64) {
        nll_loss_1d_index_bwd_impl<f64>(w, x_val, x_grd, trg, n, reduxop);
    } else {
        SYSTEM_EXIT("Unsupported dtype for nll_loss_1d_index_bwd");
    }

    HIP_ASSERT(hipPeekAtLastError());
}

#endif /* __NN_NLL_INDEX_H__ */
