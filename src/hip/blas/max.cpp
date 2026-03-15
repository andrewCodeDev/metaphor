/*
 * blas/max.cpp - Maximum reduction forward and backward
 */

#ifndef __BLAS_MAX_H__
#define __BLAS_MAX_H__

#include "utils.cpp"
#include <limits>

template <typename T>
__global__ void max_bwd_kernel(
    const void* __restrict__ x_val,
    const void* __restrict__ y_val,
    const void* __restrict__ y_grd,
    void* __restrict__ x_grd,
    len_t n
) {
    T y_v = *static_cast<const T*>(y_val);
    T y_g = *static_cast<const T*>(y_grd);
    const T* _x_val = static_cast<const T*>(x_val);
    T* _x_grd = static_cast<T*>(x_grd);
    thrust::transform(thrust::device, _x_val, _x_val + n, _x_grd, _x_grd,
        [=] __device__ (T x, T g) -> T {
            return g + ((x == y_v) ? y_g : T(0));
        }
    );
}

template <typename T>
__global__ void max_fwd_kernel(
    const void* __restrict__ x_val,
    void* __restrict__ y_val,
    len_t n
) {
    const T* src = static_cast<const T*>(x_val);
    T* dst = static_cast<T*>(y_val);
    *dst = thrust::reduce(thrust::device, src, src + n,
                          std::numeric_limits<T>::lowest(), thrust::maximum<T>());
}

extern "C" void hip_blas_max_fwd(
    Dtype id,
    StreamHandle w,
    const void* x,
    void* y,
    len_t n
) {
    hipStream_t stream = cast_stream(w);

    if (id == DTYPE_F32) {
        max_fwd_kernel<f32><<<1, 1, 0, stream>>>(x, y, n);
    } else if (id == DTYPE_F64) {
        max_fwd_kernel<f64><<<1, 1, 0, stream>>>(x, y, n);
    } else {
        SYSTEM_EXIT("Unsupported dtype for max_fwd");
    }

    HIP_ASSERT(hipPeekAtLastError());
}

extern "C" void hip_blas_max_bwd(
    Dtype id,
    StreamHandle w,
    const void* x_val,
    const void* y_val,
    const void* y_grd,
    void* x_grd,
    len_t n
) {
    hipStream_t stream = cast_stream(w);

    if (id == DTYPE_F32) {
        max_bwd_kernel<f32><<<1, 1, 0, stream>>>(x_val, y_val, y_grd, x_grd, n);
    } else if (id == DTYPE_F64) {
        max_bwd_kernel<f64><<<1, 1, 0, stream>>>(x_val, y_val, y_grd, x_grd, n);
    } else {
        SYSTEM_EXIT("Unsupported dtype for max_bwd");
    }

    HIP_ASSERT(hipPeekAtLastError());
}

#endif /* __BLAS_MAX_H__ */
