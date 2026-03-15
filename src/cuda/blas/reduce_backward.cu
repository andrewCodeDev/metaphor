/*
 * blas/reduce_backward.cu - Row/column reduce max/min backward kernels
 *
 * For reduce_max/min over rows: (M,N) -> (1,N)
 *   - For each column j, find max/min and route upstream[j] to matching positions
 *
 * For reduce_max/min over cols: (M,N) -> (M,1)
 *   - For each row i, find max/min and route upstream[i] to matching positions
 */

#ifndef __BLAS_REDUCE_BACKWARD_H__
#define __BLAS_REDUCE_BACKWARD_H__

#include "utils.cu"
#include <limits>

/* ============================================================================
 * Reduce max over columns backward: (M,N) -> (M,1)
 * Each thread handles one row, finds max, routes gradient
 * ============================================================================ */

template <typename T>
__global__ void kernel_reduce_max_cols_backward(
    const T* __restrict__ original,   // [M, N]
    const T* __restrict__ upstream,   // [M]
    T* __restrict__ grad_out,         // [M, N]
    len_t M,
    len_t N
) {
    len_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const T* row_data = original + row * N;
    T* row_grad = grad_out + row * N;
    T grad_val = upstream[row];

    // Find max in this row
    T max_val = row_data[0];
    for (len_t j = 1; j < N; j++) {
        if (row_data[j] > max_val) max_val = row_data[j];
    }

    // Route gradient to positions matching max
    for (len_t j = 0; j < N; j++) {
        row_grad[j] = (row_data[j] == max_val) ? grad_val : T(0);
    }
}

template <typename T>
__global__ void kernel_reduce_min_cols_backward(
    const T* __restrict__ original,
    const T* __restrict__ upstream,
    T* __restrict__ grad_out,
    len_t M,
    len_t N
) {
    len_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const T* row_data = original + row * N;
    T* row_grad = grad_out + row * N;
    T grad_val = upstream[row];

    // Find min in this row
    T min_val = row_data[0];
    for (len_t j = 1; j < N; j++) {
        if (row_data[j] < min_val) min_val = row_data[j];
    }

    // Route gradient to positions matching min
    for (len_t j = 0; j < N; j++) {
        row_grad[j] = (row_data[j] == min_val) ? grad_val : T(0);
    }
}

/* ============================================================================
 * Reduce max over rows backward: (M,N) -> (1,N)
 * Each thread handles one column, finds max, routes gradient
 * ============================================================================ */

template <typename T>
__global__ void kernel_reduce_max_rows_backward(
    const T* __restrict__ original,   // [M, N]
    const T* __restrict__ upstream,   // [N]
    T* __restrict__ grad_out,         // [M, N]
    len_t M,
    len_t N
) {
    len_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    T grad_val = upstream[col];

    // Find max in this column
    T max_val = original[col];
    for (len_t i = 1; i < M; i++) {
        T v = original[i * N + col];
        if (v > max_val) max_val = v;
    }

    // Route gradient to positions matching max
    for (len_t i = 0; i < M; i++) {
        len_t idx = i * N + col;
        grad_out[idx] = (original[idx] == max_val) ? grad_val : T(0);
    }
}

template <typename T>
__global__ void kernel_reduce_min_rows_backward(
    const T* __restrict__ original,
    const T* __restrict__ upstream,
    T* __restrict__ grad_out,
    len_t M,
    len_t N
) {
    len_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    T grad_val = upstream[col];

    // Find min in this column
    T min_val = original[col];
    for (len_t i = 1; i < M; i++) {
        T v = original[i * N + col];
        if (v < min_val) min_val = v;
    }

    // Route gradient to positions matching min
    for (len_t i = 0; i < M; i++) {
        len_t idx = i * N + col;
        grad_out[idx] = (original[idx] == min_val) ? grad_val : T(0);
    }
}

/* ============================================================================
 * Dispatch functions
 * ============================================================================ */

template <typename T>
static void reduce_max_cols_backward_impl(
    StreamHandle w,
    const void* original,
    const void* upstream,
    void* grad_out,
    len_t M,
    len_t N
) {
    cudaStream_t stream = cast_stream(w);
    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;
    kernel_reduce_max_cols_backward<T><<<num_blocks, block_size, 0, stream>>>(
        static_cast<const T*>(original),
        static_cast<const T*>(upstream),
        static_cast<T*>(grad_out),
        M, N
    );
}

template <typename T>
static void reduce_min_cols_backward_impl(
    StreamHandle w,
    const void* original,
    const void* upstream,
    void* grad_out,
    len_t M,
    len_t N
) {
    cudaStream_t stream = cast_stream(w);
    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;
    kernel_reduce_min_cols_backward<T><<<num_blocks, block_size, 0, stream>>>(
        static_cast<const T*>(original),
        static_cast<const T*>(upstream),
        static_cast<T*>(grad_out),
        M, N
    );
}

template <typename T>
static void reduce_max_rows_backward_impl(
    StreamHandle w,
    const void* original,
    const void* upstream,
    void* grad_out,
    len_t M,
    len_t N
) {
    cudaStream_t stream = cast_stream(w);
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    kernel_reduce_max_rows_backward<T><<<num_blocks, block_size, 0, stream>>>(
        static_cast<const T*>(original),
        static_cast<const T*>(upstream),
        static_cast<T*>(grad_out),
        M, N
    );
}

template <typename T>
static void reduce_min_rows_backward_impl(
    StreamHandle w,
    const void* original,
    const void* upstream,
    void* grad_out,
    len_t M,
    len_t N
) {
    cudaStream_t stream = cast_stream(w);
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    kernel_reduce_min_rows_backward<T><<<num_blocks, block_size, 0, stream>>>(
        static_cast<const T*>(original),
        static_cast<const T*>(upstream),
        static_cast<T*>(grad_out),
        M, N
    );
}

/* ============================================================================
 * Extern C API
 *
 * reduce_type: 0 = reduce rows (M,N) -> (1,N), 1 = reduce cols (M,N) -> (M,1)
 * is_min: 0 = max, 1 = min
 * ============================================================================ */

extern "C" void cuda_reduce_maxmin_backward(
    Dtype id,
    StreamHandle w,
    const void* original,
    const void* upstream,
    void* grad_out,
    len_t M,
    len_t N,
    int reduce_type,
    int is_min
) {
    if (reduce_type == 0) {
        // Reduce rows: (M,N) -> (1,N)
        if (is_min == 0) {
            switch (id) {
                case DTYPE_F32:
                    reduce_max_rows_backward_impl<f32>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F64:
                    reduce_max_rows_backward_impl<f64>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F16:
                    reduce_max_rows_backward_impl<f16>(w, original, upstream, grad_out, M, N);
                    return;
                default:
                    SYSTEM_EXIT("Unsupported dtype for reduce_max_rows_backward");
            }
        } else {
            switch (id) {
                case DTYPE_F32:
                    reduce_min_rows_backward_impl<f32>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F64:
                    reduce_min_rows_backward_impl<f64>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F16:
                    reduce_min_rows_backward_impl<f16>(w, original, upstream, grad_out, M, N);
                    return;
                default:
                    SYSTEM_EXIT("Unsupported dtype for reduce_min_rows_backward");
            }
        }
    } else if (reduce_type == 1) {
        // Reduce cols: (M,N) -> (M,1)
        if (is_min == 0) {
            switch (id) {
                case DTYPE_F32:
                    reduce_max_cols_backward_impl<f32>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F64:
                    reduce_max_cols_backward_impl<f64>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F16:
                    reduce_max_cols_backward_impl<f16>(w, original, upstream, grad_out, M, N);
                    return;
                default:
                    SYSTEM_EXIT("Unsupported dtype for reduce_max_cols_backward");
            }
        } else {
            switch (id) {
                case DTYPE_F32:
                    reduce_min_cols_backward_impl<f32>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F64:
                    reduce_min_cols_backward_impl<f64>(w, original, upstream, grad_out, M, N);
                    return;
                case DTYPE_F16:
                    reduce_min_cols_backward_impl<f16>(w, original, upstream, grad_out, M, N);
                    return;
                default:
                    SYSTEM_EXIT("Unsupported dtype for reduce_min_cols_backward");
            }
        }
    } else {
        SYSTEM_EXIT("Invalid reduce_type for reduce_maxmin_backward (expected 0 or 1)");
    }
}

#endif /* __BLAS_REDUCE_BACKWARD_H__ */
