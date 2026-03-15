// Scaling and Normalization Kernels for HIP
//
// Implements common data preprocessing operations matching host device API.

#include <cfloat>
#include <cmath>
#include <hip/hip_runtime.h>

// =============================================================================
// Reduction Kernels (Helper Functions)
// =============================================================================

template <typename T>
__device__ void warp_reduce_min_max(volatile T *smin, volatile T *smax,
                                    int tid) {
  if (blockDim.x >= 64) {
    smin[tid] = fminf(smin[tid], smin[tid + 32]);
    smax[tid] = fmaxf(smax[tid], smax[tid + 32]);
  }
  if (blockDim.x >= 32) {
    smin[tid] = fminf(smin[tid], smin[tid + 16]);
    smax[tid] = fmaxf(smax[tid], smax[tid + 16]);
  }
  if (blockDim.x >= 16) {
    smin[tid] = fminf(smin[tid], smin[tid + 8]);
    smax[tid] = fmaxf(smax[tid], smax[tid + 8]);
  }
  if (blockDim.x >= 8) {
    smin[tid] = fminf(smin[tid], smin[tid + 4]);
    smax[tid] = fmaxf(smax[tid], smax[tid + 4]);
  }
  if (blockDim.x >= 4) {
    smin[tid] = fminf(smin[tid], smin[tid + 2]);
    smax[tid] = fmaxf(smax[tid], smax[tid + 2]);
  }
  if (blockDim.x >= 2) {
    smin[tid] = fminf(smin[tid], smin[tid + 1]);
    smax[tid] = fmaxf(smax[tid], smax[tid + 1]);
  }
}

template <typename T>
__global__ void min_max_kernel(const T *x, size_t n, T *result) {
  extern __shared__ char shared_mem[];
  T *smin = (T *)shared_mem;
  T *smax = (T *)&smin[blockDim.x];

  int tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T local_min = (i < n) ? x[i] : FLT_MAX;
  T local_max = (i < n) ? x[i] : -FLT_MAX;

  // Grid-stride loop
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    T val = x[idx];
    local_min = fminf(local_min, val);
    local_max = fmaxf(local_max, val);
  }

  smin[tid] = local_min;
  smax[tid] = local_max;
  __syncthreads();

  // Block-level reduction
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      smin[tid] = fminf(smin[tid], smin[tid + s]);
      smax[tid] = fmaxf(smax[tid], smax[tid + s]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce_min_max(smin, smax, tid);
  }

  if (tid == 0) {
    atomicMin((int *)&result[0], __float_as_int(smin[0]));
    atomicMax((int *)&result[1], __float_as_int(smax[0]));
  }
}

template <typename T>
__global__ void mean_variance_kernel(const T *x, size_t n, T *result) {
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute sum for mean
  T sum = 0;
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    sum += x[idx];
  }

  sdata[tid] = sum;
  __syncthreads();

  // Reduce sum
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&result[0], sdata[0]);
  }
}

// =============================================================================
// Min-Max Scaling Kernels
// =============================================================================

template <typename T>
__global__ void min_max_scale_kernel(const T *x, T *y, size_t n, T min_val,
                                     T max_val) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T range = max_val - min_val;
  if (range == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      y[idx] = 0;
    }
    return;
  }

  T inv_range = 1.0f / range;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    y[idx] = (x[idx] - min_val) * inv_range;
  }
}

template <typename T>
__global__ void min_max_scale_range_kernel(const T *x, T *y, size_t n,
                                           T min_val, T max_val, T a, T b) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T input_range = max_val - min_val;
  T output_range = b - a;

  if (input_range == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      y[idx] = a;
    }
    return;
  }

  T scale = output_range / input_range;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    y[idx] = a + (x[idx] - min_val) * scale;
  }
}

// =============================================================================
// Z-Score Normalization Kernels
// =============================================================================

template <typename T>
__global__ void zscore_kernel(const T *x, T *y, size_t n, T mean, T std,
                              T epsilon) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (std < epsilon) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      y[idx] = 0;
    }
    return;
  }

  T inv_std = 1.0f / std;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    y[idx] = (x[idx] - mean) * inv_std;
  }
}

// =============================================================================
// L2 Normalization Kernels
// =============================================================================

template <typename T>
__global__ void l2_norm_kernel(const T *x, size_t n, T *result) {
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T sum_sq = 0;
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    T val = x[idx];
    sum_sq += val * val;
  }

  sdata[tid] = sum_sq;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

template <typename T>
__global__ void l2_normalize_kernel(const T *x, T *y, size_t n, T norm,
                                    T epsilon) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (norm < epsilon) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      y[idx] = 0;
    }
    return;
  }

  T inv_norm = 1.0f / norm;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    y[idx] = x[idx] * inv_norm;
  }
}

// =============================================================================
// L1 Normalization Kernels
// =============================================================================

template <typename T>
__global__ void l1_norm_kernel(const T *x, size_t n, T *result) {
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T sum_abs = 0;
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    sum_abs += fabsf(x[idx]);
  }

  sdata[tid] = sum_abs;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

template <typename T>
__global__ void l1_normalize_kernel(const T *x, T *y, size_t n, T norm,
                                    T epsilon) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (norm < epsilon) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      y[idx] = 0;
    }
    return;
  }

  T inv_norm = 1.0f / norm;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    y[idx] = x[idx] * inv_norm;
  }
}

// =============================================================================
// Max Absolute Scaling Kernels
// =============================================================================

template <typename T>
__global__ void max_abs_kernel(const T *x, size_t n, T *result) {
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T max_val = 0;
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    max_val = fmaxf(max_val, fabsf(x[idx]));
  }

  sdata[tid] = max_val;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMax((int *)result, __float_as_int(sdata[0]));
  }
}

template <typename T>
__global__ void max_abs_scale_kernel(const T *x, T *y, size_t n, T max_val,
                                     T epsilon) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (max_val < epsilon) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      y[idx] = 0;
    }
    return;
  }

  T inv_max = 1.0f / max_val;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    y[idx] = x[idx] * inv_max;
  }
}

// =============================================================================
// Clamp Kernel
// =============================================================================

template <typename T>
__global__ void clamp_kernel(const T *x, T *y, size_t n, T min_val, T max_val) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    T val = x[idx];
    y[idx] = fminf(fmaxf(val, min_val), max_val);
  }
}

// =============================================================================
// Extern "C" Interface
// =============================================================================

extern "C" {

// Min-max scaling
void hip_min_max_scale_f32(float *x, float *y, size_t n, float min_val,
                            float max_val, bool compute_stats) {
  if (compute_stats) {
    // TODO: Implement statistics computation
    // For now, assume min/max are provided
  }

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  min_max_scale_kernel<<<blocks, threads>>>(x, y, n, min_val, max_val);
}

void hip_min_max_scale_range_f32(float *x, float *y, size_t n, float min_val,
                                  float max_val, float a, float b) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  min_max_scale_range_kernel<<<blocks, threads>>>(x, y, n, min_val, max_val, a,
                                                  b);
}

// Z-score
void hip_zscore_f32(float *x, float *y, size_t n, float mean, float std,
                     float epsilon) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  zscore_kernel<<<blocks, threads>>>(x, y, n, mean, std, epsilon);
}

// L2 normalize
void hip_l2_normalize_f32(float *x, float *y, size_t n, float epsilon) {
  // Compute norm
  float *d_norm;
  HIP_ASSERT(hipMalloc(&d_norm, sizeof(float)));
  HIP_ASSERT(hipMemset(d_norm, 0, sizeof(float)));

  int threads = 256;
  int blocks = min(32, (int)((n + threads - 1) / threads));
  l2_norm_kernel<<<blocks, threads, threads * sizeof(float)>>>(x, n, d_norm);

  // Get norm and compute sqrt
  float h_norm_sq;
  HIP_ASSERT(hipMemcpy(&h_norm_sq, d_norm, sizeof(float), hipMemcpyDeviceToHost));
  float h_norm = sqrtf(h_norm_sq);

  // Normalize
  blocks = (n + threads - 1) / threads;
  l2_normalize_kernel<<<blocks, threads>>>(x, y, n, h_norm, epsilon);

  HIP_ASSERT(hipFree(d_norm));
}

// L1 normalize
void hip_l1_normalize_f32(float *x, float *y, size_t n, float epsilon) {
  float *d_norm;
  HIP_ASSERT(hipMalloc(&d_norm, sizeof(float)));
  HIP_ASSERT(hipMemset(d_norm, 0, sizeof(float)));

  int threads = 256;
  int blocks = min(32, (int)((n + threads - 1) / threads));
  l1_norm_kernel<<<blocks, threads, threads * sizeof(float)>>>(x, n, d_norm);

  float h_norm;
  HIP_ASSERT(hipMemcpy(&h_norm, d_norm, sizeof(float), hipMemcpyDeviceToHost));

  blocks = (n + threads - 1) / threads;
  l1_normalize_kernel<<<blocks, threads>>>(x, y, n, h_norm, epsilon);

  HIP_ASSERT(hipFree(d_norm));
}

// Max-abs scale
void hip_max_abs_scale_f32(float *x, float *y, size_t n, float epsilon) {
  float *d_max;
  HIP_ASSERT(hipMalloc(&d_max, sizeof(float)));
  HIP_ASSERT(hipMemset(d_max, 0, sizeof(float)));

  int threads = 256;
  int blocks = min(32, (int)((n + threads - 1) / threads));
  max_abs_kernel<<<blocks, threads, threads * sizeof(float)>>>(x, n, d_max);

  float h_max;
  HIP_ASSERT(hipMemcpy(&h_max, d_max, sizeof(float), hipMemcpyDeviceToHost));

  blocks = (n + threads - 1) / threads;
  max_abs_scale_kernel<<<blocks, threads>>>(x, y, n, h_max, epsilon);

  HIP_ASSERT(hipFree(d_max));
}

// Clamp
void hip_clamp_f32(float *x, float *y, size_t n, float min_val,
                    float max_val) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  clamp_kernel<<<blocks, threads>>>(x, y, n, min_val, max_val);
}

} // extern "C"
