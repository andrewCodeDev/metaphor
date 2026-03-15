// Backward Gradient Kernels for Scaling Operations
//
// Appends to scaling.cu - implements gradients for all scaling operations

// =============================================================================
// Helper: Dot Product Reduction
// =============================================================================

template <typename T>
__global__ void dot_product_kernel(const T *a, const T *b, size_t n,
                                   T *result) {
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T sum = 0;
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    sum += a[idx] * b[idx];
  }

  sdata[tid] = sum;
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

// =============================================================================
// Min-Max Scaling Backward
// =============================================================================
// y = (x - min) / (max - min)
// grad_x = grad_y / (max - min)

template <typename T>
__global__ void min_max_scale_backward_kernel(const T *grad_y, T *grad_x,
                                              size_t n, T min_val, T max_val) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T range = max_val - min_val;
  if (range == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      grad_x[idx] = 0;
    }
    return;
  }

  T scale = 1.0f / range;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    grad_x[idx] = grad_y[idx] * scale;
  }
}

// =============================================================================
// Z-Score Backward
// =============================================================================
// y = (x - mean) / std
// grad_x = (grad_y - mean(grad_y) - cov(y, grad_y) * y) / std
// Simplified for batch independence:
// grad_x = grad_y / std

template <typename T>
__global__ void zscore_backward_simple_kernel(const T *grad_y, T *grad_x,
                                              size_t n, T std) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (std == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      grad_x[idx] = 0;
    }
    return;
  }

  T inv_std = 1.0f / std;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    grad_x[idx] = grad_y[idx] * inv_std;
  }
}

// Full z-score backward with mean/variance correction
template <typename T>
__global__ void zscore_backward_full_kernel(const T *grad_y, const T *y,
                                            T *grad_x, size_t n, T std,
                                            T grad_y_mean, T y_grad_cov) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (std == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      grad_x[idx] = 0;
    }
    return;
  }

  T inv_std = 1.0f / std;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    grad_x[idx] = (grad_y[idx] - grad_y_mean - y_grad_cov * y[idx]) * inv_std;
  }
}

// =============================================================================
// L2 Normalize Backward
// =============================================================================
// y = x / ||x||_2
// grad_x = (grad_y - y * dot(y, grad_y)) / ||x||_2

template <typename T>
__global__ void l2_normalize_backward_kernel(const T *grad_y, const T *y,
                                             T *grad_x, size_t n, T norm,
                                             T y_dot_grad_y) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (norm == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      grad_x[idx] = 0;
    }
    return;
  }

  T inv_norm = 1.0f / norm;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    grad_x[idx] = (grad_y[idx] - y[idx] * y_dot_grad_y) * inv_norm;
  }
}

// =============================================================================
// L1 Normalize Backward
// =============================================================================
// y = x / ||x||_1
// grad_x = (grad_y - sign(x) * sum(grad_y * sign(x))) / ||x||_1

template <typename T>
__global__ void l1_normalize_backward_kernel(const T *grad_y, const T *x,
                                             T *grad_x, size_t n, T norm,
                                             T correction) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (norm == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      grad_x[idx] = 0;
    }
    return;
  }

  T inv_norm = 1.0f / norm;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    T sign_x = (x[idx] > 0) ? 1.0f : ((x[idx] < 0) ? -1.0f : 0.0f);
    grad_x[idx] = (grad_y[idx] - sign_x * correction) * inv_norm;
  }
}

// =============================================================================
// Max Absolute Scaling Backward
// =============================================================================
// y = x / max(|x|)
// grad_x = grad_y / max(|x|) - sign(x_max) * grad_y_at_max

template <typename T>
__global__ void max_abs_scale_backward_kernel(const T *grad_y, const T *x,
                                              T *grad_x, size_t n, T max_val) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (max_val == 0) {
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
      grad_x[idx] = 0;
    }
    return;
  }

  T inv_max = 1.0f / max_val;

  // Simplified: treat as scale operation (ignores max gradient contribution)
  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    grad_x[idx] = grad_y[idx] * inv_max;
  }
}

// =============================================================================
// Clamp Backward
// =============================================================================
// y = clamp(x, min, max)
// grad_x = grad_y if min <= x <= max, else 0

template <typename T>
__global__ void clamp_backward_kernel(const T *grad_y, const T *x, T *grad_x,
                                      size_t n, T min_val, T max_val) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x) {
    T val = x[idx];
    grad_x[idx] = (val >= min_val && val <= max_val) ? grad_y[idx] : 0.0f;
  }
}

// =============================================================================
// Extern "C" Backward Interface
// =============================================================================

extern "C" {

// Min-max scaling backward
void cuda_min_max_scale_backward_f32(const float *grad_y, float *grad_x,
                                     size_t n, float min_val, float max_val) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  min_max_scale_backward_kernel<<<blocks, threads>>>(grad_y, grad_x, n, min_val,
                                                     max_val);
}

// Z-score backward (simple)
void cuda_zscore_backward_f32(const float *grad_y, float *grad_x, size_t n,
                              float std) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  zscore_backward_simple_kernel<<<blocks, threads>>>(grad_y, grad_x, n, std);
}

// L2 normalize backward
void cuda_l2_normalize_backward_f32(const float *grad_y, const float *y,
                                    float *grad_x, size_t n, float norm) {
  // Compute dot(y, grad_y)
  float *d_dot;
  cudaMalloc(&d_dot, sizeof(float));
  cudaMemset(d_dot, 0, sizeof(float));

  int threads = 256;
  int blocks = std::min(32, (int)((n + threads - 1) / threads));
  dot_product_kernel<<<blocks, threads, threads * sizeof(float)>>>(y, grad_y, n,
                                                                   d_dot);

  float h_dot;
  cudaMemcpy(&h_dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost);

  // Compute gradient
  blocks = (n + threads - 1) / threads;
  l2_normalize_backward_kernel<<<blocks, threads>>>(grad_y, y, grad_x, n, norm,
                                                    h_dot);

  cudaFree(d_dot);
}

// L1 normalize backward
void cuda_l1_normalize_backward_f32(const float *grad_y, const float *x,
                                    float *grad_x, size_t n, float norm) {
  // Compute sum(grad_y * sign(x))
  float *d_sum;
  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));

  // TODO: Implement sign-weighted sum kernel
  // For now, use simplified version
  float correction = 0; // Placeholder

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  l1_normalize_backward_kernel<<<blocks, threads>>>(grad_y, x, grad_x, n, norm,
                                                    correction);

  cudaFree(d_sum);
}

// Max-abs scale backward
void cuda_max_abs_scale_backward_f32(const float *grad_y, const float *x,
                                     float *grad_x, size_t n, float max_val) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  max_abs_scale_backward_kernel<<<blocks, threads>>>(grad_y, x, grad_x, n,
                                                     max_val);
}

// Clamp backward
void cuda_clamp_backward_f32(const float *grad_y, const float *x, float *grad_x,
                             size_t n, float min_val, float max_val) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  clamp_backward_kernel<<<blocks, threads>>>(grad_y, x, grad_x, n, min_val,
                                             max_val);
}

} // extern "C"
