/*
 * loss_kernels.cpp - Graph-capture-safe loss function kernels
 *
 * These kernels do NOT use thrust or any runtime allocation,
 * making them compatible with HIP graph capture.
 */

#include "../interop.h"
#include <cfloat>
#include <cstdint>
#include <hip/hip_runtime.h>

// Use len_t and ReductionType from interop.h

// ============================================================================
// Softmax Cross-Entropy Loss (Fused, Numerically Stable)
// ============================================================================
// Forward: loss = mean(-sum(target * log_softmax(logits), axis=-1))
// where log_softmax = logits - max(logits) - log(sum(exp(logits - max)))
//
// Backward: dL/dx = softmax(x) - target (beautiful gradient!)

template <typename T, int BLOCK_SIZE = 256>
__global__ void softmax_cross_entropy_fwd_kernel(
    const T *__restrict__ logits, // [batch, classes]
    const T *__restrict__ target, // [batch, classes] one-hot
    T *__restrict__ loss,         // [1] scalar output
    len_t batch_size, len_t num_classes, ReductionType reduction) {
  // Shared memory for per-batch loss accumulation
  __shared__ T shared_loss[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size)
    return;

  const T *row_logits = logits + batch_idx * num_classes;
  const T *row_target = target + batch_idx * num_classes;

  // Step 1: Find max for numerical stability (parallel reduction)
  T local_max = -FLT_MAX;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    local_max = fmaxf(local_max, row_logits[c]);
  }
  shared_loss[tid] = local_max;
  __syncthreads();

  // Reduce to find global max
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] = fmaxf(shared_loss[tid], shared_loss[tid + s]);
    }
    __syncthreads();
  }
  T max_val = shared_loss[0];
  __syncthreads();

  // Step 2: Compute sum(exp(x - max))
  T local_sum = 0;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    local_sum += expf(row_logits[c] - max_val);
  }
  shared_loss[tid] = local_sum;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
    }
    __syncthreads();
  }
  T log_sum_exp = logf(shared_loss[0]) + max_val;
  __syncthreads();

  // Step 3: Compute -sum(target * log_softmax)
  // log_softmax(x_i) = x_i - log_sum_exp
  T local_ce = 0;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    T log_softmax = row_logits[c] - log_sum_exp;
    local_ce -= row_target[c] * log_softmax;
  }
  shared_loss[tid] = local_ce;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
    }
    __syncthreads();
  }

  // Thread 0 writes this batch's loss
  if (tid == 0) {
    T batch_loss = shared_loss[0];
    if (reduction == REDUX_MEAN) {
      batch_loss /= static_cast<T>(batch_size);
    }
    atomicAdd(loss, batch_loss);
  }
}

template <typename T>
__global__ void softmax_cross_entropy_bwd_kernel(
    const T *__restrict__ logits, // [batch, classes]
    const T *__restrict__ target, // [batch, classes] one-hot
    T *__restrict__ grad,         // [batch, classes] output gradient
    len_t batch_size, len_t num_classes, ReductionType reduction) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const len_t total = batch_size * num_classes;
  if (idx >= total)
    return;

  const len_t batch = idx / num_classes;
  const len_t cls = idx % num_classes;

  const T *row_logits = logits + batch * num_classes;

  // Compute softmax(x_cls) - need max and sum for this row
  T max_val = row_logits[0];
  for (len_t c = 1; c < num_classes; c++) {
    max_val = fmaxf(max_val, row_logits[c]);
  }

  T sum_exp = 0;
  for (len_t c = 0; c < num_classes; c++) {
    sum_exp += expf(row_logits[c] - max_val);
  }

  T softmax_val = expf(row_logits[cls] - max_val) / sum_exp;
  T target_val = target[idx];

  // Gradient: softmax - target
  T g = softmax_val - target_val;

  if (reduction == REDUX_MEAN) {
    g /= static_cast<T>(batch_size);
  }

  grad[idx] = g;
}

// ============================================================================
// MSE Loss
// ============================================================================
// Forward: loss = mean((pred - target)^2)
// Backward: dL/dpred = 2 * (pred - target) / n

template <typename T, int BLOCK_SIZE = 256>
__global__ void
mse_fwd_kernel(const T *__restrict__ pred, const T *__restrict__ target,
               T *__restrict__ loss, len_t n, ReductionType reduction) {
  __shared__ T shared[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const len_t gid = blockIdx.x * BLOCK_SIZE + tid;

  // Compute local squared diff
  T local_sum = 0;
  for (len_t i = gid; i < n; i += gridDim.x * BLOCK_SIZE) {
    T diff = pred[i] - target[i];
    local_sum += diff * diff;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // Block-level reduction
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    T block_sum = shared[0];
    if (reduction == REDUX_MEAN) {
      block_sum /= static_cast<T>(n);
    }
    atomicAdd(loss, block_sum);
  }
}

template <typename T>
__global__ void
mse_bwd_kernel(const T *__restrict__ pred, const T *__restrict__ target,
               T *__restrict__ grad, len_t n, ReductionType reduction) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  T diff = pred[idx] - target[idx];
  T g = T(2) * diff;

  if (reduction == REDUX_MEAN) {
    g /= static_cast<T>(n);
  }

  grad[idx] = g;
}

// ============================================================================
// Binary Cross-Entropy Loss
// ============================================================================
// Forward: loss = -mean(target * log(pred) + (1 - target) * log(1 - pred))
// Backward: dL/dpred = -(target/pred - (1-target)/(1-pred)) / n

template <typename T, int BLOCK_SIZE = 256>
__global__ void
bce_fwd_kernel(const T *__restrict__ pred, const T *__restrict__ target,
               T *__restrict__ loss, len_t n, ReductionType reduction) {
  __shared__ T shared[BLOCK_SIZE];

  const T eps = T(1e-7);
  const int tid = threadIdx.x;
  const len_t gid = blockIdx.x * BLOCK_SIZE + tid;

  T local_sum = 0;
  for (len_t i = gid; i < n; i += gridDim.x * BLOCK_SIZE) {
    T p = fminf(fmaxf(pred[i], eps), T(1) - eps); // clamp to [eps, 1-eps]
    T t = target[i];
    local_sum -= t * logf(p) + (T(1) - t) * logf(T(1) - p);
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    T block_sum = shared[0];
    if (reduction == REDUX_MEAN) {
      block_sum /= static_cast<T>(n);
    }
    atomicAdd(loss, block_sum);
  }
}

template <typename T>
__global__ void
bce_bwd_kernel(const T *__restrict__ pred, const T *__restrict__ target,
               T *__restrict__ grad, len_t n, ReductionType reduction) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const T eps = T(1e-7);
  T p = fminf(fmaxf(pred[idx], eps), T(1) - eps);
  T t = target[idx];

  // Gradient: -(t/p - (1-t)/(1-p))
  T g = -(t / p - (T(1) - t) / (T(1) - p));

  if (reduction == REDUX_MEAN) {
    g /= static_cast<T>(n);
  }

  grad[idx] = g;
}

// ============================================================================
// Weighted MSE Loss
// ============================================================================
// Forward: loss = sum(weight * (pred - target)^2) / sum(weight)
// Backward: dL/dpred = 2 * weight * (pred - target) / sum(weight)

template <typename T, int BLOCK_SIZE = 256>
__global__ void weighted_mse_fwd_kernel(const T *__restrict__ pred,
                                        const T *__restrict__ target,
                                        const T *__restrict__ weight,
                                        T *__restrict__ loss,
                                        T *__restrict__ weight_sum, len_t n) {
  __shared__ T shared_loss[BLOCK_SIZE];
  __shared__ T shared_weight[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const len_t gid = blockIdx.x * BLOCK_SIZE + tid;

  T local_loss = 0;
  T local_weight = 0;
  for (len_t i = gid; i < n; i += gridDim.x * BLOCK_SIZE) {
    T diff = pred[i] - target[i];
    T w = weight[i];
    local_loss += w * diff * diff;
    local_weight += w;
  }
  shared_loss[tid] = local_loss;
  shared_weight[tid] = local_weight;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
      shared_weight[tid] += shared_weight[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(loss, shared_loss[0]);
    atomicAdd(weight_sum, shared_weight[0]);
  }
}

// Finalize weighted loss by dividing by weight sum
template <typename T>
__global__ void weighted_loss_finalize_kernel(T *__restrict__ loss,
                                              const T *__restrict__ weight_sum) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *loss = *loss / *weight_sum;
  }
}

template <typename T>
__global__ void weighted_mse_bwd_kernel(const T *__restrict__ pred,
                                        const T *__restrict__ target,
                                        const T *__restrict__ weight,
                                        const T *__restrict__ weight_sum,
                                        T *__restrict__ grad, len_t n) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  T diff = pred[idx] - target[idx];
  T w = weight[idx];
  grad[idx] = T(2) * w * diff / (*weight_sum);
}

// ============================================================================
// Weighted Binary Cross-Entropy Loss
// ============================================================================
// Forward: loss = -sum(weight * (target * log(pred) + (1 - target) * log(1 - pred))) / sum(weight)
// Backward: dL/dpred = -weight * (target/pred - (1-target)/(1-pred)) / sum(weight)

template <typename T, int BLOCK_SIZE = 256>
__global__ void weighted_bce_fwd_kernel(const T *__restrict__ pred,
                                        const T *__restrict__ target,
                                        const T *__restrict__ weight,
                                        T *__restrict__ loss,
                                        T *__restrict__ weight_sum, len_t n) {
  __shared__ T shared_loss[BLOCK_SIZE];
  __shared__ T shared_weight[BLOCK_SIZE];

  const T eps = T(1e-7);
  const int tid = threadIdx.x;
  const len_t gid = blockIdx.x * BLOCK_SIZE + tid;

  T local_loss = 0;
  T local_weight = 0;
  for (len_t i = gid; i < n; i += gridDim.x * BLOCK_SIZE) {
    T p = fminf(fmaxf(pred[i], eps), T(1) - eps);
    T t = target[i];
    T w = weight[i];
    local_loss -= w * (t * logf(p) + (T(1) - t) * logf(T(1) - p));
    local_weight += w;
  }
  shared_loss[tid] = local_loss;
  shared_weight[tid] = local_weight;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
      shared_weight[tid] += shared_weight[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(loss, shared_loss[0]);
    atomicAdd(weight_sum, shared_weight[0]);
  }
}

template <typename T>
__global__ void weighted_bce_bwd_kernel(const T *__restrict__ pred,
                                        const T *__restrict__ target,
                                        const T *__restrict__ weight,
                                        const T *__restrict__ weight_sum,
                                        T *__restrict__ grad, len_t n) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const T eps = T(1e-7);
  T p = fminf(fmaxf(pred[idx], eps), T(1) - eps);
  T t = target[idx];
  T w = weight[idx];

  // Gradient: -weight * (t/p - (1-t)/(1-p)) / sum(weight)
  T g = -w * (t / p - (T(1) - t) / (T(1) - p)) / (*weight_sum);
  grad[idx] = g;
}

// ============================================================================
// Weighted Softmax Cross-Entropy Loss (per-sample weights)
// ============================================================================
// Forward: loss = -sum(weight * sum(target * log_softmax(logits), axis=-1)) / sum(weight)

template <typename T, int BLOCK_SIZE = 256>
__global__ void weighted_softmax_cross_entropy_fwd_kernel(
    const T *__restrict__ logits, const T *__restrict__ target,
    const T *__restrict__ weight, // [batch] per-sample weights
    T *__restrict__ loss, T *__restrict__ weight_sum, len_t batch_size,
    len_t num_classes) {
  __shared__ T shared_loss[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size)
    return;

  const T *row_logits = logits + batch_idx * num_classes;
  const T *row_target = target + batch_idx * num_classes;
  T sample_weight = weight[batch_idx];

  // Step 1: Find max for numerical stability
  T local_max = -FLT_MAX;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    local_max = fmaxf(local_max, row_logits[c]);
  }
  shared_loss[tid] = local_max;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] = fmaxf(shared_loss[tid], shared_loss[tid + s]);
    }
    __syncthreads();
  }
  T max_val = shared_loss[0];
  __syncthreads();

  // Step 2: Compute sum(exp(x - max))
  T local_sum = 0;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    local_sum += expf(row_logits[c] - max_val);
  }
  shared_loss[tid] = local_sum;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
    }
    __syncthreads();
  }
  T log_sum_exp = logf(shared_loss[0]) + max_val;
  __syncthreads();

  // Step 3: Compute -sum(target * log_softmax)
  T local_ce = 0;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    T log_softmax = row_logits[c] - log_sum_exp;
    local_ce -= row_target[c] * log_softmax;
  }
  shared_loss[tid] = local_ce;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    T batch_loss = sample_weight * shared_loss[0];
    atomicAdd(loss, batch_loss);
    atomicAdd(weight_sum, sample_weight);
  }
}

template <typename T>
__global__ void weighted_softmax_cross_entropy_bwd_kernel(
    const T *__restrict__ logits, const T *__restrict__ target,
    const T *__restrict__ weight, const T *__restrict__ weight_sum,
    T *__restrict__ grad, len_t batch_size, len_t num_classes) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const len_t total = batch_size * num_classes;
  if (idx >= total)
    return;

  const len_t batch = idx / num_classes;
  const len_t cls = idx % num_classes;

  const T *row_logits = logits + batch * num_classes;
  T sample_weight = weight[batch];

  // Compute softmax for this element
  T max_val = row_logits[0];
  for (len_t c = 1; c < num_classes; c++) {
    max_val = fmaxf(max_val, row_logits[c]);
  }

  T sum_exp = 0;
  for (len_t c = 0; c < num_classes; c++) {
    sum_exp += expf(row_logits[c] - max_val);
  }

  T softmax_val = expf(row_logits[cls] - max_val) / sum_exp;
  T target_val = target[idx];

  // Gradient: weight * (softmax - target) / sum(weight)
  grad[idx] = sample_weight * (softmax_val - target_val) / (*weight_sum);
}

// ============================================================================
// Class-Weighted Softmax Cross-Entropy Loss (per-class weights)
// ============================================================================
// Forward: loss = mean(-sum(class_weight * target * log_softmax(logits), axis=-1))

template <typename T, int BLOCK_SIZE = 256>
__global__ void class_weighted_softmax_cross_entropy_fwd_kernel(
    const T *__restrict__ logits, const T *__restrict__ target,
    const T *__restrict__ class_weight, // [classes] per-class weights
    T *__restrict__ loss, len_t batch_size, len_t num_classes,
    ReductionType reduction) {
  __shared__ T shared_loss[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int batch_idx = blockIdx.x;

  if (batch_idx >= batch_size)
    return;

  const T *row_logits = logits + batch_idx * num_classes;
  const T *row_target = target + batch_idx * num_classes;

  // Step 1: Find max
  T local_max = -FLT_MAX;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    local_max = fmaxf(local_max, row_logits[c]);
  }
  shared_loss[tid] = local_max;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] = fmaxf(shared_loss[tid], shared_loss[tid + s]);
    }
    __syncthreads();
  }
  T max_val = shared_loss[0];
  __syncthreads();

  // Step 2: sum(exp(x - max))
  T local_sum = 0;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    local_sum += expf(row_logits[c] - max_val);
  }
  shared_loss[tid] = local_sum;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
    }
    __syncthreads();
  }
  T log_sum_exp = logf(shared_loss[0]) + max_val;
  __syncthreads();

  // Step 3: -sum(class_weight * target * log_softmax)
  T local_ce = 0;
  for (len_t c = tid; c < num_classes; c += BLOCK_SIZE) {
    T log_softmax = row_logits[c] - log_sum_exp;
    local_ce -= class_weight[c] * row_target[c] * log_softmax;
  }
  shared_loss[tid] = local_ce;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_loss[tid] += shared_loss[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    T batch_loss = shared_loss[0];
    if (reduction == REDUX_MEAN) {
      batch_loss /= static_cast<T>(batch_size);
    }
    atomicAdd(loss, batch_loss);
  }
}

template <typename T>
__global__ void class_weighted_softmax_cross_entropy_bwd_kernel(
    const T *__restrict__ logits, const T *__restrict__ target,
    const T *__restrict__ class_weight, T *__restrict__ grad, len_t batch_size,
    len_t num_classes, ReductionType reduction) {
  const len_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const len_t total = batch_size * num_classes;
  if (idx >= total)
    return;

  const len_t batch = idx / num_classes;
  const len_t cls = idx % num_classes;

  const T *row_logits = logits + batch * num_classes;
  const T *row_target = target + batch * num_classes;

  // Compute softmax
  T max_val = row_logits[0];
  for (len_t c = 1; c < num_classes; c++) {
    max_val = fmaxf(max_val, row_logits[c]);
  }

  T sum_exp = 0;
  for (len_t c = 0; c < num_classes; c++) {
    sum_exp += expf(row_logits[c] - max_val);
  }

  T softmax_val = expf(row_logits[cls] - max_val) / sum_exp;

  // For class-weighted CE, the gradient is more complex:
  // dL/dx_j = softmax_j * sum_k(class_weight_k * target_k) - class_weight_j * target_j
  T weighted_target_sum = 0;
  for (len_t c = 0; c < num_classes; c++) {
    weighted_target_sum += class_weight[c] * row_target[c];
  }

  T g = softmax_val * weighted_target_sum - class_weight[cls] * target[idx];

  if (reduction == REDUX_MEAN) {
    g /= static_cast<T>(batch_size);
  }

  grad[idx] = g;
}

// ============================================================================
// C Interface
// ============================================================================

extern "C" void
hip_loss_softmax_cross_entropy_fwd(int dtype, const void *logits,
                                    const void *target, void *loss,
                                    uint64_t batch_size, uint64_t num_classes,
                                    int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  // Zero the output
  HIP_ASSERT(hipMemsetAsync(loss, 0, (dtype == 0) ? sizeof(float) : sizeof(double),
                  stream));

  dim3 grid(batch_size);
  dim3 block(256);

  if (dtype == 0) { // f32
    softmax_cross_entropy_fwd_kernel<float, 256><<<grid, block, 0, stream>>>(
        static_cast<const float *>(logits), static_cast<const float *>(target),
        static_cast<float *>(loss), batch_size, num_classes,
        static_cast<ReductionType>(reduction));
  } else { // f64
    softmax_cross_entropy_fwd_kernel<double, 256><<<grid, block, 0, stream>>>(
        static_cast<const double *>(logits),
        static_cast<const double *>(target), static_cast<double *>(loss),
        batch_size, num_classes, static_cast<ReductionType>(reduction));
  }
}

extern "C" void
hip_loss_softmax_cross_entropy_bwd(int dtype, const void *logits,
                                    const void *target, void *grad,
                                    uint64_t batch_size, uint64_t num_classes,
                                    int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  const uint64_t total = batch_size * num_classes;
  dim3 grid((total + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    softmax_cross_entropy_bwd_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float *>(logits), static_cast<const float *>(target),
        static_cast<float *>(grad), batch_size, num_classes,
        static_cast<ReductionType>(reduction));
  } else {
    softmax_cross_entropy_bwd_kernel<double><<<grid, block, 0, stream>>>(
        static_cast<const double *>(logits),
        static_cast<const double *>(target), static_cast<double *>(grad),
        batch_size, num_classes, static_cast<ReductionType>(reduction));
  }
}

extern "C" void hip_loss_mse_fwd(int dtype, const void *pred,
                                  const void *target, void *loss, uint64_t n,
                                  int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  HIP_ASSERT(hipMemsetAsync(loss, 0, (dtype == 0) ? sizeof(float) : sizeof(double),
                  stream));

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    mse_fwd_kernel<float, 256><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<float *>(loss), n, static_cast<ReductionType>(reduction));
  } else {
    mse_fwd_kernel<double, 256><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<double *>(loss), n, static_cast<ReductionType>(reduction));
  }
}

extern "C" void hip_loss_mse_bwd(int dtype, const void *pred,
                                  const void *target, void *grad, uint64_t n,
                                  int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    mse_bwd_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<float *>(grad), n, static_cast<ReductionType>(reduction));
  } else {
    mse_bwd_kernel<double><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<double *>(grad), n, static_cast<ReductionType>(reduction));
  }
}

extern "C" void hip_loss_bce_fwd(int dtype, const void *pred,
                                  const void *target, void *loss, uint64_t n,
                                  int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  HIP_ASSERT(hipMemsetAsync(loss, 0, (dtype == 0) ? sizeof(float) : sizeof(double),
                  stream));

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    bce_fwd_kernel<float, 256><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<float *>(loss), n, static_cast<ReductionType>(reduction));
  } else {
    bce_fwd_kernel<double, 256><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<double *>(loss), n, static_cast<ReductionType>(reduction));
  }
}

extern "C" void hip_loss_bce_bwd(int dtype, const void *pred,
                                  const void *target, void *grad, uint64_t n,
                                  int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    bce_bwd_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<float *>(grad), n, static_cast<ReductionType>(reduction));
  } else {
    bce_bwd_kernel<double><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<double *>(grad), n, static_cast<ReductionType>(reduction));
  }
}

// ============================================================================
// Weighted Loss C Interface
// ============================================================================

extern "C" void hip_loss_weighted_mse_fwd(int dtype, const void *pred,
                                           const void *target,
                                           const void *weight, void *loss,
                                           void *weight_sum, uint64_t n,
                                           void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  // Zero outputs
  size_t elem_size = (dtype == 0) ? sizeof(float) : sizeof(double);
  HIP_ASSERT(hipMemsetAsync(loss, 0, elem_size, stream));
  HIP_ASSERT(hipMemsetAsync(weight_sum, 0, elem_size, stream));

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    weighted_mse_fwd_kernel<float, 256><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<const float *>(weight), static_cast<float *>(loss),
        static_cast<float *>(weight_sum), n);
    weighted_loss_finalize_kernel<float><<<1, 1, 0, stream>>>(
        static_cast<float *>(loss), static_cast<const float *>(weight_sum));
  } else {
    weighted_mse_fwd_kernel<double, 256><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<const double *>(weight), static_cast<double *>(loss),
        static_cast<double *>(weight_sum), n);
    weighted_loss_finalize_kernel<double><<<1, 1, 0, stream>>>(
        static_cast<double *>(loss), static_cast<const double *>(weight_sum));
  }
}

extern "C" void hip_loss_weighted_mse_bwd(int dtype, const void *pred,
                                           const void *target,
                                           const void *weight,
                                           const void *weight_sum, void *grad,
                                           uint64_t n, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    weighted_mse_bwd_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<const float *>(weight),
        static_cast<const float *>(weight_sum), static_cast<float *>(grad), n);
  } else {
    weighted_mse_bwd_kernel<double><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<const double *>(weight),
        static_cast<const double *>(weight_sum), static_cast<double *>(grad),
        n);
  }
}

extern "C" void hip_loss_weighted_bce_fwd(int dtype, const void *pred,
                                           const void *target,
                                           const void *weight, void *loss,
                                           void *weight_sum, uint64_t n,
                                           void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  size_t elem_size = (dtype == 0) ? sizeof(float) : sizeof(double);
  HIP_ASSERT(hipMemsetAsync(loss, 0, elem_size, stream));
  HIP_ASSERT(hipMemsetAsync(weight_sum, 0, elem_size, stream));

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    weighted_bce_fwd_kernel<float, 256><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<const float *>(weight), static_cast<float *>(loss),
        static_cast<float *>(weight_sum), n);
    weighted_loss_finalize_kernel<float><<<1, 1, 0, stream>>>(
        static_cast<float *>(loss), static_cast<const float *>(weight_sum));
  } else {
    weighted_bce_fwd_kernel<double, 256><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<const double *>(weight), static_cast<double *>(loss),
        static_cast<double *>(weight_sum), n);
    weighted_loss_finalize_kernel<double><<<1, 1, 0, stream>>>(
        static_cast<double *>(loss), static_cast<const double *>(weight_sum));
  }
}

extern "C" void hip_loss_weighted_bce_bwd(int dtype, const void *pred,
                                           const void *target,
                                           const void *weight,
                                           const void *weight_sum, void *grad,
                                           uint64_t n, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  dim3 grid((n + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    weighted_bce_bwd_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float *>(pred), static_cast<const float *>(target),
        static_cast<const float *>(weight),
        static_cast<const float *>(weight_sum), static_cast<float *>(grad), n);
  } else {
    weighted_bce_bwd_kernel<double><<<grid, block, 0, stream>>>(
        static_cast<const double *>(pred), static_cast<const double *>(target),
        static_cast<const double *>(weight),
        static_cast<const double *>(weight_sum), static_cast<double *>(grad),
        n);
  }
}

extern "C" void hip_loss_weighted_softmax_ce_fwd(
    int dtype, const void *logits, const void *target, const void *weight,
    void *loss, void *weight_sum, uint64_t batch_size, uint64_t num_classes,
    void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  size_t elem_size = (dtype == 0) ? sizeof(float) : sizeof(double);
  HIP_ASSERT(hipMemsetAsync(loss, 0, elem_size, stream));
  HIP_ASSERT(hipMemsetAsync(weight_sum, 0, elem_size, stream));

  dim3 grid(batch_size);
  dim3 block(256);

  if (dtype == 0) {
    weighted_softmax_cross_entropy_fwd_kernel<float, 256>
        <<<grid, block, 0, stream>>>(
            static_cast<const float *>(logits),
            static_cast<const float *>(target),
            static_cast<const float *>(weight), static_cast<float *>(loss),
            static_cast<float *>(weight_sum), batch_size, num_classes);
    weighted_loss_finalize_kernel<float><<<1, 1, 0, stream>>>(
        static_cast<float *>(loss), static_cast<const float *>(weight_sum));
  } else {
    weighted_softmax_cross_entropy_fwd_kernel<double, 256>
        <<<grid, block, 0, stream>>>(
            static_cast<const double *>(logits),
            static_cast<const double *>(target),
            static_cast<const double *>(weight), static_cast<double *>(loss),
            static_cast<double *>(weight_sum), batch_size, num_classes);
    weighted_loss_finalize_kernel<double><<<1, 1, 0, stream>>>(
        static_cast<double *>(loss), static_cast<const double *>(weight_sum));
  }
}

extern "C" void hip_loss_weighted_softmax_ce_bwd(
    int dtype, const void *logits, const void *target, const void *weight,
    const void *weight_sum, void *grad, uint64_t batch_size,
    uint64_t num_classes, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  const uint64_t total = batch_size * num_classes;
  dim3 grid((total + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    weighted_softmax_cross_entropy_bwd_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const float *>(logits), static_cast<const float *>(target),
        static_cast<const float *>(weight),
        static_cast<const float *>(weight_sum), static_cast<float *>(grad),
        batch_size, num_classes);
  } else {
    weighted_softmax_cross_entropy_bwd_kernel<double>
        <<<grid, block, 0, stream>>>(
            static_cast<const double *>(logits),
            static_cast<const double *>(target),
            static_cast<const double *>(weight),
            static_cast<const double *>(weight_sum),
            static_cast<double *>(grad), batch_size, num_classes);
  }
}

extern "C" void hip_loss_class_weighted_softmax_ce_fwd(
    int dtype, const void *logits, const void *target,
    const void *class_weight, void *loss, uint64_t batch_size,
    uint64_t num_classes, int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  size_t elem_size = (dtype == 0) ? sizeof(float) : sizeof(double);
  HIP_ASSERT(hipMemsetAsync(loss, 0, elem_size, stream));

  dim3 grid(batch_size);
  dim3 block(256);

  if (dtype == 0) {
    class_weighted_softmax_cross_entropy_fwd_kernel<float, 256>
        <<<grid, block, 0, stream>>>(
            static_cast<const float *>(logits),
            static_cast<const float *>(target),
            static_cast<const float *>(class_weight),
            static_cast<float *>(loss), batch_size, num_classes,
            static_cast<ReductionType>(reduction));
  } else {
    class_weighted_softmax_cross_entropy_fwd_kernel<double, 256>
        <<<grid, block, 0, stream>>>(
            static_cast<const double *>(logits),
            static_cast<const double *>(target),
            static_cast<const double *>(class_weight),
            static_cast<double *>(loss), batch_size, num_classes,
            static_cast<ReductionType>(reduction));
  }
}

extern "C" void hip_loss_class_weighted_softmax_ce_bwd(
    int dtype, const void *logits, const void *target,
    const void *class_weight, void *grad, uint64_t batch_size,
    uint64_t num_classes, int reduction, void *stream_ptr) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  const uint64_t total = batch_size * num_classes;
  dim3 grid((total + 255) / 256);
  dim3 block(256);

  if (dtype == 0) {
    class_weighted_softmax_cross_entropy_bwd_kernel<float>
        <<<grid, block, 0, stream>>>(
            static_cast<const float *>(logits),
            static_cast<const float *>(target),
            static_cast<const float *>(class_weight),
            static_cast<float *>(grad), batch_size, num_classes,
            static_cast<ReductionType>(reduction));
  } else {
    class_weighted_softmax_cross_entropy_bwd_kernel<double>
        <<<grid, block, 0, stream>>>(
            static_cast<const double *>(logits),
            static_cast<const double *>(target),
            static_cast<const double *>(class_weight),
            static_cast<double *>(grad), batch_size, num_classes,
            static_cast<ReductionType>(reduction));
  }
}
