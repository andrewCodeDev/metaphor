/*
 * nn/optimizers.cpp - Optimizer update kernels
 *
 * CUSTOM KERNEL REQUIREMENTS FOR F16/BF16:
 *
 * These optimizers use Thrust templates with standard arithmetic operators,
 * which do NOT work directly with __half or hip_bfloat16 types. Supporting
 * F16/BF16 would require:
 *
 * Option 1: Custom HIP kernels using half intrinsics (__hadd, __hmul, etc.)
 * Option 2: Mixed precision training (recommended):
 *   - Parameters stored in F16 for memory savings and fast GEMM
 *   - Optimizer state (m, v) stored in FP32 for numerical stability
 *   - Optimizer updates computed in FP32
 *   - Final param update cast back to F16
 *
 * Mixed precision is the industry standard for training (see AMP, apex).
 * The forward/backward pass uses F16, but optimizer math uses FP32.
 *
 * Current status: F16/BF16 cases will SYSTEM_EXIT - custom kernels needed.
 */

#ifndef __NN_OPTIMIZERS_H__
#define __NN_OPTIMIZERS_H__

#include "../blas/utils.cpp"

/* ============================================================================
 * SGD: param -= lr * grad
 * ============================================================================ */

template <typename T>
struct SgdFunctor {
    T lr;
    __device__ T operator()(T param, T grad) const {
        return param - lr * grad;
    }
};

extern "C" void hip_optim_sgd(
    Dtype id,
    StreamHandle w,
    void* param,
    const void* grad,
    len_t n,
    f64 lr
) {
    hipStream_t stream = cast_stream(w);

    switch (id) {
        case DTYPE_F32: {
            f32* p = static_cast<f32*>(param);
            const f32* g = static_cast<const f32*>(grad);
            thrust::transform(
                thrust::hip::par.on(stream),
                p, p + n, g, p,
                SgdFunctor<f32>{static_cast<f32>(lr)}
            );
            return;
        }
        case DTYPE_F64: {
            f64* p = static_cast<f64*>(param);
            const f64* g = static_cast<const f64*>(grad);
            thrust::transform(
                thrust::hip::par.on(stream),
                p, p + n, g, p,
                SgdFunctor<f64>{lr}
            );
            return;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for sgd");
    }
}

/* ============================================================================
 * Momentum: v = momentum * v + grad; param -= lr * v
 * ============================================================================ */

template <typename T>
struct MomentumVelocityFunctor {
    T momentum;
    __device__ T operator()(T vel, T grad) const {
        return momentum * vel + grad;
    }
};

template <typename T>
struct MomentumParamFunctor {
    T lr;
    __device__ T operator()(T param, T vel) const {
        return param - lr * vel;
    }
};

extern "C" void hip_optim_momentum(
    Dtype id,
    StreamHandle w,
    void* param,
    const void* grad,
    void* velocity,
    len_t n,
    f64 lr,
    f64 momentum
) {
    hipStream_t stream = cast_stream(w);

    switch (id) {
        case DTYPE_F32: {
            f32* p = static_cast<f32*>(param);
            const f32* g = static_cast<const f32*>(grad);
            f32* v = static_cast<f32*>(velocity);
            // Update velocity: v = momentum * v + grad
            thrust::transform(
                thrust::hip::par.on(stream),
                v, v + n, g, v,
                MomentumVelocityFunctor<f32>{static_cast<f32>(momentum)}
            );
            // Update param: param -= lr * v
            thrust::transform(
                thrust::hip::par.on(stream),
                p, p + n, v, p,
                MomentumParamFunctor<f32>{static_cast<f32>(lr)}
            );
            return;
        }
        case DTYPE_F64: {
            f64* p = static_cast<f64*>(param);
            const f64* g = static_cast<const f64*>(grad);
            f64* v = static_cast<f64*>(velocity);
            thrust::transform(
                thrust::hip::par.on(stream),
                v, v + n, g, v,
                MomentumVelocityFunctor<f64>{momentum}
            );
            thrust::transform(
                thrust::hip::par.on(stream),
                p, p + n, v, p,
                MomentumParamFunctor<f64>{lr}
            );
            return;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for momentum");
    }
}

/* ============================================================================
 * RMSProp: cache = decay * cache + (1 - decay) * grad^2
 *          param -= lr * grad / (sqrt(cache) + eps)
 * ============================================================================ */

template <typename T>
struct RMSPropFunctor {
    T lr;
    T decay;
    T epsilon;
    __device__ thrust::tuple<T, T> operator()(
        thrust::tuple<T, T, T> tup
    ) const {
        T param = thrust::get<0>(tup);
        T grad = thrust::get<1>(tup);
        T cache = thrust::get<2>(tup);
        // Update cache
        T new_cache = decay * cache + (T{1} - decay) * grad * grad;
        // Update param
        T new_param = param - lr * grad / (sqrt(new_cache) + epsilon);
        return thrust::make_tuple(new_param, new_cache);
    }
};

extern "C" void hip_optim_rmsprop(
    Dtype id,
    StreamHandle w,
    void* param,
    const void* grad,
    void* cache,
    len_t n,
    f64 lr,
    f64 decay,
    f64 epsilon
) {
    hipStream_t stream = cast_stream(w);

    switch (id) {
        case DTYPE_F32: {
            f32* p = static_cast<f32*>(param);
            const f32* g = static_cast<const f32*>(grad);
            f32* c = static_cast<f32*>(cache);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(p, g, c));
            auto out = thrust::make_zip_iterator(thrust::make_tuple(p, c));
            thrust::transform(
                thrust::hip::par.on(stream),
                in, in + n, out,
                RMSPropFunctor<f32>{
                    static_cast<f32>(lr),
                    static_cast<f32>(decay),
                    static_cast<f32>(epsilon)
                }
            );
            return;
        }
        case DTYPE_F64: {
            f64* p = static_cast<f64*>(param);
            const f64* g = static_cast<const f64*>(grad);
            f64* c = static_cast<f64*>(cache);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(p, g, c));
            auto out = thrust::make_zip_iterator(thrust::make_tuple(p, c));
            thrust::transform(
                thrust::hip::par.on(stream),
                in, in + n, out,
                RMSPropFunctor<f64>{lr, decay, epsilon}
            );
            return;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for rmsprop");
    }
}

/* ============================================================================
 * Adadelta: grad_cache = rho * grad_cache + (1 - rho) * grad^2
 *           delta = sqrt(delta_cache + eps) / sqrt(grad_cache + eps) * grad
 *           param -= delta
 *           delta_cache = rho * delta_cache + (1 - rho) * delta^2
 * ============================================================================ */

template <typename T>
struct AdadeltaFunctor {
    T rho;
    T epsilon;
    __device__ thrust::tuple<T, T, T> operator()(
        thrust::tuple<T, T, T, T> tup
    ) const {
        T param = thrust::get<0>(tup);
        T grad = thrust::get<1>(tup);
        T grad_cache = thrust::get<2>(tup);
        T delta_cache = thrust::get<3>(tup);
        // Update grad_cache
        T new_grad_cache = rho * grad_cache + (T{1} - rho) * grad * grad;
        // Compute delta
        T rms_delta = sqrt(delta_cache + epsilon);
        T rms_grad = sqrt(new_grad_cache + epsilon);
        T delta = rms_delta / rms_grad * grad;
        // Update param
        T new_param = param - delta;
        // Update delta_cache
        T new_delta_cache = rho * delta_cache + (T{1} - rho) * delta * delta;
        return thrust::make_tuple(new_param, new_grad_cache, new_delta_cache);
    }
};

extern "C" void hip_optim_adadelta(
    Dtype id,
    StreamHandle w,
    void* param,
    const void* grad,
    void* grad_cache,
    void* delta_cache,
    len_t n,
    f64 rho,
    f64 epsilon
) {
    hipStream_t stream = cast_stream(w);

    switch (id) {
        case DTYPE_F32: {
            f32* p = static_cast<f32*>(param);
            const f32* g = static_cast<const f32*>(grad);
            f32* gc = static_cast<f32*>(grad_cache);
            f32* dc = static_cast<f32*>(delta_cache);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(p, g, gc, dc));
            auto out = thrust::make_zip_iterator(thrust::make_tuple(p, gc, dc));
            thrust::transform(
                thrust::hip::par.on(stream),
                in, in + n, out,
                AdadeltaFunctor<f32>{
                    static_cast<f32>(rho),
                    static_cast<f32>(epsilon)
                }
            );
            return;
        }
        case DTYPE_F64: {
            f64* p = static_cast<f64*>(param);
            const f64* g = static_cast<const f64*>(grad);
            f64* gc = static_cast<f64*>(grad_cache);
            f64* dc = static_cast<f64*>(delta_cache);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(p, g, gc, dc));
            auto out = thrust::make_zip_iterator(thrust::make_tuple(p, gc, dc));
            thrust::transform(
                thrust::hip::par.on(stream),
                in, in + n, out,
                AdadeltaFunctor<f64>{rho, epsilon}
            );
            return;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for adadelta");
    }
}

/* ============================================================================
 * Adam: m = beta1 * m + (1 - beta1) * grad
 *       v = beta2 * v + (1 - beta2) * grad^2
 *       m_hat = m / (1 - beta1^t)
 *       v_hat = v / (1 - beta2^t)
 *       param -= lr * m_hat / (sqrt(v_hat) + eps)
 * ============================================================================ */

template <typename T>
struct AdamFunctor {
    T lr;
    T beta1;
    T beta2;
    T epsilon;
    T bc1; // 1 / (1 - beta1^t)
    T bc2; // 1 / (1 - beta2^t)
    T wd;  // decoupled weight decay (AdamW)
    __device__ thrust::tuple<T, T, T> operator()(
        thrust::tuple<T, T, T, T> tup
    ) const {
        T param = thrust::get<0>(tup);
        T grad = thrust::get<1>(tup);
        T m = thrust::get<2>(tup);
        T v = thrust::get<3>(tup);
        // Update moments
        T new_m = beta1 * m + (T{1} - beta1) * grad;
        T new_v = beta2 * v + (T{1} - beta2) * grad * grad;
        // Bias-corrected estimates
        T m_hat = new_m * bc1;
        T v_hat = new_v * bc2;
        // AdamW: decoupled weight decay + Adam update
        T new_param = param * (T{1} - lr * wd) - lr * m_hat / (sqrt(v_hat) + epsilon);
        return thrust::make_tuple(new_param, new_m, new_v);
    }
};

extern "C" void hip_optim_adam(
    Dtype id,
    StreamHandle w,
    void* param,
    const void* grad,
    void* m,
    void* v,
    len_t n,
    f64 lr,
    f64 beta1,
    f64 beta2,
    f64 epsilon,
    u64 t,
    f64 weight_decay
) {
    hipStream_t stream = cast_stream(w);

    // Compute bias correction factors
    f64 bc1 = 1.0 / (1.0 - pow(beta1, static_cast<f64>(t)));
    f64 bc2 = 1.0 / (1.0 - pow(beta2, static_cast<f64>(t)));

    switch (id) {
        case DTYPE_F32: {
            f32* p = static_cast<f32*>(param);
            const f32* g = static_cast<const f32*>(grad);
            f32* m_ptr = static_cast<f32*>(m);
            f32* v_ptr = static_cast<f32*>(v);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(p, g, m_ptr, v_ptr));
            auto out = thrust::make_zip_iterator(thrust::make_tuple(p, m_ptr, v_ptr));
            thrust::transform(
                thrust::hip::par.on(stream),
                in, in + n, out,
                AdamFunctor<f32>{
                    static_cast<f32>(lr),
                    static_cast<f32>(beta1),
                    static_cast<f32>(beta2),
                    static_cast<f32>(epsilon),
                    static_cast<f32>(bc1),
                    static_cast<f32>(bc2),
                    static_cast<f32>(weight_decay)
                }
            );
            return;
        }
        case DTYPE_F64: {
            f64* p = static_cast<f64*>(param);
            const f64* g = static_cast<const f64*>(grad);
            f64* m_ptr = static_cast<f64*>(m);
            f64* v_ptr = static_cast<f64*>(v);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(p, g, m_ptr, v_ptr));
            auto out = thrust::make_zip_iterator(thrust::make_tuple(p, m_ptr, v_ptr));
            thrust::transform(
                thrust::hip::par.on(stream),
                in, in + n, out,
                AdamFunctor<f64>{lr, beta1, beta2, epsilon, bc1, bc2, weight_decay}
            );
            return;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for adam");
    }
}

/* ============================================================================
 * F16/BF16 Mixed Precision Support
 *
 * For half precision training, we maintain F32 master weights for numerical
 * stability. The visible weights are in F16/BF16, but optimizer updates are
 * computed in F32 on the master weights, then cast back.
 *
 * This is the industry-standard approach (see AMP, apex, DeepSpeed).
 * ============================================================================ */

#define OPTIM_BLOCK_SIZE 256

/* --------------------------------------------------------------------------
 * SGD with Master Weights (F16/BF16)
 *
 * Updates master weights in F32, then casts result to visible weights.
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * Gradient Accumulation (mixed-type: any grad dtype → F32 accum)
 *
 * accum[i] += scale * (f32)grad[i]
 * -------------------------------------------------------------------------- */

template <typename GradT>
__global__ void grad_accum_kernel(
    f32* accum,
    const GradT* grad,
    len_t n,
    f32 scale
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        accum[i] += scale * static_cast<f32>(grad[i]);
    }
}

extern "C" void hip_optim_grad_accum(
    Dtype grad_dtype,
    StreamHandle w,
    void* accum_f32,
    const void* grad,
    len_t n,
    f64 scale
) {
    hipStream_t stream = cast_stream(w);
    const len_t blocks = (n + OPTIM_BLOCK_SIZE - 1) / OPTIM_BLOCK_SIZE;
    f32 sc = static_cast<f32>(scale);

    switch (grad_dtype) {
        case DTYPE_F32:
            grad_accum_kernel<f32><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f32*>(accum_f32),
                static_cast<const f32*>(grad),
                n, sc);
            break;
        case DTYPE_F64:
            grad_accum_kernel<f64><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f32*>(accum_f32),
                static_cast<const f64*>(grad),
                n, sc);
            break;
        case DTYPE_F16:
            grad_accum_kernel<f16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f32*>(accum_f32),
                static_cast<const f16*>(grad),
                n, sc);
            break;
        case DTYPE_BF16:
            grad_accum_kernel<bf16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f32*>(accum_f32),
                static_cast<const bf16*>(grad),
                n, sc);
            break;
        default:
            SYSTEM_EXIT("Unsupported dtype for grad_accum");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* -------------------------------------------------------------------------- */

template <typename HalfT>
__global__ void sgd_master_kernel(
    HalfT* visible,         // F16/BF16 visible weights (output)
    f32* master,            // F32 master weights (in/out)
    const HalfT* grad,      // F16/BF16 gradients
    len_t n,
    f32 lr
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Cast grad to f32
        f32 g = static_cast<f32>(grad[i]);
        // Update master weights
        master[i] -= lr * g;
        // Cast back to half
        visible[i] = static_cast<HalfT>(master[i]);
    }
}

extern "C" void hip_optim_sgd_master(
    Dtype visible_dtype,
    StreamHandle w,
    void* visible,          // F16/BF16 visible weights
    void* master,           // F32 master weights
    const void* grad,       // F16/BF16 gradients
    len_t n,
    f64 lr
) {
    hipStream_t stream = cast_stream(w);
    const len_t blocks = (n + OPTIM_BLOCK_SIZE - 1) / OPTIM_BLOCK_SIZE;

    switch (visible_dtype) {
        case DTYPE_F16:
            sgd_master_kernel<f16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f16*>(visible),
                static_cast<f32*>(master),
                static_cast<const f16*>(grad),
                n,
                static_cast<f32>(lr)
            );
            break;
        case DTYPE_BF16:
            sgd_master_kernel<bf16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<bf16*>(visible),
                static_cast<f32*>(master),
                static_cast<const bf16*>(grad),
                n,
                static_cast<f32>(lr)
            );
            break;
        default:
            SYSTEM_EXIT("sgd_master only supports F16/BF16 visible weights");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* --------------------------------------------------------------------------
 * Adam with Master Weights (F16/BF16)
 *
 * Moments (m, v) and master weights are in F32 for numerical stability.
 * Gradients are cast from F16/BF16 to F32, update computed, result cast back.
 * -------------------------------------------------------------------------- */

template <typename HalfT>
__global__ void adam_master_kernel(
    HalfT* visible,         // F16/BF16 visible weights (output)
    f32* master,            // F32 master weights (in/out)
    const HalfT* grad,      // F16/BF16 gradients
    f32* m,                 // First moment (F32)
    f32* v,                 // Second moment (F32)
    len_t n,
    f32 lr,
    f32 beta1,
    f32 beta2,
    f32 epsilon,
    f32 bc1,                // 1 / (1 - beta1^t)
    f32 bc2,                // 1 / (1 - beta2^t)
    f32 wd                  // decoupled weight decay
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Cast grad to f32
        f32 g = static_cast<f32>(grad[i]);

        // Update moments
        f32 m_new = beta1 * m[i] + (1.0f - beta1) * g;
        f32 v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = m_new;
        v[i] = v_new;

        // Bias-corrected estimates
        f32 m_hat = m_new * bc1;
        f32 v_hat = v_new * bc2;

        // AdamW: decoupled weight decay + Adam update on master
        master[i] = master[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + epsilon);

        // Cast back to half
        visible[i] = static_cast<HalfT>(master[i]);
    }
}

extern "C" void hip_optim_adam_master(
    Dtype visible_dtype,
    StreamHandle w,
    void* visible,          // F16/BF16 visible weights
    void* master,           // F32 master weights
    const void* grad,       // F16/BF16 gradients
    void* m,                // F32 first moment
    void* v,                // F32 second moment
    len_t n,
    f64 lr,
    f64 beta1,
    f64 beta2,
    f64 epsilon,
    u64 t,
    f64 weight_decay
) {
    hipStream_t stream = cast_stream(w);
    const len_t blocks = (n + OPTIM_BLOCK_SIZE - 1) / OPTIM_BLOCK_SIZE;

    // Compute bias correction factors
    f32 bc1 = 1.0f / (1.0f - powf(static_cast<f32>(beta1), static_cast<f32>(t)));
    f32 bc2 = 1.0f / (1.0f - powf(static_cast<f32>(beta2), static_cast<f32>(t)));
    f32 wd = static_cast<f32>(weight_decay);

    switch (visible_dtype) {
        case DTYPE_F16:
            adam_master_kernel<f16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f16*>(visible),
                static_cast<f32*>(master),
                static_cast<const f16*>(grad),
                static_cast<f32*>(m),
                static_cast<f32*>(v),
                n,
                static_cast<f32>(lr),
                static_cast<f32>(beta1),
                static_cast<f32>(beta2),
                static_cast<f32>(epsilon),
                bc1, bc2, wd
            );
            break;
        case DTYPE_BF16:
            adam_master_kernel<bf16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<bf16*>(visible),
                static_cast<f32*>(master),
                static_cast<const bf16*>(grad),
                static_cast<f32*>(m),
                static_cast<f32*>(v),
                n,
                static_cast<f32>(lr),
                static_cast<f32>(beta1),
                static_cast<f32>(beta2),
                static_cast<f32>(epsilon),
                bc1, bc2, wd
            );
            break;
        default:
            SYSTEM_EXIT("adam_master only supports F16/BF16 visible weights");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* --------------------------------------------------------------------------
 * Initialize Master Weights from Visible Weights
 *
 * Copies F16/BF16 visible weights to F32 master weights.
 * Called once when optimizer is created for half precision params.
 * -------------------------------------------------------------------------- */

template <typename HalfT>
__global__ void init_master_kernel(
    f32* master,
    const HalfT* visible,
    len_t n
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        master[i] = static_cast<f32>(visible[i]);
    }
}

extern "C" void hip_optim_init_master(
    Dtype visible_dtype,
    StreamHandle w,
    void* master,           // F32 master weights (output)
    const void* visible,    // F16/BF16 visible weights (input)
    len_t n
) {
    hipStream_t stream = cast_stream(w);
    const len_t blocks = (n + OPTIM_BLOCK_SIZE - 1) / OPTIM_BLOCK_SIZE;

    switch (visible_dtype) {
        case DTYPE_F16:
            init_master_kernel<f16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f32*>(master),
                static_cast<const f16*>(visible),
                n
            );
            break;
        case DTYPE_BF16:
            init_master_kernel<bf16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f32*>(master),
                static_cast<const bf16*>(visible),
                n
            );
            break;
        default:
            SYSTEM_EXIT("init_master only supports F16/BF16 visible weights");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* --------------------------------------------------------------------------
 * Copy Master Weights to Visible Weights (F32 → F16/BF16)
 *
 * After updating F32 master weights, copies them back to half-precision
 * visible weights. Used by gradient accumulation with master weights.
 * -------------------------------------------------------------------------- */

template <typename HalfT>
__global__ void copy_master_to_visible_kernel(
    HalfT* visible,
    const f32* master,
    len_t n
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        visible[i] = static_cast<HalfT>(master[i]);
    }
}

extern "C" void hip_optim_copy_master_to_visible(
    Dtype visible_dtype,
    StreamHandle w,
    void* visible,          // F16/BF16 visible weights (output)
    const void* master,     // F32 master weights (input)
    len_t n
) {
    hipStream_t stream = cast_stream(w);
    const len_t blocks = (n + OPTIM_BLOCK_SIZE - 1) / OPTIM_BLOCK_SIZE;

    switch (visible_dtype) {
        case DTYPE_F16:
            copy_master_to_visible_kernel<f16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<f16*>(visible),
                static_cast<const f32*>(master),
                n
            );
            break;
        case DTYPE_BF16:
            copy_master_to_visible_kernel<bf16><<<blocks, OPTIM_BLOCK_SIZE, 0, stream>>>(
                static_cast<bf16*>(visible),
                static_cast<const f32*>(master),
                n
            );
            break;
        default:
            SYSTEM_EXIT("copy_master_to_visible only supports F16/BF16 visible weights");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Gradient Norm Clipping (Fully Async GPU Implementation)
 *
 * Clips gradients by L2 norm: if ||g|| > max_norm, g = g * (max_norm / ||g||)
 *
 * Uses thrust which handles all the complexity internally.
 * Note: thrust::transform_reduce does sync to return value to host,
 * but this is unavoidable without cooperative groups or complex multi-kernel schemes.
 * The alternative would be much more complex CUB code.
 * ============================================================================ */

// Functor to square a value (cast to f32 for accumulation)
template <typename T>
struct SquareFunctor {
    __host__ __device__ f32 operator()(const T& x) const {
        return static_cast<f32>(x) * static_cast<f32>(x);
    }
};

// Functor to scale a gradient value
template <typename T>
struct ScaleFunctor {
    f32 scale;
    __host__ __device__ T operator()(const T& x) const {
        return static_cast<T>(static_cast<f32>(x) * scale);
    }
};

// workspace is not used in this implementation (kept for API compatibility)
extern "C" void hip_optim_clip_grad_norm(
    Dtype id,
    StreamHandle w,
    void* grad,
    len_t n,
    f64 max_norm,
    void* workspace
) {
    (void)workspace; // unused
    hipStream_t stream = cast_stream(w);
    f32 max_norm_f = static_cast<f32>(max_norm);

    // Step 1: Compute sum of squares using thrust::transform_reduce
    f32 sum_sq = 0.0f;
    switch (id) {
        case DTYPE_F32: {
            f32* g = static_cast<f32*>(grad);
            sum_sq = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<f32>{},
                0.0f,
                thrust::plus<f32>()
            );
            break;
        }
        case DTYPE_F16: {
            f16* g = static_cast<f16*>(grad);
            sum_sq = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<f16>{},
                0.0f,
                thrust::plus<f32>()
            );
            break;
        }
        case DTYPE_BF16: {
            bf16* g = static_cast<bf16*>(grad);
            sum_sq = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<bf16>{},
                0.0f,
                thrust::plus<f32>()
            );
            break;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for clip_grad_norm");
    }

    // Step 2: Compute scale factor
    f32 norm = sqrtf(sum_sq);
    f32 scale = (norm > max_norm_f && norm > 0.0f) ? (max_norm_f / norm) : 1.0f;

    // Step 3: Apply scale if needed
    if (scale != 1.0f) {
        switch (id) {
            case DTYPE_F32: {
                f32* g = static_cast<f32*>(grad);
                thrust::transform(
                    thrust::hip::par.on(stream),
                    g, g + n, g,
                    ScaleFunctor<f32>{scale}
                );
                break;
            }
            case DTYPE_F16: {
                f16* g = static_cast<f16*>(grad);
                thrust::transform(
                    thrust::hip::par.on(stream),
                    g, g + n, g,
                    ScaleFunctor<f16>{scale}
                );
                break;
            }
            case DTYPE_BF16: {
                bf16* g = static_cast<bf16*>(grad);
                thrust::transform(
                    thrust::hip::par.on(stream),
                    g, g + n, g,
                    ScaleFunctor<bf16>{scale}
                );
                break;
            }
            default:
                break;
        }
    }

    HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Global Gradient Clipping Support
 *
 * These functions split the clipping into two parts for global norm computation:
 * 1. hip_optim_grad_norm_sq: Compute sum of squares for one gradient tensor
 * 2. hip_optim_scale_grad: Scale gradients by a pre-computed factor
 *
 * Usage for global clipping:
 *   total_sq = 0
 *   for each param: total_sq += hip_optim_grad_norm_sq(grad)
 *   total_norm = sqrt(total_sq)
 *   if total_norm > max_norm: scale = max_norm / total_norm
 *   for each param: hip_optim_scale_grad(grad, scale)
 * ============================================================================ */

// Compute sum of squares for a gradient tensor (returns to host)
extern "C" f32 hip_optim_grad_norm_sq(
    Dtype id,
    StreamHandle w,
    const void* grad,
    len_t n
) {
    hipStream_t stream = cast_stream(w);
    f32 sum_sq = 0.0f;

    switch (id) {
        case DTYPE_F32: {
            const f32* g = static_cast<const f32*>(grad);
            sum_sq = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<f32>{},
                0.0f,
                thrust::plus<f32>()
            );
            break;
        }
        case DTYPE_F64: {
            const f64* g = static_cast<const f64*>(grad);
            // Accumulate in f64 for precision, return as f32
            f64 sum_sq_64 = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<f64>{},
                0.0,
                thrust::plus<f64>()
            );
            sum_sq = static_cast<f32>(sum_sq_64);
            break;
        }
        case DTYPE_F16: {
            const f16* g = static_cast<const f16*>(grad);
            sum_sq = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<f16>{},
                0.0f,
                thrust::plus<f32>()
            );
            break;
        }
        case DTYPE_BF16: {
            const bf16* g = static_cast<const bf16*>(grad);
            sum_sq = thrust::transform_reduce(
                thrust::hip::par.on(stream),
                g, g + n,
                SquareFunctor<bf16>{},
                0.0f,
                thrust::plus<f32>()
            );
            break;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for grad_norm_sq");
    }

    return sum_sq;
}

// Scale gradients by a factor (for global clipping)
extern "C" void hip_optim_scale_grad(
    Dtype id,
    StreamHandle w,
    void* grad,
    len_t n,
    f32 scale
) {
    hipStream_t stream = cast_stream(w);

    switch (id) {
        case DTYPE_F32: {
            f32* g = static_cast<f32*>(grad);
            thrust::transform(
                thrust::hip::par.on(stream),
                g, g + n, g,
                ScaleFunctor<f32>{scale}
            );
            break;
        }
        case DTYPE_F64: {
            f64* g = static_cast<f64*>(grad);
            thrust::transform(
                thrust::hip::par.on(stream),
                g, g + n, g,
                ScaleFunctor<f64>{scale}
            );
            break;
        }
        case DTYPE_F16: {
            f16* g = static_cast<f16*>(grad);
            thrust::transform(
                thrust::hip::par.on(stream),
                g, g + n, g,
                ScaleFunctor<f16>{scale}
            );
            break;
        }
        case DTYPE_BF16: {
            bf16* g = static_cast<bf16*>(grad);
            thrust::transform(
                thrust::hip::par.on(stream),
                g, g + n, g,
                ScaleFunctor<bf16>{scale}
            );
            break;
        }
        default:
            SYSTEM_EXIT("Unsupported dtype for scale_grad");
    }

    HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Device-side Gradient Norm² Reduction
 *
 * Reduces grad² to a single float on the device (no DtoH sync).
 * Caller must zero *norm_sq_out before launch (hipMemsetAsync).
 * ============================================================================ */

template <typename T>
__global__ void grad_norm_sq_reduce_kernel(
    const T* __restrict__ grad,
    float* __restrict__ norm_sq_out,
    len_t n
) {
    extern __shared__ float sdata[];

    len_t tid = threadIdx.x;
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and square
    float val = 0.0f;
    if (i < n) {
        float g = static_cast<float>(grad[i]);
        val = g * g;
    }
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory
    for (len_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block leader atomically adds to output
    if (tid == 0) {
        atomicAdd(norm_sq_out, sdata[0]);
    }
}

/* --------------------------------------------------------------------------
 * Clipped Adam update kernel
 *
 * Each thread reads the device-side norm², computes scale = max_norm / norm
 * if clipping is needed, then applies scaled gradient to Adam update.
 * -------------------------------------------------------------------------- */

template <typename T>
__global__ void adam_clipped_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ norm_sq_ptr,
    len_t n,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bc1,
    float bc2,
    float max_norm
) {
    len_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = static_cast<float>(grad[i]);

    float norm_sq = *norm_sq_ptr;
    float norm = sqrtf(norm_sq);
    if (norm > max_norm && norm > 0.0f) {
        g *= (max_norm / norm);
    }

    float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    float m_hat = m_new * bc1;
    float v_hat = v_new * bc2;

    float p = static_cast<float>(param[i]);
    p -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    param[i] = static_cast<T>(p);
}

/* --------------------------------------------------------------------------
 * C Interface: Reduce grad norm² to device memory
 * -------------------------------------------------------------------------- */

#define LAUNCH_NORM_REDUCE(T) do { \
    const int threads = OPTIM_BLOCK_SIZE; \
    const int blocks = (n + threads - 1) / threads; \
    size_t smem = threads * sizeof(float); \
    grad_norm_sq_reduce_kernel<T><<<blocks, threads, smem, stream>>>( \
        static_cast<const T*>(grad), \
        static_cast<float*>(norm_sq_out), \
        n \
    ); \
} while(0)

extern "C" void hip_optim_grad_norm_sq_device(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad,
    void* norm_sq_out,
    len_t n
) {
    hipStream_t stream = cast_stream(stream_handle);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_NORM_REDUCE(f32);  break;
        case DTYPE_F64:  LAUNCH_NORM_REDUCE(f64);  break;
        case DTYPE_F16:  LAUNCH_NORM_REDUCE(f16);  break;
        case DTYPE_BF16: LAUNCH_NORM_REDUCE(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for grad_norm_sq_device");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* --------------------------------------------------------------------------
 * C Interface: Clipped Adam update (reads device-side norm)
 * -------------------------------------------------------------------------- */

#define LAUNCH_ADAM_CLIPPED(T) do { \
    const int threads = OPTIM_BLOCK_SIZE; \
    const int blocks_n = (n + threads - 1) / threads; \
    adam_clipped_kernel<T><<<blocks_n, threads, 0, stream>>>( \
        static_cast<T*>(param), \
        static_cast<const T*>(grad), \
        static_cast<float*>(m), \
        static_cast<float*>(v), \
        static_cast<const float*>(norm_sq_ptr), \
        n, \
        static_cast<float>(lr), \
        static_cast<float>(beta1), \
        static_cast<float>(beta2), \
        static_cast<float>(epsilon), \
        bc1, bc2, \
        static_cast<float>(max_norm) \
    ); \
} while(0)

extern "C" void hip_optim_adam_clipped(
    Dtype dtype,
    StreamHandle stream_handle,
    void* param,
    const void* grad,
    void* m,
    void* v,
    const void* norm_sq_ptr,
    len_t n,
    f64 lr,
    f64 beta1,
    f64 beta2,
    f64 epsilon,
    u64 t,
    f64 max_norm
) {
    hipStream_t stream = cast_stream(stream_handle);

    f32 bc1 = 1.0f / (1.0f - powf(static_cast<f32>(beta1), static_cast<f32>(t)));
    f32 bc2 = 1.0f / (1.0f - powf(static_cast<f32>(beta2), static_cast<f32>(t)));

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_ADAM_CLIPPED(f32);  break;
        case DTYPE_F64:  LAUNCH_ADAM_CLIPPED(f64);  break;
        case DTYPE_F16:  LAUNCH_ADAM_CLIPPED(f16);  break;
        case DTYPE_BF16: LAUNCH_ADAM_CLIPPED(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for adam_clipped");
    }
    HIP_ASSERT(hipPeekAtLastError());
}

/* ============================================================================
 * Debug: Check GPU buffer for NaN values
 * ============================================================================ */

extern "C" int hip_optim_debug_check_nan(
    Dtype dtype,
    StreamHandle w,
    const void* data,
    len_t n,
    const char* label
) {
    hipStream_t stream = cast_stream(w);
    HIP_ASSERT(hipStreamSynchronize(stream));

    int nan_count = 0;
    int inf_count = 0;

    if (dtype == DTYPE_BF16) {
        len_t check_n = (n < 16) ? n : 16;
        std::vector<bf16> host_buf(n);
        HIP_ASSERT(hipMemcpy(host_buf.data(), data, n * sizeof(bf16), hipMemcpyDeviceToHost));
        for (len_t i = 0; i < n; i++) {
            float val = static_cast<float>(host_buf[i]);
            if (std::isnan(val)) nan_count++;
            if (std::isinf(val)) inf_count++;
        }
        printf("  [dbg] %s: n=%llu nan=%d inf=%d first8=[", label, (unsigned long long)n, nan_count, inf_count);
        for (len_t i = 0; i < check_n && i < n; i++) {
            printf("%.4f%s", static_cast<float>(host_buf[i]), (i < check_n-1) ? "," : "");
        }
        printf("]\n");
        fflush(stdout);
    } else if (dtype == DTYPE_F32) {
        len_t check_n = (n < 16) ? n : 16;
        std::vector<f32> host_buf(n);
        HIP_ASSERT(hipMemcpy(host_buf.data(), data, n * sizeof(f32), hipMemcpyDeviceToHost));
        for (len_t i = 0; i < n; i++) {
            if (std::isnan(host_buf[i])) nan_count++;
            if (std::isinf(host_buf[i])) inf_count++;
        }
        printf("  [dbg] %s: n=%llu nan=%d inf=%d first8=[", label, (unsigned long long)n, nan_count, inf_count);
        for (len_t i = 0; i < check_n && i < n; i++) {
            printf("%.6f%s", host_buf[i], (i < check_n-1) ? "," : "");
        }
        printf("]\n");
        fflush(stdout);
    }

    return nan_count;
}

#endif /* __NN_OPTIMIZERS_H__ */
