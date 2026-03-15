/*
 * jit/cache.cpp - Kernel cache for compiled JIT kernels
 *
 * Caches compiled kernels by (op, dtype, ndim, contiguous) to avoid
 * recompilation on repeated use.
 */

#ifndef __METAPHOR_JIT_CACHE_H__
#define __METAPHOR_JIT_CACHE_H__

#include "codegen.cpp"
#include "compiler.cpp"
#include <unordered_map>
#include <mutex>

/* ============================================================================
 * Cache Key
 * ============================================================================ */

struct KernelKey {
    KernelType type;
    Dtype dtype;
    uint16_t op;      /* MapOp or BinOp */
    uint8_t ndim;

    bool operator==(const KernelKey& other) const {
        return type == other.type &&
               dtype == other.dtype &&
               op == other.op &&
               ndim == other.ndim;
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const {
        /* Simple hash combining all fields */
        size_t h = static_cast<size_t>(k.type);
        h = h * 31 + static_cast<size_t>(k.dtype);
        h = h * 31 + static_cast<size_t>(k.op);
        h = h * 31 + static_cast<size_t>(k.ndim);
        return h;
    }
};

/* ============================================================================
 * Kernel Cache
 * ============================================================================ */

class JitKernelCache {
public:
    JitKernelCache() : compute_cap_(0) {}

    ~JitKernelCache() {
        clear();
    }

    /*
     * Get or compile a unary elementwise kernel.
     */
    CompiledKernel* get_unary(Dtype dtype, MapOp op, bool contiguous, len_t ndim) {
        KernelKey key = {
            contiguous ? KERNEL_UNARY_CONTIGUOUS : KERNEL_UNARY_STRIDED,
            dtype,
            static_cast<uint16_t>(op),
            static_cast<uint8_t>(ndim)
        };

        /* Check cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                LOG_DEBUG("JitKernelCache: hit for unary op=%d dtype=%d", op, dtype);
                return it->second;
            }
        }

        /* Cache miss - generate and compile */
        LOG_DEBUG("JitKernelCache: miss for unary op=%d dtype=%d, compiling...", op, dtype);

        GeneratedKernel gen = codegen_unary(dtype, op, contiguous, ndim);
        CompiledKernel* kernel = jit_compile(
            gen.source,
            gen.source_len,
            "kernel",
            get_compute_cap()
        );

        if (!kernel) {
            LOG_ERROR("JitKernelCache: failed to compile unary kernel");
            return nullptr;
        }

        /* Insert into cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_[key] = kernel;
        }

        return kernel;
    }

    /*
     * Get or compile a binary elementwise kernel.
     */
    CompiledKernel* get_binary(Dtype dtype, BinOp op, bool contiguous, len_t ndim) {
        KernelKey key = {
            contiguous ? KERNEL_BINARY_CONTIGUOUS : KERNEL_BINARY_STRIDED,
            dtype,
            static_cast<uint16_t>(op),
            static_cast<uint8_t>(ndim)
        };

        /* Check cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                LOG_DEBUG("JitKernelCache: hit for binary op=%d dtype=%d", op, dtype);
                return it->second;
            }
        }

        /* Cache miss - generate and compile */
        LOG_DEBUG("JitKernelCache: miss for binary op=%d dtype=%d, compiling...", op, dtype);

        GeneratedKernel gen = codegen_binary(dtype, op, contiguous, ndim);
        CompiledKernel* kernel = jit_compile(
            gen.source,
            gen.source_len,
            "kernel",
            get_compute_cap()
        );

        if (!kernel) {
            LOG_ERROR("JitKernelCache: failed to compile binary kernel");
            return nullptr;
        }

        /* Insert into cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_[key] = kernel;
        }

        return kernel;
    }

    /*
     * Get or compile a comparison kernel (float/double input -> u8 output).
     * input_dtype: dtype of input tensors (f32 or f64)
     */
    CompiledKernel* get_comparison(Dtype input_dtype, BinOp op, bool contiguous, len_t ndim) {
        KernelKey key = {
            contiguous ? KERNEL_COMPARISON_CONTIGUOUS : KERNEL_COMPARISON_STRIDED,
            input_dtype,  /* Key by input dtype */
            static_cast<uint16_t>(op),
            static_cast<uint8_t>(ndim)
        };

        /* Check cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                LOG_DEBUG("JitKernelCache: hit for comparison op=%d input_dtype=%d", op, input_dtype);
                return it->second;
            }
        }

        /* Cache miss - generate and compile */
        LOG_DEBUG("JitKernelCache: miss for comparison op=%d input_dtype=%d, compiling...", op, input_dtype);

        GeneratedKernel gen = codegen_comparison(input_dtype, op, contiguous, ndim);
        CompiledKernel* kernel = jit_compile(
            gen.source,
            gen.source_len,
            "kernel",
            get_compute_cap()
        );

        if (!kernel) {
            LOG_ERROR("JitKernelCache: failed to compile comparison kernel");
            return nullptr;
        }

        /* Insert into cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_[key] = kernel;
        }

        return kernel;
    }

    /*
     * Get or compile a logical kernel (u8 input -> u8 output).
     */
    CompiledKernel* get_logical(BinOp op, bool contiguous, len_t ndim) {
        KernelKey key = {
            contiguous ? KERNEL_LOGICAL_CONTIGUOUS : KERNEL_LOGICAL_STRIDED,
            DTYPE_U8,  /* Always u8 */
            static_cast<uint16_t>(op),
            static_cast<uint8_t>(ndim)
        };

        /* Check cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                LOG_DEBUG("JitKernelCache: hit for logical op=%d", op);
                return it->second;
            }
        }

        /* Cache miss - generate and compile */
        LOG_DEBUG("JitKernelCache: miss for logical op=%d, compiling...", op);

        GeneratedKernel gen = codegen_logical(op, contiguous, ndim);
        CompiledKernel* kernel = jit_compile(
            gen.source,
            gen.source_len,
            "kernel",
            get_compute_cap()
        );

        if (!kernel) {
            LOG_ERROR("JitKernelCache: failed to compile logical kernel");
            return nullptr;
        }

        /* Insert into cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_[key] = kernel;
        }

        return kernel;
    }

    /*
     * Get or compile a select kernel (ternary: u8 mask, T true, T false -> T).
     */
    CompiledKernel* get_select(Dtype dtype, bool contiguous, len_t ndim) {
        KernelKey key = {
            contiguous ? KERNEL_SELECT_CONTIGUOUS : KERNEL_SELECT_STRIDED,
            dtype,
            0,  /* No op for select */
            static_cast<uint8_t>(ndim)
        };

        /* Check cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                LOG_DEBUG("JitKernelCache: hit for select dtype=%d", dtype);
                return it->second;
            }
        }

        /* Cache miss - generate and compile */
        LOG_DEBUG("JitKernelCache: miss for select dtype=%d, compiling...", dtype);

        GeneratedKernel gen = codegen_select(dtype, contiguous, ndim);
        CompiledKernel* kernel = jit_compile(
            gen.source,
            gen.source_len,
            "kernel",
            get_compute_cap()
        );

        if (!kernel) {
            LOG_ERROR("JitKernelCache: failed to compile select kernel");
            return nullptr;
        }

        /* Insert into cache */
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_[key] = kernel;
        }

        return kernel;
    }

    /*
     * Compile an inplace chain kernel (not cached - each chain is unique).
     * Caller is responsible for freeing the returned kernel.
     *
     * ops: array of binary operations in execution order
     * num_ops: number of operations (>= 1)
     */
    CompiledKernel* compile_inplace_chain(
        Dtype dtype,
        const BinOp* ops,
        size_t num_ops,
        bool contiguous,
        len_t ndim
    ) {
        LOG_DEBUG("JitKernelCache: compiling inplace chain with %zu ops, dtype=%d", num_ops, dtype);

        InplaceChainKernel gen = codegen_inplace_chain(dtype, ops, num_ops, contiguous, ndim);
        CompiledKernel* kernel = jit_compile(
            gen.source,
            gen.source_len,
            "kernel",
            get_compute_cap()
        );

        if (!kernel) {
            LOG_ERROR("JitKernelCache: failed to compile inplace chain kernel");
            return nullptr;
        }

        return kernel;
    }

    /*
     * Clear all cached kernels.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : cache_) {
            jit_kernel_free(pair.second);
        }
        cache_.clear();
    }

    /*
     * Get cache size.
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

private:
    int get_compute_cap() {
        if (compute_cap_ == 0) {
            compute_cap_ = get_compute_capability();
        }
        return compute_cap_;
    }

    mutable std::mutex mutex_;
    std::unordered_map<KernelKey, CompiledKernel*, KernelKeyHash> cache_;
    int compute_cap_;
};

/* ============================================================================
 * Global Cache Instance
 * ============================================================================ */

static JitKernelCache* g_jit_cache = nullptr;

static JitKernelCache* jit_cache_get() {
    if (!g_jit_cache) {
        g_jit_cache = new JitKernelCache();
    }
    return g_jit_cache;
}

static void jit_cache_destroy() {
    if (g_jit_cache) {
        delete g_jit_cache;
        g_jit_cache = nullptr;
    }
}

#endif /* __METAPHOR_JIT_CACHE_H__ */
