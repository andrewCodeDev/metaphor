/*
 * devices/graph_device.cu - CUDA Graph-based device
 *
 * Captures operations into a CUDA Graph, then replays with minimal overhead.
 * Best for static computation graphs (inference, fixed training loops).
 *
 * Owns all CUDA resources including cuTENSOR plan cache.
 * C3 sees only the opaque CudaDeviceHandle.
 */


#ifndef __DEVICES_GRAPH_DEVICE_CU__
#define __DEVICES_GRAPH_DEVICE_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"
#include "../cutensor/backend.cu"
#include "../cutensor/contraction.cu"
#include "../cutensor/reduce.cu"
#include "../cutensor/permutate.cu"
#include "../cutensor/elementwise_binary.cu"

#include <unordered_map>
#include <variant>

/* ============================================================================
 * Plan Cache Types
 * ============================================================================ */

typedef enum {
    PLAN_CONTRACTION,
    PLAN_REDUCE,
    PLAN_PERMUTATE,
    PLAN_ELEMENTWISE_BINARY,
} PlanType;

typedef std::variant<
    CutensorContractionPlan,
    CutensorReducePlan,
    CutensorPermutatePlan,
    CutensorElementwiseBinaryPlan
> CachedPlan;

/* ============================================================================
 * Plan Cache Key
 *
 * Hashes: operation type + dtype + dimensions + symbols
 * ============================================================================ */

struct PlanCacheKey {
    static constexpr std::size_t MAX_HASH_PARTS = 64;
    len_t parts[MAX_HASH_PARTS];
    std::size_t len = 0;

    PlanCacheKey() = default;

    void append(len_t val) {
        CHECK_INVARIANT(len < MAX_HASH_PARTS, "PlanCacheKey overflow");
        parts[len++] = val;
    }

    void append_array(const len_t* vals, len_t n) {
        for (len_t i = 0; i < n; ++i) append(vals[i]);
    }

    void append_array(const u8* vals, len_t n) {
        for (len_t i = 0; i < n; ++i) append(static_cast<len_t>(vals[i]));
    }

    bool operator==(const PlanCacheKey& other) const {
        if (len != other.len) return false;
        for (std::size_t i = 0; i < len; ++i) {
            if (parts[i] != other.parts[i]) return false;
        }
        return true;
    }

    struct Hash {
        std::size_t operator()(const PlanCacheKey& key) const {
            std::size_t seed = key.len;
            for (std::size_t i = 0; i < key.len; ++i) {
                len_t x = key.parts[i];
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = (x >> 16) ^ x;
                seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
};

/* ============================================================================
 * Plan Cache
 *
 * Caches cuTENSOR plans by operation signature.
 * Plans are created once and reused for same-shape operations.
 * ============================================================================ */

class PlanCache {
private:
    std::unordered_map<PlanCacheKey, CachedPlan, PlanCacheKey::Hash> cache;
    CutensorBackend* backend;
    size_t max_scratch_size = 0;  /* Track max scratch needed across all plans */

public:
    explicit PlanCache(CutensorBackend* b) : backend(b), max_scratch_size(0) {}

    /* Get max scratch size needed by any plan in this cache */
    size_t get_max_scratch_size() const { return max_scratch_size; }

    ~PlanCache() {
        for (auto& [key, plan] : cache) {
            std::visit([](auto& p) {
                if (p.plan) {
                    cutensorDestroyPlan(p.plan);
                }
            }, plan);
        }
    }

    PlanCache(const PlanCache&) = delete;
    PlanCache& operator=(const PlanCache&) = delete;

    /* Get or create contraction plan */
    CutensorContractionPlan* get_contraction(
        Dtype dtype,
        const len_t* x_dims, const u8* x_syms, len_t x_len,
        const len_t* y_dims, const u8* y_syms, len_t y_len,
        const len_t* z_dims, const u8* z_syms, len_t z_len
    ) {
        PlanCacheKey key;
        key.append(PLAN_CONTRACTION);
        key.append(static_cast<len_t>(dtype));
        key.append(x_len);
        key.append_array(x_dims, x_len);
        key.append_array(x_syms, x_len);
        key.append(y_len);
        key.append_array(y_dims, y_len);
        key.append_array(y_syms, y_len);
        key.append(z_len);
        key.append_array(z_dims, z_len);
        key.append_array(z_syms, z_len);

        auto it = cache.find(key);
        if (it != cache.end()) {
            return &std::get<CutensorContractionPlan>(it->second);
        }

        auto plan = cutensor_contraction_plan_create(
            wrap_cutensor(backend), dtype,
            x_dims, x_syms, x_len,
            y_dims, y_syms, y_len,
            z_dims, z_syms, z_len
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<CutensorContractionPlan>(inserted->second);
    }

    /* Get or create reduce plan */
    CutensorReducePlan* get_reduce(
        Dtype dtype,
        const len_t* src_dims, const u8* src_syms, len_t src_len,
        const len_t* dst_dims, const u8* dst_syms, len_t dst_len,
        BinaryOp op
    ) {
        PlanCacheKey key;
        key.append(PLAN_REDUCE);
        key.append(static_cast<len_t>(dtype));
        key.append(static_cast<len_t>(op));
        key.append(src_len);
        key.append_array(src_dims, src_len);
        key.append_array(src_syms, src_len);
        key.append(dst_len);
        key.append_array(dst_dims, dst_len);
        key.append_array(dst_syms, dst_len);

        auto it = cache.find(key);
        if (it != cache.end()) {
            return &std::get<CutensorReducePlan>(it->second);
        }

        auto plan = cutensor_reduce_plan_create(
            wrap_cutensor(backend), dtype,
            src_dims, src_syms, src_len,
            dst_dims, dst_syms, dst_len,
            op
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<CutensorReducePlan>(inserted->second);
    }

    /* Get or create permutate plan */
    CutensorPermutatePlan* get_permutate(
        Dtype dtype,
        const len_t* src_dims, const u8* src_syms, len_t src_len,
        const len_t* dst_dims, const u8* dst_syms, len_t dst_len
    ) {
        PlanCacheKey key;
        key.append(PLAN_PERMUTATE);
        key.append(static_cast<len_t>(dtype));
        key.append(src_len);
        key.append_array(src_dims, src_len);
        key.append_array(src_syms, src_len);
        key.append(dst_len);
        key.append_array(dst_dims, dst_len);
        key.append_array(dst_syms, dst_len);

        auto it = cache.find(key);
        if (it != cache.end()) {
            return &std::get<CutensorPermutatePlan>(it->second);
        }

        auto plan = cutensor_permutate_plan_create(
            wrap_cutensor(backend), dtype,
            src_dims, src_syms, src_len,
            dst_dims, dst_syms, dst_len
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<CutensorPermutatePlan>(inserted->second);
    }

    /* Get or create elementwise binary plan */
    CutensorElementwiseBinaryPlan* get_elementwise_binary(
        Dtype dtype,
        const len_t* a_dims, const u8* a_syms, len_t a_len,
        const len_t* c_dims, const u8* c_syms, len_t c_len,
        const len_t* d_dims, const u8* d_syms, len_t d_len,
        BinaryOp op
    ) {
        PlanCacheKey key;
        key.append(PLAN_ELEMENTWISE_BINARY);
        key.append(static_cast<len_t>(dtype));
        key.append(static_cast<len_t>(op));
        key.append(a_len);
        key.append_array(a_dims, a_len);
        key.append_array(a_syms, a_len);
        key.append(c_len);
        key.append_array(c_dims, c_len);
        key.append_array(c_syms, c_len);
        key.append(d_len);
        key.append_array(d_dims, d_len);
        key.append_array(d_syms, d_len);

        auto it = cache.find(key);
        if (it != cache.end()) {
            return &std::get<CutensorElementwiseBinaryPlan>(it->second);
        }

        auto plan = cutensor_elementwise_binary_plan_create(
            wrap_cutensor(backend), dtype,
            a_dims, a_syms, a_len,
            c_dims, c_syms, c_len,
            d_dims, d_syms, d_len,
            op
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<CutensorElementwiseBinaryPlan>(inserted->second);
    }

    void clear() {
        for (auto& [key, plan] : cache) {
            std::visit([](auto& p) {
                if (p.plan) {
                    cutensorDestroyPlan(p.plan);
                    p.plan = nullptr;
                }
            }, plan);
        }
        cache.clear();
    }

    std::size_t size() const { return cache.size(); }
};

/* ============================================================================
 * Graph Device State
 *
 * Owns all CUDA resources. Only CudaDeviceHandle crosses the Zig boundary.
 * ============================================================================ */

struct GraphDeviceState {
    unsigned device_id;

    /* Core handles */
    cudaStream_t stream;
    cudaStream_t alloc_stream;  /* Separate stream for allocations during capture */
    cublasHandle_t cublas;
    cudnnHandle_t cudnn;
    CutensorBackend* cutensor;
    PlanCache* plan_cache;

    /* Memory pool for stream-ordered allocation */
    cudaMemPool_t mem_pool;

    /* CUDA Graph capture state (for current capture in progress) */
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;


    /* Per-observable graph cache: observable_id -> executable graph */
    std::unordered_map<uintptr_t, cudaGraphExec_t> graph_cache;

    /* Persistent scratch buffer for cuTENSOR operations.
     * Grows as needed but never shrinks - ensures stable addresses for graph capture. */
    void* scratch_buffer;
    size_t scratch_capacity;

    /* Initialization */
    static GraphDeviceState* create(unsigned device_id);
    void destroy();

    /* Stream access */
    cudaStream_t get_stream() { return this->stream; }

    /* Memory allocation (uses pool) */
    void* alloc(size_t size);
    void free(void* ptr);

    /* Persistent scratch buffer for cuTENSOR (grows as needed, never freed until destroy) */
    void* get_scratch(size_t size);

    /* Graph capture (legacy single-graph API) */
    void begin_capture();
    void end_capture();
    void replay();
    bool has_graph() { return this->graph_exec != nullptr; }

    /* Per-observable graph cache API (legacy - use owned graph API instead) */
    bool has_cached_graph(uintptr_t id);
    void replay_cached_graph(uintptr_t id);
    void end_capture_and_cache(uintptr_t id);
    void invalidate_cached_graph(uintptr_t id);
    void clear_graph_cache();

    /* Owned graph API - caller owns the returned handle */
    cudaGraphExec_t capture_to_exec();
    void launch_graph_exec(cudaGraphExec_t exec);
    void destroy_graph_exec(cudaGraphExec_t exec);
};

/* ============================================================================
 * Device Lifecycle
 * ============================================================================ */

GraphDeviceState* GraphDeviceState::create(unsigned device_id) {
    auto* state = new GraphDeviceState();
    state->device_id = device_id;
    state->graph = nullptr;
    state->graph_exec = nullptr;

    state->scratch_buffer = nullptr;
    state->scratch_capacity = 0;

    /* Initialize CUDA context */
    CUdevice device;
    CUcontext context;
    int device_count = 0;

    CUDRIVER_ASSERT(cuInit(0));
    CUDRIVER_ASSERT(cuDeviceGetCount(&device_count));
    CHECK_INVARIANT(device_count > 0, "No CUDA devices found");
    CHECK_INVARIANT((int)device_id < device_count, "Device ID out of range");

    CUDRIVER_ASSERT(cuDeviceGet(&device, device_id));
    // TODO: NULL ctxCreateParams means no execution affinity or CIG (CUDA in Graphics) settings
    CUDRIVER_ASSERT(cuCtxCreate(&context, NULL, 0, device));

    /* Create streams */
    CUDRIVER_ASSERT(cuStreamCreate(&state->stream, CU_STREAM_DEFAULT));
    CUDA_ASSERT(cudaStreamCreate(&state->alloc_stream));

    /* Create memory pool with device's default pool as base */
    CUDA_ASSERT(cudaDeviceGetDefaultMemPool(&state->mem_pool, device_id));

    /* Configure pool: don't release memory back to OS during graph lifetime.
     * Setting release threshold to max prevents automatic trimming.
     * Memory is only released when explicitly requested or on pool destruction. */
    uint64_t release_threshold = UINT64_MAX;
    CUDA_ASSERT(cudaMemPoolSetAttribute(
        state->mem_pool,
        cudaMemPoolAttrReleaseThreshold,
        &release_threshold
    ));

    /* Create cuBLAS handle */
    CUBLAS_ASSERT(cublasCreate(&state->cublas));
    CUBLAS_ASSERT(cublasSetStream(state->cublas, state->stream));

    /* Create cuDNN handle */
    CUDNN_ASSERT(cudnnCreate(&state->cudnn));
    CUDNN_ASSERT(cudnnSetStream(state->cudnn, state->stream));

    /* Create cuTENSOR backend and plan cache */
    state->cutensor = new CutensorBackend(state->stream);
    state->plan_cache = new PlanCache(state->cutensor);

    return state;
}

/* ============================================================================
 * Memory Allocation (pool-based, stream-ordered)
 * ============================================================================ */

void* GraphDeviceState::alloc(size_t size) {
    CHECK_INVARIANT(size > 0, "Zero-size allocation");
    void* ptr = nullptr;
    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(this->stream, &status);
    if (status == cudaStreamCaptureStatusActive) {
        CUDA_ASSERT(cudaMallocFromPoolAsync(&ptr, size, this->mem_pool, this->alloc_stream));
        CUDA_ASSERT(cudaStreamSynchronize(this->alloc_stream));
    } else {
        CUDA_ASSERT(cudaMallocFromPoolAsync(&ptr, size, this->mem_pool, this->stream));
    }
    CHECK_INVARIANT(ptr != nullptr, "Allocation returned null");
    return ptr;
}

void GraphDeviceState::free(void* ptr) {
    CHECK_INVARIANT(ptr != nullptr, "Freeing null pointer");
    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(this->stream, &status);
    if (status == cudaStreamCaptureStatusActive) {
        CUDA_ASSERT(cudaFreeAsync(ptr, this->alloc_stream));
    } else {
        CUDA_ASSERT(cudaFreeAsync(ptr, this->stream));
    }
}

/* Get persistent scratch buffer, growing if needed (only outside capture).
 * The buffer is never freed until device destruction - this ensures stable
 * addresses for CUDA graph capture/replay.
 *
 * IMPORTANT: During capture, the buffer must already be large enough.
 * Run a warmup pass before capture to ensure scratch is pre-allocated. */
void* GraphDeviceState::get_scratch(size_t size) {
    if (size == 0) return nullptr;

    if (size > this->scratch_capacity) {
        /* Check if we're in stream capture mode */
        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(this->stream, &capture_status);

        if (capture_status == cudaStreamCaptureStatusActive) {
            /* Cannot grow during capture - addresses would be invalidated */
            fprintf(stderr, "FATAL: scratch buffer needs to grow during CUDA graph capture.\n");
            fprintf(stderr, "  Required: %zu bytes, Available: %zu bytes\n", size, this->scratch_capacity);
            fprintf(stderr, "  Ensure scratch is allocated before capture (call ensure_scratch after compilation).\n");
            abort();
        }

        /* Not capturing - safe to grow */
        if (this->scratch_buffer) {
            /* Free via pool (async) */
            CUDA_ASSERT(cudaFreeAsync(this->scratch_buffer, this->stream));
        }
        /* Allocate from pool (stream-ordered) */
        CUDA_ASSERT(cudaMallocFromPoolAsync(&this->scratch_buffer, size, this->mem_pool, this->stream));
        this->scratch_capacity = size;
        LOG_DEBUG("get_scratch: grew buffer to %zu bytes at %p\n", size, this->scratch_buffer);
    }

    return this->scratch_buffer;
}

void GraphDeviceState::destroy() {
    /* Synchronize streams before cleanup */
    CUDA_ASSERT(cudaStreamSynchronize(this->stream));
    CUDA_ASSERT(cudaStreamSynchronize(this->alloc_stream));

    /* Clear per-observable graph cache */
    clear_graph_cache();

    /* Destroy legacy single graph if exists */
    if (this->graph_exec) {
        CUDA_ASSERT(cudaGraphExecDestroy(this->graph_exec));
    }
    if (this->graph) {
        CUDA_ASSERT(cudaGraphDestroy(this->graph));
    }

    /* Free persistent scratch buffer (pool-allocated) */
    if (this->scratch_buffer) {
        CUDA_ASSERT(cudaFreeAsync(this->scratch_buffer, this->stream));
        this->scratch_buffer = nullptr;
        this->scratch_capacity = 0;
    }

    /* Trim memory pool - release unused memory back to OS */
    CUDA_ASSERT(cudaMemPoolTrimTo(this->mem_pool, 0));

    /* Destroy handles (plan cache before cutensor backend) */
    delete this->plan_cache;
    delete this->cutensor;
    CUDNN_ASSERT(cudnnDestroy(this->cudnn));
    CUBLAS_ASSERT(cublasDestroy(this->cublas));
    CUDRIVER_ASSERT(cuStreamDestroy(this->stream));

    delete this;
}

/* ============================================================================
 * Graph Capture
 * ============================================================================ */

void GraphDeviceState::begin_capture() {
    CUDA_ASSERT(cudaStreamBeginCapture(this->stream, cudaStreamCaptureModeRelaxed));
}

void GraphDeviceState::end_capture() {
    /* Destroy previous graph if exists */
    if (this->graph_exec) {
        CUDA_ASSERT(cudaGraphExecDestroy(this->graph_exec));
        this->graph_exec = nullptr;
    }
    if (this->graph) {
        CUDA_ASSERT(cudaGraphDestroy(this->graph));
        this->graph = nullptr;
    }

    CUDA_ASSERT(cudaStreamEndCapture(this->stream, &this->graph));
    CUDA_ASSERT(cudaGraphInstantiate(&this->graph_exec, this->graph, NULL, NULL, 0));
}

void GraphDeviceState::replay() {
    CHECK_INVARIANT(this->graph_exec != nullptr, "No graph to replay");
    CUDA_ASSERT(cudaGraphLaunch(this->graph_exec, this->stream));
}

/* ============================================================================
 * Per-Observable Graph Cache
 * ============================================================================ */

bool GraphDeviceState::has_cached_graph(uintptr_t id) {
    return graph_cache.find(id) != graph_cache.end();
}

void GraphDeviceState::replay_cached_graph(uintptr_t id) {
    auto it = graph_cache.find(id);
    CHECK_INVARIANT(it != graph_cache.end(), "No cached graph for observable");
    CUDA_ASSERT(cudaGraphLaunch(it->second, this->stream));
}

void GraphDeviceState::end_capture_and_cache(uintptr_t id) {
    /* End capture */
    cudaGraph_t captured_graph;
    CUDA_ASSERT(cudaStreamEndCapture(this->stream, &captured_graph));

    /* Instantiate executable graph */
    cudaGraphExec_t exec;
    CUDA_ASSERT(cudaGraphInstantiate(&exec, captured_graph, NULL, NULL, 0));

    /* Destroy the graph (we keep only the executable) */
    CUDA_ASSERT(cudaGraphDestroy(captured_graph));

    /* Remove old cached graph if exists */
    auto it = graph_cache.find(id);
    if (it != graph_cache.end()) {
        CUDA_ASSERT(cudaGraphExecDestroy(it->second));
        graph_cache.erase(it);
    }

    /* Cache the new executable graph */
    graph_cache[id] = exec;

    /* Launch it immediately */
    CUDA_ASSERT(cudaGraphLaunch(exec, this->stream));
}

void GraphDeviceState::invalidate_cached_graph(uintptr_t id) {
    auto it = graph_cache.find(id);
    if (it != graph_cache.end()) {
        CUDA_ASSERT(cudaGraphExecDestroy(it->second));
        graph_cache.erase(it);
    }
}

void GraphDeviceState::clear_graph_cache() {
    for (auto& [id, exec] : graph_cache) {
        CUDA_ASSERT(cudaGraphExecDestroy(exec));
    }
    graph_cache.clear();
}

/* End stream capture and return executable graph handle.
 * Caller owns the returned handle and must call destroy_graph_exec to free. */
cudaGraphExec_t GraphDeviceState::capture_to_exec() {
    cudaGraph_t captured_graph;
    CUDA_ASSERT(cudaStreamEndCapture(this->stream, &captured_graph));

    size_t num_nodes = 0;
    CUDA_ASSERT(cudaGraphGetNodes(captured_graph, nullptr, &num_nodes));
    fprintf(stderr, "[cuda_graph] Captured graph has %zu nodes\n", num_nodes);

    cudaGraphExec_t exec;
    CUDA_ASSERT(cudaGraphInstantiate(&exec, captured_graph, NULL, NULL, 0));

    cudaGraphDebugDotPrint(captured_graph, "/tmp/metaphor_graph.dot", cudaGraphDebugDotFlagsVerbose);

    CUDA_ASSERT(cudaGraphDestroy(captured_graph));

    return exec;
}

/* Launch an executable graph on this device's stream */
void GraphDeviceState::launch_graph_exec(cudaGraphExec_t exec) {
    CHECK_INVARIANT(exec != nullptr, "Null graph exec handle");
    CUDA_ASSERT(cudaGraphLaunch(exec, this->stream));
}

/* Destroy an executable graph (caller gives up ownership) */
void GraphDeviceState::destroy_graph_exec(cudaGraphExec_t exec) {
    if (exec != nullptr) {
        CUDA_ASSERT(cudaGraphExecDestroy(exec));
    }
}

/* ============================================================================
 * Unwrap Helper
 * ============================================================================ */

static inline GraphDeviceState* unwrap_device(CudaDeviceHandle h) {
    return static_cast<GraphDeviceState*>(h.ptr);
}

static inline CudaDeviceHandle wrap_device(GraphDeviceState* s) {
    return {.ptr = s};
}

/* ============================================================================
 * Extern C API - Device Lifecycle
 * ============================================================================ */

extern "C" CudaDeviceHandle cuda_device_init(unsigned device_id) {
    return wrap_device(GraphDeviceState::create(device_id));
}

extern "C" void cuda_device_deinit(CudaDeviceHandle handle) {
    unwrap_device(handle)->destroy();
}

extern "C" int cuda_device_check_capture_status(CudaDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    cudaStreamCaptureStatus cap_status;
    cudaStreamGetCaptureInfo(state->stream, &cap_status, nullptr);
    return (int)cap_status;
}

extern "C" void cuda_device_handle_sync(CudaDeviceHandle handle) {
    CUDRIVER_ASSERT(cuStreamSynchronize(unwrap_device(handle)->stream));
}

/* ============================================================================
 * Extern C API - Memory Allocation (pool-based, stream-ordered)
 * ============================================================================ */

extern "C" DevicePtr cuda_device_alloc(CudaDeviceHandle handle, len_t size) {
    void* ptr = unwrap_device(handle)->alloc(size);
    return {.ptr = ptr};
}

extern "C" void cuda_device_free(CudaDeviceHandle handle, DevicePtr ptr) {
    unwrap_device(handle)->free(ptr.ptr);
}

extern "C" void cuda_device_pool_trim(CudaDeviceHandle handle, len_t min_bytes_to_keep) {
    auto* state = unwrap_device(handle);
    CUDA_ASSERT(cudaStreamSynchronize(state->stream));
    CUDA_ASSERT(cudaMemPoolTrimTo(state->mem_pool, min_bytes_to_keep));
}

extern "C" len_t cuda_device_pool_reserved(CudaDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    size_t reserved = 0;
    CUDA_ASSERT(cudaMemPoolGetAttribute(
        state->mem_pool,
        cudaMemPoolAttrReservedMemCurrent,
        &reserved
    ));
    return reserved;
}

extern "C" len_t cuda_device_pool_used(CudaDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    size_t used = 0;
    CUDA_ASSERT(cudaMemPoolGetAttribute(
        state->mem_pool,
        cudaMemPoolAttrUsedMemCurrent,
        &used
    ));
    return used;
}

/* ============================================================================
 * Extern C API - Graph Capture
 * ============================================================================ */

extern "C" void cuda_device_begin_capture(CudaDeviceHandle handle) {
    unwrap_device(handle)->begin_capture();
}

extern "C" void cuda_device_end_capture(CudaDeviceHandle handle) {
    unwrap_device(handle)->end_capture();
}

extern "C" void cuda_device_replay(CudaDeviceHandle handle) {
    unwrap_device(handle)->replay();
}

extern "C" bool cuda_device_has_graph(CudaDeviceHandle handle) {
    return unwrap_device(handle)->has_graph();
}



/* ============================================================================
 * Extern C API - Per-Observable Graph Cache (Legacy)
 *
 * Each observable can have its own cached CUDA graph, keyed by pointer.
 * Consider using capture mode API instead for proper ExecutionUnit lifecycle.
 * ============================================================================ */

extern "C" bool cuda_device_has_cached_graph(CudaDeviceHandle handle, uintptr_t id) {
    return unwrap_device(handle)->has_cached_graph(id);
}

extern "C" void cuda_device_replay_cached_graph(CudaDeviceHandle handle, uintptr_t id) {
    unwrap_device(handle)->replay_cached_graph(id);
}

extern "C" void cuda_device_end_capture_and_cache(CudaDeviceHandle handle, uintptr_t id) {
    unwrap_device(handle)->end_capture_and_cache(id);
}

extern "C" void cuda_device_invalidate_cached_graph(CudaDeviceHandle handle, uintptr_t id) {
    unwrap_device(handle)->invalidate_cached_graph(id);
}

extern "C" void cuda_device_clear_graph_cache(CudaDeviceHandle handle) {
    unwrap_device(handle)->clear_graph_cache();
}

/* ============================================================================
 * Extern C API - Owned Graph Handles
 *
 * These APIs return ownership of CUDA graph handles to the caller.
 * The caller must call destroy_graph_exec when done.
 * ============================================================================ */

extern "C" void* cuda_device_capture_to_exec(CudaDeviceHandle handle) {
    return unwrap_device(handle)->capture_to_exec();
}

extern "C" void cuda_device_launch_exec(CudaDeviceHandle handle, void* graph_exec) {
    unwrap_device(handle)->launch_graph_exec(static_cast<cudaGraphExec_t>(graph_exec));
}

extern "C" void cuda_device_destroy_exec(CudaDeviceHandle handle, void* graph_exec) {
    unwrap_device(handle)->destroy_graph_exec(static_cast<cudaGraphExec_t>(graph_exec));
}

/* ============================================================================
 * Extern C API - Handle Accessors (stream, cublas, cudnn for other ops)
 * ============================================================================ */

extern "C" void* cuda_device_get_stream(CudaDeviceHandle handle) {
    return unwrap_device(handle)->stream;
}

extern "C" void* cuda_device_get_cublas(CudaDeviceHandle handle) {
    return unwrap_device(handle)->cublas;
}

extern "C" void* cuda_device_get_cudnn(CudaDeviceHandle handle) {
    return unwrap_device(handle)->cudnn;
}

/* ============================================================================
 * Extern C API - Plan Cache Stats (for debugging/monitoring)
 * ============================================================================ */

extern "C" len_t cuda_device_plan_cache_size(CudaDeviceHandle handle) {
    return static_cast<len_t>(unwrap_device(handle)->plan_cache->size());
}

extern "C" void cuda_device_plan_cache_clear(CudaDeviceHandle handle) {
    unwrap_device(handle)->plan_cache->clear();
}

/* ============================================================================
 * Extern C API - Scratch Buffer Management
 *
 * Call ensure_scratch() after compilation (plan creation) but before capture.
 * This pre-allocates the scratch buffer based on max size needed by plans.
 * ============================================================================ */

extern "C" len_t cuda_device_get_max_scratch_size(CudaDeviceHandle handle) {
    return static_cast<len_t>(unwrap_device(handle)->plan_cache->get_max_scratch_size());
}

extern "C" len_t cuda_device_get_scratch_capacity(CudaDeviceHandle handle) {
    return static_cast<len_t>(unwrap_device(handle)->scratch_capacity);
}

extern "C" void cuda_device_ensure_scratch(CudaDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    size_t needed = state->plan_cache->get_max_scratch_size();
    if (needed > 0) {
        state->get_scratch(needed);  /* Allocates if not already large enough */
    }
}

/* ============================================================================
 * Extern C API - cuTENSOR Operations
 *
 * Individual operations that can be called between begin_capture/end_capture.
 * Each takes DenseCore pointers plus einsum symbols (operation-specific).
 * ============================================================================ */

extern "C" void cuda_device_contraction(
    CudaDeviceHandle handle,
    const DenseCore* x, const u8* x_syms,
    const DenseCore* y, const u8* y_syms,
    DenseCore* z, const u8* z_syms
) {
    auto* state = unwrap_device(handle);

    LOG_DEBUG("contraction called");
    LOG_TENSOR("x", x, x_syms);
    LOG_TENSOR("y", y, y_syms);
    LOG_TENSOR("z", z, z_syms);

    Dtype x_dtype = dense_core_dtype(x);

    /* Get or create plan from cache */
    CutensorContractionPlan* plan = state->plan_cache->get_contraction(
        x_dtype,
        x->shape.buffer, x_syms, x->shape.len,
        y->shape.buffer, y_syms, y->shape.len,
        z->shape.buffer, z_syms, z->shape.len
    );

    /* Get persistent scratch buffer (grows if needed, never freed until device destruction) */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalars for contraction: C = alpha * A @ B + beta * C */
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    const void* alpha = (x_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;
    const void* beta = (x_dtype == DTYPE_F64) ? (const void*)&beta_d : (const void*)&beta_f;

    /* Execute contraction */
    cutensor_contract(
        wrap_cutensor(state->cutensor),
        *plan,
        x->data, y->data, z->data,
        scratch,
        alpha, beta
    );

    /* Scratch is persistent - not freed here */
}

/* Prepare contraction plan without execution (for pre-compilation) */
extern "C" void cuda_device_prepare_contraction(
    CudaDeviceHandle handle,
    const DenseCore* x, const u8* x_syms,
    const DenseCore* y, const u8* y_syms,
    const DenseCore* z, const u8* z_syms
) {
    auto* state = unwrap_device(handle);

    LOG_DEBUG("prepare_contraction called");
    LOG_TENSOR("x", x, x_syms);
    LOG_TENSOR("y", y, y_syms);
    LOG_TENSOR("z", z, z_syms);

    /* Create plan in cache (if not already cached) */
    state->plan_cache->get_contraction(
        dense_core_dtype(x),
        x->shape.buffer, x_syms, x->shape.len,
        y->shape.buffer, y_syms, y->shape.len,
        z->shape.buffer, z_syms, z->shape.len
    );
    /* Plan is now cached and max_scratch_size is updated */
}

/* Scalar reduction kernel: reads T_in, accumulates in T_acc.
   T_acc must support atomicAdd (float or double). Output is T_acc. */
template<typename T_in, typename T_acc>
__global__ void scalar_reduce_sum(const T_in* __restrict__ input, T_acc* output, len_t n) {
    __shared__ T_acc sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    T_acc sum = T_acc(0);
    while (i < n) {
        sum += T_acc(input[i]);
        i += blockDim.x * gridDim.x;
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
        atomicAdd(output, sdata[0]);
    }
}

/* Copy single element with type conversion */
template<typename T_out, typename T_acc>
__global__ void scalar_convert(const T_acc* input, T_out* output) {
    if (threadIdx.x == 0) {
        *output = T_out(*input);
    }
}

extern "C" void cuda_device_reduce(
    CudaDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    DenseCore* dst, const u8* dst_syms,
    BinaryOp reduce_op
) {
    auto* state = unwrap_device(handle);
    Dtype src_dtype = dense_core_dtype(src);

    /* For scalar reductions (dst is 0-dim), use a simple kernel since cuTENSOR
     * doesn't support 0-dimensional output. */
    if (dst->shape.len == 0 && reduce_op == BINARY_ADD) {
        len_t n = src->num_elements;
        int threads = 256;
        int blocks = std::min((n + threads - 1) / threads, (len_t)256);

        LOG_DEBUG("Scalar reduction: src numel=%lu, dst ptr=%p, src ptr=%p\n",
            n, dst->data, src->data);

        // Check for null pointers
        if (dst->data == nullptr) {
            LOG_ERROR("Scalar reduction: dst pointer is null!\n");
            return;
        }
        if (src->data == nullptr) {
            LOG_ERROR("Scalar reduction: src pointer is null!\n");
            return;
        }

        switch (src_dtype) {
            case DTYPE_F64:
                CUDA_ASSERT(cudaMemsetAsync(dst->data, 0, sizeof(double), state->stream));
                scalar_reduce_sum<double, double><<<blocks, threads, 0, state->stream>>>(
                    (const double*)src->data, (double*)dst->data, n);
                break;
            case DTYPE_F32:
                CUDA_ASSERT(cudaMemsetAsync(dst->data, 0, sizeof(float), state->stream));
                scalar_reduce_sum<float, float><<<blocks, threads, 0, state->stream>>>(
                    (const float*)src->data, (float*)dst->data, n);
                break;
            case DTYPE_BF16: {
                float* scratch = (float*)state->get_scratch(sizeof(float));
                CUDA_ASSERT(cudaMemsetAsync(scratch, 0, sizeof(float), state->stream));
                scalar_reduce_sum<bf16, float><<<blocks, threads, 0, state->stream>>>(
                    (const bf16*)src->data, scratch, n);
                scalar_convert<bf16, float><<<1, 1, 0, state->stream>>>(scratch, (bf16*)dst->data);
                break;
            }
            case DTYPE_F16: {
                float* scratch = (float*)state->get_scratch(sizeof(float));
                CUDA_ASSERT(cudaMemsetAsync(scratch, 0, sizeof(float), state->stream));
                scalar_reduce_sum<f16, float><<<blocks, threads, 0, state->stream>>>(
                    (const f16*)src->data, scratch, n);
                scalar_convert<f16, float><<<1, 1, 0, state->stream>>>(scratch, (f16*)dst->data);
                break;
            }
            default:
                assert(false && "Scalar reduction: unsupported dtype");
                break;
        }
        return;
    }

    /* Get or create plan from cache */
    CutensorReducePlan* plan = state->plan_cache->get_reduce(
        src_dtype,
        src->shape.buffer, src_syms, src->shape.len,
        dst->shape.buffer, dst_syms, dst->shape.len,
        reduce_op
    );

    /* Get persistent scratch buffer (grows if needed, never freed until device destruction) */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalars for reduction: Y = alpha * reduce(X) + beta * Y */
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    const void* alpha = (src_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;
    const void* beta = (src_dtype == DTYPE_F64) ? (const void*)&beta_d : (const void*)&beta_f;

    /* Execute reduction */
    cutensor_reduce(
        wrap_cutensor(state->cutensor),
        *plan,
        src->data, dst->data,
        scratch,
        alpha, beta
    );
    /* Scratch is persistent - not freed here */
}

/* Prepare reduce plan without execution (for pre-compilation) */
extern "C" void cuda_device_prepare_reduce(
    CudaDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    const DenseCore* dst, const u8* dst_syms,
    BinaryOp reduce_op
) {
    auto* state = unwrap_device(handle);

    LOG_DEBUG("prepare_reduce called");

    /* Scalar reductions don't use cuTENSOR plans */
    if (dst->shape.len == 0) {
        return;
    }

    /* Create plan in cache (if not already cached) */
    state->plan_cache->get_reduce(
        dense_core_dtype(src),
        src->shape.buffer, src_syms, src->shape.len,
        dst->shape.buffer, dst_syms, dst->shape.len,
        reduce_op
    );
    /* Plan is now cached and max_scratch_size is updated */
}

extern "C" void cuda_device_permutate(
    CudaDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    DenseCore* dst, const u8* dst_syms
) {
    auto* state = unwrap_device(handle);
    Dtype src_dtype = dense_core_dtype(src);

    LOG_DEBUG("cuda_device_permutate: src->ndim=%lu, dst->ndim=%lu, dtype=%d",
              src->shape.len, dst->shape.len, src_dtype);
    LOG_DEBUG("  src->data=%p, dst->data=%p", src->data, dst->data);
    for (size_t i = 0; i < src->shape.len; i++) {
        LOG_DEBUG("  src shape[%zu]=%lu, sym=%d", i, src->shape.buffer[i], src_syms[i]);
    }
    for (size_t i = 0; i < dst->shape.len; i++) {
        LOG_DEBUG("  dst shape[%zu]=%lu, sym=%d", i, dst->shape.buffer[i], dst_syms[i]);
    }

    /* Get or create plan from cache */
    CutensorPermutatePlan* plan = state->plan_cache->get_permutate(
        src_dtype,
        src->shape.buffer, src_syms, src->shape.len,
        dst->shape.buffer, dst_syms, dst->shape.len
    );

    LOG_DEBUG("  plan=%p, plan->scratch_len=%lu", (void*)plan, plan->scratch_len);

    /* Get persistent scratch buffer (grows if needed, never freed until device destruction) */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalar for permutation: Y = alpha * permute(X) */
    float alpha_f = 1.0f;
    double alpha_d = 1.0;
    const void* alpha = (src_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;

    LOG_DEBUG("  calling cutensor_permutate with scratch=%p", scratch);

    /* Execute permutation */
    cutensor_permutate(
        wrap_cutensor(state->cutensor),
        *plan,
        src->data, dst->data,
        scratch,
        alpha
    );
    LOG_DEBUG("  permutate completed successfully");
    /* Scratch is persistent - not freed here */
}

/* Prepare permutate plan without execution (for pre-compilation) */
extern "C" void cuda_device_prepare_permutate(
    CudaDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    const DenseCore* dst, const u8* dst_syms
) {
    auto* state = unwrap_device(handle);

    LOG_DEBUG("prepare_permutate called");

    /* Create plan in cache (if not already cached) */
    state->plan_cache->get_permutate(
        dense_core_dtype(src),
        src->shape.buffer, src_syms, src->shape.len,
        dst->shape.buffer, dst_syms, dst->shape.len
    );
    /* Plan is now cached and max_scratch_size is updated */
}

/* ============================================================================
 * Elementwise Binary (cuTENSOR) - handles einsum-style broadcasting
 * ============================================================================ */

extern "C" void cuda_device_elementwise_binary(
    CudaDeviceHandle handle,
    const DenseCore* a, const u8* a_syms,
    const DenseCore* c, const u8* c_syms,
    DenseCore* d, const u8* d_syms,
    BinaryOp op
) {
    auto* state = unwrap_device(handle);
    Dtype a_dtype = dense_core_dtype(a);

    LOG_DEBUG("cuda_device_elementwise_binary: a->ndim=%lu, c->ndim=%lu, d->ndim=%lu, dtype=%d, op=%d",
              a->shape.len, c->shape.len, d->shape.len, a_dtype, op);

    /* Get or create plan from cache */
    CutensorElementwiseBinaryPlan* plan = state->plan_cache->get_elementwise_binary(
        a_dtype,
        a->shape.buffer, a_syms, a->shape.len,
        c->shape.buffer, c_syms, c->shape.len,
        d->shape.buffer, d_syms, d->shape.len,
        op
    );

    LOG_DEBUG("  plan=%p, plan->scratch_len=%lu", (void*)plan, plan->scratch_len);

    /* Get persistent scratch buffer */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalars: D = alpha * A op gamma * C */
    float alpha_f = 1.0f, gamma_f = 1.0f;
    double alpha_d = 1.0, gamma_d = 1.0;
    const void* alpha = (a_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;
    const void* gamma = (a_dtype == DTYPE_F64) ? (const void*)&gamma_d : (const void*)&gamma_f;

    /* Execute elementwise binary */
    cutensor_elementwise_binary(
        wrap_cutensor(state->cutensor),
        *plan,
        a->data, c->data, d->data,
        scratch,
        alpha, gamma
    );
    LOG_DEBUG("  elementwise_binary completed successfully");
}

/* Prepare elementwise binary plan without execution (for pre-compilation) */
extern "C" void cuda_device_prepare_elementwise_binary(
    CudaDeviceHandle handle,
    const DenseCore* a, const u8* a_syms,
    const DenseCore* c, const u8* c_syms,
    const DenseCore* d, const u8* d_syms,
    BinaryOp op
) {
    auto* state = unwrap_device(handle);

    LOG_DEBUG("prepare_elementwise_binary called");

    /* Create plan in cache (if not already cached) */
    state->plan_cache->get_elementwise_binary(
        dense_core_dtype(a),
        a->shape.buffer, a_syms, a->shape.len,
        c->shape.buffer, c_syms, c->shape.len,
        d->shape.buffer, d_syms, d->shape.len,
        op
    );
    /* Plan is now cached and max_scratch_size is updated */
}

#endif /* __DEVICES_GRAPH_DEVICE_CU__ */
