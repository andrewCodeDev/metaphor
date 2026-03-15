/*
 * devices/graph_device.cpp - HIP Graph-based device
 *
 * Captures operations into a HIP Graph, then replays with minimal overhead.
 * Best for static computation graphs (inference, fixed training loops).
 *
 * Owns all HIP resources including hipTENSOR plan cache.
 * C3 sees only the opaque HipDeviceHandle.
 */


#ifndef __DEVICES_GRAPH_DEVICE_CU__
#define __DEVICES_GRAPH_DEVICE_CU__
#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"
#include "../hiptensor/backend.cpp"
#include "../hiptensor/contraction.cpp"
#include "../hiptensor/reduce.cpp"
#include "../hiptensor/permutate.cpp"
#include "../hiptensor/elementwise_binary.cpp"
#include "../blas/einsum.cpp"

#include <set>
#include <unordered_map>
#include <variant>

/* Include hipTensor internal types for plan cleanup (mOpDesc, mPref fields) */
#include "data_types.hpp"

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
    HiptensorContractionPlan,
    HiptensorReducePlan,
    HiptensorPermutatePlan,
    HiptensorElementwiseBinaryPlan
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
 * Caches hipTENSOR plans by operation signature.
 * Plans are created once and reused for same-shape operations.
 * ============================================================================ */

class PlanCache {
private:
    std::unordered_map<PlanCacheKey, CachedPlan, PlanCacheKey::Hash> cache;
    HiptensorBackend* backend;
    size_t max_scratch_size = 0;  /* Track max scratch needed across all plans */

public:
    explicit PlanCache(HiptensorBackend* b) : backend(b), max_scratch_size(0) {}

    /* Get max scratch size needed by any plan in this cache */
    size_t get_max_scratch_size() const { return max_scratch_size; }

    ~PlanCache() {
        for (auto& [key, plan] : cache) {
            std::visit([](auto& p) {
                if (p.plan) {
                    /* hiptensorDestroyPlan only frees the plan struct itself.
                     * We must also free the nested operation descriptor,
                     * tensor descriptors, and plan preference since we don't
                     * destroy them after plan creation (the plan stores raw
                     * pointers to all of them). */
                    auto* opDesc = p.plan->mOpDesc;
                    if (opDesc) {
                        /* Collect unique descriptor pointers to avoid double-free
                         * (e.g. contraction sets mDescC == mDescD). */
                        std::set<hiptensorTensorDescriptor_t> descs;
                        if (opDesc->mDescA) descs.insert(opDesc->mDescA);
                        if (opDesc->mDescB) descs.insert(opDesc->mDescB);
                        if (opDesc->mDescC) descs.insert(opDesc->mDescC);
                        if (opDesc->mDescD) descs.insert(opDesc->mDescD);
                        for (auto* d : descs) hiptensorDestroyTensorDescriptor(d);
                        hiptensorDestroyOperationDescriptor(opDesc);
                    }
                    if (p.plan->mPref) hiptensorDestroyPlanPreference(p.plan->mPref);
                    hiptensorDestroyPlan(p.plan);
                }
            }, plan);
        }
    }

    PlanCache(const PlanCache&) = delete;
    PlanCache& operator=(const PlanCache&) = delete;

    /* Get or create contraction plan */
    HiptensorContractionPlan* get_contraction(
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
            return &std::get<HiptensorContractionPlan>(it->second);
        }

        auto plan = hiptensor_contraction_plan_create(
            wrap_hiptensor(backend), dtype,
            x_dims, x_syms, x_len,
            y_dims, y_syms, y_len,
            z_dims, z_syms, z_len
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<HiptensorContractionPlan>(inserted->second);
    }

    /* Get or create reduce plan */
    HiptensorReducePlan* get_reduce(
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
            return &std::get<HiptensorReducePlan>(it->second);
        }

        auto plan = hiptensor_reduce_plan_create(
            wrap_hiptensor(backend), dtype,
            src_dims, src_syms, src_len,
            dst_dims, dst_syms, dst_len,
            op
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<HiptensorReducePlan>(inserted->second);
    }

    /* Get or create permutate plan */
    HiptensorPermutatePlan* get_permutate(
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
            return &std::get<HiptensorPermutatePlan>(it->second);
        }

        auto plan = hiptensor_permutate_plan_create(
            wrap_hiptensor(backend), dtype,
            src_dims, src_syms, src_len,
            dst_dims, dst_syms, dst_len
        );
        /* Track max scratch size */
        if (plan.scratch_len > max_scratch_size) {
            max_scratch_size = plan.scratch_len;
        }
        auto [inserted, _] = cache.emplace(key, plan);
        return &std::get<HiptensorPermutatePlan>(inserted->second);
    }

    /* Get or create elementwise binary plan */
    HiptensorElementwiseBinaryPlan* get_elementwise_binary(
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
            return &std::get<HiptensorElementwiseBinaryPlan>(it->second);
        }

        auto plan = hiptensor_elementwise_binary_plan_create(
            wrap_hiptensor(backend), dtype,
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
        return &std::get<HiptensorElementwiseBinaryPlan>(inserted->second);
    }

    void clear() {
        for (auto& [key, plan] : cache) {
            std::visit([](auto& p) {
                if (p.plan) {
                    hiptensorDestroyPlan(p.plan);
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
 * Owns all HIP resources. Only HipDeviceHandle crosses the C3 boundary.
 * ============================================================================ */

struct GraphDeviceState {
    unsigned device_id;

    /* Core handles */
    hipStream_t stream;
    hipStream_t alloc_stream;  /* Separate stream for allocations during capture */
    hipblasHandle_t hipblas;
    miopenHandle_t miopen;
    HiptensorBackend* hiptensor;
    PlanCache* plan_cache;

    /* Memory pool for stream-ordered allocation */
    hipMemPool_t mem_pool;

    /* HIP Graph capture state (for current capture in progress) */
    hipGraph_t graph;
    hipGraphExec_t graph_exec;


    /* Per-observable graph cache: observable_id -> executable graph */
    std::unordered_map<uintptr_t, hipGraphExec_t> graph_cache;

    /* Persistent scratch buffer for hipTENSOR operations.
     * Grows as needed but never shrinks - ensures stable addresses for graph capture. */
    void* scratch_buffer;
    size_t scratch_capacity;

    /* Initialization */
    static GraphDeviceState* create(unsigned device_id);
    void destroy();

    /* Stream access */
    hipStream_t get_stream() { return this->stream; }

    /* Memory allocation (uses pool) */
    void* alloc(size_t size);
    void free(void* ptr);

    /* Persistent scratch buffer for hipTENSOR (grows as needed, never freed until destroy) */
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
    hipGraphExec_t capture_to_exec();
    void launch_graph_exec(hipGraphExec_t exec);
    void destroy_graph_exec(hipGraphExec_t exec);
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

    /* Initialize HIP device */
    int device_count = 0;

    HIPDRIVER_ASSERT(hipInit(0));
    HIPDRIVER_ASSERT(hipGetDeviceCount(&device_count));
    CHECK_INVARIANT(device_count > 0, "No HIP/ROCm devices found");
    CHECK_INVARIANT((int)device_id < device_count, "Device ID out of range");

    HIP_ASSERT(hipSetDevice(device_id));

    /* Create streams */
    HIPDRIVER_ASSERT(hipStreamCreateWithFlags(&state->stream, hipStreamDefault));
    HIP_ASSERT(hipStreamCreate(&state->alloc_stream));

    /* Create memory pool with device's default pool as base */
    HIP_ASSERT(hipDeviceGetDefaultMemPool(&state->mem_pool, device_id));

    /* Configure pool: don't release memory back to OS during graph lifetime.
     * Setting release threshold to max prevents automatic trimming.
     * Memory is only released when explicitly requested or on pool destruction. */
    uint64_t release_threshold = UINT64_MAX;
    HIP_ASSERT(hipMemPoolSetAttribute(
        state->mem_pool,
        hipMemPoolAttrReleaseThreshold,
        &release_threshold
    ));

    /* Create hipBLAS handle */
    HIPBLAS_ASSERT(hipblasCreate(&state->hipblas));
    HIPBLAS_ASSERT(hipblasSetStream(state->hipblas, state->stream));

    /* Create MIOpen handle */
    MIOPEN_ASSERT(miopenCreate(&state->miopen));
    MIOPEN_ASSERT(miopenSetStream(state->miopen, state->stream));

    /* Create hipTENSOR backend and plan cache */
    state->hiptensor = new HiptensorBackend(state->stream);
    state->plan_cache = new PlanCache(state->hiptensor);

    return state;
}

/* ============================================================================
 * Memory Allocation (pool-based, stream-ordered)
 * ============================================================================ */

void* GraphDeviceState::alloc(size_t size) {
    CHECK_INVARIANT(size > 0, "Zero-size allocation");
    // static size_t total_allocated = 0;
    // total_allocated += size;
    // if (size > 64 * 1024 * 1024) {
    //     fprintf(stderr, "[ALLOC_DEBUG] large alloc: %zu bytes (%.1f MB), total: %.1f MB\n",
    //             size, size / (1024.0 * 1024.0), total_allocated / (1024.0 * 1024.0));
    // }
    void* ptr = nullptr;
    hipStreamCaptureStatus status;
    HIP_ASSERT(hipStreamIsCapturing(this->stream, &status));
    if (status == hipStreamCaptureStatusActive) {
        HIP_ASSERT(hipMallocFromPoolAsync(&ptr, size, this->mem_pool, this->alloc_stream));
        HIP_ASSERT(hipStreamSynchronize(this->alloc_stream));
    } else {
        HIP_ASSERT(hipMallocFromPoolAsync(&ptr, size, this->mem_pool, this->stream));
    }
    CHECK_INVARIANT(ptr != nullptr, "Allocation returned null");
    return ptr;
}

void GraphDeviceState::free(void* ptr) {
    CHECK_INVARIANT(ptr != nullptr, "Freeing null pointer");
    hipStreamCaptureStatus status;
    HIP_ASSERT(hipStreamIsCapturing(this->stream, &status));
    if (status == hipStreamCaptureStatusActive) {
        HIP_ASSERT(hipFreeAsync(ptr, this->alloc_stream));
    } else {
        HIP_ASSERT(hipFreeAsync(ptr, this->stream));
    }
}

/* Get persistent scratch buffer, growing if needed (only outside capture).
 * The buffer is never freed until device destruction - this ensures stable
 * addresses for HIP graph capture/replay.
 *
 * IMPORTANT: During capture, the buffer must already be large enough.
 * Run a warmup pass before capture to ensure scratch is pre-allocated. */
void* GraphDeviceState::get_scratch(size_t size) {
    if (size == 0) return nullptr;

    if (size > this->scratch_capacity) {
        /* Check if we're in stream capture mode */
        hipStreamCaptureStatus capture_status;
        HIP_ASSERT(hipStreamIsCapturing(this->stream, &capture_status));

        if (capture_status == hipStreamCaptureStatusActive) {
            /* Cannot grow during capture - addresses would be invalidated */
            fprintf(stderr, "FATAL: scratch buffer needs to grow during HIP graph capture.\n");
            fprintf(stderr, "  Required: %zu bytes, Available: %zu bytes\n", size, this->scratch_capacity);
            fprintf(stderr, "  Ensure scratch is allocated before capture (call ensure_scratch after compilation).\n");
            abort();
        }

        /* Not capturing - safe to grow */
        if (this->scratch_buffer) {
            /* Free via pool (async) */
            HIP_ASSERT(hipFreeAsync(this->scratch_buffer, this->stream));
        }
        /* Allocate from pool (stream-ordered) */
        HIP_ASSERT(hipMallocFromPoolAsync(&this->scratch_buffer, size, this->mem_pool, this->stream));
        this->scratch_capacity = size;
        LOG_DEBUG("get_scratch: grew buffer to %zu bytes at %p\n", size, this->scratch_buffer);
    }

    return this->scratch_buffer;
}

void GraphDeviceState::destroy() {
    /* Synchronize streams before cleanup */
    HIP_ASSERT(hipStreamSynchronize(this->stream));
    HIP_ASSERT(hipStreamSynchronize(this->alloc_stream));

    /* Clear per-observable graph cache */
    clear_graph_cache();

    /* Destroy legacy single graph if exists */
    if (this->graph_exec) {
        HIP_ASSERT(hipGraphExecDestroy(this->graph_exec));
    }
    if (this->graph) {
        HIP_ASSERT(hipGraphDestroy(this->graph));
    }

    /* Free persistent scratch buffer (pool-allocated) */
    if (this->scratch_buffer) {
        HIP_ASSERT(hipFreeAsync(this->scratch_buffer, this->stream));
        this->scratch_buffer = nullptr;
        this->scratch_capacity = 0;
    }

    /* Trim memory pool - release unused memory back to OS */
    HIP_ASSERT(hipMemPoolTrimTo(this->mem_pool, 0));

    /* Destroy handles (plan cache before hiptensor backend) */
    delete this->plan_cache;
    delete this->hiptensor;
    MIOPEN_ASSERT(miopenDestroy(this->miopen));
    HIPBLAS_ASSERT(hipblasDestroy(this->hipblas));
    HIPDRIVER_ASSERT(hipStreamDestroy(this->stream));

    delete this;
}

/* ============================================================================
 * Graph Capture
 * ============================================================================ */

void GraphDeviceState::begin_capture() {
    HIP_ASSERT(hipStreamBeginCapture(this->stream, hipStreamCaptureModeRelaxed));
}

void GraphDeviceState::end_capture() {
    /* Destroy previous graph if exists */
    if (this->graph_exec) {
        HIP_ASSERT(hipGraphExecDestroy(this->graph_exec));
        this->graph_exec = nullptr;
    }
    if (this->graph) {
        HIP_ASSERT(hipGraphDestroy(this->graph));
        this->graph = nullptr;
    }

    HIP_ASSERT(hipStreamEndCapture(this->stream, &this->graph));
    HIP_ASSERT(hipGraphInstantiate(&this->graph_exec, this->graph, NULL, NULL, 0));
}

void GraphDeviceState::replay() {
    CHECK_INVARIANT(this->graph_exec != nullptr, "No graph to replay");
    HIP_ASSERT(hipGraphLaunch(this->graph_exec, this->stream));
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
    HIP_ASSERT(hipGraphLaunch(it->second, this->stream));
}

void GraphDeviceState::end_capture_and_cache(uintptr_t id) {
    /* End capture */
    hipGraph_t captured_graph;
    HIP_ASSERT(hipStreamEndCapture(this->stream, &captured_graph));

    /* Instantiate executable graph */
    hipGraphExec_t exec;
    HIP_ASSERT(hipGraphInstantiate(&exec, captured_graph, NULL, NULL, 0));

    /* Destroy the graph (we keep only the executable) */
    HIP_ASSERT(hipGraphDestroy(captured_graph));

    /* Remove old cached graph if exists */
    auto it = graph_cache.find(id);
    if (it != graph_cache.end()) {
        HIP_ASSERT(hipGraphExecDestroy(it->second));
        graph_cache.erase(it);
    }

    /* Cache the new executable graph */
    graph_cache[id] = exec;

    /* Launch it immediately */
    HIP_ASSERT(hipGraphLaunch(exec, this->stream));
}

void GraphDeviceState::invalidate_cached_graph(uintptr_t id) {
    auto it = graph_cache.find(id);
    if (it != graph_cache.end()) {
        HIP_ASSERT(hipGraphExecDestroy(it->second));
        graph_cache.erase(it);
    }
}

void GraphDeviceState::clear_graph_cache() {
    for (auto& [id, exec] : graph_cache) {
        HIP_ASSERT(hipGraphExecDestroy(exec));
    }
    graph_cache.clear();
}

/* End stream capture and return executable graph handle.
 * Caller owns the returned handle and must call destroy_graph_exec to free. */
hipGraphExec_t GraphDeviceState::capture_to_exec() {
    hipGraph_t captured_graph;
    HIP_ASSERT(hipStreamEndCapture(this->stream, &captured_graph));

    size_t num_nodes = 0;
    HIP_ASSERT(hipGraphGetNodes(captured_graph, nullptr, &num_nodes));
    fprintf(stderr, "[hip_graph] Captured graph has %zu nodes\n", num_nodes);

    hipGraphExec_t exec;
    HIP_ASSERT(hipGraphInstantiate(&exec, captured_graph, NULL, NULL, 0));

    HIP_ASSERT(hipGraphDebugDotPrint(captured_graph, "/tmp/metaphor_graph.dot", hipGraphDebugDotFlagsVerbose));

    HIP_ASSERT(hipGraphDestroy(captured_graph));

    return exec;
}

/* Launch an executable graph on this device's stream */
void GraphDeviceState::launch_graph_exec(hipGraphExec_t exec) {
    CHECK_INVARIANT(exec != nullptr, "Null graph exec handle");
    HIP_ASSERT(hipGraphLaunch(exec, this->stream));
}

/* Destroy an executable graph (caller gives up ownership) */
void GraphDeviceState::destroy_graph_exec(hipGraphExec_t exec) {
    if (exec != nullptr) {
        HIP_ASSERT(hipGraphExecDestroy(exec));
    }
}

/* ============================================================================
 * Unwrap Helper
 * ============================================================================ */

static inline GraphDeviceState* unwrap_device(HipDeviceHandle h) {
    return static_cast<GraphDeviceState*>(h.ptr);
}

static inline HipDeviceHandle wrap_device(GraphDeviceState* s) {
    return {.ptr = s};
}

/* ============================================================================
 * Extern C API - Device Lifecycle
 * ============================================================================ */

extern "C" HipDeviceHandle hip_device_init(unsigned device_id) {
    return wrap_device(GraphDeviceState::create(device_id));
}

extern "C" void hip_device_deinit(HipDeviceHandle handle) {
    unwrap_device(handle)->destroy();
}

extern "C" int hip_device_check_capture_status(HipDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    hipStreamCaptureStatus cap_status;
    HIP_ASSERT(hipStreamGetCaptureInfo(state->stream, &cap_status, nullptr));
    return (int)cap_status;
}

extern "C" void hip_device_handle_sync(HipDeviceHandle handle) {
    HIPDRIVER_ASSERT(hipStreamSynchronize(unwrap_device(handle)->stream));
}

/* ============================================================================
 * Extern C API - Memory Allocation (pool-based, stream-ordered)
 * ============================================================================ */

extern "C" DevicePtr hip_device_alloc(HipDeviceHandle handle, len_t size) {
    void* ptr = unwrap_device(handle)->alloc(size);
    return {.ptr = ptr};
}

extern "C" void hip_device_free(HipDeviceHandle handle, DevicePtr ptr) {
    unwrap_device(handle)->free(ptr.ptr);
}

extern "C" void hip_device_pool_trim(HipDeviceHandle handle, len_t min_bytes_to_keep) {
    auto* state = unwrap_device(handle);
    HIP_ASSERT(hipStreamSynchronize(state->stream));
    HIP_ASSERT(hipMemPoolTrimTo(state->mem_pool, min_bytes_to_keep));
}

extern "C" len_t hip_device_pool_reserved(HipDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    size_t reserved = 0;
    HIP_ASSERT(hipMemPoolGetAttribute(
        state->mem_pool,
        hipMemPoolAttrReservedMemCurrent,
        &reserved
    ));
    return reserved;
}

extern "C" len_t hip_device_pool_used(HipDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    size_t used = 0;
    HIP_ASSERT(hipMemPoolGetAttribute(
        state->mem_pool,
        hipMemPoolAttrUsedMemCurrent,
        &used
    ));
    return used;
}

/* ============================================================================
 * Extern C API - Graph Capture
 * ============================================================================ */

extern "C" void hip_device_begin_capture(HipDeviceHandle handle) {
    unwrap_device(handle)->begin_capture();
}

extern "C" void hip_device_end_capture(HipDeviceHandle handle) {
    unwrap_device(handle)->end_capture();
}

extern "C" void hip_device_replay(HipDeviceHandle handle) {
    unwrap_device(handle)->replay();
}

extern "C" bool hip_device_has_graph(HipDeviceHandle handle) {
    return unwrap_device(handle)->has_graph();
}



/* ============================================================================
 * Extern C API - Per-Observable Graph Cache (Legacy)
 *
 * Each observable can have its own cached HIP graph, keyed by pointer.
 * Consider using capture mode API instead for proper ExecutionUnit lifecycle.
 * ============================================================================ */

extern "C" bool hip_device_has_cached_graph(HipDeviceHandle handle, uintptr_t id) {
    return unwrap_device(handle)->has_cached_graph(id);
}

extern "C" void hip_device_replay_cached_graph(HipDeviceHandle handle, uintptr_t id) {
    unwrap_device(handle)->replay_cached_graph(id);
}

extern "C" void hip_device_end_capture_and_cache(HipDeviceHandle handle, uintptr_t id) {
    unwrap_device(handle)->end_capture_and_cache(id);
}

extern "C" void hip_device_invalidate_cached_graph(HipDeviceHandle handle, uintptr_t id) {
    unwrap_device(handle)->invalidate_cached_graph(id);
}

extern "C" void hip_device_clear_graph_cache(HipDeviceHandle handle) {
    unwrap_device(handle)->clear_graph_cache();
}

/* ============================================================================
 * Extern C API - Owned Graph Handles
 *
 * These APIs return ownership of HIP graph handles to the caller.
 * The caller must call destroy_graph_exec when done.
 * ============================================================================ */

extern "C" void* hip_device_capture_to_exec(HipDeviceHandle handle) {
    return unwrap_device(handle)->capture_to_exec();
}

extern "C" void hip_device_launch_exec(HipDeviceHandle handle, void* graph_exec) {
    unwrap_device(handle)->launch_graph_exec(static_cast<hipGraphExec_t>(graph_exec));
}

extern "C" void hip_device_destroy_exec(HipDeviceHandle handle, void* graph_exec) {
    unwrap_device(handle)->destroy_graph_exec(static_cast<hipGraphExec_t>(graph_exec));
}

/* ============================================================================
 * Extern C API - Handle Accessors (stream, hipblas, miopen for other ops)
 * ============================================================================ */

extern "C" void* hip_device_get_stream(HipDeviceHandle handle) {
    return unwrap_device(handle)->stream;
}

extern "C" void* hip_device_get_hipblas(HipDeviceHandle handle) {
    return unwrap_device(handle)->hipblas;
}

extern "C" void* hip_device_get_miopen(HipDeviceHandle handle) {
    return unwrap_device(handle)->miopen;
}

/* ============================================================================
 * Extern C API - Plan Cache Stats (for debugging/monitoring)
 * ============================================================================ */

extern "C" len_t hip_device_plan_cache_size(HipDeviceHandle handle) {
    return static_cast<len_t>(unwrap_device(handle)->plan_cache->size());
}

extern "C" void hip_device_plan_cache_clear(HipDeviceHandle handle) {
    unwrap_device(handle)->plan_cache->clear();
}

/* ============================================================================
 * Extern C API - Scratch Buffer Management
 *
 * Call ensure_scratch() after compilation (plan creation) but before capture.
 * This pre-allocates the scratch buffer based on max size needed by plans.
 * ============================================================================ */

extern "C" len_t hip_device_get_max_scratch_size(HipDeviceHandle handle) {
    return static_cast<len_t>(unwrap_device(handle)->plan_cache->get_max_scratch_size());
}

extern "C" len_t hip_device_get_scratch_capacity(HipDeviceHandle handle) {
    return static_cast<len_t>(unwrap_device(handle)->scratch_capacity);
}

extern "C" void hip_device_ensure_scratch(HipDeviceHandle handle) {
    auto* state = unwrap_device(handle);
    size_t needed = state->plan_cache->get_max_scratch_size();
    if (needed > 0) {
        state->get_scratch(needed);  /* Allocates if not already large enough */
    }
}

/* ============================================================================
 * Extern C API - hipTENSOR Operations
 *
 * Individual operations that can be called between begin_capture/end_capture.
 * Each takes DenseCore pointers plus einsum symbols (operation-specific).
 * ============================================================================ */

extern "C" void hip_device_contraction(
    HipDeviceHandle handle,
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

    /* Fast path: try hipBLAS for GEMM-like patterns (batched matmul, etc.)
     * This avoids hipTensor's normalizeTensorModes bug with batch dimensions. */
    if (hip_einsum_can_dispatch(x, x_syms, y, y_syms, z, z_syms)) {
        LOG_DEBUG("contraction: dispatching to hipBLAS GEMM");

        /* Zero output buffer before GEMM.
         * hipBLAS beta=0 may still read C (IEEE 754: 0*NaN=NaN).
         * Uninitialized/recycled output buffers can contain NaN from prior ops. */
        {
            size_t z_bytes = 1;
            for (size_t i = 0; i < z->shape.len; i++) z_bytes *= z->shape.buffer[i];
            switch (x_dtype) {
                case DTYPE_F64:  z_bytes *= sizeof(double); break;
                case DTYPE_BF16: z_bytes *= sizeof(hip_bfloat16); break;
                case DTYPE_F16:  z_bytes *= sizeof(__half); break;
                default:         z_bytes *= sizeof(float); break;
            }
            HIP_ASSERT(hipMemsetAsync(z->data, 0, z_bytes, state->stream));
        }

        hip_einsum_contract(
            state->hipblas, state->stream, x_dtype,
            x, x_syms, y, y_syms, z, z_syms
        );
        return;
    }

    /* Slow path: hipTensor for non-GEMM contractions */
    LOG_DEBUG("contraction: falling back to hipTensor");
    /* Check for deferred GPU errors from previous operations */
    {
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "[HIP FATAL] deferred GPU error BEFORE contraction: %s\n", hipGetErrorString(err));
            fprintf(stderr, "  x: dtype=%d ndim=%lu shape=[", x_dtype, x->shape.len);
            for (size_t i = 0; i < x->shape.len; i++) fprintf(stderr, "%s%lu", i?",":"", x->shape.buffer[i]);
            fprintf(stderr, "] syms=[");
            for (size_t i = 0; i < x->shape.len; i++) fprintf(stderr, "%s%c", i?",":"", (char)x_syms[i]);
            fprintf(stderr, "]\n");
            abort();
        }
    }

    /* Get or create plan from cache */
    HiptensorContractionPlan* plan;
    try {
        plan = state->plan_cache->get_contraction(
            x_dtype,
            x->shape.buffer, x_syms, x->shape.len,
            y->shape.buffer, y_syms, y->shape.len,
            z->shape.buffer, z_syms, z->shape.len
        );
    } catch (const std::exception& e) {
        fprintf(stderr, "[HIP FATAL] contraction plan creation failed: %s\n", e.what());
        fprintf(stderr, "  x: dtype=%d ndim=%lu shape=[", x_dtype, x->shape.len);
        for (size_t i = 0; i < x->shape.len; i++) fprintf(stderr, "%s%lu", i?",":"", x->shape.buffer[i]);
        fprintf(stderr, "] syms=[");
        for (size_t i = 0; i < x->shape.len; i++) fprintf(stderr, "%s%c", i?",":"", (char)x_syms[i]);
        fprintf(stderr, "]\n  y: ndim=%lu shape=[", y->shape.len);
        for (size_t i = 0; i < y->shape.len; i++) fprintf(stderr, "%s%lu", i?",":"", y->shape.buffer[i]);
        fprintf(stderr, "] syms=[");
        for (size_t i = 0; i < y->shape.len; i++) fprintf(stderr, "%s%c", i?",":"", (char)y_syms[i]);
        fprintf(stderr, "]\n  z: ndim=%lu shape=[", z->shape.len);
        for (size_t i = 0; i < z->shape.len; i++) fprintf(stderr, "%s%lu", i?",":"", z->shape.buffer[i]);
        fprintf(stderr, "] syms=[");
        for (size_t i = 0; i < z->shape.len; i++) fprintf(stderr, "%s%c", i?",":"", (char)z_syms[i]);
        fprintf(stderr, "]\n");
        throw;
    }

    /* Get persistent scratch buffer (grows if needed, never freed until device destruction) */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalars for contraction: C = alpha * A @ B + beta * C
     * hipTensor reads alpha/beta according to the compute descriptor type.
     * Contraction uses native half compute on RDNA3 (F32 gives ARCH_MISMATCH). */
    hip_bfloat16 alpha_bf16 = hip_bfloat16(1.0f), beta_bf16 = hip_bfloat16(0.0f);
    __half alpha_f16 = __float2half(1.0f), beta_f16 = __float2half(0.0f);
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    const void* alpha;
    const void* beta;
    switch (x_dtype) {
        case DTYPE_F64:  alpha = &alpha_d;     beta = &beta_d;     break;
        case DTYPE_BF16: alpha = &alpha_bf16;   beta = &beta_bf16;  break;
        case DTYPE_F16:  alpha = &alpha_f16;    beta = &beta_f16;   break;
        default:         alpha = &alpha_f;      beta = &beta_f;     break;
    }

    /* Zero output buffer before contraction.
     * hipTensor on RDNA3 may not skip reading C when beta=0,
     * so 0 * NaN (uninitialized) = NaN would poison the result. */
    {
        size_t z_bytes = 1;
        for (size_t i = 0; i < z->shape.len; i++) z_bytes *= z->shape.buffer[i];
        switch (x_dtype) {
            case DTYPE_F64:  z_bytes *= sizeof(double); break;
            case DTYPE_BF16: z_bytes *= sizeof(hip_bfloat16); break;
            case DTYPE_F16:  z_bytes *= sizeof(__half); break;
            default:         z_bytes *= sizeof(float); break;
        }
        HIP_ASSERT(hipMemsetAsync(z->data, 0, z_bytes, state->stream));
    }

    /* Execute contraction */
    hiptensor_contract(
        wrap_hiptensor(state->hiptensor),
        *plan,
        x->data, y->data, z->data,
        scratch,
        alpha, beta
    );

    /* Scratch is persistent - not freed here */
}

/* Prepare contraction plan without execution (for pre-compilation) */
extern "C" void hip_device_prepare_contraction(
    HipDeviceHandle handle,
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

/* Simple scalar reduction kernel (sum all elements to single value) */
__global__ void scalar_reduce_sum_f32(const float* __restrict__ input, float* output, len_t n) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and sum into shared memory
    float sum = 0.0f;
    while (i < n) {
        sum += input[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void scalar_reduce_sum_bf16(const hip_bfloat16* __restrict__ input, hip_bfloat16* output, len_t n) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    while (i < n) {
        sum += (float)input[i];
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
        /* atomicAdd not available for bf16 — use atomicCAS loop on the
         * underlying uint16_t to add our f32 partial sum. */
        float old_val = (float)(*output);
        float new_val = old_val + sdata[0];
        *output = hip_bfloat16(new_val);
    }
}

__global__ void scalar_reduce_sum_f64(const double* __restrict__ input, double* output, len_t n) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    while (i < n) {
        sum += input[i];
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

extern "C" void hip_device_reduce(
    HipDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    DenseCore* dst, const u8* dst_syms,
    BinaryOp reduce_op
) {
    auto* state = unwrap_device(handle);
    Dtype src_dtype = dense_core_dtype(src);

    /* For scalar reductions (dst is 0-dim), use a simple kernel since hipTENSOR
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

        // Zero the output and launch the appropriate kernel
        switch (src_dtype) {
        case DTYPE_F64:
            HIP_ASSERT(hipMemsetAsync(dst->data, 0, sizeof(double), state->stream));
            scalar_reduce_sum_f64<<<blocks, threads, 0, state->stream>>>(
                (const double*)src->data, (double*)dst->data, n);
            break;
        case DTYPE_BF16:
            HIP_ASSERT(hipMemsetAsync(dst->data, 0, sizeof(hip_bfloat16), state->stream));
            scalar_reduce_sum_bf16<<<blocks, threads, 0, state->stream>>>(
                (const hip_bfloat16*)src->data, (hip_bfloat16*)dst->data, n);
            break;
        default:
            HIP_ASSERT(hipMemsetAsync(dst->data, 0, sizeof(float), state->stream));
            scalar_reduce_sum_f32<<<blocks, threads, 0, state->stream>>>(
                (const float*)src->data, (float*)dst->data, n);
            break;
        }
        return;
    }

    /* Get or create plan from cache */
    HiptensorReducePlan* plan = state->plan_cache->get_reduce(
        src_dtype,
        src->shape.buffer, src_syms, src->shape.len,
        dst->shape.buffer, dst_syms, dst->shape.len,
        reduce_op
    );

    /* Get persistent scratch buffer (grows if needed, never freed until device destruction) */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalars for reduction: Y = alpha * reduce(X) + beta * Y
     * Reduction uses F32 compute for half types, so alpha/beta are
     * always float for non-f64 types. */
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    const void* alpha = (src_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;
    const void* beta  = (src_dtype == DTYPE_F64) ? (const void*)&beta_d  : (const void*)&beta_f;

    /* Execute reduction */
    hiptensor_reduce(
        wrap_hiptensor(state->hiptensor),
        *plan,
        src->data, dst->data,
        scratch,
        alpha, beta
    );
    /* Scratch is persistent - not freed here */
}

/* Prepare reduce plan without execution (for pre-compilation) */
extern "C" void hip_device_prepare_reduce(
    HipDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    const DenseCore* dst, const u8* dst_syms,
    BinaryOp reduce_op
) {
    auto* state = unwrap_device(handle);

    LOG_DEBUG("prepare_reduce called");

    /* Scalar reductions don't use hipTENSOR plans */
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

extern "C" void hip_device_permutate(
    HipDeviceHandle handle,
    const DenseCore* src, const u8* src_syms,
    DenseCore* dst, const u8* dst_syms
) {
    auto* state = unwrap_device(handle);
    Dtype src_dtype = dense_core_dtype(src);

    LOG_DEBUG("hip_device_permutate: src->ndim=%lu, dst->ndim=%lu, dtype=%d",
              src->shape.len, dst->shape.len, src_dtype);
    LOG_DEBUG("  src->data=%p, dst->data=%p", src->data, dst->data);
    for (size_t i = 0; i < src->shape.len; i++) {
        LOG_DEBUG("  src shape[%zu]=%lu, sym=%d", i, src->shape.buffer[i], src_syms[i]);
    }
    for (size_t i = 0; i < dst->shape.len; i++) {
        LOG_DEBUG("  dst shape[%zu]=%lu, sym=%d", i, dst->shape.buffer[i], dst_syms[i]);
    }

    /* Get or create plan from cache */
    HiptensorPermutatePlan* plan = state->plan_cache->get_permutate(
        src_dtype,
        src->shape.buffer, src_syms, src->shape.len,
        dst->shape.buffer, dst_syms, dst->shape.len
    );

    LOG_DEBUG("  plan=%p, plan->scratch_len=%lu", (void*)plan, plan->scratch_len);

    /* Get persistent scratch buffer (grows if needed, never freed until device destruction) */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalar for permutation: Y = alpha * permute(X)
     * Permutation uses F32 compute for half types (hipTensor RDNA3 bug
     * produces zeros with native half compute), so alpha is always float
     * for non-f64 types. */
    float alpha_f = 1.0f;
    double alpha_d = 1.0;
    const void* alpha = (src_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;

    LOG_DEBUG("  calling hiptensor_permutate with scratch=%p", scratch);

    /* Execute permutation */
    hiptensor_permutate(
        wrap_hiptensor(state->hiptensor),
        *plan,
        src->data, dst->data,
        scratch,
        alpha
    );
    LOG_DEBUG("  permutate completed successfully");
    /* Scratch is persistent - not freed here */
}

/* Prepare permutate plan without execution (for pre-compilation) */
extern "C" void hip_device_prepare_permutate(
    HipDeviceHandle handle,
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
 * Elementwise Binary (hipTENSOR) - handles einsum-style broadcasting
 * ============================================================================ */

extern "C" void hip_device_elementwise_binary(
    HipDeviceHandle handle,
    const DenseCore* a, const u8* a_syms,
    const DenseCore* c, const u8* c_syms,
    DenseCore* d, const u8* d_syms,
    BinaryOp op
) {
    auto* state = unwrap_device(handle);
    Dtype a_dtype = dense_core_dtype(a);

    LOG_DEBUG("hip_device_elementwise_binary: a->ndim=%lu, c->ndim=%lu, d->ndim=%lu, dtype=%d, op=%d",
              a->shape.len, c->shape.len, d->shape.len, a_dtype, op);

    /* Get or create plan from cache */
    HiptensorElementwiseBinaryPlan* plan = state->plan_cache->get_elementwise_binary(
        a_dtype,
        a->shape.buffer, a_syms, a->shape.len,
        c->shape.buffer, c_syms, c->shape.len,
        d->shape.buffer, d_syms, d->shape.len,
        op
    );

    LOG_DEBUG("  plan=%p, plan->scratch_len=%lu", (void*)plan, plan->scratch_len);

    /* Get persistent scratch buffer */
    void* scratch = state->get_scratch(plan->scratch_len);

    /* Scalars: D = alpha * A op gamma * C
     * Elementwise binary uses F32 compute for half types, so alpha/gamma
     * are always float for non-f64 types. */
    float alpha_f = 1.0f, gamma_f = 1.0f;
    double alpha_d = 1.0, gamma_d = 1.0;
    const void* alpha = (a_dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;
    const void* gamma = (a_dtype == DTYPE_F64) ? (const void*)&gamma_d : (const void*)&gamma_f;

    /* Execute elementwise binary */
    hiptensor_elementwise_binary(
        wrap_hiptensor(state->hiptensor),
        *plan,
        a->data, c->data, d->data,
        scratch,
        alpha, gamma
    );
    LOG_DEBUG("  elementwise_binary completed successfully");
}

/* Prepare elementwise binary plan without execution (for pre-compilation) */
extern "C" void hip_device_prepare_elementwise_binary(
    HipDeviceHandle handle,
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
