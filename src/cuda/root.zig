//! CUDA C Interop Module
//!
//! This module imports C headers and re-exports types for use by Zig code.
//! The include path is set by the build system.

const c = @cImport({
    @cInclude("interop.h");
});

// Re-export all C types and constants
pub const CudaDeviceHandle = c.CudaDeviceHandle;
pub const DevicePtr = c.DevicePtr;
pub const HostPtr = c.HostPtr;
pub const CudaTensor = c.CudaTensor;
pub const Dtype = c.Dtype;
pub const BinaryOp = c.BinaryOp;
pub const MapOp = c.MapOp;
pub const RandType = c.RandType;
pub const ReductionType = c.ReductionType;
pub const SoftmaxType = c.SoftmaxType;
pub const StreamHandle = c.StreamHandle;
pub const CublasHandle = c.CublasHandle;
pub const CudnnHandle = c.CudnnHandle;
pub const CutensorPlanHandle = c.CutensorPlanHandle;
pub const EventHandle = c.EventHandle;

pub const CUDA_MAX_DIMS = c.CUDA_MAX_DIMS;

// Dtype constants
pub const DTYPE_F32 = c.DTYPE_F32;
pub const DTYPE_F64 = c.DTYPE_F64;
pub const DTYPE_U8 = c.DTYPE_U8;
pub const DTYPE_F16 = c.DTYPE_F16;
pub const DTYPE_BF16 = c.DTYPE_BF16;

// BinaryOp constants
pub const BINARY_ADD = c.BINARY_ADD;
pub const BINARY_MIN = c.BINARY_MIN;
pub const BINARY_MAX = c.BINARY_MAX;
pub const BINARY_MUL = c.BINARY_MUL;

// MapOp constants
pub const MAP_IDENTITY = c.MAP_IDENTITY;
pub const MAP_NEG = c.MAP_NEG;
pub const MAP_ABS = c.MAP_ABS;
pub const MAP_SQR = c.MAP_SQR;
pub const MAP_SQRT = c.MAP_SQRT;
pub const MAP_RECIP = c.MAP_RECIP;
pub const MAP_EXP = c.MAP_EXP;
pub const MAP_LOG = c.MAP_LOG;
pub const MAP_SIN = c.MAP_SIN;
pub const MAP_COS = c.MAP_COS;
pub const MAP_TANH = c.MAP_TANH;
pub const MAP_RELU = c.MAP_RELU;
pub const MAP_HEAVISIDE = c.MAP_HEAVISIDE;
pub const MAP_SIGMOID = c.MAP_SIGMOID;
pub const MAP_GELU = c.MAP_GELU;
pub const MAP_SILU = c.MAP_SILU;
pub const MAP_SOFTPLUS = c.MAP_SOFTPLUS;
pub const MAP_NOT = c.MAP_NOT;
pub const MAP_SIGN = c.MAP_SIGN;

// BinOp constants (for JIT elementwise)
pub const BinOp = c.BinOp;
pub const ChainOp = c.ChainOp;
pub const MAX_FUSED_CHAIN_OPS = c.MAX_FUSED_CHAIN_OPS;
pub const BINOP_ADD = c.BINOP_ADD;
pub const BINOP_SUB = c.BINOP_SUB;
pub const BINOP_MUL = c.BINOP_MUL;
pub const BINOP_DIV = c.BINOP_DIV;
pub const BINOP_MAX = c.BINOP_MAX;
pub const BINOP_MIN = c.BINOP_MIN;
pub const BINOP_EQ = c.BINOP_EQ;
pub const BINOP_NE = c.BINOP_NE;
pub const BINOP_GT = c.BINOP_GT;
pub const BINOP_LT = c.BINOP_LT;
pub const BINOP_GTE = c.BINOP_GTE;
pub const BINOP_LTE = c.BINOP_LTE;
pub const BINOP_AND = c.BINOP_AND;
pub const BINOP_OR = c.BINOP_OR;
pub const BINOP_XOR = c.BINOP_XOR;

// RandType constants
pub const RAND_UNIFORM = c.RAND_UNIFORM;
pub const RAND_NORMAL = c.RAND_NORMAL;

// ReductionType constants
pub const REDUX_NONE = c.REDUX_NONE;
pub const REDUX_MEAN = c.REDUX_MEAN;
pub const REDUX_SUM = c.REDUX_SUM;

// SoftmaxType constants
pub const SMAX_FAST = c.SMAX_FAST;
pub const SMAX_MAX = c.SMAX_MAX;
pub const SMAX_LOG = c.SMAX_LOG;

// ============================================================================
// Event Functions (for synchronization)
// ============================================================================

pub extern "C" fn cuda_event_create() EventHandle;
pub extern "C" fn cuda_event_destroy(handle: EventHandle) void;
pub extern "C" fn cuda_event_record(event: EventHandle, stream: StreamHandle) void;
pub extern "C" fn cuda_event_query(handle: EventHandle) bool;
pub extern "C" fn cuda_event_sync(handle: EventHandle) void;
pub extern "C" fn cuda_stream_wait_event(stream: StreamHandle, event: EventHandle) void;

// ============================================================================
// Memory Cast (dtype conversion)
// ============================================================================

pub extern "C" fn cuda_mem_cast(
    stream: StreamHandle,
    src_dtype: Dtype,
    dst_dtype: Dtype,
    src: *const anyopaque,
    dst: *anyopaque,
    n: u64,
) void;

// ============================================================================
// Slice Operations (strided copy for slice forward/backward)
// ============================================================================

pub extern "C" fn cuda_slice_forward(
    device: CudaDeviceHandle,
    out: *CudaTensor,
    in: *const CudaTensor,
    offsets: [*]const u64,
    steps: [*]const u64,
    ndim: u64,
    stream: StreamHandle,
) void;

pub extern "C" fn cuda_slice_backward(
    device: CudaDeviceHandle,
    out: *CudaTensor,
    grad: *const CudaTensor,
    offsets: [*]const u64,
    steps: [*]const u64,
    ndim: u64,
    stream: StreamHandle,
) void;

// ============================================================================
// Concat/Stack Operations
// ============================================================================

pub extern "C" fn cuda_concat(
    device: CudaDeviceHandle,
    out: *CudaTensor,
    inputs: [*]const DevicePtr,
    num_inputs: u64,
    in_shape: [*]const u64,
    offsets: [*]const u64,
    concat_dim: u64,
    ndim: u64,
    stream: StreamHandle,
) void;

pub extern "C" fn cuda_stack(
    device: CudaDeviceHandle,
    out: *CudaTensor,
    inputs: [*]const DevicePtr,
    num_inputs: u64,
    in_shape: [*]const u64,
    stack_dim: u64,
    in_ndim: u64,
    stream: StreamHandle,
) void;

// ============================================================================
// Gather/Scatter Operations
// ============================================================================

pub extern "C" fn cuda_gather(
    device: CudaDeviceHandle,
    out: *CudaTensor,
    in: *const CudaTensor,
    indices: DevicePtr,
    in_shape: [*]const u64,
    idx_shape: [*]const u64,
    out_shape: [*]const u64,
    gather_dim: u64,
    in_ndim: u64,
    idx_ndim: u64,
    out_ndim: u64,
    numel: u64,
    idx_dtype: Dtype,
    stream: StreamHandle,
) void;

pub extern "C" fn cuda_scatter_add(
    device: CudaDeviceHandle,
    out: *CudaTensor,
    grad: *const CudaTensor,
    indices: DevicePtr,
    out_shape: [*]const u64,
    idx_shape: [*]const u64,
    grad_shape: [*]const u64,
    gather_dim: u64,
    out_ndim: u64,
    idx_ndim: u64,
    grad_ndim: u64,
    grad_numel: u64,
    idx_dtype: Dtype,
    stream: StreamHandle,
) void;

// ============================================================================
// Sampling Operations (for text generation)
// ============================================================================

pub extern "C" fn cuda_sampling_temperature(
    dtype: Dtype,
    stream: StreamHandle,
    logits: *anyopaque,
    n: u64,
    temperature: f32,
) void;

pub extern "C" fn cuda_sampling_repetition_penalty(
    dtype: Dtype,
    stream: StreamHandle,
    logits: *anyopaque,
    context: [*]const u32,
    context_len: u64,
    vocab_size: u64,
    penalty: f32,
) void;

pub extern "C" fn cuda_sampling_softmax(
    dtype: Dtype,
    stream: StreamHandle,
    data: *anyopaque,
    n: u64,
) void;

pub extern "C" fn cuda_sampling_argmax(
    dtype: Dtype,
    stream: StreamHandle,
    data: *const anyopaque,
    n: u64,
    result: *u32,
) void;

pub extern "C" fn cuda_sampling_init_indices(
    stream: StreamHandle,
    indices: [*]u32,
    n: u64,
) void;

pub extern "C" fn cuda_sampling_topk_workspace_size(
    n: u64,
) usize;

pub extern "C" fn cuda_sampling_topk(
    dtype: Dtype,
    stream: StreamHandle,
    probs: *anyopaque,
    indices: [*]u32,
    probs_out: *anyopaque,
    indices_out: [*]u32,
    temp_storage: *anyopaque,
    temp_storage_bytes: usize,
    n: u64,
) void;

pub extern "C" fn cuda_sampling_multinomial(
    dtype: Dtype,
    stream: StreamHandle,
    probs: *const anyopaque,
    indices: [*]const u32,
    n: u64,
    random_val: f32,
    result: *u32,
) void;

pub extern "C" fn cuda_sampling_cumsum(
    dtype: Dtype,
    stream: StreamHandle,
    probs: *const anyopaque,
    cumsum: *anyopaque,
    n: u64,
) void;

pub extern "C" fn cuda_sampling_find_topp_cutoff(
    dtype: Dtype,
    stream: StreamHandle,
    cumsum: *const anyopaque,
    n: u64,
    p: f32,
    cutoff: *u64,
) void;

pub extern "C" fn cuda_sampling_renormalize(
    dtype: Dtype,
    stream: StreamHandle,
    probs: *anyopaque,
    cutoff: u64,
) void;

// ============================================================================
// Optimizer Operations
// ============================================================================

/// SGD update: param -= lr * grad
pub extern "C" fn cuda_optim_sgd(
    dtype: Dtype,
    stream: StreamHandle,
    param: *anyopaque,
    grad: *const anyopaque,
    n: u64,
    lr: f64,
) void;

/// Adam update: Updates param, m, v in-place
pub extern "C" fn cuda_optim_adam(
    dtype: Dtype,
    stream: StreamHandle,
    param: *anyopaque,
    grad: *const anyopaque,
    m: *anyopaque,
    v: *anyopaque,
    n: u64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: u64,
) void;

/// SGD with master weights (for F16/BF16)
/// Updates master weights in F32, casts back to visible weights
pub extern "C" fn cuda_optim_sgd_master(
    visible_dtype: Dtype,
    stream: StreamHandle,
    visible: *anyopaque,
    master: *anyopaque,
    grad: *const anyopaque,
    n: u64,
    lr: f64,
) void;

/// Adam with master weights (for F16/BF16)
/// Updates master weights and moments in F32, casts back to visible weights
pub extern "C" fn cuda_optim_adam_master(
    visible_dtype: Dtype,
    stream: StreamHandle,
    visible: *anyopaque,
    master: *anyopaque,
    grad: *const anyopaque,
    m: *anyopaque,
    v: *anyopaque,
    n: u64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: u64,
) void;

/// Initialize F32 master weights from F16/BF16 visible weights
pub extern "C" fn cuda_optim_init_master(
    visible_dtype: Dtype,
    stream: StreamHandle,
    master: *anyopaque,
    visible: *const anyopaque,
    n: u64,
) void;

/// Clip gradient by L2 norm: if ||g|| > max_norm, g = g * (max_norm / ||g||)
/// workspace is currently unused (kept for API compatibility)
pub extern "C" fn cuda_optim_clip_grad_norm(
    dtype: Dtype,
    stream: StreamHandle,
    grad: *anyopaque,
    n: u64,
    max_norm: f64,
    workspace: ?*anyopaque,
) void;

/// Compute sum of squares for a gradient tensor (for global norm computation).
/// Returns the sum to host (involves sync).
pub extern "C" fn cuda_optim_grad_norm_sq(
    dtype: Dtype,
    stream: StreamHandle,
    grad: *const anyopaque,
    n: u64,
) f32;

/// Scale gradients by a factor (for global clipping after norm is computed).
pub extern "C" fn cuda_optim_scale_grad(
    dtype: Dtype,
    stream: StreamHandle,
    grad: *anyopaque,
    n: u64,
    scale: f32,
) void;

// ============================================================================
// RMS Normalization (with f32 internal accumulation for numerical stability)
// ============================================================================

/// RMS Norm forward pass.
/// x: [num_rows, d_model], weight: [d_model] -> y: [num_rows, d_model], rms_out: [num_rows]
/// rms_out stores the computed RMS values for use in the backward pass.
pub extern "C" fn cuda_nn_rms_norm_forward(
    dtype: Dtype,
    stream: StreamHandle,
    x: *const anyopaque,
    weight: *const anyopaque,
    y: *anyopaque,
    rms_out: *anyopaque,
    num_rows: u64,
    d_model: u64,
    eps: f32,
) void;

/// RMS Norm backward pass for input gradient.
/// Computes grad_x from grad_y, x, weight. Recomputes RMS internally for numerical stability.
pub extern "C" fn cuda_nn_rms_norm_backward_x(
    dtype: Dtype,
    stream: StreamHandle,
    grad_y: *const anyopaque,
    x: *const anyopaque,
    weight: *const anyopaque,
    grad_x: *anyopaque,
    num_rows: u64,
    d_model: u64,
    eps: f32,
) void;

/// Compute RMS values for each row (helper for backward_weight).
/// rms_out is f32 regardless of input dtype.
pub extern "C" fn cuda_nn_rms_norm_compute_rms(
    dtype: Dtype,
    stream: StreamHandle,
    x: *const anyopaque,
    rms_out: *anyopaque,
    num_rows: u64,
    d_model: u64,
    eps: f32,
) void;

/// RMS Norm backward pass for weight gradient.
/// Computes grad_weight by summing over all rows. Requires pre-computed RMS values.
pub extern "C" fn cuda_nn_rms_norm_backward_weight(
    dtype: Dtype,
    stream: StreamHandle,
    grad_y: *const anyopaque,
    x: *const anyopaque,
    rms: *const anyopaque,
    grad_weight: *anyopaque,
    num_rows: u64,
    d_model: u64,
) void;
