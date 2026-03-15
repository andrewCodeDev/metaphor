//! HIP Device
//!
//! GPU-based device implementation using HIP for AMD GPUs.
//! Best for static computation graphs (inference, fixed training loops).
//!
//! Memory allocated by this device resides in GPU memory and is NOT
//! safe to dereference directly from user code. Use transfer operations
//! to move data between host and device.
//!
//! Port of src/device/cuda_device/cuda_device.c3 for AMD/HIP.

module metaphor::device::hip_device;
import metaphor::device::hip_device::utils;
import metaphor::device::hip_device::hip_graph_unit;
import metaphor::device::hip_device::types;
import metaphor::device::common::jit_cache;
import metaphor::device::device_reference;
import metaphor::graph::execution_sequence;
import metaphor::tensor::tensor_core;
import metaphor::tensor::dense_core;
import metaphor::rtti;
import metaphor::utils::sampling;
import metaphor::graph;
import metaphor::cas;
import metaphor::device::hip_device::compile;
import std::math;
import std::core::mem;
import std::io;

// C API Bindings — Device Lifecycle

extern fn DeviceHandle hip_device_init(uint device_id);
extern fn void hip_device_deinit(DeviceHandle handle);
extern fn void hip_device_handle_sync(DeviceHandle handle);
extern fn int hip_device_check_capture_status(DeviceHandle handle);



// C API Bindings — Graph Capture (legacy single-graph API)

extern fn void hip_device_begin_capture(DeviceHandle handle);
extern fn void hip_device_end_capture(DeviceHandle handle);
extern fn void hip_device_replay(DeviceHandle handle);
extern fn bool hip_device_has_graph(DeviceHandle handle);

// C API Bindings — Per-Observable Graph Cache (legacy)

extern fn bool hip_device_has_cached_graph(DeviceHandle handle, usz id);
extern fn void hip_device_replay_cached_graph(DeviceHandle handle, usz id);
extern fn void hip_device_end_capture_and_cache(DeviceHandle handle, usz id);
extern fn void hip_device_invalidate_cached_graph(DeviceHandle handle, usz id);
extern fn void hip_device_clear_graph_cache(DeviceHandle handle);

// C API Bindings — Owned Graph API

extern fn GraphExecHandle hip_device_capture_to_exec(DeviceHandle handle);
extern fn void hip_device_launch_exec(DeviceHandle handle, GraphExecHandle graph_exec);
extern fn void hip_device_destroy_exec(DeviceHandle handle, GraphExecHandle graph_exec);

// C API Bindings — Plan Cache

extern fn usz hip_device_plan_cache_size(DeviceHandle handle);
extern fn void hip_device_plan_cache_clear(DeviceHandle handle);
extern fn StreamHandle hip_device_get_stream(DeviceHandle handle);
extern fn MiopenHandle hip_device_get_miopen(DeviceHandle handle);

// C API Bindings — Scratch Buffer Management

extern fn usz hip_device_get_max_scratch_size(DeviceHandle handle);
extern fn usz hip_device_get_scratch_capacity(DeviceHandle handle);
extern fn void hip_device_ensure_scratch(DeviceHandle handle);

// C API Bindings — Memory Allocation (pool-based, stream-ordered)

extern fn void* hip_device_alloc(DeviceHandle handle, ulong size);
extern fn void hip_device_free(DeviceHandle handle, void* ptr);
extern fn void hip_device_pool_trim(DeviceHandle handle, ulong min_bytes);
extern fn ulong hip_device_pool_reserved(DeviceHandle handle);
extern fn ulong hip_device_pool_used(DeviceHandle handle);

// C API Bindings — Device Query (static)

extern fn uint hip_device_count();
extern fn ulong hip_device_total_memory(uint device_id);

// C API Bindings — hipTENSOR Operations

extern fn void hip_device_contraction(
	DeviceHandle handle,
	DenseCore* x, char* x_syms,
	DenseCore* y, char* y_syms,
	DenseCore* z, char* z_syms
);

extern fn void hip_device_reduce(
	DeviceHandle handle,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms,
	int reduce_op
);

extern fn void hip_device_permutate(
	DeviceHandle handle,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms
);

extern fn void hip_device_elementwise_binary(
	DeviceHandle handle,
	DenseCore* a, char* a_syms,
	DenseCore* c, char* c_syms,
	DenseCore* d, char* d_syms,
	int op
);

// Plan preparation (compile-time, no execution)
extern fn void hip_device_prepare_contraction(
	DeviceHandle handle,
	DenseCore* x, char* x_syms,
	DenseCore* y, char* y_syms,
	DenseCore* z, char* z_syms
);

extern fn void hip_device_prepare_reduce(
	DeviceHandle handle,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms,
	int reduce_op
);

extern fn void hip_device_prepare_permutate(
	DeviceHandle handle,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms
);

extern fn void hip_device_prepare_elementwise_binary(
	DeviceHandle handle,
	DenseCore* a, char* a_syms,
	DenseCore* c, char* c_syms,
	DenseCore* d, char* d_syms,
	int op
);

// Constants

/// Handle to a captured HIP graph execution instance.
struct GraphExecHandle { void* ptr; }

/// Memory alignment required for optimal hipTENSOR/rocBLAS performance.
const usz ALIGNMENT = 256;

// Enums

/// Graph capture mode for compilation.
enum GraphCaptureMode : char
{
	/// Operations execute directly on the stream (default).
	DISABLED,
	/// Operations are collected into a meta unit for capture/replay.
	ENABLED,
}

// Device Query API (static — no instance required)

/// Returns the number of HIP devices available on the system.
fn uint device_count() {
	return hip_device_count();
}

/// Returns the total global memory (in bytes) for the specified device.
fn ulong total_memory(uint device_id) {
	return hip_device_total_memory(device_id);
}

/// Round up a byte size to the next aligned boundary.
fn usz align_size(usz size, usz alignment) {
	return ((size + alignment - 1) / alignment) * alignment;
}

/// Calculate the aligned stride for batched data.
fn usz aligned_stride(usz batch_size, usz features, usz elem_size) {
	usz raw_stride = batch_size * features * elem_size;
	return align_size(raw_stride, ALIGNMENT);
}

/// Calculate the padding (in elements) needed per batch for alignment.
fn usz padding_elements(usz batch_size, usz features, usz elem_size) {
	usz raw_stride = batch_size * features * elem_size;
	usz aligned = align_size(raw_stride, ALIGNMENT);
	return (aligned - raw_stride) / elem_size;
}

/// Calculate total buffer size for N batches with alignment padding.
fn usz aligned_buffer_size(usz num_batches, usz batch_size, usz features, usz elem_size) {
	return num_batches * aligned_stride(batch_size, features, elem_size);
}

// HipDevice — Instance

struct HipDevice
{
	/// Opaque handle to C++ backend
	DeviceHandle handle;

	/// Device ID
	uint device_id;

	/// Graph capture mode
	GraphCaptureMode graph_capture_mode;

	/// True while stream is in capture mode
	bool capturing;

	/// Device reference for interface dispatch
	DeviceReference device_ref;

	/// Memory tracking (capture-safe — no pool queries)
	usz mem_used_val;
	usz mem_peak_val;

	/// JIT kernel cache — keyed by canonical JitKey, valued by compiled kernel handle
	jit_cache::JitCache{types::KernelHandle} jit_kernel_cache;
}

// Options

struct HipDeviceOptions
{
	uint device_id;
	GraphCaptureMode graph_capture_mode;
}

const HipDeviceOptions DEFAULT_HIP_OPTIONS = {
	.device_id = 0,
	.graph_capture_mode = GraphCaptureMode.DISABLED,
};

// Lifecycle

fn HipDevice hip_device_create(HipDeviceOptions opts) {
	HipDevice dev = {
		.handle = hip_device_init(opts.device_id),
		.device_id = opts.device_id,
		.graph_capture_mode = opts.graph_capture_mode,
	};
	dev.jit_kernel_cache.init();
	return dev;
}

fn void HipDevice.deinit(&self) {
	self.jit_kernel_cache.free();
	hip_device_deinit(self.handle);
}

// Synchronization

fn void HipDevice.sync(&self) @dynamic {
	if (self.capturing) return;
	hip_device_handle_sync(self.handle);
}

fn StreamHandle HipDevice.get_stream(&self) {
	return hip_device_get_stream(self.handle);
}

fn MiopenHandle HipDevice.get_miopen(&self) {
	return hip_device_get_miopen(self.handle);
}

fn DeviceHandle HipDevice.get_handle(&self) {
	return self.handle;
}

// Execution Helpers

/// Prepare a tensor for HIP execution by initializing its sync primitive.
fn void HipDevice.prepare_execute(&self, device_reference::TensorCorePtr core) @dynamic {
	// TODO: Check core.sync.inner variant and init HipSync if needed
}

/// Finalize execution for a tensor by recording the completion event.
fn void HipDevice.finalize_execute(&self, device_reference::TensorCorePtr core) @dynamic {
	// TODO: core.sync.complete();
}

/// Execute an operation on this device.
/// Rebind is handled by the execution sequence before this is called.
fn void HipDevice.execute(&self, any operation, device_reference::TensorCorePtr[] sources, device_reference::TensorCorePtr target) @dynamic {
	Executable e = (Executable)operation;
	e.execute(sources, target);
}



fn void HipDevice.begin_capture(&self) {
	hip_device_begin_capture(self.handle);
}

fn void HipDevice.end_capture(&self) {
	hip_device_end_capture(self.handle);
}

fn void HipDevice.replay(&self) {
	hip_device_replay(self.handle);
}

fn bool HipDevice.has_graph(&self) {
	return hip_device_has_graph(self.handle);
}

// Per-Observable Graph Cache (Legacy)

fn bool HipDevice.has_cached_graph(&self, usz observable_id) {
	return hip_device_has_cached_graph(self.handle, observable_id);
}

fn void HipDevice.replay_cached_graph(&self, usz observable_id) {
	hip_device_replay_cached_graph(self.handle, observable_id);
}

fn void HipDevice.end_capture_and_cache(&self, usz observable_id) {
	hip_device_end_capture_and_cache(self.handle, observable_id);
}

fn void HipDevice.invalidate_cached_graph(&self, usz observable_id) {
	hip_device_invalidate_cached_graph(self.handle, observable_id);
}

fn void HipDevice.clear_graph_cache(&self) {
	hip_device_clear_graph_cache(self.handle);
}

// Owned Graph API

fn GraphExecHandle HipDevice.capture_to_exec(&self) {
	return hip_device_capture_to_exec(self.handle);
}

fn void HipDevice.launch_exec(&self, GraphExecHandle graph_exec) {
	hip_device_launch_exec(self.handle, graph_exec);
}

fn void HipDevice.destroy_exec(&self, GraphExecHandle graph_exec) {
	hip_device_destroy_exec(self.handle, graph_exec);
}

// User-Facing Graph Capture API

/// A captured HIP graph that can be replayed efficiently.
struct HipGraphCapture
{
	HipDevice* device;
	GraphExecHandle graph_exec;
}

fn void HipGraphCapture.replay(&self) {
	self.device.ensure_scratch();
	self.device.launch_exec(self.graph_exec);
}

fn void HipGraphCapture.deinit(&self) {
	self.device.destroy_exec(self.graph_exec);
}

/// Capture all HIP operations in the body into a replayable graph.
///
/// Usage:
///   HipGraphCapture graph = device.@hip_graph()
///   {
///       // ops captured here
///   };
///   graph.replay();
///   defer graph.deinit();
macro HipGraphCapture HipDevice.@hip_graph(&self; @body())
{
	// Pass 1: pre-allocate output tensors + warm JIT caches (runs normally).
	@body();
	self.sync();

	// Pass 2: capture kernel launches only.
	// Alloc/free route to alloc_stream (not captured) via graph_device.cpp.
	self.capturing = true;
	defer self.capturing = false;
	hip_device_begin_capture(self.handle);
	@body();
	GraphExecHandle graph_exec = hip_device_capture_to_exec(self.handle);

	return { .device = self, .graph_exec = graph_exec };
}

// Plan Cache

fn usz HipDevice.plan_cache_size(&self) {
	return hip_device_plan_cache_size(self.handle);
}

fn void HipDevice.plan_cache_clear(&self) {
	hip_device_plan_cache_clear(self.handle);
}

// Scratch Buffer Management

fn usz HipDevice.get_max_scratch_size(&self) {
	return hip_device_get_max_scratch_size(self.handle);
}

fn usz HipDevice.get_scratch_capacity(&self) {
	return hip_device_get_scratch_capacity(self.handle);
}

fn void HipDevice.ensure_scratch(&self) {
	hip_device_ensure_scratch(self.handle);
}

// hipTENSOR Operations

fn void HipDevice.contraction(
	&self,
	DenseCore* x, char* x_syms,
	DenseCore* y, char* y_syms,
	DenseCore* z, char* z_syms
) {
	hip_device_contraction(self.handle, x, x_syms, y, y_syms, z, z_syms);
}

fn void HipDevice.reduce(
	&self,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms,
	int reduce_op
) {
	hip_device_reduce(self.handle, src, src_syms, dst, dst_syms, reduce_op);
}

fn void HipDevice.permutate(
	&self,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms
) {
	hip_device_permutate(self.handle, src, src_syms, dst, dst_syms);
}

fn void HipDevice.hiptensor_elementwise_binary(
	&self,
	DenseCore* a, char* a_syms,
	DenseCore* b, char* b_syms,
	DenseCore* out, char* out_syms,
	int op
) {
	hip_device_elementwise_binary(self.handle, a, a_syms, b, b_syms, out, out_syms, op);
}

// Plan preparation (compile-time, no execution)
fn void HipDevice.prepare_contraction(
	&self,
	DenseCore* x, char* x_syms,
	DenseCore* y, char* y_syms,
	DenseCore* z, char* z_syms
) {
	hip_device_prepare_contraction(self.handle, x, x_syms, y, y_syms, z, z_syms);
}

fn void HipDevice.prepare_reduce(
	&self,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms,
	int reduce_op
) {
	hip_device_prepare_reduce(self.handle, src, src_syms, dst, dst_syms, reduce_op);
}

fn void HipDevice.prepare_permutate(
	&self,
	DenseCore* src, char* src_syms,
	DenseCore* dst, char* dst_syms
) {
	hip_device_prepare_permutate(self.handle, src, src_syms, dst, dst_syms);
}

fn void HipDevice.prepare_elementwise_binary(
	&self,
	DenseCore* a, char* a_syms,
	DenseCore* b, char* b_syms,
	DenseCore* out, char* out_syms,
	int op
) {
	hip_device_prepare_elementwise_binary(self.handle, a, a_syms, b, b_syms, out, out_syms, op);
}

// Memory Management (pool-based, stream-ordered)

fn char[] HipDevice.mem_alloc(&self, Datatype dtype, usz n) @dynamic {
	usz byte_size = n * dtype.size_of();
	if (byte_size == 0) return {};

	// io::eprintfn("[HIP_ALLOC] %d bytes (%d elems × %s) used=%d MB",
	// 	(int)byte_size, (int)n, dtype.name(), (int)(self.mem_used_val / (1024 * 1024)));

	void* ptr = hip_device_alloc(self.handle, (ulong)byte_size);
	assert(ptr != null, "HIP mem_alloc: out of memory");
	self.mem_used_val += byte_size;
	if (self.mem_used_val > self.mem_peak_val) { self.mem_peak_val = self.mem_used_val; }
	return ((char*)ptr)[:byte_size];
}

fn void HipDevice.mem_free(&self, char[] slice) @dynamic {
	if (slice.len == 0) return;

	hip_device_free(self.handle, slice.ptr);

	self.mem_used_val -= slice.len;
}

fn char[] HipDevice.mem_dupe(&self, Datatype dtype, char[] src) @dynamic {
	if (src.len == 0) return {};
	char[] dst = self.mem_alloc(dtype, src.len / dtype.size_of());
	if (dst.len == 0) return {};
	utils::memcpy_dtod(self.handle, dst.ptr, src.ptr, src.len);
	return dst;
}

fn void HipDevice.mem_copy(&self, char[] src, char[] dst) @dynamic {
	utils::memcpy_dtod(self.handle, dst.ptr, src.ptr, src.len);
}

fn void HipDevice.mem_transfer(&self, char[] src, char[] dst, TransferDirection direction) @dynamic {
	switch (direction) {
		case HTOH: dst[:src.len] = src[:src.len];
		case HTOD: utils::memcpy_htod(self.handle, dst.ptr, src.ptr, src.len);
		case DTOH: utils::memcpy_dtoh(self.handle, dst.ptr, src.ptr, src.len);
		case DTOD: utils::memcpy_dtod(self.handle, dst.ptr, src.ptr, src.len);
	}
}

fn void HipDevice.mem_fill_f32(&self, char[] slice, float value) {
	utils::mem_fill_f32(self.handle, slice.ptr, (usz)(slice.len / 4), value);
}

fn void HipDevice.mem_fill_f64(&self, char[] slice, double value) {
	utils::mem_fill_f64(self.handle, slice.ptr, (usz)(slice.len / 8), value);
}

fn void HipDevice.mem_sequence_f32(&self, char[] slice, float initial, float step) {
	utils::mem_sequence_f32(self.handle, slice.ptr, (usz)(slice.len / 4), initial, step);
}

fn void HipDevice.mem_sequence_f64(&self, char[] slice, double initial, double step) {
	utils::mem_sequence_f64(self.handle, slice.ptr, (usz)(slice.len / 8), initial, step);
}

// Memory Pool Info

fn void HipDevice.pool_trim(&self, usz min_bytes_to_keep) {
	hip_device_pool_trim(self.handle, (ulong)min_bytes_to_keep);
}

fn usz HipDevice.mem_reserved(&self) @dynamic {
	return (usz)hip_device_pool_reserved(self.handle);
}

fn usz HipDevice.mem_used(&self) @dynamic {
	return self.mem_used_val;
}

fn usz HipDevice.mem_peak(&self) @dynamic {
	return self.mem_peak_val;
}

// Device Reference

fn ZString HipDevice.name(&self) @dynamic {
	return "hip";
}

fn DeviceReference HipDevice.reference(&self) {
	return { .device = (Device)self };
}

fn void HipDevice.mem_random(&self, Datatype dtype, char[] slice, device_reference::RandType rand_type, device_reference::Scale scale = device_reference::DEFAULT_SCALE) @dynamic
{
	sampling::Prng* rng = sampling::get_rng();
	uint seed = (uint)rng.next();
	usz n = slice.len / dtype.size_of();
	double resolved = scale.resolve();

	if (resolved != 1.0)
	{
		// Scaled normal — used for Kaiming init, custom scales, etc.
		switch (dtype)
		{
			case Datatype.F32: utils::mem_random_scaled_normal_f32(self.handle, slice.ptr, n, resolved, seed);
			case Datatype.F64: utils::mem_random_scaled_normal_f64(self.handle, slice.ptr, n, resolved, seed);
			case Datatype.F16: utils::mem_random_scaled_normal_f16(self.handle, slice.ptr, n, resolved, seed);
			case Datatype.BF16: utils::mem_random_scaled_normal_bf16(self.handle, slice.ptr, n, resolved, seed);
			default: unreachable("mem_random: unsupported dtype for HIP");
		}
	}
	else
	{
		int op = (rand_type == RandType.UNIFORM) ? utils::RAND_UNIFORM : utils::RAND_NORMAL;
		switch (dtype)
		{
			case Datatype.F32: utils::mem_random_f32(self.handle, slice.ptr, n, op, seed);
			case Datatype.F64: utils::mem_random_f64(self.handle, slice.ptr, n, op, seed);
			case Datatype.F16: utils::mem_random_f16(self.handle, slice.ptr, n, op, seed);
			case Datatype.BF16: utils::mem_random_bf16(self.handle, slice.ptr, n, op, seed);
			default: unreachable("mem_random: unsupported dtype for HIP");
		}
	}
}

fn void HipDevice.mem_fill(&self, Datatype dtype, char[] slice, double value) @dynamic
{
	switch (dtype)
	{
		case Datatype.F16: utils::mem_fill_f16(self.handle, slice.ptr, (usz)(slice.len / 2), (float)value);
		case Datatype.BF16: utils::mem_fill_bf16(self.handle, slice.ptr, (usz)(slice.len / 2), (float)value);
		case Datatype.F32: utils::mem_fill_f32(self.handle, slice.ptr, (usz)(slice.len / 4), (float)value);
		case Datatype.F64: utils::mem_fill_f64(self.handle, slice.ptr, (usz)(slice.len / 8), value);
		default: unreachable("mem_fill: unsupported dtype for HIP");
	}
}

fn void HipDevice.mem_sequence(&self, Datatype dtype, char[] dst, double initial, double step) @dynamic
{
	switch (dtype)
	{
		case Datatype.F32:
			utils::mem_sequence_f32(self.handle, dst.ptr, (usz)(dst.len / 4), (float)initial, (float)step);
		case Datatype.F64:
			utils::mem_sequence_f64(self.handle, dst.ptr, (usz)(dst.len / 8), initial, step);
		default: unreachable("mem_sequence: unsupported dtype for HIP");
	}
}

fn void HipDevice.mem_cast(&self, Datatype src_dtype, Datatype dst_dtype, char[] src, char[] dst) @dynamic
{
	usz n = src.len / src_dtype.size_of();
	int src_dt = utils::dtype_from_datatype(src_dtype);
	int dst_dt = utils::dtype_from_datatype(dst_dtype);
	assert(src_dt >= 0, "mem_cast: unsupported source dtype");
	assert(dst_dt >= 0, "mem_cast: unsupported target dtype");

	StreamHandle stream = hip_device_get_stream(self.handle);
	utils::mem_cast(stream, src_dt, dst_dt, src.ptr, dst.ptr, (ulong)n);
}

fn void HipDevice.compile(&self, void* graph_ptr, void* entry_ptr) @dynamic
{
	// Delegate to HIP graph compiler.
	// The HIP compile module takes (HipDevice*, ExecutionSequence*, IREntry*).
	// We extract the execution sequence from the graph.
	metaphor::graph::Graph* g = (metaphor::graph::Graph*)graph_ptr;
	metaphor::cas::IREntry* entry = (metaphor::cas::IREntry*)entry_ptr;
	metaphor::device::hip_device::compile::compile(self, g, entry);
}

fn uint HipDevice.sample(&self, Datatype dtype, char[] logits, usz vocab_size, char[] context, usz context_len, void* config_ptr) @dynamic
{
	// Transfer logits to host, sample there
	usz byte_size = vocab_size * dtype.size_of();
	char* buf_ptr = (char*)mem::malloc(byte_size);

	char[] host_logits = buf_ptr[:byte_size];
	utils::memcpy_dtoh(self.handle, host_logits.ptr, logits.ptr, byte_size);
	hip_device_handle_sync(self.handle);

	// Use host sampling
	sampling::SamplingConfig* config = (sampling::SamplingConfig*)config_ptr;
	uint result;
	switch (dtype)
	{
		case Datatype.F32:
			float[] flogits = ((float*)host_logits.ptr)[:vocab_size];
			uint[] ctx;
			if (context.len > 0 && context_len > 0)
			{
				ctx = ((uint*)context.ptr)[:context_len];
			}
			sampling::Sampler sampler;
			sampler.init((uint)vocab_size, *config);
			result = sampler.sample_with_context(flogits, ctx);
			sampler.deinit();
		default:
			unreachable("sample: only f32 logits supported");
	}

	mem::free(buf_ptr);
	return result;
}

fn uint HipDevice.argmax(&self, Datatype dtype, char[] logits, usz vocab_size) @dynamic
{
	// Transfer logits to host, argmax there
	usz byte_size = vocab_size * dtype.size_of();
	char* buf_ptr = (char*)mem::malloc(byte_size);

	char[] host_logits = buf_ptr[:byte_size];
	utils::memcpy_dtoh(self.handle, host_logits.ptr, logits.ptr, byte_size);
	hip_device_handle_sync(self.handle);

	uint result;
	switch (dtype)
	{
		case Datatype.F32:
			float[] flogits = ((float*)host_logits.ptr)[:vocab_size];
			result = sampling::argmax(flogits);
		case Datatype.F64:
			double[] dlogits = ((double*)host_logits.ptr)[:vocab_size];
			uint max_idx = 0;
			double max_val = dlogits[0];
			for (usz i = 1; i < vocab_size; i++)
			{
				if (dlogits[i] > max_val)
				{
					max_val = dlogits[i];
					max_idx = (uint)i;
				}
			}
			result = max_idx;
		default:
			unreachable("argmax: only f32/f64 logits supported");
	}

	mem::free(buf_ptr);
	return result;
}
