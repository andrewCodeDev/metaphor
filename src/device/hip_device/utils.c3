module metaphor::device::hip_device::utils;
import metaphor::rtti;
import metaphor::device::common::operations;
import metaphor::tensor::dense_core;

// Typed Handles — thin wrappers for type safety at the C boundary

struct DeviceHandle { void* ptr; }
struct StreamHandle { void* ptr; }
struct RocblasHandle { void* ptr; }
struct MiopenHandle  { void* ptr; }
struct EventHandle   { void* ptr; }

const DeviceHandle NULL_DEVICE  = { .ptr = null };
const StreamHandle NULL_STREAM  = { .ptr = null };
const MiopenHandle NULL_MIOPEN  = { .ptr = null };
const EventHandle  NULL_EVENT   = { .ptr = null };

// C Dtype constants matching interop.h

const int DTYPE_F32  = 0;
const int DTYPE_F64  = 1;
const int DTYPE_U8   = 2;
const int DTYPE_F16  = 3;
const int DTYPE_BF16 = 4;

/// Binary reduction op constants matching interop.h
const int BINARY_ADD = 0;
const int BINARY_MIN = 1;
const int BINARY_MAX = 2;
const int BINARY_MUL = 3;

/// Reduction type constants matching interop.h
const int REDUX_NONE = 0;
const int REDUX_MEAN = 1;
const int REDUX_SUM  = 2;

/// Random distribution type constants
const int RAND_UNIFORM = 0;
const int RAND_NORMAL  = 1;

// Dtype Mapping

/// Convert rtti::Datatype to HIP Dtype int.
fn int dtype_from_datatype(rtti::Datatype dtype) {
	switch (dtype) {
		case F32:  return DTYPE_F32;
		case F64:  return DTYPE_F64;
		case F16:  return DTYPE_F16;
		case BF16: return DTYPE_BF16;
		case BOOL: return DTYPE_U8;
		case U8:   return DTYPE_U8;
		default:   unreachable("dtype_from_datatype: unsupported dtype");
	}
}

// Memory Allocation

extern fn void* hip_alloc(DeviceHandle device, usz size);
extern fn void hip_free(DeviceHandle device, void* ptr);

fn void* alloc(DeviceHandle device, usz size) {
	if (size == 0) { return null; }
	return hip_alloc(device, size);
}

fn void free(DeviceHandle device, void* ptr) {
	if (ptr == null) { return; }
	hip_free(device, ptr);
}

// Memory Transfer

extern fn void hip_memcpy_htod(DeviceHandle device, void* dst, void* src, usz size);
extern fn void hip_memcpy_dtoh(DeviceHandle device, void* dst, void* src, usz size);
extern fn void hip_memcpy_dtod(DeviceHandle device, void* dst, void* src, usz size);

fn void memcpy_htod(DeviceHandle device, void* dst, void* src, usz size) {
	if (size == 0) { return; }
	hip_memcpy_htod(device, dst, src, size);
}

fn void memcpy_dtoh(DeviceHandle device, void* dst, void* src, usz size) {
	if (size == 0) { return; }
	hip_memcpy_dtoh(device, dst, src, size);
}

fn void memcpy_dtod(DeviceHandle device, void* dst, void* src, usz size) {
	if (size == 0) { return; }
	hip_memcpy_dtod(device, dst, src, size);
}

// Memory Fill / Sequence / Random / Cast

extern fn void hip_mem_fill(DeviceHandle device, int dtype, void* data, usz n, void* value);
extern fn void hip_mem_broadcast_scalar(DeviceHandle device, int dtype, void* src, void* dst, usz n);
extern fn void hip_mem_sequence(DeviceHandle device, int dtype, void* data, usz n, void* init, void* step);
extern fn void hip_mem_random(DeviceHandle device, int dtype, void* data, usz n, int op, uint seed);
extern fn void hip_mem_random_scaled_normal(DeviceHandle device, int dtype, void* data, usz n, double scale, uint seed);
extern fn void hip_mem_cast(StreamHandle stream, int src_dtype, int dst_dtype, void* src, void* dst, ulong n);

fn void mem_fill_f32(DeviceHandle device, void* data, usz n, float value) {
	hip_mem_fill(device, DTYPE_F32, data, n, &value);
}

fn void mem_fill_f64(DeviceHandle device, void* data, usz n, double value) {
	hip_mem_fill(device, DTYPE_F64, data, n, &value);
}

fn void mem_fill_f16(DeviceHandle device, void* data, usz n, float value) {
	hip_mem_fill(device, DTYPE_F16, data, n, &value);
}

fn void mem_fill_bf16(DeviceHandle device, void* data, usz n, float value) {
	hip_mem_fill(device, DTYPE_BF16, data, n, &value);
}

fn void mem_sequence_f32(DeviceHandle device, void* data, usz n, float initial, float step) {
	hip_mem_sequence(device, DTYPE_F32, data, n, &initial, &step);
}

fn void mem_sequence_f64(DeviceHandle device, void* data, usz n, double initial, double step) {
	hip_mem_sequence(device, DTYPE_F64, data, n, &initial, &step);
}

fn void mem_random_f32(DeviceHandle device, void* data, usz n, int op, uint seed) {
	hip_mem_random(device, DTYPE_F32, data, n, op, seed);
}

fn void mem_random_f64(DeviceHandle device, void* data, usz n, int op, uint seed) {
	hip_mem_random(device, DTYPE_F64, data, n, op, seed);
}

fn void mem_random_f16(DeviceHandle device, void* data, usz n, int op, uint seed) {
	hip_mem_random(device, DTYPE_F16, data, n, op, seed);
}

fn void mem_random_scaled_normal_f32(DeviceHandle device, void* data, usz n, double scale, uint seed) {
	hip_mem_random_scaled_normal(device, DTYPE_F32, data, n, scale, seed);
}

fn void mem_random_scaled_normal_f64(DeviceHandle device, void* data, usz n, double scale, uint seed) {
	hip_mem_random_scaled_normal(device, DTYPE_F64, data, n, scale, seed);
}

fn void mem_random_scaled_normal_f16(DeviceHandle device, void* data, usz n, double scale, uint seed) {
	hip_mem_random_scaled_normal(device, DTYPE_F16, data, n, scale, seed);
}

fn void mem_random_bf16(DeviceHandle device, void* data, usz n, int op, uint seed) {
	hip_mem_random(device, DTYPE_BF16, data, n, op, seed);
}

fn void mem_random_scaled_normal_bf16(DeviceHandle device, void* data, usz n, double scale, uint seed) {
	hip_mem_random_scaled_normal(device, DTYPE_BF16, data, n, scale, seed);
}

fn void mem_cast(StreamHandle stream, int src_dtype, int dst_dtype, void* src, void* dst, ulong n) {
	hip_mem_cast(stream, src_dtype, dst_dtype, src, dst, n);
}

// HIP Synchronization

extern fn EventHandle hip_event_create();
extern fn void hip_event_destroy(EventHandle event);
extern fn void hip_event_record(EventHandle event, StreamHandle stream);
extern fn void hip_event_sync(EventHandle event);
extern fn bool hip_event_query(EventHandle event);

/// HIP event-based synchronization primitive.
/// Uses HIP events for efficient GPU synchronization without busy-waiting.
struct HipSync
{
	EventHandle event;
	StreamHandle stream;
}

fn HipSync hip_sync_init(StreamHandle stream) {
	return { .event = hip_event_create(), .stream = stream };
}

fn void HipSync.deinit(&self) {
	if (self.event.ptr != null) {
		hip_event_destroy(self.event);
		self.event = NULL_EVENT;
	}
}

fn void HipSync.begin(&self) {
	// No-op for HIP — work is tracked by events
}

fn void HipSync.complete(&self) {
	if (self.event.ptr != null && self.stream.ptr != null) {
		hip_event_record(self.event, self.stream);
	}
}

fn void HipSync.await_completion(&self) {
	if (self.event.ptr != null) {
		hip_event_sync(self.event);
	}
}

fn bool HipSync.is_ready(&self) {
	if (self.event.ptr == null) { return true; }
	return hip_event_query(self.event);
}

fn void HipSync.reset(&self) {
	if (self.event.ptr != null) {
		hip_event_destroy(self.event);
	}
	self.event = hip_event_create();
}

// JIT Elementwise Operations

// UnaryOp/BinaryOp enums match interop.h MapOp/BinOp by value — cast directly.

extern fn void jit_elementwise_unary(DeviceHandle device, DenseCore* out, DenseCore* input, int op, StreamHandle stream);
extern fn void jit_elementwise_binary(DeviceHandle device, DenseCore* out, DenseCore* a, DenseCore* b, int op, StreamHandle stream);
extern fn void jit_elementwise_select(DeviceHandle device, DenseCore* out, DenseCore* mask, DenseCore* true_vals, DenseCore* false_vals, StreamHandle stream);
extern fn void jit_elementwise_fused_chain(DeviceHandle device, DenseCore* out, DenseCore* inputs, usz num_inputs, void* ops, usz num_ops, StreamHandle stream);
extern fn void jit_elementwise_from_source(DeviceHandle device, DenseCore* out, DenseCore* inputs, usz num_inputs, char* source, usz source_len, bool needs_broadcast, StreamHandle stream);
extern fn char* jit_compile_to_device_code(char* source, usz source_len, usz* code_size_out);
extern fn void jit_free_device_code(char* code);
extern fn void jit_elementwise_from_device_code(DeviceHandle device, DenseCore* out, DenseCore* inputs, usz num_inputs, char* code, usz code_size, bool needs_broadcast, StreamHandle stream);

// Handle-based device code API for C3-driven codegen
extern fn void* jit_load_device_code(char* code, usz code_size);

extern fn void jit_launch_kernel(void* kernel_handle, void** args, usz num_elements, StreamHandle stream);
extern fn void jit_unload_kernel(void* kernel_handle);

extern fn usz jit_codegen_fused_chain_source(int dtype, void* ops, usz num_ops, bool contiguous, usz ndim, char* buf, usz buf_max);

extern fn ulong jit_cache_size();
extern fn void jit_cache_clear();

fn void elementwise_unary(DeviceHandle device, StreamHandle stream, DenseCore* out, DenseCore* input, operations::UnaryOp op) {
	jit_elementwise_unary(device, out, input, (int)op, stream);
}

fn void elementwise_binary(DeviceHandle device, StreamHandle stream, DenseCore* out, DenseCore* a, DenseCore* b, operations::BinaryOp op) {
	jit_elementwise_binary(device, out, a, b, (int)op, stream);
}

fn void elementwise_select(DeviceHandle device, StreamHandle stream, DenseCore* out, DenseCore* mask, DenseCore* true_vals, DenseCore* false_vals) {
	jit_elementwise_select(device, out, mask, true_vals, false_vals, stream);
}

fn void elementwise_fused_chain(DeviceHandle device, StreamHandle stream, DenseCore* out, DenseCore* inputs, usz num_inputs, void* ops, usz num_ops) {
	jit_elementwise_fused_chain(device, out, inputs, num_inputs, ops, num_ops, stream);
}

fn void elementwise_from_source(DeviceHandle device, StreamHandle stream, DenseCore* out, DenseCore* inputs, usz num_inputs, char* source, usz source_len, bool needs_broadcast) {
	jit_elementwise_from_source(device, out, inputs, num_inputs, source, source_len, needs_broadcast, stream);
}

fn void elementwise_from_device_code(DeviceHandle device, StreamHandle stream, DenseCore* out, DenseCore* inputs, usz num_inputs, char* code, usz code_size, bool needs_broadcast) {
	jit_elementwise_from_device_code(device, out, inputs, num_inputs, code, code_size, needs_broadcast, stream);
}

/// Load device code into a persistent kernel handle that can be cached.
fn void* load_device_code(char* code, usz code_size) {
	return jit_load_device_code(code, code_size);
}

/// Launch a cached kernel with the given args array.
fn void launch_kernel(void* kernel_handle, void** args, usz num_elements, StreamHandle stream) {
	jit_launch_kernel(kernel_handle, args, num_elements, stream);
}

/// Unload a kernel module and free its handle.
fn void unload_kernel(void* kernel_handle) {
	jit_unload_kernel(kernel_handle);
}

fn ulong get_jit_cache_size() {
	return jit_cache_size();
}

fn void clear_jit_cache() {
	jit_cache_clear();
}
