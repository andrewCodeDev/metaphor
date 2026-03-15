//! HIP Executor
//!
//! Dispatches HipOp to HIP kernel calls. The central execute_single_op
//! function switches on OpType and calls the appropriate kernel or JIT function.
//!
//! Passes DenseCore* directly to HIP — no marshalling intermediary.

module metaphor::device::hip_device::executor;
import metaphor::device::hip_device::types;
import metaphor::device::hip_device::utils;
import metaphor::device::hip_device::kernels;
import metaphor::device::hip_device::codegen;
import metaphor::device::hip_device;
import metaphor::device::common::operations;
import metaphor::utils::hash;
import metaphor::tensor::tensor_core;
import metaphor::tensor::dense_core;
import metaphor::rtti;
import metaphor::utils::hash_map;
import metaphor::logging;
import std::io;
import std::os::posix;
import std::core::string;
import std::core::mem::allocator;
import std::thread;

// Helpers

/// Get DenseCore* from a void* that is actually a TensorCore*.
macro DenseCore* dc(void* ptr) { return &((TensorCore*)ptr).state.dense; }

/// Resolve a kernel handle: use cached_handle if available, otherwise
/// compile from source and cache the result for subsequent executions.
macro KernelHandle resolve_kernel(HipOp* op, char* source, usz source_len)
{
	KernelHandle handle = op.cached_handle;
	if ((void*)handle == null)
	{
		handle = get_or_load_kernel(source, source_len);
		op.cached_handle = handle;
	}
	assert((void*)handle != null, "HIP JIT: kernel compilation failed");
	return handle;
}

/// Ensure the HIP HSACO cache directory exists.
bool hip_cache_dir_created = false;
fn void ensure_hip_cache_dir() {
	if (!hip_cache_dir_created) {
		posix::mkdir("/tmp/metaphor", 0o755);
		posix::mkdir("/tmp/metaphor/hip", 0o755);
		hip_cache_dir_created = true;
	}
}

// Broadcast Stride Computation (ported from C++ compute_broadcast_strides)

/// Compute broadcast strides for a tensor with shape `tensor_shape[0..tensor_ndim]`
/// relative to the output shape `out_shape[0..out_ndim]`.
/// Returns true on success, false on shape mismatch.
fn bool compute_broadcast_strides(
	ulong* tensor_shape, usz tensor_ndim,
	ulong* out_shape, usz out_ndim,
	usz* out_strides)
{
	int offset = (int)out_ndim - (int)tensor_ndim;
	usz stride = 1;
	for (int d = (int)out_ndim - 1; d >= 0; d--)
	{
		int tensor_d = d - offset;
		if (tensor_d < 0)
		{
			out_strides[d] = 0;
		}
		else
		{
			ulong t_dim = tensor_shape[tensor_d];
			ulong o_dim = out_shape[d];
			if (t_dim == o_dim)
			{
				out_strides[d] = stride;
				stride *= (usz)t_dim;
			}
			else if (t_dim == 1)
			{
				out_strides[d] = 0;
			}
			else
			{
				return false; // shape mismatch
			}
		}
	}
	return true;
}

/// Check if two DenseCores have the same shape.
fn bool same_shape(DenseCore* a, DenseCore* b)
{
	if (a.shape.len != b.shape.len) return false;
	for (usz d = 0; d < a.shape.len; d++)
	{
		if (a.shape.buffer[d] != b.shape.buffer[d]) return false;
	}
	return true;
}

// In-Memory Kernel Handle Cache
//
// HashMap-based cache: hash → KernelHandle. No size limit, no collisions,
// no silent eviction. Mirrors the host device's kernel_cache.c3 approach.

alias KernelCacheMap = hash_map::HashMap{ulong, KernelHandle};

allocator::LibcAllocator kernel_libc_alloc;
Mutex kernel_cache_mutex;
KernelCacheMap kernel_cache;
bool kernel_cache_initialized = false;
bool kernel_mutex_initialized = false;

fn void ensure_kernel_cache()
{
	if (!kernel_mutex_initialized)
	{
		kernel_cache_mutex.init()!!;
		kernel_mutex_initialized = true;
	}
	if (!kernel_cache_initialized)
	{
		kernel_cache_mutex.lock()!!;
		defer kernel_cache_mutex.unlock();
		if (!kernel_cache_initialized)
		{
			kernel_cache.init(&kernel_libc_alloc);
			kernel_cache_initialized = true;
		}
	}
}

fn KernelHandle kernel_cache_get(ulong hash)
{
	ensure_kernel_cache();
	kernel_cache_mutex.lock()!!;
	defer kernel_cache_mutex.unlock();
	if (try handle = kernel_cache.get(hash))
	{
		return handle;
	}
	return (KernelHandle)null;
}

fn void kernel_cache_put(ulong hash, KernelHandle handle)
{
	ensure_kernel_cache();
	kernel_cache_mutex.lock()!!;
	defer kernel_cache_mutex.unlock();
	kernel_cache.set(hash, handle);
}

// Get-or-Load Kernel (3-tier: memory → disk → compile)

/// Given HIP source in buf[0..source_len], return a kernel handle.
/// 1. Hash the source and check in-memory cache.
/// 2. If miss, check disk for /tmp/metaphor/hip/{hash}.hsaco.
/// 3. If miss, compile source → PTX, save to disk, load handle.
/// Result is cached in memory for subsequent calls.
fn KernelHandle get_or_load_kernel(char* source, usz source_len)
{
	ulong hash = hash::fnv1a(source[:source_len]);

	// Tier 1: in-memory cache
	KernelHandle handle = kernel_cache_get(hash);
	if ((void*)handle != null)
	{
		logging::@debug(CODEGEN, "kernel cache hit (memory): hash=%x", hash);
		return handle;
	}

	// Build disk path
	char[256] ptx_path_buf;
	String ptx_path = string::bformat(&ptx_path_buf, "/tmp/metaphor/hip/%x.hsaco", hash);

	// Tier 2: disk cache
	char[]? ptx_data = io::file::load_temp(ptx_path);
	if (try ptx = ptx_data)
	{
		if (ptx.len > 0)
		{
			handle = (KernelHandle)utils::load_device_code(ptx.ptr, ptx.len);
			if ((void*)handle != null)
			{
				logging::@debug(CODEGEN, "kernel cache hit (disk): hash=%x", hash);
				kernel_cache_put(hash, handle);
				return handle;
			}
		}
	}

	// Tier 3: compile from source, save HSACO, load handle
	// io::eprintfn("JIT compiling HIP kernel %x (%d bytes)", hash, (int)source_len);
	// // Debug: dump JIT source to /tmp/metaphor/hip/{hash}.hip for inspection
	// {
	// 	char[256] src_path_buf;
	// 	String src_path = string::bformat(&src_path_buf, "/tmp/metaphor/hip/%x.hip", hash);
	// 	io::File? src_file = io::file::open(src_path, "wb");
	// 	if (try sf = src_file)
	// 	{
	// 		(void)sf.write(source[:source_len]);
	// 		(void)sf.close();
	// 	}
	// }
	logging::@debug(CODEGEN, "kernel cache miss — compiling HSACO: hash=%x source_len=%d", hash, (int)source_len);
	ensure_hip_cache_dir();
	usz ptx_size;
	char* ptx = utils::jit_compile_to_device_code(source, source_len, &ptx_size);
	// io::eprintfn("[JIT_DEBUG] compile returned: ptx=%p ptx_size=%d", ptx, (int)ptx_size);
	assert(ptx != null && ptx_size > 0, "HIP JIT: compilation returned null");

	// Save to disk
	// io::eprintfn("[JIT_DEBUG] writing %d bytes to %s", (int)ptx_size, ptx_path);
	io::File? out_file = io::file::open(ptx_path, "wb");
	if (try f = out_file)
	{
		(void)f.write(ptx[:ptx_size]);
		// io::eprintfn("[JIT_DEBUG] f.write returned: %d", (int)written);
		(void)f.close();
	}
	// else
	// {
	// 	io::eprintfn("[JIT_DEBUG] failed to open file for writing!");
	// }

	handle = (KernelHandle)utils::load_device_code(ptx, ptx_size);
	utils::jit_free_device_code(ptx);
	assert((void*)handle != null, "HIP JIT: failed to load compiled HSACO");

	kernel_cache_put(hash, handle);
	return handle;
}

// Main Dispatch

/// Execute a single HIP operation.
<*
 @require device != null : "execute_single_op: device must not be null"
 @require op != null : "execute_single_op: op must not be null"
 @require op.result != null : "execute_single_op: op result must not be null"
*>
fn void execute_single_op(hip_device::HipDevice* device, types::HipOp* op) {
	switch (op.op_type) {
		case CONTRACTION:                  execute_contraction(device, op);
		case REDUCTION:                    execute_reduction(device, op);
		case PERMUTATION:                  execute_permutation(device, op);
		case ELEMENTWISE_BINARY_TENSOR:  execute_elementwise_binary_hiptensor(device, op);
		case ELEMENTWISE_UNARY:            execute_elementwise_unary(device, op);
		case ELEMENTWISE_BINARY:           execute_elementwise_binary(device, op);
		case ELEMENTWISE_SELECT:           execute_elementwise_select(device, op);
		case ELEMENTWISE_FUSED_CHAIN:      execute_elementwise_fused_chain(device, op);
		case ELEMENTWISE_CAS_EXPR:         execute_elementwise_cas_expr(device, op);
		case SOFTMAX:                      execute_softmax(device, op);
		case CAST:                         execute_cast(device, op);
		case TRANSFER:                     execute_transfer(device, op);
		case REDUCE_MAXMIN_BACKWARD:       execute_reduce_maxmin_backward(device, op);
		case SLICE:                        execute_slice(device, op);
		case CONCAT:                       execute_concat(device, op);
		case STACK:                        execute_stack(device, op);
		case GATHER:                       execute_gather(device, op);
		case SCATTER_ADD:                  execute_scatter_add(device, op);
		case RMS_NORM:                     execute_rms_norm(device, op);
		case RMS_NORM_BWD_X:               execute_rms_norm_bwd_x(device, op);
		case RMS_NORM_BWD_WEIGHT:          execute_rms_norm_bwd_weight(device, op);
		case SELECTIVE_SCAN:               execute_selective_scan(device, op);
		case SELECTIVE_SCAN_BWD_A:         execute_selective_scan_bwd_a(device, op);
		case SELECTIVE_SCAN_BWD_BX:        execute_selective_scan_bwd_bx(device, op);
		case CAUSAL_CONV1D:                execute_causal_conv1d(device, op);
		case CAUSAL_CONV1D_BWD_X:          execute_causal_conv1d_bwd_x(device, op);
		case CAUSAL_CONV1D_BWD_WEIGHT:     execute_causal_conv1d_bwd_weight(device, op);
		case CAUSAL_CONV1D_BWD_BIAS:       execute_causal_conv1d_bwd_bias(device, op);
		case DROPOUT:                      execute_dropout(device, op);
		default: unreachable("execute_single_op: unhandled op_type");
	}
}

// hipTENSOR Operations

/// Convert BinaryOp to hipTENSOR reduce op constant (BINARY_ADD/MIN/MAX/MUL).
fn int hiptensor_reduce_op(operations::BinaryOp op) {
	switch (op) {
		case ADD: return utils::BINARY_ADD;
		case MUL: return utils::BINARY_MUL;
		case MAX: return utils::BINARY_MAX;
		case MIN: return utils::BINARY_MIN;
		default:  return utils::BINARY_ADD;
	}
}

fn void execute_contraction(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "contraction: input[0] is null");
	assert(op.inputs[1] != null, "contraction: input[1] is null");
	DenseCore* x = dc(op.inputs[0]);
	DenseCore* y = dc(op.inputs[1]);
	DenseCore* z = dc(op.result);
	assert(x.data != null, "contraction: input[0] data is null");
	assert(y.data != null, "contraction: input[1] data is null");
	char[] x_pat = op.get_input_pattern(0);
	char[] y_pat = op.get_input_pattern(1);
	char[] z_pat = op.get_output_pattern();
	assert(x_pat.len > 0, "contraction: input[0] pattern is empty");
	assert(y_pat.len > 0, "contraction: input[1] pattern is empty");
	assert(z_pat.len > 0, "contraction: output pattern is empty");

	device.contraction(x, x_pat.ptr, y, y_pat.ptr, z, z_pat.ptr);
}

fn void execute_reduction(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "reduction: input[0] is null");
	DenseCore* src = dc(op.inputs[0]);
	DenseCore* dst = dc(op.result);
	assert(src.data != null, "reduction: input data is null");
	char[] src_pat = op.get_input_pattern(0);
	char[] dst_pat = op.get_output_pattern();

	// Log reduction dispatch
	if (logging::is_enabled(logging::Scope.EXECUTOR, logging::Level.DEBUG))
	{
		logging::@debug(EXECUTOR, "execute_reduction: src_ndim=%d dst_ndim=%d src_pat_len=%d dst_pat_len=%d",
			(int)src.shape.len, (int)dst.shape.len, (int)src_pat.len, (int)dst_pat.len);
		for (usz di = 0; di < src.shape.len; di++)
		{
			logging::@debug(EXECUTOR, "  src dim[%d]=%d sym='%c'(0x%02x)",
				(int)di, (int)src.shape.buffer[di],
				di < src_pat.len ? src_pat[di] : '?',
				di < src_pat.len ? (int)src_pat[di] : 0);
		}
		for (usz di = 0; di < dst.shape.len; di++)
		{
			logging::@debug(EXECUTOR, "  dst dim[%d]=%d sym='%c'(0x%02x)",
				(int)di, (int)dst.shape.buffer[di],
				di < dst_pat.len ? dst_pat[di] : '?',
				di < dst_pat.len ? (int)dst_pat[di] : 0);
		}
	}

	assert(src_pat.len > 0, "reduction: input pattern is empty");
	// Note: dst_pat.len == 0 is valid for scalar reductions (0-dim output)

	// All scalar reductions (dst 0-dim) go directly to C++ scalar kernel path,
	// regardless of input rank. The scalar kernel treats data as flat memory.
	// hipTENSOR doesn't support 0-dim output and wrapping as [1] with a fake
	// symbol not in the input pattern causes NOT_SUPPORTED errors.
	device.reduce(src, src_pat.ptr, dst, dst_pat.ptr, hiptensor_reduce_op(op.reduction.reduce_op));
}

fn void execute_permutation(hip_device::HipDevice* device, types::HipOp* op) {
	DenseCore* src = dc(op.inputs[0]);
	DenseCore* dst = dc(op.result);

	// hipTensor permutate cannot broadcast. When src has fewer elements
	// than dst (e.g. scalar→1D expand), broadcast the single source value
	// into all destination elements.
	if (src.num_elements == 1 && dst.num_elements > 1) {
		int dtype = utils::dtype_from_datatype(dst.datatype());
		utils::hip_mem_broadcast_scalar(device.handle, dtype, src.data, dst.data, dst.num_elements);
		return;
	}

	char[] src_pat = op.get_input_pattern(0);
	char[] dst_pat = op.get_output_pattern();

	// General broadcast: hipTENSOR permutate cannot handle expanding dimensions
	// (e.g. "a" → "ab" for [1023] → [1023, 128256]). Use a JIT strided-copy
	// kernel with pattern-aware broadcast strides instead.
	if (src_pat.len < dst_pat.len) {
		// Compute source row-major strides
		usz[8] src_strides;
		usz s = 1;
		for (usz d = src.shape.len; d > 0; d--) {
			src_strides[d - 1] = s;
			s *= (usz)src.shape.buffer[d - 1];
		}

		// Map each output dimension to an input stride via pattern matching.
		// Output dims whose pattern char is NOT in the source pattern get stride 0 (broadcast).
		usz[8] in_strides;
		for (usz od = 0; od < dst.shape.len; od++) {
			in_strides[od] = 0;  // default: broadcast
			for (usz id = 0; id < src_pat.len; id++) {
				if (dst_pat[od] == src_pat[id]) {
					in_strides[od] = src_strides[id];
					break;
				}
			}
		}

		// Generate strided IDENTITY kernel (same as unary broadcast path)
		rtti::Datatype dtype = dst.datatype();
		char[codegen::KERNEL_SOURCE_MAX] source_buf;
		usz source_len = codegen::gen_unary_strided(
			&source_buf, codegen::KERNEL_SOURCE_MAX,
			dtype, operations::UnaryOp.IDENTITY, dst.shape.len);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)dst.num_elements;
		void* out_ptr = dst.data;
		void* in_ptr = src.data;
		usz ndim = dst.shape.len;

		usz[8] out_shape;
		usz[8] in_strides_padded;
		for (usz d = 0; d < 8; d++) {
			out_shape[d] = (d < dst.shape.len) ? (usz)dst.shape.buffer[d] : 0;
			in_strides_padded[d] = (d < dst.shape.len) ? in_strides[d] : 0;
		}

		void*[20] args = {
			&out_ptr, &in_ptr,
			&out_shape[0], &out_shape[1], &out_shape[2], &out_shape[3],
			&out_shape[4], &out_shape[5], &out_shape[6], &out_shape[7],
			&in_strides_padded[0], &in_strides_padded[1], &in_strides_padded[2], &in_strides_padded[3],
			&in_strides_padded[4], &in_strides_padded[5], &in_strides_padded[6], &in_strides_padded[7],
			&ndim, &n
		};
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
		return;
	}

	device.permutate(src, src_pat.ptr, dst, dst_pat.ptr);
}

fn void execute_elementwise_binary_hiptensor(hip_device::HipDevice* device, types::HipOp* op) {
	DenseCore* a = dc(op.inputs[0]);
	DenseCore* c = dc(op.inputs[1]);
	DenseCore* d = dc(op.result);
	char[] a_pat = op.get_input_pattern(0);
	char[] c_pat = op.get_input_pattern(1);
	char[] d_pat = op.get_output_pattern();
	device.hiptensor_elementwise_binary(a, a_pat.ptr, c, c_pat.ptr, d, d_pat.ptr, (int)op.elem_binary_tensor.binary_op);
}

// Elementwise Operations — C3 codegen + handle-based cached launch

fn void execute_elementwise_unary(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "elementwise_unary: input[0] is null");
	DenseCore* out = dc(op.result);
	DenseCore* inp = dc(op.inputs[0]);
	assert(inp.data != null, "elementwise_unary: input data is null");
	assert(out.num_elements > 0, "elementwise_unary: output has zero elements");
	rtti::Datatype dtype = out.datatype();

	char[codegen::KERNEL_SOURCE_MAX] source_buf;
	usz source_len;

	bool contiguous = same_shape(out, inp);

	if (contiguous)
	{
		source_len = codegen::gen_unary_contiguous(&source_buf, codegen::KERNEL_SOURCE_MAX, dtype, op.elem_unary.map_op);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)out.num_elements;
		void* out_ptr = out.data;
		void* in_ptr = inp.data;
		void*[3] args = { &out_ptr, &in_ptr, &n };
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
	}
	else
	{
		// Broadcast case
		usz[8] in_strides;
		if (!compute_broadcast_strides(&inp.shape.buffer, inp.shape.len, &out.shape.buffer, out.shape.len, &in_strides)) { return; }

		source_len = codegen::gen_unary_strided(&source_buf, codegen::KERNEL_SOURCE_MAX, dtype, op.elem_unary.map_op, out.shape.len);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)out.num_elements;
		void* out_ptr = out.data;
		void* in_ptr = inp.data;
		usz ndim = out.shape.len;

		// Pad shape/strides to 8
		usz[8] out_shape;
		usz[8] in_strides_padded;
		for (usz d = 0; d < 8; d++)
		{
			out_shape[d] = (d < out.shape.len) ? (usz)out.shape.buffer[d] : 0;
			in_strides_padded[d] = (d < out.shape.len) ? in_strides[d] : 0;
		}

		void*[20] args = {
			&out_ptr, &in_ptr,
			&out_shape[0], &out_shape[1], &out_shape[2], &out_shape[3],
			&out_shape[4], &out_shape[5], &out_shape[6], &out_shape[7],
			&in_strides_padded[0], &in_strides_padded[1], &in_strides_padded[2], &in_strides_padded[3],
			&in_strides_padded[4], &in_strides_padded[5], &in_strides_padded[6], &in_strides_padded[7],
			&ndim, &n
		};
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
	}
}

fn void execute_elementwise_binary(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "elementwise_binary: input[0] is null");
	assert(op.inputs[1] != null, "elementwise_binary: input[1] is null");
	DenseCore* out = dc(op.result);
	DenseCore* a = dc(op.inputs[0]);
	DenseCore* b = dc(op.inputs[1]);
	assert(a.data != null, "elementwise_binary: input[0] data is null");
	assert(b.data != null, "elementwise_binary: input[1] data is null");
	assert(out.num_elements > 0, "elementwise_binary: output has zero elements");
	rtti::Datatype dtype = out.datatype();

	char[codegen::KERNEL_SOURCE_MAX] source_buf;
	usz source_len;

	bool contiguous = same_shape(out, a) && same_shape(out, b);

	if (contiguous)
	{
		source_len = codegen::gen_binary_contiguous(&source_buf, codegen::KERNEL_SOURCE_MAX, dtype, op.elem_binary.binary_op);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)out.num_elements;
		void* out_ptr = out.data;
		void* a_ptr = a.data;
		void* b_ptr = b.data;
		void*[4] args = { &out_ptr, &a_ptr, &b_ptr, &n };
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
	}
	else
	{
		// Broadcast case
		usz[8] a_strides;
		usz[8] b_strides;
		if (!compute_broadcast_strides(&a.shape.buffer, a.shape.len, &out.shape.buffer, out.shape.len, &a_strides)) { return; }
		if (!compute_broadcast_strides(&b.shape.buffer, b.shape.len, &out.shape.buffer, out.shape.len, &b_strides)) { return; }

		source_len = codegen::gen_binary_strided(&source_buf, codegen::KERNEL_SOURCE_MAX, dtype, op.elem_binary.binary_op, out.shape.len);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)out.num_elements;
		void* out_ptr = out.data;
		void* a_ptr = a.data;
		void* b_ptr = b.data;
		usz ndim = out.shape.len;

		usz[8] out_shape;
		usz[8] a_strides_padded;
		usz[8] b_strides_padded;
		for (usz d = 0; d < 8; d++)
		{
			out_shape[d] = (d < out.shape.len) ? (usz)out.shape.buffer[d] : 0;
			a_strides_padded[d] = (d < out.shape.len) ? a_strides[d] : 0;
			b_strides_padded[d] = (d < out.shape.len) ? b_strides[d] : 0;
		}

		void*[29] args = {
			&out_ptr, &a_ptr, &b_ptr,
			&out_shape[0], &out_shape[1], &out_shape[2], &out_shape[3],
			&out_shape[4], &out_shape[5], &out_shape[6], &out_shape[7],
			&a_strides_padded[0], &a_strides_padded[1], &a_strides_padded[2], &a_strides_padded[3],
			&a_strides_padded[4], &a_strides_padded[5], &a_strides_padded[6], &a_strides_padded[7],
			&b_strides_padded[0], &b_strides_padded[1], &b_strides_padded[2], &b_strides_padded[3],
			&b_strides_padded[4], &b_strides_padded[5], &b_strides_padded[6], &b_strides_padded[7],
			&ndim, &n
		};
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
	}
}

fn void execute_elementwise_select(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "elementwise_select: input[0] (mask) is null");
	assert(op.inputs[1] != null, "elementwise_select: input[1] (true vals) is null");
	assert(op.inputs[2] != null, "elementwise_select: input[2] (false vals) is null");
	DenseCore* out = dc(op.result);
	DenseCore* mask = dc(op.inputs[0]);
	DenseCore* true_vals = dc(op.inputs[1]);
	DenseCore* false_vals = dc(op.inputs[2]);
	assert(mask.data != null, "elementwise_select: mask data is null");
	assert(true_vals.data != null, "elementwise_select: true_vals data is null");
	assert(false_vals.data != null, "elementwise_select: false_vals data is null");
	rtti::Datatype dtype = out.datatype();

	char[codegen::KERNEL_SOURCE_MAX] source_buf;
	usz source_len;

	bool contiguous = same_shape(out, mask) && same_shape(out, true_vals) && same_shape(out, false_vals);

	if (contiguous)
	{
		source_len = codegen::gen_select_contiguous(&source_buf, codegen::KERNEL_SOURCE_MAX, dtype);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)out.num_elements;
		void* out_ptr = out.data;
		void* mask_ptr = mask.data;
		void* true_ptr = true_vals.data;
		void* false_ptr = false_vals.data;
		void*[5] args = { &out_ptr, &mask_ptr, &true_ptr, &false_ptr, &n };
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
	}
	else
	{
		// Broadcast case
		usz[8] mask_strides;
		usz[8] true_strides;
		usz[8] false_strides;
		if (!compute_broadcast_strides(&mask.shape.buffer, mask.shape.len, &out.shape.buffer, out.shape.len, &mask_strides)) { return; }
		if (!compute_broadcast_strides(&true_vals.shape.buffer, true_vals.shape.len, &out.shape.buffer, out.shape.len, &true_strides)) { return; }
		if (!compute_broadcast_strides(&false_vals.shape.buffer, false_vals.shape.len, &out.shape.buffer, out.shape.len, &false_strides)) { return; }

		source_len = codegen::gen_select_strided(&source_buf, codegen::KERNEL_SOURCE_MAX, dtype, out.shape.len);

		KernelHandle handle = resolve_kernel(op, &source_buf, source_len);

		usz n = (usz)out.num_elements;
		void* out_ptr = out.data;
		void* mask_ptr = mask.data;
		void* true_ptr = true_vals.data;
		void* false_ptr = false_vals.data;
		usz ndim = out.shape.len;

		usz[8] out_shape;
		usz[8] mask_strides_padded;
		usz[8] true_strides_padded;
		usz[8] false_strides_padded;
		for (usz d = 0; d < 8; d++)
		{
			out_shape[d] = (d < out.shape.len) ? (usz)out.shape.buffer[d] : 0;
			mask_strides_padded[d] = (d < out.shape.len) ? mask_strides[d] : 0;
			true_strides_padded[d] = (d < out.shape.len) ? true_strides[d] : 0;
			false_strides_padded[d] = (d < out.shape.len) ? false_strides[d] : 0;
		}

		void*[38] args = {
			&out_ptr, &mask_ptr, &true_ptr, &false_ptr,
			&out_shape[0], &out_shape[1], &out_shape[2], &out_shape[3],
			&out_shape[4], &out_shape[5], &out_shape[6], &out_shape[7],
			&mask_strides_padded[0], &mask_strides_padded[1], &mask_strides_padded[2], &mask_strides_padded[3],
			&mask_strides_padded[4], &mask_strides_padded[5], &mask_strides_padded[6], &mask_strides_padded[7],
			&true_strides_padded[0], &true_strides_padded[1], &true_strides_padded[2], &true_strides_padded[3],
			&true_strides_padded[4], &true_strides_padded[5], &true_strides_padded[6], &true_strides_padded[7],
			&false_strides_padded[0], &false_strides_padded[1], &false_strides_padded[2], &false_strides_padded[3],
			&false_strides_padded[4], &false_strides_padded[5], &false_strides_padded[6], &false_strides_padded[7],
			&ndim, &n
		};
		utils::launch_kernel((void*)handle, &args, n, device.get_stream());
	}
}

fn void execute_elementwise_fused_chain(hip_device::HipDevice* device, types::HipOp* op) {
	// Handle was resolved at compile time in register_hip_op_with_inputs.
	KernelHandle handle = op.cached_handle;
	assert((void*)handle != null, "HIP fused chain: kernel handle must be resolved at compile time");

	usz num_inputs = (usz)op.fused_chain.inputs_len;
	DenseCore* out = dc(op.result);
	usz n = (usz)out.num_elements;
	void* out_ptr = out.data;

	if (!op.cas_expr.jit_needs_broadcast)
	{
		// Contiguous: args = out, in_0, ..., in_N, n
		void*[types::MAX_FUSED_CHAIN + 3] arg_ptrs;
		arg_ptrs[0] = &out_ptr;

		void*[types::MAX_FUSED_CHAIN] in_ptrs;
		for (usz i = 0; i < num_inputs; i++)
		{
			in_ptrs[i] = ((TensorCore*)op.fused_chain.chain_inputs[i]).state.dense.data;
			arg_ptrs[1 + i] = &in_ptrs[i];
		}
		arg_ptrs[1 + num_inputs] = &n;

		utils::launch_kernel((void*)handle, &arg_ptrs, n, device.get_stream());
	}
	else
	{
		// Broadcast: args = out, in_0..in_N, shape_0..shape_D, stride_in0_0..strideInN_D, n
		DenseCore* out_dc = dc(op.result);
		usz ndim = out_dc.shape.len;

		// Compute broadcast strides for each input
		ulong[types::MAX_FUSED_CHAIN][8] strides;
		for (usz j = 0; j < num_inputs; j++)
		{
			DenseCore* in_dc = &((TensorCore*)op.fused_chain.chain_inputs[j]).state.dense;
			int offset = (int)ndim - (int)in_dc.shape.len;
			ulong stride = 1;
			for (int d = (int)ndim - 1; d >= 0; d--)
			{
				int td = d - offset;
				if (td < 0) { strides[j][d] = 0; }
				else if (in_dc.shape.buffer[td] == out_dc.shape.buffer[d])
				{
					strides[j][d] = stride;
					stride *= in_dc.shape.buffer[td];
				}
				else { strides[j][d] = 0; } // broadcast dim
			}
		}

		// Build args
		ulong[8] shape_vals;
		for (usz d = 0; d < ndim; d++) { shape_vals[d] = out_dc.shape.buffer[d]; }

		// Max args: 1 out + MAX_FUSED inputs + 8 shape + MAX_FUSED*8 strides + 1 n
		void*[types::MAX_FUSED_CHAIN * 9 + 12] arg_ptrs;
		usz arg_idx = 0;
		arg_ptrs[arg_idx++] = &out_ptr;

		void*[types::MAX_FUSED_CHAIN] in_ptrs;
		for (usz j = 0; j < num_inputs; j++)
		{
			in_ptrs[j] = ((TensorCore*)op.fused_chain.chain_inputs[j]).state.dense.data;
			arg_ptrs[arg_idx++] = &in_ptrs[j];
		}
		for (usz d = 0; d < ndim; d++) { arg_ptrs[arg_idx++] = &shape_vals[d]; }
		for (usz j = 0; j < num_inputs; j++)
		{
			for (usz d = 0; d < ndim; d++) { arg_ptrs[arg_idx++] = &strides[j][d]; }
		}
		arg_ptrs[arg_idx++] = &n;

		utils::launch_kernel((void*)handle, &arg_ptrs, n, device.get_stream());
	}
}

fn void execute_elementwise_cas_expr(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.cas_expr.jit_num_inputs > 0, "cas_expr: JIT has zero inputs");

	DenseCore[types::MAX_JIT_INPUTS] input_dcs;
	for (usz i = 0; i < (usz)op.cas_expr.jit_num_inputs; i++) {
		assert(op.cas_expr.jit_inputs[i] != null, "cas_expr: JIT input tensor is null");
		input_dcs[i] = ((TensorCore*)op.cas_expr.jit_inputs[i]).state.dense;
		assert(input_dcs[i].data != null, "cas_expr: JIT input data is null");
	}

	KernelHandle handle = op.cached_handle;
	assert((void*)handle != null, "HIP JIT: kernel handle must be resolved at compile time");

	// Build args: out, in_0, ..., in_N, n, p0, ..., pM
	usz n = (usz)dc(op.result).num_elements;
	void* out_ptr = dc(op.result).data;

	void*[types::MAX_JIT_INPUTS + 3 + 16] arg_ptrs;
	arg_ptrs[0] = &out_ptr;

	void*[types::MAX_JIT_INPUTS] in_ptrs;
	for (usz i = 0; i < (usz)op.cas_expr.jit_num_inputs; i++)
	{
		in_ptrs[i] = input_dcs[i].data;
		arg_ptrs[1 + i] = &in_ptrs[i];
	}

	usz arg_idx = 1 + (usz)op.cas_expr.jit_num_inputs;
	arg_ptrs[arg_idx] = &n;
	arg_idx++;

	// Dynamic params (non-pow2 broadcast strides/sizes)
	for (usz i = 0; i < (usz)op.cas_expr.jit_num_dynamic_params; i++)
	{
		arg_ptrs[arg_idx] = &op.cas_expr.jit_dynamic_params[i];
		arg_idx++;
	}

	utils::launch_kernel((void*)handle, &arg_ptrs, n, device.get_stream());
}

// Softmax

fn void execute_softmax(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "softmax: input[0] is null");
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* input_tc  = (TensorCore*)op.inputs[0];

	void* out_data = result_tc.state.dense.data;
	void* in_data  = input_tc.state.dense.data;
	assert(out_data != null, "softmax: output data is null");
	assert(in_data != null, "softmax: input data is null");
	int dtype = utils::dtype_from_datatype(result_tc.dtype());

	// Resolve axis: negative values index from end (torch convention)
	usz ndim = result_tc.shape().len;
	usz resolved_axis;
	if (op.softmax.axis < 0)
	{
		resolved_axis = (usz)((int)ndim + op.softmax.axis);
	}
	else
	{
		resolved_axis = (usz)op.softmax.axis;
	}

	// Flatten to 2D (m, n): m = product of dims before axis, n = dim at axis
	ulong m = 1;
	for (usz d = 0; d < resolved_axis; d++)
	{
		m *= result_tc.shape().get(d);
	}
	ulong n = result_tc.shape().get(resolved_axis);

	bool is_log = op.softmax.is_log;
	bool is_backward = (op.input_count > 1);

	if (is_backward) {
		// inputs[0] = y (forward output), inputs[1] = grad_y (upstream gradient)
		TensorCore* y_val_tc = (TensorCore*)op.inputs[0];
		TensorCore* grad_y_tc = (TensorCore*)op.inputs[1];
		void* y_val = y_val_tc.state.dense.data;
		void* grad_y = grad_y_tc.state.dense.data;
		assert(y_val != null, "softmax backward: y_val data is null");
		assert(grad_y != null, "softmax backward: grad_y data is null");

		if (is_log) {
			kernels::hip_nn_log_smax_2d_row_bwd(dtype, device.get_miopen(), y_val, grad_y, out_data, m, n);
		} else {
			kernels::hip_nn_smax_2d_row_bwd(dtype, device.get_miopen(), y_val, grad_y, out_data, m, n);
		}
	} else {
		if (is_log) {
			kernels::hip_nn_log_smax_2d_row_fwd(dtype, device.get_miopen(), in_data, out_data, m, n);
		} else {
			kernels::hip_nn_smax_2d_row_fwd(dtype, device.get_miopen(), in_data, out_data, m, n);
		}
	}
}

// Cast

fn void execute_cast(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* input_tc  = (TensorCore*)op.inputs[0];

	int src_dtype = utils::dtype_from_datatype(input_tc.dtype());
	int dst_dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong n = (ulong)result_tc.shape().size();

	utils::mem_cast(
		device.get_stream(), src_dtype, dst_dtype,
		input_tc.state.dense.data,
		result_tc.state.dense.data,
		n
	);
}

// Transfer (memcpy between host/device)

fn void execute_transfer(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "transfer: input[0] is null");
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* input_tc  = (TensorCore*)op.inputs[0];

	usz size = result_tc.shape().size() * result_tc.dtype().size_of();
	assert(size > 0, "transfer: tensor byte size is zero");
	void* dst = result_tc.state.dense.data;
	void* src = input_tc.state.dense.data;
	assert(dst != null, "transfer: destination data is null");
	assert(src != null, "transfer: source data is null");

	switch (op.transfer.direction) {
		case HTOD: utils::memcpy_htod(device.get_handle(), dst, src, size);
		case DTOH: utils::memcpy_dtoh(device.get_handle(), dst, src, size);
		case DTOD: utils::memcpy_dtod(device.get_handle(), dst, src, size);
		case HTOH: unreachable("execute_transfer: HTOH not supported on HIP");
	}
}

// Reduce Max/Min Backward

fn void execute_reduce_maxmin_backward(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc   = (TensorCore*)op.result;
	TensorCore* original_tc = (TensorCore*)op.inputs[0];
	TensorCore* upstream_tc = (TensorCore*)op.inputs[1];

	int dtype = utils::dtype_from_datatype(result_tc.dtype());

	kernels::hip_reduce_maxmin_backward(
		dtype, device.get_stream(),
		original_tc.state.dense.data,
		upstream_tc.state.dense.data,
		result_tc.state.dense.data,
		(ulong)op.reduce_maxmin_bwd.row, (ulong)op.reduce_maxmin_bwd.col,
		(int)op.reduce_maxmin_bwd.reduce_type, op.reduce_maxmin_bwd.is_min ? 1 : 0
	);
}

// Slice

fn void execute_slice(hip_device::HipDevice* device, types::HipOp* op) {
	if (op.slice.is_scatter) {
		kernels::hip_slice_backward(
			device.get_handle(), dc(op.result), dc(op.inputs[0]),
			&op.slice.offsets[0], &op.slice.steps[0],
			device.get_stream()
		);
	} else {
		kernels::hip_slice_forward(
			device.get_handle(), dc(op.result), dc(op.inputs[0]),
			&op.slice.offsets[0], &op.slice.steps[0],
			device.get_stream()
		);
	}
}

// Concat / Stack

fn void execute_concat(hip_device::HipDevice* device, types::HipOp* op) {
	DenseCore*[types::MAX_MULTI_INPUTS] input_dcs;
	for (usz i = 0; i < (usz)op.concat.num_inputs; i++) {
		input_dcs[i] = &((TensorCore*)op.concat.multi_inputs[i]).state.dense;
	}

	kernels::hip_concat(
		device.get_handle(), dc(op.result), &input_dcs[0],
		(ulong)op.concat.num_inputs, &op.concat.offsets[0],
		(ulong)op.concat.dim,
		device.get_stream()
	);
}

fn void execute_stack(hip_device::HipDevice* device, types::HipOp* op) {
	DenseCore*[types::MAX_MULTI_INPUTS] input_dcs;
	for (usz i = 0; i < (usz)op.stack.num_inputs; i++) {
		input_dcs[i] = &((TensorCore*)op.stack.multi_inputs[i]).state.dense;
	}

	kernels::hip_stack(
		device.get_handle(), dc(op.result), &input_dcs[0],
		(ulong)op.stack.num_inputs,
		(ulong)op.stack.dim,
		device.get_stream()
	);
}

// Gather / Scatter Add

fn void execute_gather(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "gather: input[0] is null");
	assert(op.gather.indices_tensor != null, "gather: indices tensor is null");
	assert(dc(op.inputs[0]).data != null, "gather: input data is null");
	assert(dc(op.gather.indices_tensor).data != null, "gather: indices data is null");
	kernels::hip_gather(
		device.get_handle(), dc(op.result), dc(op.inputs[0]),
		dc(op.gather.indices_tensor),
		(ulong)op.gather.dim,
		device.get_stream()
	);
}

fn void execute_scatter_add(hip_device::HipDevice* device, types::HipOp* op) {
	assert(op.inputs[0] != null, "scatter_add: input[0] is null");
	assert(op.scatter_add.indices_tensor != null, "scatter_add: indices tensor is null");
	assert(dc(op.inputs[0]).data != null, "scatter_add: input data is null");
	assert(dc(op.scatter_add.indices_tensor).data != null, "scatter_add: indices data is null");
	kernels::hip_scatter_add(
		device.get_handle(), dc(op.result), dc(op.inputs[0]),
		dc(op.scatter_add.indices_tensor),
		(ulong)op.scatter_add.dim,
		device.get_stream()
	);
}

// RMS Norm

fn void rms_norm_2d(TensorCore* input_tc, ulong* num_rows, ulong* d_model)
{
	usz ndim = input_tc.shape().len;
	*d_model = input_tc.shape().get(ndim - 1);
	*num_rows = 1;
	for (usz d = 0; d < ndim - 1; d++) { *num_rows *= input_tc.shape().get(d); }
}

fn void execute_rms_norm(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* input_tc  = (TensorCore*)op.inputs[0];
	TensorCore* weight_tc = (TensorCore*)op.inputs[1];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong num_rows; ulong d_model;
	rms_norm_2d(input_tc, &num_rows, &d_model);

	char[] rms_buf = device.mem_alloc(F32, (usz)num_rows);
	defer device.mem_free(rms_buf);

	kernels::hip_nn_rms_norm_forward(
		dtype, device.get_stream(),
		input_tc.state.dense.data, weight_tc.state.dense.data,
		result_tc.state.dense.data, rms_buf.ptr,
		num_rows, d_model, op.rms_norm.eps
	);
}

fn void execute_rms_norm_bwd_x(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* grad_tc   = (TensorCore*)op.inputs[0];
	TensorCore* x_tc      = (TensorCore*)op.inputs[1];
	TensorCore* w_tc      = (TensorCore*)op.inputs[2];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong num_rows; ulong d_model;
	rms_norm_2d(grad_tc, &num_rows, &d_model);

	kernels::hip_nn_rms_norm_backward_x(
		dtype, device.get_stream(),
		grad_tc.state.dense.data, x_tc.state.dense.data,
		w_tc.state.dense.data, result_tc.state.dense.data,
		num_rows, d_model, op.rms_norm.eps
	);
}

fn void execute_rms_norm_bwd_weight(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* grad_tc   = (TensorCore*)op.inputs[0];
	TensorCore* x_tc      = (TensorCore*)op.inputs[1];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong num_rows; ulong d_model;
	rms_norm_2d(grad_tc, &num_rows, &d_model);

	char[] rms_buf = device.mem_alloc(F32, (usz)num_rows);
	defer device.mem_free(rms_buf);

	kernels::hip_nn_rms_norm_compute_rms(
		dtype, device.get_stream(),
		x_tc.state.dense.data, rms_buf.ptr,
		num_rows, d_model, op.rms_norm.eps
	);
	kernels::hip_nn_rms_norm_backward_weight(
		dtype, device.get_stream(),
		grad_tc.state.dense.data, x_tc.state.dense.data,
		rms_buf.ptr, result_tc.state.dense.data,
		num_rows, d_model
	);
}

// Selective Scan

fn void sscan_4d(TensorCore* tc, ulong* batch, ulong* seq_len, ulong* d_inner, ulong* d_state)
{
	assert(tc.shape().len == 4, "selective_scan: input must be 4D");
	*batch = tc.shape().get(0); *seq_len = tc.shape().get(1);
	*d_inner = tc.shape().get(2); *d_state = tc.shape().get(3);
}

fn void execute_selective_scan(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* a_bar_tc  = (TensorCore*)op.inputs[0];
	TensorCore* b_bar_x_tc = (TensorCore*)op.inputs[1];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong batch, seq_len, d_inner, d_state;
	sscan_4d(a_bar_tc, &batch, &seq_len, &d_inner, &d_state);
	kernels::hip_nn_selective_scan_forward(dtype, device.get_stream(),
		a_bar_tc.state.dense.data, b_bar_x_tc.state.dense.data,
		result_tc.state.dense.data, batch, seq_len, d_inner, d_state);
}

fn void execute_selective_scan_bwd_a(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* grad_tc   = (TensorCore*)op.inputs[0];
	TensorCore* a_bar_tc  = (TensorCore*)op.inputs[1];
	TensorCore* h_tc      = (TensorCore*)op.inputs[2];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong batch, seq_len, d_inner, d_state;
	sscan_4d(grad_tc, &batch, &seq_len, &d_inner, &d_state);
	kernels::hip_nn_selective_scan_backward_a_bar(dtype, device.get_stream(),
		grad_tc.state.dense.data, a_bar_tc.state.dense.data,
		h_tc.state.dense.data, result_tc.state.dense.data,
		batch, seq_len, d_inner, d_state);
}

fn void execute_selective_scan_bwd_bx(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* grad_tc   = (TensorCore*)op.inputs[0];
	TensorCore* a_bar_tc  = (TensorCore*)op.inputs[1];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong batch, seq_len, d_inner, d_state;
	sscan_4d(grad_tc, &batch, &seq_len, &d_inner, &d_state);
	kernels::hip_nn_selective_scan_backward_b_bar_x(dtype, device.get_stream(),
		grad_tc.state.dense.data, a_bar_tc.state.dense.data,
		result_tc.state.dense.data, batch, seq_len, d_inner, d_state);
}

// Causal Conv1D

fn void execute_causal_conv1d(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc  = (TensorCore*)op.result;
	TensorCore* x_tc       = (TensorCore*)op.inputs[0];
	TensorCore* weight_tc  = (TensorCore*)op.inputs[1];
	TensorCore* bias_tc    = (TensorCore*)op.inputs[2];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	assert(x_tc.shape().len == 3);
	ulong batch = x_tc.shape().get(0); ulong channels = x_tc.shape().get(1); ulong seq_len = x_tc.shape().get(2);
	assert(weight_tc.shape().len == 3);
	ulong kernel_size = weight_tc.shape().get(2);
	kernels::hip_nn_causal_conv1d_forward(dtype, device.get_stream(),
		x_tc.state.dense.data, weight_tc.state.dense.data, bias_tc.state.dense.data,
		result_tc.state.dense.data, batch, channels, seq_len, kernel_size);
}

fn void execute_causal_conv1d_bwd_x(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc  = (TensorCore*)op.result;
	TensorCore* grad_tc    = (TensorCore*)op.inputs[0];
	TensorCore* weight_tc  = (TensorCore*)op.inputs[1];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	assert(grad_tc.shape().len == 3);
	ulong batch = grad_tc.shape().get(0); ulong channels = grad_tc.shape().get(1); ulong seq_len = grad_tc.shape().get(2);
	ulong kernel_size = weight_tc.shape().get(2);
	kernels::hip_nn_causal_conv1d_backward_x(dtype, device.get_stream(),
		grad_tc.state.dense.data, weight_tc.state.dense.data,
		result_tc.state.dense.data, batch, channels, seq_len, kernel_size);
}

fn void execute_causal_conv1d_bwd_weight(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc  = (TensorCore*)op.result;
	TensorCore* grad_tc    = (TensorCore*)op.inputs[0];
	TensorCore* x_tc       = (TensorCore*)op.inputs[1];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	assert(grad_tc.shape().len == 3);
	ulong batch = grad_tc.shape().get(0); ulong channels = grad_tc.shape().get(1); ulong seq_len = grad_tc.shape().get(2);
	ulong kernel_size = result_tc.shape().get(2);
	kernels::hip_nn_causal_conv1d_backward_weight(dtype, device.get_stream(),
		grad_tc.state.dense.data, x_tc.state.dense.data,
		result_tc.state.dense.data, batch, channels, seq_len, kernel_size);
}

fn void execute_causal_conv1d_bwd_bias(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc  = (TensorCore*)op.result;
	TensorCore* grad_tc    = (TensorCore*)op.inputs[0];
	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	assert(grad_tc.shape().len == 3);
	ulong batch = grad_tc.shape().get(0); ulong channels = grad_tc.shape().get(1); ulong seq_len = grad_tc.shape().get(2);
	kernels::hip_nn_causal_conv1d_backward_bias(dtype, device.get_stream(),
		grad_tc.state.dense.data, result_tc.state.dense.data,
		batch, channels, seq_len);
}

// Dropout

fn void execute_dropout(hip_device::HipDevice* device, types::HipOp* op) {
	TensorCore* result_tc = (TensorCore*)op.result;
	TensorCore* input0_tc = (TensorCore*)op.inputs[0];

	int dtype = utils::dtype_from_datatype(result_tc.dtype());
	ulong n = (ulong)result_tc.shape().size();

	if (op.dropout.is_bwd) {
		// Backward: regenerate mask from same seed, then apply
		// inputs[0] = grad (only input — mask determined by seed alone)

		// Allocate temp mask
		char[] mask_buf = device.mem_alloc(U8, (usz)n);

		// Regenerate the same mask by running forward kernel with grad as dummy input
		// (mask is determined by seed, not input values — we discard the output)
		char[] dummy_buf = device.mem_alloc(result_tc.dtype(), (usz)n);
		kernels::hip_nn_dropout_fwd(
			dtype, device.get_stream(),
			input0_tc.state.dense.data,  // grad data as dummy input
			dummy_buf.ptr,
			mask_buf.ptr,
			n, op.dropout.p, op.dropout.seed
		);

		// Apply backward: grad_input = grad * mask / (1 - p)
		kernels::hip_nn_dropout_bwd(
			dtype, device.get_stream(),
			input0_tc.state.dense.data,
			mask_buf.ptr,
			result_tc.state.dense.data,
			n, op.dropout.p
		);

		device.mem_free(dummy_buf);
		device.mem_free(mask_buf);
	} else {
		// Forward: generate mask and apply
		usz mask_bytes = (usz)n;
		char[] mask_buf = device.mem_alloc(U8, (usz)n);

		kernels::hip_nn_dropout_fwd(
			dtype, device.get_stream(),
			input0_tc.state.dense.data,
			result_tc.state.dense.data,
			mask_buf.ptr,
			n, op.dropout.p, op.dropout.seed
		);

		device.mem_free(mask_buf);
	}
}
