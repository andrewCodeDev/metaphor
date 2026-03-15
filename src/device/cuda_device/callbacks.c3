//! CUDA Compiler Callbacks
//!
//! Callback structs that implement the Executable interface for CUDA operations.
//! Each callback wraps a specific CUDA operation (cuDNN softmax, RMS norm, conv2d,
//! device transfer, graph replay, etc.) and is stored as the `operation` field
//! of an ExecutionUnit.
//!

module metaphor::device::cuda_device::callbacks;
import metaphor::device::cuda_device;
import metaphor::device::cuda_device::kernels;
import metaphor::device::cuda_device::utils;
import metaphor::device::device_reference;
import metaphor::device::common::operations;
import metaphor::graph::execution_sequence;
import metaphor::tensor::tensor_core;
import metaphor::rtti;

// TensorCore pointer alias — matches execution_sequence convention

alias TensorCorePtr = execution_sequence::TensorCorePtr;

// No-Op Callback

/// No-op for operations pre-computed during compile (e.g., conv via cuDNN).
struct NoOpCallback (Executable)
{
	char _pad;  // C3 structs must have at least one field
}

fn void NoOpCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic {}
fn void NoOpCallback.destroy(&self) @dynamic {}

// Softmax Forward Callback

/// CUDA-specific softmax/log_softmax forward using cuDNN.
struct SoftmaxForwardCallback (Executable)
{
	CudaDevice* device;
	TensorCorePtr input;
	TensorCorePtr output;
	bool is_log;
}

fn void SoftmaxForwardCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* in_tc = self.input;
	TensorCore* out_tc = self.output;

	assert(in_tc.shape().len > 0, "CUDA softmax: cannot softmax a scalar");

	// Flatten to 2D: m = product of all dims except last, n = last dim
	ulong n = in_tc.shape().get(in_tc.shape().len - 1);
	ulong m = 1;
	for (usz i = 0; i < in_tc.shape().len - 1; i++) { m *= in_tc.shape().get(i); }

	int dtype = utils::dtype_from_datatype(in_tc.dtype());
	CudnnHandle cudnn = self.device.get_cudnn();
	void* in_data = in_tc.state.dense.data;
	void* out_data = out_tc.state.dense.data;

	if (self.is_log) {
		kernels::cuda_nn_log_smax_2d_row_fwd(dtype, cudnn, in_data, out_data, m, n);
	} else {
		kernels::cuda_nn_smax_2d_row_fwd(dtype, cudnn, in_data, out_data, m, n);
	}
}

fn void SoftmaxForwardCallback.destroy(&self) @dynamic {}

// Softmax Backward Callback

/// CUDA-specific softmax backward using cuDNN.
/// grad_x = y * (grad_y - sum(y * grad_y))
struct SoftmaxBackwardCallback (Executable)
{
	CudaDevice* device;
}

fn void SoftmaxBackwardCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* y_tc = sources[0];
	TensorCore* gy_tc = sources[1];
	TensorCore* gx_tc = target;

	assert(y_tc.shape().len > 0, "CUDA softmax backward: invalid shape");

	ulong n = y_tc.shape().get(y_tc.shape().len - 1);
	ulong m = 1;
	for (usz i = 0; i < y_tc.shape().len - 1; i++) { m *= y_tc.shape().get(i); }

	int dtype = utils::dtype_from_datatype(y_tc.dtype());
	CudnnHandle cudnn = self.device.get_cudnn();

	kernels::cuda_nn_smax_2d_row_bwd(
		dtype, cudnn,
		y_tc.state.dense.data,
		gy_tc.state.dense.data,
		gx_tc.state.dense.data,
		m, n
	);
}

fn void SoftmaxBackwardCallback.destroy(&self) @dynamic {}

// Log-Softmax Backward Callback

/// CUDA-specific log-softmax backward using cuDNN.
struct LogSoftmaxBackwardCallback (Executable)
{
	CudaDevice* device;
	/// Log-softmax output (y)
	TensorCorePtr y_val;
	/// Incoming gradient (grad_y)
	TensorCorePtr grad_y;
	/// Output gradient (grad_x)
	TensorCorePtr grad_x;
}

fn void LogSoftmaxBackwardCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* y_tc = self.y_val;
	TensorCore* gy_tc = self.grad_y;
	TensorCore* gx_tc = self.grad_x;

	assert(y_tc.shape().len > 0, "CUDA log_softmax backward: invalid shape");

	ulong n = y_tc.shape().get(y_tc.shape().len - 1);
	ulong m = 1;
	for (usz i = 0; i < y_tc.shape().len - 1; i++) { m *= y_tc.shape().get(i); }

	int dtype = utils::dtype_from_datatype(y_tc.dtype());
	CudnnHandle cudnn = self.device.get_cudnn();

	kernels::cuda_nn_log_smax_2d_row_bwd(
		dtype, cudnn,
		y_tc.state.dense.data,
		gy_tc.state.dense.data,
		gx_tc.state.dense.data,
		m, n
	);
}

fn void LogSoftmaxBackwardCallback.destroy(&self) @dynamic {}

// RMS Norm Forward Callback

/// CUDA-specific RMS normalization forward.
/// y = (x / rms) * weight, where rms = sqrt(mean(x^2) + eps)
struct RmsNormForwardCallback (Executable)
{
	CudaDevice* device;
	/// Input tensor [batch*seq, d_model]
	TensorCorePtr x;
	/// Weight tensor [d_model]
	TensorCorePtr weight;
	/// Output tensor [batch*seq, d_model]
	TensorCorePtr output;
	float eps;
}

fn void RmsNormForwardCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* x_tc = self.x;
	TensorCore* w_tc = self.weight;
	TensorCore* out_tc = self.output;

	int dtype = utils::dtype_from_datatype(x_tc.dtype());
	StreamHandle stream = self.device.get_stream();

	ulong d_model = x_tc.shape().get(x_tc.shape().len - 1);
	ulong num_rows = 1;
	for (usz i = 0; i < x_tc.shape().len - 1; i++) { num_rows *= x_tc.shape().get(i); }

	// Allocate temporary buffer for per-row RMS values (f32)
	char[] rms_buf = self.device.mem_alloc(F32, (usz)num_rows);
	assert(rms_buf.len > 0, "RmsNormForward: failed to allocate RMS buffer");

	kernels::cuda_nn_rms_norm_forward(
		dtype, stream,
		x_tc.state.dense.data,
		w_tc.state.dense.data,
		out_tc.state.dense.data,
		rms_buf.ptr,
		num_rows, d_model, self.eps
	);

	self.device.mem_free(rms_buf);
}

fn void RmsNormForwardCallback.destroy(&self) @dynamic {}

// RMS Norm Backward X Callback

/// CUDA-specific RMS normalization backward for input gradient.
struct RmsNormBackwardXCallback (Executable)
{
	CudaDevice* device;
	/// Upstream gradient [batch*seq, d_model]
	TensorCorePtr grad_y;
	/// Original input [batch*seq, d_model]
	TensorCorePtr x;
	/// Weight [d_model]
	TensorCorePtr weight;
	/// Output gradient [batch*seq, d_model]
	TensorCorePtr grad_x;
	float eps;
}

fn void RmsNormBackwardXCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* gy_tc = self.grad_y;
	TensorCore* x_tc = self.x;
	TensorCore* w_tc = self.weight;
	TensorCore* gx_tc = self.grad_x;

	int dtype = utils::dtype_from_datatype(x_tc.dtype());
	StreamHandle stream = self.device.get_stream();

	ulong d_model = x_tc.shape().get(x_tc.shape().len - 1);
	ulong num_rows = 1;
	for (usz i = 0; i < x_tc.shape().len - 1; i++) { num_rows *= x_tc.shape().get(i); }

	// backward_x recomputes RMS internally (one block per row)
	kernels::cuda_nn_rms_norm_backward_x(
		dtype, stream,
		gy_tc.state.dense.data,
		x_tc.state.dense.data,
		w_tc.state.dense.data,
		gx_tc.state.dense.data,
		num_rows, d_model, self.eps
	);
}

fn void RmsNormBackwardXCallback.destroy(&self) @dynamic {}

// RMS Norm Backward Weight Callback

/// CUDA-specific RMS normalization backward for weight gradient.
struct RmsNormBackwardWeightCallback (Executable)
{
	CudaDevice* device;
	/// Upstream gradient [batch*seq, d_model]
	TensorCorePtr grad_y;
	/// Original input [batch*seq, d_model]
	TensorCorePtr x;
	/// Output gradient [d_model]
	TensorCorePtr grad_weight;
	float eps;
}

fn void RmsNormBackwardWeightCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* gy_tc = self.grad_y;
	TensorCore* x_tc = self.x;
	TensorCore* gw_tc = self.grad_weight;

	int dtype = utils::dtype_from_datatype(x_tc.dtype());
	StreamHandle stream = self.device.get_stream();

	ulong d_model = x_tc.shape().get(x_tc.shape().len - 1);
	ulong num_rows = 1;
	for (usz i = 0; i < x_tc.shape().len - 1; i++) { num_rows *= x_tc.shape().get(i); }

	// Allocate temporary buffer for per-row RMS values (f32)
	char[] rms_buf = self.device.mem_alloc(F32, (usz)num_rows);
	assert(rms_buf.len > 0, "RmsNormBackwardWeight: failed to allocate RMS buffer");

	// First compute RMS for all rows
	kernels::cuda_nn_rms_norm_compute_rms(
		dtype, stream,
		x_tc.state.dense.data,
		rms_buf.ptr,
		num_rows, d_model, self.eps
	);

	// Then compute grad_weight using pre-computed RMS
	kernels::cuda_nn_rms_norm_backward_weight(
		dtype, stream,
		gy_tc.state.dense.data,
		x_tc.state.dense.data,
		rms_buf.ptr,
		gw_tc.state.dense.data,
		num_rows, d_model
	);

	self.device.mem_free(rms_buf);
}

fn void RmsNormBackwardWeightCallback.destroy(&self) @dynamic {}

// CUDA Graph Replay Callback
//
// The ONLY ExecutableClosure for a CUDA observable.
// Owns and replays a captured CUDA graph.

struct GraphReplayCallback (Executable)
{
	CudaDevice* device;
	GraphExecHandle graph_exec;
}

fn void GraphReplayCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	self.device.launch_exec(self.graph_exec);
}

fn void GraphReplayCallback.destroy(&self) @dynamic
{
	self.device.destroy_exec(self.graph_exec);
}

// Transfer Callback
//
// Device-to-device, host-to-device, and device-to-host transfers.
// Runs outside of CUDA graph capture since it crosses device boundaries.

struct TransferCallback (Executable)
{
	CudaDevice* device;
	device_reference::TransferDirection direction;
}

fn void TransferCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* src_tc = sources[0];
	TensorCore* dst_tc = target;

	assert(src_tc.has_data(), "transfer: source has no data");

	usz size = dst_tc.shape().size() * dst_tc.dtype().size_of();
	void* src_data = src_tc.state.dense.data;
	void* dst_data = dst_tc.state.dense.data;

	switch (self.direction) {
		case HTOD: utils::memcpy_htod(self.device.get_handle(), dst_data, src_data, size);
		case DTOH: utils::memcpy_dtoh(self.device.get_handle(), dst_data, src_data, size);
		case DTOD: utils::memcpy_dtod(self.device.get_handle(), dst_data, src_data, size);
		case HTOH: unreachable("TransferCallback: HTOH not supported on CUDA");
	}
}

fn void TransferCallback.destroy(&self) @dynamic {}

// Conv2d Forward Callback

struct Conv2dFwdCallback (Executable)
{
	CudaDevice* device;
	ConvParams params;
}

fn void Conv2dFwdCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* in_tc = sources[0];
	TensorCore* w_tc = sources[1];
	TensorCore* out_tc = target;

	int dtype = utils::dtype_from_datatype(out_tc.dtype());
	CudnnHandle cudnn = self.device.get_cudnn();
	ConvParams p = self.params;

	kernels::cuda_nn_conv2d_fwd(
		dtype, cudnn,
		in_tc.state.dense.data,
		w_tc.state.dense.data,
		out_tc.state.dense.data,
		null, 0,  // workspace
		(int)in_tc.shape().get(0),  // n_batch
		(int)in_tc.shape().get(1),  // c_in
		(int)in_tc.shape().get(2),  // h_in
		(int)in_tc.shape().get(3),  // w_in
		(int)w_tc.shape().get(0),   // c_out
		(int)w_tc.shape().get(2),   // kh
		(int)w_tc.shape().get(3),   // kw
		(int)p.padding[0], (int)p.padding[1],
		(int)p.stride[0], (int)p.stride[1],
		(int)p.dilation[0], (int)p.dilation[1],
		(int)p.groups,
		(int)out_tc.shape().get(2), // h_out
		(int)out_tc.shape().get(3)  // w_out
	);
}

fn void Conv2dFwdCallback.destroy(&self) @dynamic {}

// Conv2d Backward Data Callback

struct Conv2dBwdDataCallback (Executable)
{
	CudaDevice* device;
	ConvParams params;
}

fn void Conv2dBwdDataCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* go_tc = sources[0];
	TensorCore* w_tc = sources[1];
	TensorCore* gi_tc = target;

	int dtype = utils::dtype_from_datatype(gi_tc.dtype());
	CudnnHandle cudnn = self.device.get_cudnn();
	ConvParams p = self.params;

	kernels::cuda_nn_conv2d_bwd_data(
		dtype, cudnn,
		w_tc.state.dense.data,
		go_tc.state.dense.data,
		gi_tc.state.dense.data,
		null, 0,  // workspace
		(int)gi_tc.shape().get(0),  // n_batch
		(int)gi_tc.shape().get(1),  // c_in
		(int)gi_tc.shape().get(2),  // h_in
		(int)gi_tc.shape().get(3),  // w_in
		(int)w_tc.shape().get(0),   // c_out
		(int)w_tc.shape().get(2),   // kh
		(int)w_tc.shape().get(3),   // kw
		(int)p.padding[0], (int)p.padding[1],
		(int)p.stride[0], (int)p.stride[1],
		(int)p.dilation[0], (int)p.dilation[1],
		(int)p.groups,
		(int)go_tc.shape().get(2),  // h_out
		(int)go_tc.shape().get(3)   // w_out
	);
}

fn void Conv2dBwdDataCallback.destroy(&self) @dynamic {}

// Conv2d Backward Filter Callback

struct Conv2dBwdFilterCallback (Executable)
{
	CudaDevice* device;
	ConvParams params;
}

fn void Conv2dBwdFilterCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* in_tc = sources[0];
	TensorCore* go_tc = sources[1];
	TensorCore* gw_tc = target;

	int dtype = utils::dtype_from_datatype(gw_tc.dtype());
	CudnnHandle cudnn = self.device.get_cudnn();
	ConvParams p = self.params;

	kernels::cuda_nn_conv2d_bwd_filter(
		dtype, cudnn,
		in_tc.state.dense.data,
		go_tc.state.dense.data,
		gw_tc.state.dense.data,
		null, 0,  // workspace
		(int)in_tc.shape().get(0),  // n_batch
		(int)in_tc.shape().get(1),  // c_in
		(int)in_tc.shape().get(2),  // h_in
		(int)in_tc.shape().get(3),  // w_in
		(int)gw_tc.shape().get(0),  // c_out
		(int)gw_tc.shape().get(2),  // kh
		(int)gw_tc.shape().get(3),  // kw
		(int)p.padding[0], (int)p.padding[1],
		(int)p.stride[0], (int)p.stride[1],
		(int)p.dilation[0], (int)p.dilation[1],
		(int)p.groups,
		(int)go_tc.shape().get(2),  // h_out
		(int)go_tc.shape().get(3)   // w_out
	);
}

fn void Conv2dBwdFilterCallback.destroy(&self) @dynamic {}
