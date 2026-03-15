module metaphor::device::hip_device::kernels;
import metaphor::device::hip_device::utils;
import metaphor::tensor::dense_core;

// Softmax Kernels (MIOpen-based)

/// 2D row-wise softmax forward: y = softmax(x) over each row [m, n].
extern fn void hip_nn_smax_2d_row_fwd(int dtype, MiopenHandle miopen, void* x, void* y, ulong m, ulong n);

/// 2D row-wise log-softmax forward.
extern fn void hip_nn_log_smax_2d_row_fwd(int dtype, MiopenHandle miopen, void* x, void* y, ulong m, ulong n);

/// 2D row-wise softmax backward.
extern fn void hip_nn_smax_2d_row_bwd(int dtype, MiopenHandle miopen, void* y_val, void* y_grd, void* x_grd, ulong m, ulong n);

/// 2D row-wise log-softmax backward.
extern fn void hip_nn_log_smax_2d_row_bwd(int dtype, MiopenHandle miopen, void* y_val, void* y_grd, void* x_grd, ulong m, ulong n);

// ReLU / Exp Kernels

extern fn void hip_nn_relu_fwd(int dtype, StreamHandle stream, void* x, void* y, ulong n);
extern fn void hip_nn_relu_bwd(int dtype, StreamHandle stream, void* x, void* y_grd, void* x_grd, ulong n);
extern fn void hip_nn_exp_fwd(int dtype, StreamHandle stream, void* x, void* y, ulong n);

// Clamp / Dropout Kernels

extern fn void hip_nn_clamp_fwd(int dtype, StreamHandle stream, void* x, void* y, ulong n, void* min_val, void* max_val);
extern fn void hip_nn_dropout_fwd(int dtype, StreamHandle stream, void* x, void* y, void* mask, ulong n, float p, ulong seed);
extern fn void hip_nn_dropout_bwd(int dtype, StreamHandle stream, void* y_grd, void* mask, void* x_grd, ulong n, float p);

// NLL Loss Kernels

extern fn void hip_nn_nll_loss_1d_index_fwd(int dtype, StreamHandle stream, void* log_probs, void* targets, void* output, ulong batch_size, ulong num_classes, int reduction);
extern fn void hip_nn_nll_loss_1d_index_bwd(int dtype, StreamHandle stream, void* targets, void* grad_output, void* grad_input, ulong batch_size, ulong num_classes, int reduction);

// 2D Convolution Kernels (MIOpen-based)

extern fn void hip_nn_conv2d_fwd(
	int dtype, MiopenHandle miopen,
	void* input, void* weight, void* output, void* workspace, usz workspace_size,
	int n_batch, int c_in, int h_in, int w_in, int c_out, int kh, int kw,
	int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
	int groups, int h_out, int w_out
);

extern fn void hip_nn_conv2d_bwd_data(
	int dtype, MiopenHandle miopen,
	void* weight, void* grad_output, void* grad_input, void* workspace, usz workspace_size,
	int n_batch, int c_in, int h_in, int w_in, int c_out, int kh, int kw,
	int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
	int groups, int h_out, int w_out
);

extern fn void hip_nn_conv2d_bwd_filter(
	int dtype, MiopenHandle miopen,
	void* input, void* grad_output, void* grad_weight, void* workspace, usz workspace_size,
	int n_batch, int c_in, int h_in, int w_in, int c_out, int kh, int kw,
	int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
	int groups, int h_out, int w_out
);

extern fn usz hip_nn_conv2d_get_workspace_size(
	int dtype, MiopenHandle miopen,
	int n_batch, int c_in, int h_in, int w_in, int c_out, int kh, int kw,
	int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
	int groups, int h_out, int w_out
);

// RMS Norm Kernels

extern fn void hip_nn_rms_norm_forward(int dtype, StreamHandle stream, void* x, void* weight, void* out, void* rms_buf, ulong num_rows, ulong d_model, float eps);
extern fn void hip_nn_rms_norm_backward_x(int dtype, StreamHandle stream, void* grad_y, void* x, void* weight, void* grad_x, ulong num_rows, ulong d_model, float eps);
extern fn void hip_nn_rms_norm_compute_rms(int dtype, StreamHandle stream, void* x, void* rms_buf, ulong num_rows, ulong d_model, float eps);
extern fn void hip_nn_rms_norm_backward_weight(int dtype, StreamHandle stream, void* grad_y, void* x, void* rms_buf, void* grad_weight, ulong num_rows, ulong d_model);

// Selective Scan Kernels (fused SSM recurrence)

extern fn void hip_nn_selective_scan_forward(int dtype, StreamHandle stream, void* a_bar, void* b_bar_x, void* h_out, ulong batch, ulong seq_len, ulong d_inner, ulong d_state);
extern fn void hip_nn_selective_scan_backward_a_bar(int dtype, StreamHandle stream, void* grad_output, void* a_bar, void* h, void* grad_a_bar, ulong batch, ulong seq_len, ulong d_inner, ulong d_state);
extern fn void hip_nn_selective_scan_backward_b_bar_x(int dtype, StreamHandle stream, void* grad_output, void* a_bar, void* grad_b_bar_x, ulong batch, ulong seq_len, ulong d_inner, ulong d_state);

// SSD Chunk Scan Kernels (Mamba-2 structured state space duality)

/// Cumulative sum of dt*A within each chunk, with optional softplus on dt.
extern fn void hip_nn_ssd_chunk_cumsum_forward(int dtype, StreamHandle stream, void* dt, void* a_param, void* dt_out, void* dA_cumsum, ulong batch, ulong seqlen, ulong nheads, ulong chunk_size, ulong nchunks, int dt_softplus);

/// C @ B^T within each chunk (intra-chunk attention matrix).
extern fn void hip_nn_ssd_bmm_chunk_forward(int dtype, StreamHandle stream, void* c_in, void* b_in, void* cb, ulong batch, ulong seqlen, ulong ngroups, ulong dstate, ulong chunk_size, ulong nchunks);

/// Compute SSM states at chunk boundaries.
extern fn void hip_nn_ssd_chunk_state_forward(int dtype, StreamHandle stream, void* b_in, void* x, void* dt, void* dA_cumsum, void* states, ulong batch, ulong seqlen, ulong nheads, ulong headdim, ulong ngroups, ulong dstate, ulong chunk_size, ulong nchunks);

/// Sequential scan across chunk boundaries (inter-chunk recurrence).
extern fn void hip_nn_ssd_state_passing_forward(int dtype, StreamHandle stream, void* states, void* dA_chunk_cumsum, void* initial_states, void* out, void* final_states, ulong batch, ulong nchunks, ulong nheads, ulong dim);

/// Combine intra-chunk and inter-chunk contributions to produce output.
extern fn void hip_nn_ssd_chunk_scan_forward(int dtype, StreamHandle stream, void* cb, void* x, void* dt, void* dA_cumsum, void* c_in, void* prev_states, void* d_param, void* out, ulong batch, ulong seqlen, ulong nheads, ulong headdim, ulong ngroups, ulong dstate, ulong chunk_size, ulong nchunks);

/// Backward pass for state_passing (reverse scan for gradient flow).
extern fn void hip_nn_ssd_state_passing_backward(int dtype, StreamHandle stream, void* grad_out, void* dA_chunk_cumsum, void* final_grad, void* grad_states, void* grad_initial, ulong batch, ulong nchunks, ulong nheads, ulong dim);

// Causal Conv1D Kernels (fused depthwise)

extern fn void hip_nn_causal_conv1d_forward(int dtype, StreamHandle stream, void* x, void* weight, void* bias, void* output, ulong batch, ulong channels, ulong seq_len, ulong kernel_size);
extern fn void hip_nn_causal_conv1d_backward_x(int dtype, StreamHandle stream, void* grad, void* weight, void* grad_x, ulong batch, ulong channels, ulong seq_len, ulong kernel_size);
extern fn void hip_nn_causal_conv1d_backward_weight(int dtype, StreamHandle stream, void* grad, void* x, void* grad_weight, ulong batch, ulong channels, ulong seq_len, ulong kernel_size);
extern fn void hip_nn_causal_conv1d_backward_bias(int dtype, StreamHandle stream, void* grad, void* grad_bias, ulong batch, ulong channels, ulong seq_len);

// Graph-Capture-Safe Loss Kernels

extern fn void hip_loss_softmax_cross_entropy_fwd(int dtype, void* logits, void* target, void* loss, ulong batch_size, ulong num_classes, int reduction, StreamHandle stream);
extern fn void hip_loss_softmax_cross_entropy_bwd(int dtype, void* logits, void* target, void* grad, ulong batch_size, ulong num_classes, int reduction, StreamHandle stream);

extern fn void hip_loss_mse_fwd(int dtype, void* pred, void* target, void* loss, ulong n, int reduction, StreamHandle stream);
extern fn void hip_loss_mse_bwd(int dtype, void* pred, void* target, void* grad, ulong n, int reduction, StreamHandle stream);

extern fn void hip_loss_bce_fwd(int dtype, void* pred, void* target, void* loss, ulong n, int reduction, StreamHandle stream);
extern fn void hip_loss_bce_bwd(int dtype, void* pred, void* target, void* grad, ulong n, int reduction, StreamHandle stream);

// Weighted Loss Kernels

extern fn void hip_loss_weighted_mse_fwd(int dtype, void* pred, void* target, void* weight, void* loss, void* weight_sum, ulong n, StreamHandle stream);
extern fn void hip_loss_weighted_mse_bwd(int dtype, void* pred, void* target, void* weight, void* weight_sum, void* grad, ulong n, StreamHandle stream);

extern fn void hip_loss_weighted_bce_fwd(int dtype, void* pred, void* target, void* weight, void* loss, void* weight_sum, ulong n, StreamHandle stream);
extern fn void hip_loss_weighted_bce_bwd(int dtype, void* pred, void* target, void* weight, void* weight_sum, void* grad, ulong n, StreamHandle stream);

extern fn void hip_loss_weighted_softmax_ce_fwd(int dtype, void* logits, void* target, void* weight, void* loss, void* weight_sum, ulong batch_size, ulong num_classes, StreamHandle stream);
extern fn void hip_loss_weighted_softmax_ce_bwd(int dtype, void* logits, void* target, void* weight, void* weight_sum, void* grad, ulong batch_size, ulong num_classes, StreamHandle stream);

extern fn void hip_loss_class_weighted_softmax_ce_fwd(int dtype, void* logits, void* target, void* class_weight, void* loss, ulong batch_size, ulong num_classes, int reduction, StreamHandle stream);
extern fn void hip_loss_class_weighted_softmax_ce_bwd(int dtype, void* logits, void* target, void* class_weight, void* grad, ulong batch_size, ulong num_classes, int reduction, StreamHandle stream);

// Max/Min Reduction Kernels (Thrust-based)

extern fn void hip_blas_max_fwd(int dtype, StreamHandle stream, void* x, void* y, ulong n);
extern fn void hip_blas_max_bwd(int dtype, StreamHandle stream, void* x_val, void* y_val, void* y_grd, void* x_grd, ulong n);
extern fn void hip_reduce_maxmin_backward(int dtype, StreamHandle stream, void* original, void* upstream, void* grad_out, ulong m, ulong n, int reduce_type, int is_min);

// Slice / Concat / Stack / Gather / Scatter Kernels

extern fn void hip_slice_forward(DeviceHandle device, DenseCore* out, DenseCore* input, ulong* offsets, ulong* steps, StreamHandle stream);
extern fn void hip_slice_backward(DeviceHandle device, DenseCore* out, DenseCore* input, ulong* offsets, ulong* steps, StreamHandle stream);
extern fn void hip_concat(DeviceHandle device, DenseCore* out, DenseCore** inputs, ulong num_inputs, ulong* offsets, ulong concat_dim, StreamHandle stream);
extern fn void hip_stack(DeviceHandle device, DenseCore* out, DenseCore** inputs, ulong num_inputs, ulong stack_dim, StreamHandle stream);
extern fn void hip_gather(DeviceHandle device, DenseCore* out, DenseCore* input, DenseCore* indices, ulong gather_dim, StreamHandle stream);
extern fn void hip_scatter_add(DeviceHandle device, DenseCore* out, DenseCore* grad, DenseCore* indices, ulong gather_dim, StreamHandle stream);

// Optimizer Kernels

/// SGD: param -= lr * grad (F32/F64 on device)
extern fn void hip_optim_sgd(int dtype, StreamHandle stream, void* param, void* grad, ulong n, double lr);

/// Adam (AdamW): m/v moment update + decoupled weight decay + param update (F32/F64 on device)
extern fn void hip_optim_adam(int dtype, StreamHandle stream, void* param, void* grad, void* m, void* v, ulong n, double lr, double beta1, double beta2, double epsilon, ulong t, double weight_decay);

/// Compute sum of squared gradient values (returns to host via rocThrust reduce)
extern fn float hip_optim_grad_norm_sq(int dtype, StreamHandle stream, void* grad, ulong n);

/// Scale all gradient values by a factor
extern fn void hip_optim_scale_grad(int dtype, StreamHandle stream, void* grad, ulong n, float scale);

/// Reduce grad² to a device-side float (no DtoH sync).
/// norm_sq_out must be zeroed before call (hipMemsetAsync).
extern fn void hip_optim_grad_norm_sq_device(int dtype, StreamHandle stream, void* grad, void* norm_sq_out, ulong n);

/// Clipped Adam: reads device-side norm² from norm_sq_ptr, clips inline, then updates.
/// No DtoH transfer required — norm_sq_ptr is a device float*.
extern fn void hip_optim_adam_clipped(int dtype, StreamHandle stream, void* param, void* grad, void* m, void* v, void* norm_sq_ptr, ulong n, double lr, double beta1, double beta2, double epsilon, ulong t, double max_norm);

/// SGD with F32 master weights for F16/BF16 visible weights
extern fn void hip_optim_sgd_master(int visible_dtype, StreamHandle stream, void* visible, void* master, void* grad, ulong n, double lr);

/// Adam (AdamW) with F32 master weights for F16/BF16 visible weights
extern fn void hip_optim_adam_master(int visible_dtype, StreamHandle stream, void* visible, void* master, void* grad, void* m, void* v, ulong n, double lr, double beta1, double beta2, double epsilon, ulong t, double weight_decay);

/// Initialize F32 master weights from F16/BF16 visible weights
extern fn void hip_optim_init_master(int visible_dtype, StreamHandle stream, void* master, void* visible, ulong n);

/// Gradient accumulation: accum_f32[i] += scale * (f32)grad[i]
/// Supports F32/F64/F16/BF16 gradient → F32 accumulation buffer.
extern fn void hip_optim_grad_accum(int grad_dtype, StreamHandle stream, void* accum_f32, void* grad, ulong n, double scale);

/// Copy F32 master weights back to F16/BF16 visible weights.
extern fn void hip_optim_copy_master_to_visible(int visible_dtype, StreamHandle stream, void* visible, void* master, ulong n);

/// Debug: check GPU buffer for NaN, returns count. Syncs stream.
extern fn int hip_optim_debug_check_nan(int dtype, StreamHandle stream, void* data, ulong n, char* label);
