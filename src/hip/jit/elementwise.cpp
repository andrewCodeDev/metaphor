/*
 * jit/elementwise.cpp - Elementwise kernel execution via JIT
 *
 * Extern C API for C3 to execute JIT-compiled elementwise operations.
 */

#ifndef __METAPHOR_JIT_ELEMENTWISE_H__
#define __METAPHOR_JIT_ELEMENTWISE_H__

#include "../interop.h"
#include "../logging.h"
#include "cache.cpp"



/* ============================================================================
 * Stride Computation Helpers
 * ============================================================================
 */

/*
 * Compute broadcast strides for a tensor with given shape into output shape.
 *
 * For each dimension:
 *   - If tensor dim == output dim: use normal stride
 *   - If tensor dim == 1: stride = 0 (broadcast)
 *   - Otherwise: shape mismatch error
 *
 * tensor_shape: shape of the input tensor (may have fewer dims)
 * tensor_ndim: number of dims in input tensor
 * output_shape: shape of the output (broadcast result)
 * output_ndim: number of dims in output
 * out_strides: output array for computed strides (size = output_ndim)
 *
 * Returns true on success, false on shape mismatch.
 */
static bool compute_broadcast_strides(const len_t *tensor_shape,
                                      len_t tensor_ndim,
                                      const len_t *output_shape,
                                      len_t output_ndim, len_t *out_strides) {
  /* Right-align tensor shape with output shape */
  int offset = output_ndim - tensor_ndim;

  /* Initialize strides */
  len_t stride = 1;
  for (int d = output_ndim - 1; d >= 0; d--) {
    int tensor_d = d - offset;

    if (tensor_d < 0) {
      /* This dimension doesn't exist in tensor - broadcast */
      out_strides[d] = 0;
    } else {
      len_t t_dim = tensor_shape[tensor_d];
      len_t o_dim = output_shape[d];

      if (t_dim == o_dim) {
        out_strides[d] = stride;
        stride *= t_dim;
      } else if (t_dim == 1) {
        /* Broadcast */
        out_strides[d] = 0;
      } else {
        LOG_ERROR("Shape mismatch: tensor[%d]=%lu vs output[%d]=%lu", tensor_d,
                  t_dim, d, o_dim);
        return false;
      }
    }
  }

  return true;
}

/*
 * Check if a tensor is contiguous given its strides and shape.
 */
static bool is_contiguous(const len_t *shape, const len_t *strides,
                          len_t ndim) {
  len_t expected_stride = 1;
  for (int d = ndim - 1; d >= 0; d--) {
    if (strides[d] != expected_stride && shape[d] != 1) {
      return false;
    }
    expected_stride *= shape[d];
  }
  return true;
}

/* ============================================================================
 * Extern C API
 * ============================================================================
 */

/*
 * Execute a unary elementwise operation.
 *
 * out = op(in)
 *
 * Handles broadcasting if shapes differ. Both tensors must have same dtype.
 */
extern "C" void jit_elementwise_unary(HipDeviceHandle device_handle,
                                      DenseCore *out, const DenseCore *in,
                                      MapOp op, StreamHandle stream_handle) {
  LOG_DEBUG("jit_elementwise_unary: op=%d numel=%lu ndim=%lu", op, out->num_elements,
            out->shape.len);

  hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);
  JitKernelCache *cache = jit_cache_get();

  /* Check if contiguous and same shape */
  bool same_shape = (out->shape.len == in->shape.len);
  if (same_shape) {
    for (len_t d = 0; d < out->shape.len; d++) {
      if (out->shape.buffer[d] != in->shape.buffer[d]) {
        same_shape = false;
        break;
      }
    }
  }

  if (same_shape) {
    /* Simple contiguous case */
    CompiledKernel *kernel = cache->get_unary(dense_core_dtype(out), op, true, out->shape.len);
    if (!kernel)
      return;

    size_t n = out->num_elements;
    void *out_ptr = out->data;
    void *in_ptr = const_cast<void *>(in->data);
    void *args[] = {&out_ptr, &in_ptr, &n};
    jit_launch(kernel, n, args, stream);
  } else {
    /* Broadcasting case - compute strides */
    len_t in_strides[HIP_MAX_DIMS] = {0};
    if (!compute_broadcast_strides(in->shape.buffer, in->shape.len, out->shape.buffer, out->shape.len,
                                   in_strides)) {
      LOG_ERROR("jit_elementwise_unary: broadcast failed");
      return;
    }

    CompiledKernel *kernel = cache->get_unary(dense_core_dtype(out), op, false, out->shape.len);
    if (!kernel)
      return;

    /* Pad shape/strides to 8 elements (kernel expects fixed-size arrays) */
    len_t out_shape[8] = {0};
    len_t in_strides_padded[8] = {0};
    for (len_t d = 0; d < out->shape.len && d < 8; d++) {
      out_shape[d] = out->shape.buffer[d];
      in_strides_padded[d] = in_strides[d];
    }

    size_t ndim = out->shape.len;
    size_t n = out->num_elements;
    void *out_ptr = out->data;
    void *in_ptr = const_cast<void *>(in->data);

    /* Pass shape/stride arrays directly as individual parameters - graph
     * capture safe */
    void *args[] = {&out_ptr,
                    &in_ptr,
                    &out_shape[0],
                    &out_shape[1],
                    &out_shape[2],
                    &out_shape[3],
                    &out_shape[4],
                    &out_shape[5],
                    &out_shape[6],
                    &out_shape[7],
                    &in_strides_padded[0],
                    &in_strides_padded[1],
                    &in_strides_padded[2],
                    &in_strides_padded[3],
                    &in_strides_padded[4],
                    &in_strides_padded[5],
                    &in_strides_padded[6],
                    &in_strides_padded[7],
                    &ndim,
                    &n};
    jit_launch(kernel, n, args, stream);
  }
}

/*
 * Execute a binary elementwise operation.
 *
 * out = op(a, b)
 *
 * Handles broadcasting. All tensors must have same dtype.
 * Output shape must be the broadcast result of a and b shapes.
 */
extern "C" void jit_elementwise_binary(HipDeviceHandle device_handle,
                                       DenseCore *out, const DenseCore *a,
                                       const DenseCore *b, BinOp op,
                                       StreamHandle stream_handle) {
  LOG_DEBUG(
      "jit_elementwise_binary: op=%d out_numel=%lu a_numel=%lu b_numel=%lu", op,
      out->num_elements, a->num_elements, b->num_elements);

  hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);
  JitKernelCache *cache = jit_cache_get();

  /* Determine kernel type:
   * - Comparison ops (EQ, NE, GT, etc): float/double input -> u8 output
   * - Logical ops (AND, OR, XOR): u8 input -> u8 output
   * - Arithmetic ops: same dtype for input and output
   */
  bool is_comparison = binop_is_comparison(op);
  bool is_logical = binop_is_logical(op);
  Dtype kernel_dtype = is_comparison ? dense_core_dtype(a) : dense_core_dtype(out);

  /* Check if all same shape (no broadcasting needed) */
  bool same_shape = (out->shape.len == a->shape.len) && (out->shape.len == b->shape.len);
  if (same_shape) {
    for (len_t d = 0; d < out->shape.len; d++) {
      if (out->shape.buffer[d] != a->shape.buffer[d] || out->shape.buffer[d] != b->shape.buffer[d]) {
        same_shape = false;
        break;
      }
    }
  }

  if (same_shape) {
    /* Simple contiguous case */
    CompiledKernel *kernel;
    if (is_comparison) {
      kernel = cache->get_comparison(kernel_dtype, op, true, out->shape.len);
    } else if (is_logical) {
      kernel = cache->get_logical(op, true, out->shape.len);
    } else {
      kernel = cache->get_binary(kernel_dtype, op, true, out->shape.len);
    }
    if (!kernel)
      return;

    size_t n = out->num_elements;
    void *out_ptr = out->data;
    void *a_ptr = const_cast<void *>(a->data);
    void *b_ptr = const_cast<void *>(b->data);
    void *args[] = {&out_ptr, &a_ptr, &b_ptr, &n};
    jit_launch(kernel, n, args, stream);
  } else {
    /* Broadcasting case - compute strides for both inputs */
    len_t a_strides[HIP_MAX_DIMS];
    len_t b_strides[HIP_MAX_DIMS];

    if (!compute_broadcast_strides(a->shape.buffer, a->shape.len, out->shape.buffer, out->shape.len,
                                   a_strides)) {
      LOG_ERROR("jit_elementwise_binary: broadcast failed for input a");
      return;
    }
    if (!compute_broadcast_strides(b->shape.buffer, b->shape.len, out->shape.buffer, out->shape.len,
                                   b_strides)) {
      LOG_ERROR("jit_elementwise_binary: broadcast failed for input b");
      return;
    }

    CompiledKernel *kernel;
    if (is_comparison) {
      kernel = cache->get_comparison(kernel_dtype, op, false, out->shape.len);
    } else if (is_logical) {
      kernel = cache->get_logical(op, false, out->shape.len);
    } else {
      kernel = cache->get_binary(kernel_dtype, op, false, out->shape.len);
    }
    if (!kernel)
      return;

    /* Pad shape/strides to 8 elements (kernel expects fixed-size arrays) */
    len_t out_shape[8] = {0};
    len_t a_strides_padded[8] = {0};
    len_t b_strides_padded[8] = {0};
    for (len_t d = 0; d < out->shape.len && d < 8; d++) {
      out_shape[d] = out->shape.buffer[d];
      a_strides_padded[d] = a_strides[d];
      b_strides_padded[d] = b_strides[d];
    }

    size_t ndim = out->shape.len;
    size_t n = out->num_elements;
    void *out_ptr = out->data;
    void *a_ptr = const_cast<void *>(a->data);
    void *b_ptr = const_cast<void *>(b->data);

    /* Pass shape/stride arrays directly as individual parameters - graph
     * capture safe */
    void *args[] = {&out_ptr,
                    &a_ptr,
                    &b_ptr,
                    &out_shape[0],
                    &out_shape[1],
                    &out_shape[2],
                    &out_shape[3],
                    &out_shape[4],
                    &out_shape[5],
                    &out_shape[6],
                    &out_shape[7],
                    &a_strides_padded[0],
                    &a_strides_padded[1],
                    &a_strides_padded[2],
                    &a_strides_padded[3],
                    &a_strides_padded[4],
                    &a_strides_padded[5],
                    &a_strides_padded[6],
                    &a_strides_padded[7],
                    &b_strides_padded[0],
                    &b_strides_padded[1],
                    &b_strides_padded[2],
                    &b_strides_padded[3],
                    &b_strides_padded[4],
                    &b_strides_padded[5],
                    &b_strides_padded[6],
                    &b_strides_padded[7],
                    &ndim,
                    &n};
    jit_launch(kernel, n, args, stream);
  }
}

/*
 * Execute a select (ternary) operation.
 *
 * out = mask ? true_vals : false_vals
 *
 * mask: u8 tensor (0 or 1)
 * true_vals, false_vals: same dtype tensors
 * out: same dtype as true_vals/false_vals
 *
 * All tensors must have same shape (no broadcasting for now).
 */
extern "C" void jit_elementwise_select(HipDeviceHandle device_handle,
                                       DenseCore *out, const DenseCore *mask,
                                       const DenseCore *true_vals,
                                       const DenseCore *false_vals,
                                       StreamHandle stream_handle) {
  LOG_DEBUG("jit_elementwise_select: out_numel=%lu mask_numel=%lu", out->num_elements,
            mask->num_elements);

  hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);
  JitKernelCache *cache = jit_cache_get();

  /* Check if all same shape (no broadcasting needed) */
  bool same_shape = (out->shape.len == mask->shape.len) &&
                    (out->shape.len == true_vals->shape.len) &&
                    (out->shape.len == false_vals->shape.len);
  if (same_shape) {
    for (len_t d = 0; d < out->shape.len; d++) {
      if (out->shape.buffer[d] != mask->shape.buffer[d] ||
          out->shape.buffer[d] != true_vals->shape.buffer[d] ||
          out->shape.buffer[d] != false_vals->shape.buffer[d]) {
        same_shape = false;
        break;
      }
    }
  }

  if (same_shape) {
    /* Simple contiguous case */
    CompiledKernel *kernel = cache->get_select(dense_core_dtype(out), true, out->shape.len);
    if (!kernel)
      return;

    size_t n = out->num_elements;
    void *out_ptr = out->data;
    void *mask_ptr = const_cast<void *>(mask->data);
    void *true_ptr = const_cast<void *>(true_vals->data);
    void *false_ptr = const_cast<void *>(false_vals->data);
    void *args[] = {&out_ptr, &mask_ptr, &true_ptr, &false_ptr, &n};
    jit_launch(kernel, n, args, stream);
  } else {
    /* Broadcasting case - compute strides for all inputs */
    len_t mask_strides[HIP_MAX_DIMS];
    len_t true_strides[HIP_MAX_DIMS];
    len_t false_strides[HIP_MAX_DIMS];

    if (!compute_broadcast_strides(mask->shape.buffer, mask->shape.len, out->shape.buffer,
                                   out->shape.len, mask_strides)) {
      LOG_ERROR("jit_elementwise_select: broadcast failed for mask");
      return;
    }
    if (!compute_broadcast_strides(true_vals->shape.buffer, true_vals->shape.len,
                                   out->shape.buffer, out->shape.len, true_strides)) {
      LOG_ERROR("jit_elementwise_select: broadcast failed for true_vals");
      return;
    }
    if (!compute_broadcast_strides(false_vals->shape.buffer, false_vals->shape.len,
                                   out->shape.buffer, out->shape.len, false_strides)) {
      LOG_ERROR("jit_elementwise_select: broadcast failed for false_vals");
      return;
    }

    CompiledKernel *kernel = cache->get_select(dense_core_dtype(out), false, out->shape.len);
    if (!kernel)
      return;

    /* Pad shape/strides to 8 elements (kernel expects fixed-size arrays) */
    len_t out_shape[8] = {0};
    len_t mask_strides_padded[8] = {0};
    len_t true_strides_padded[8] = {0};
    len_t false_strides_padded[8] = {0};
    for (len_t d = 0; d < out->shape.len && d < 8; d++) {
      out_shape[d] = out->shape.buffer[d];
      mask_strides_padded[d] = mask_strides[d];
      true_strides_padded[d] = true_strides[d];
      false_strides_padded[d] = false_strides[d];
    }

    size_t ndim = out->shape.len;
    size_t n = out->num_elements;
    void *out_ptr = out->data;
    void *mask_ptr = const_cast<void *>(mask->data);
    void *true_ptr = const_cast<void *>(true_vals->data);
    void *false_ptr = const_cast<void *>(false_vals->data);

    /* Pass shape/stride arrays directly as individual parameters - graph
     * capture safe */
    void *args[] = {&out_ptr,
                    &mask_ptr,
                    &true_ptr,
                    &false_ptr,
                    &out_shape[0],
                    &out_shape[1],
                    &out_shape[2],
                    &out_shape[3],
                    &out_shape[4],
                    &out_shape[5],
                    &out_shape[6],
                    &out_shape[7],
                    &mask_strides_padded[0],
                    &mask_strides_padded[1],
                    &mask_strides_padded[2],
                    &mask_strides_padded[3],
                    &mask_strides_padded[4],
                    &mask_strides_padded[5],
                    &mask_strides_padded[6],
                    &mask_strides_padded[7],
                    &true_strides_padded[0],
                    &true_strides_padded[1],
                    &true_strides_padded[2],
                    &true_strides_padded[3],
                    &true_strides_padded[4],
                    &true_strides_padded[5],
                    &true_strides_padded[6],
                    &true_strides_padded[7],
                    &false_strides_padded[0],
                    &false_strides_padded[1],
                    &false_strides_padded[2],
                    &false_strides_padded[3],
                    &false_strides_padded[4],
                    &false_strides_padded[5],
                    &false_strides_padded[6],
                    &false_strides_padded[7],
                    &ndim,
                    &n};
    jit_launch(kernel, n, args, stream);
  }
}

/*
 * Execute an inplace chain operation (fused sequence of binary ops).
 *
 * For clamp (min then max):
 *   out = fmax(fmin(base, children[0]), children[1])
 *
 * base: the tensor being modified
 * children: array of operand tensors, one per operation
 * ops: array of binary operations in execution order
 * num_ops: number of operations (must be >= 1)
 *
 * All tensors must have compatible shapes for broadcasting.
 */
extern "C" void jit_elementwise_inplace_chain(HipDeviceHandle device_handle,
                                              DenseCore *out,
                                              const DenseCore *base,
                                              const DenseCore *children,
                                              const BinOp *ops, size_t num_ops,
                                              StreamHandle stream_handle) {
  LOG_DEBUG("jit_elementwise_inplace_chain: num_ops=%zu out_numel=%lu", num_ops,
            out->num_elements);

  if (num_ops == 0) {
    LOG_ERROR("jit_elementwise_inplace_chain: num_ops must be >= 1");
    return;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);
  JitKernelCache *cache = jit_cache_get();

  size_t num_inputs = num_ops + 1; /* base + one child per op */

  /* Check if all same shape (no broadcasting needed) */
  bool same_shape = (out->shape.len == base->shape.len);
  if (same_shape) {
    for (len_t d = 0; d < out->shape.len; d++) {
      if (out->shape.buffer[d] != base->shape.buffer[d]) {
        same_shape = false;
        break;
      }
    }
  }
  if (same_shape) {
    for (size_t j = 0; j < num_ops; j++) {
      if (out->shape.len != children[j].shape.len) {
        same_shape = false;
        break;
      }
      for (len_t d = 0; d < out->shape.len; d++) {
        if (out->shape.buffer[d] != children[j].shape.buffer[d]) {
          same_shape = false;
          break;
        }
      }
      if (!same_shape)
        break;
    }
  }

  if (same_shape) {
    /* Simple contiguous case */
    CompiledKernel *kernel =
        cache->compile_inplace_chain(dense_core_dtype(out), ops, num_ops, true, out->shape.len);
    if (!kernel)
      return;

    size_t n = out->num_elements;
    void *out_ptr = out->data;

    /* Build args array: out, in_0 (base), in_1, in_2, ..., n */
    void *
        args[MAX_INPLACE_CHAIN_OPS + 3]; /* out + base + up to 8 children + n */
    args[0] = &out_ptr;

    void *base_ptr = const_cast<void *>(base->data);
    args[1] = &base_ptr;

    void *child_ptrs[MAX_INPLACE_CHAIN_OPS];
    for (size_t j = 0; j < num_ops; j++) {
      child_ptrs[j] = const_cast<void *>(children[j].data);
      args[2 + j] = &child_ptrs[j];
    }
    args[2 + num_ops] = &n;

    jit_launch(kernel, n, args, stream);

    /* Free the kernel after use (not cached) */
    jit_kernel_free(kernel);
  } else {
    /* Broadcasting case - compute strides for all inputs */
    len_t base_strides[HIP_MAX_DIMS];
    len_t child_strides[MAX_INPLACE_CHAIN_OPS][HIP_MAX_DIMS];

    if (!compute_broadcast_strides(base->shape.buffer, base->shape.len, out->shape.buffer,
                                   out->shape.len, base_strides)) {
      LOG_ERROR("jit_elementwise_inplace_chain: broadcast failed for base");
      return;
    }
    for (size_t j = 0; j < num_ops; j++) {
      if (!compute_broadcast_strides(children[j].shape.buffer, children[j].shape.len,
                                     out->shape.buffer, out->shape.len, child_strides[j])) {
        LOG_ERROR(
            "jit_elementwise_inplace_chain: broadcast failed for child %zu", j);
        return;
      }
    }

    CompiledKernel *kernel = cache->compile_inplace_chain(
        dense_core_dtype(out), ops, num_ops, false, out->shape.len);
    if (!kernel)
      return;

    size_t ndim = out->shape.len;
    size_t n = out->num_elements;
    void *out_ptr = out->data;

    /* Shape/stride values passed as scalars (no device alloc needed) */
    len_t shape_vals[HIP_MAX_DIMS];
    len_t stride_vals[MAX_INPLACE_CHAIN_OPS + 1][HIP_MAX_DIMS];
    for (size_t d = 0; d < ndim; d++) {
      shape_vals[d] = out->shape.buffer[d];
    }
    /* base strides at index 0, child strides at 1..num_ops */
    for (size_t d = 0; d < ndim; d++) {
      stride_vals[0][d] = base_strides[d];
    }
    for (size_t j = 0; j < num_ops; j++) {
      for (size_t d = 0; d < ndim; d++) {
        stride_vals[1 + j][d] = child_strides[j][d];
      }
    }

    /* Max args: 1 out + num_inputs ptrs + ndim shape + num_inputs*ndim strides + 1 n */
    void *args[(MAX_INPLACE_CHAIN_OPS + 1) * HIP_MAX_DIMS + MAX_INPLACE_CHAIN_OPS + 10];
    size_t arg_idx = 0;

    args[arg_idx++] = &out_ptr;

    /* Input pointers */
    void *base_ptr = const_cast<void *>(base->data);
    args[arg_idx++] = &base_ptr;

    void *child_ptrs[MAX_INPLACE_CHAIN_OPS];
    for (size_t j = 0; j < num_ops; j++) {
      child_ptrs[j] = const_cast<void *>(children[j].data);
      args[arg_idx++] = &child_ptrs[j];
    }

    /* Shape values as individual args */
    for (size_t d = 0; d < ndim; d++) {
      args[arg_idx++] = &shape_vals[d];
    }

    /* Stride values as individual args */
    for (size_t j = 0; j < num_inputs; j++) {
      for (size_t d = 0; d < ndim; d++) {
        args[arg_idx++] = &stride_vals[j][d];
      }
    }

    args[arg_idx++] = &n;

    jit_launch(kernel, n, args, stream);

    /* Free the kernel after use (not cached) */
    jit_kernel_free(kernel);
  }
}

/* ============================================================================
 * Fused Chain Execution (mixed unary/binary)
 *
 * Executes a chain of mixed unary and binary operations in a single kernel.
 *
 * Parameters:
 * - out: output tensor
 * - inputs: array of input tensors (base + one per binary op)
 * - num_inputs: number of input tensors
 * - ops: array of ChainOp (unary or binary)
 * - num_ops: number of operations
 *
 * Example: exp(x) * y
 *   inputs = [x, y], num_inputs = 2
 *   ops = [{unary, MAP_EXP}, {binary, BINOP_MUL}], num_ops = 2
 * ============================================================================
 */
extern "C" void jit_elementwise_fused_chain(HipDeviceHandle device_handle,
                                            DenseCore *out,
                                            const DenseCore *inputs,
                                            size_t num_inputs,
                                            const ChainOp *ops,
                                            size_t num_ops,
                                            StreamHandle stream_handle) {
  LOG_DEBUG("jit_elementwise_fused_chain: num_ops=%zu num_inputs=%zu out_numel=%lu",
            num_ops, num_inputs, out->num_elements);

  if (num_ops == 0 || num_ops > MAX_FUSED_CHAIN_OPS) {
    LOG_ERROR("jit_elementwise_fused_chain: invalid num_ops=%zu", num_ops);
    return;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);

  /* Check if all same shape (no broadcasting needed) */
  bool same_shape = true;
  for (size_t j = 0; j < num_inputs && same_shape; j++) {
    if (out->shape.len != inputs[j].shape.len) {
      same_shape = false;
      break;
    }
    for (len_t d = 0; d < out->shape.len; d++) {
      if (out->shape.buffer[d] != inputs[j].shape.buffer[d]) {
        same_shape = false;
        break;
      }
    }
  }

  if (same_shape) {
    /* Simple contiguous case */

    /* Debug: check capture status at each step */
    auto check_cap = [&](const char* label) {
      hipStreamCaptureStatus cap_status;
      HIP_ASSERT(hipStreamGetCaptureInfo(stream, &cap_status, nullptr));
      LOG_DEBUG("  [fused_chain] %s: capture_status=%d", label, (int)cap_status);
    };

    check_cap("before codegen");
    FusedChainKernel gen = codegen_fused_chain(dense_core_dtype(out), ops, num_ops, true, out->shape.len);
    check_cap("after codegen");

    CompiledKernel *kernel = jit_compile(gen.source, gen.source_len, "kernel", get_compute_capability());
    check_cap("after jit_compile");
    if (!kernel) {
      LOG_ERROR("jit_elementwise_fused_chain: failed to compile kernel");
      return;
    }

    size_t n = out->num_elements;
    void *out_ptr = out->data;

    /* Build args array: out, in_0, in_1, ..., n */
    void *args[MAX_FUSED_CHAIN_OPS + 3];
    size_t arg_idx = 0;
    args[arg_idx++] = &out_ptr;

    void *input_ptrs[MAX_FUSED_CHAIN_OPS + 1];
    for (size_t j = 0; j < num_inputs; j++) {
      input_ptrs[j] = const_cast<void *>(inputs[j].data);
      args[arg_idx++] = &input_ptrs[j];
    }
    args[arg_idx++] = &n;

    jit_launch(kernel, n, args, stream);
    check_cap("after jit_launch");
    jit_kernel_free(kernel);
    check_cap("after jit_kernel_free");
  } else {
    /* Broadcasting case - compute strides for all inputs */
    len_t input_strides[MAX_FUSED_CHAIN_OPS + 1][HIP_MAX_DIMS];

    for (size_t j = 0; j < num_inputs; j++) {
      if (!compute_broadcast_strides(inputs[j].shape.buffer, inputs[j].shape.len,
                                     out->shape.buffer, out->shape.len, input_strides[j])) {
        LOG_ERROR("jit_elementwise_fused_chain: broadcast failed for input %zu", j);
        return;
      }
    }

    FusedChainKernel gen = codegen_fused_chain(dense_core_dtype(out), ops, num_ops, false, out->shape.len);
    CompiledKernel *kernel = jit_compile(gen.source, gen.source_len, "kernel", get_compute_capability());
    if (!kernel) {
      LOG_ERROR("jit_elementwise_fused_chain: failed to compile strided kernel");
      return;
    }

    size_t ndim = out->shape.len;
    size_t n = out->num_elements;
    void *out_ptr = out->data;

    /* Build args: out, in_0, ..., in_N, shape_0..shape_N, stride_0_0..stride_N_D, n
     * All shape/stride values passed as scalars (no device alloc needed) */
    len_t shape_vals[HIP_MAX_DIMS];
    len_t stride_vals[MAX_FUSED_CHAIN_OPS + 1][HIP_MAX_DIMS];
    for (size_t d = 0; d < ndim; d++) {
      shape_vals[d] = out->shape.buffer[d];
    }
    for (size_t j = 0; j < num_inputs; j++) {
      for (size_t d = 0; d < ndim; d++) {
        stride_vals[j][d] = input_strides[j][d];
      }
    }

    /* Max args: 1 out + MAX_FUSED inputs + MAX_DIMS shape + MAX_FUSED*MAX_DIMS strides + 1 n */
    void *args[(MAX_FUSED_CHAIN_OPS + 1) * HIP_MAX_DIMS + MAX_FUSED_CHAIN_OPS + 10];
    size_t arg_idx = 0;

    args[arg_idx++] = &out_ptr;

    void *input_ptrs[MAX_FUSED_CHAIN_OPS + 1];
    for (size_t j = 0; j < num_inputs; j++) {
      input_ptrs[j] = const_cast<void *>(inputs[j].data);
      args[arg_idx++] = &input_ptrs[j];
    }

    /* Shape values as individual args */
    for (size_t d = 0; d < ndim; d++) {
      args[arg_idx++] = &shape_vals[d];
    }

    /* Stride values as individual args */
    for (size_t j = 0; j < num_inputs; j++) {
      for (size_t d = 0; d < ndim; d++) {
        args[arg_idx++] = &stride_vals[j][d];
      }
    }

    args[arg_idx++] = &n;

    jit_launch(kernel, n, args, stream);

    jit_kernel_free(kernel);
  }
}

/* ============================================================================
 * Custom Source Kernel Execution
 *
 * Compiles and launches a pre-generated HIP kernel source string.
 * Used for complex expressions (diamond patterns, deep trees) that can't
 * be represented as flat chain operations.
 *
 * Kernel signature must be:
 *   extern "C" __global__ void kernel(T* out, const T* in_0, ..., size_t n
 *                                     [, size_t _dim0, size_t _dim1, ...])
 *
 * When needs_broadcast is true, the output dimension sizes are appended as
 * additional kernel arguments after n, enabling runtime-parameterized
 * broadcast index computation.
 * ============================================================================
 */
extern "C" void jit_elementwise_from_source(HipDeviceHandle device_handle,
                                            DenseCore *out,
                                            const DenseCore *inputs,
                                            size_t num_inputs,
                                            const char *source,
                                            size_t source_len,
                                            bool needs_broadcast,
                                            StreamHandle stream_handle) {
  LOG_DEBUG("jit_elementwise_from_source: num_inputs=%zu out_numel=%lu source_len=%zu broadcast=%d",
            num_inputs, out->num_elements, source_len, (int)needs_broadcast);

  if (source == nullptr || source_len == 0) {
    LOG_ERROR("jit_elementwise_from_source: null or empty source");
    return;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);

  CompiledKernel *kernel = jit_compile(source, source_len, "kernel", get_compute_capability());
  if (!kernel) {
    LOG_ERROR("jit_elementwise_from_source: failed to compile kernel");
    return;
  }

  size_t n = out->num_elements;
  void *out_ptr = out->data;

  /* Build args array: out, in_0, ..., n [, _dim0, _dim1, ...] */
  void *args[34 + HIP_MAX_DIMS]; /* max 32 inputs + out + n + dims */
  size_t arg_idx = 0;
  args[arg_idx++] = &out_ptr;

  void *input_ptrs[32];
  for (size_t j = 0; j < num_inputs && j < 32; j++) {
    input_ptrs[j] = const_cast<void *>(inputs[j].data);
    args[arg_idx++] = &input_ptrs[j];
  }
  args[arg_idx++] = &n;

  /* Append output dimension sizes for broadcast kernels */
  size_t dim_sizes[HIP_MAX_DIMS];
  if (needs_broadcast) {
    for (size_t d = 0; d < out->shape.len && d < HIP_MAX_DIMS; d++) {
      dim_sizes[d] = out->shape.buffer[d];
      args[arg_idx++] = &dim_sizes[d];
    }
  }

  jit_launch(kernel, n, args, stream);
  jit_kernel_free(kernel);
}

/*
 * Get JIT cache statistics.
 */
extern "C" len_t jit_cache_size() { return jit_cache_get()->size(); }

/*
 * Clear JIT cache.
 */
extern "C" void jit_cache_clear() { jit_cache_get()->clear(); }

/*
 * Compile HIP source to device binary without executing.
 * Returns heap-allocated device code (caller must free with jit_free_device_code).
 * Sets *code_size_out to the device code size.
 * Returns nullptr on compilation failure.
 */
extern "C" const char* jit_compile_to_device_code(
    const char* source,
    size_t source_len,
    size_t* code_size_out
) {
    CompiledKernel* kernel = jit_compile(source, source_len, "kernel", get_compute_capability());
    if (!kernel) {
        *code_size_out = 0;
        return nullptr;
    }

    /* Copy device code to a new allocation for the caller */
    // fprintf(stderr, "[JIT_DEBUG_CPP] kernel->code=%p kernel->code_size=%zu\n",
    //         (void*)kernel->code, kernel->code_size);
    char* code_copy = (char*)malloc(kernel->code_size);
    memcpy(code_copy, kernel->code, kernel->code_size);
    *code_size_out = kernel->code_size;
    // fprintf(stderr, "[JIT_DEBUG_CPP] returning code_copy=%p code_size_out=%zu\n",
    //         (void*)code_copy, *code_size_out);

    jit_kernel_free(kernel);
    return code_copy;
}

/*
 * Free device code returned by jit_compile_to_device_code.
 */
extern "C" void jit_free_device_code(const char* device_code) {
    free(const_cast<char*>(device_code));
}

/*
 * Execute a kernel from cached device binary (no HIPRTC compilation).
 * Same signature as jit_elementwise_from_source but takes device binary instead of HIP source.
 */
extern "C" void jit_elementwise_from_device_code(
    HipDeviceHandle device_handle,
    DenseCore* out,
    const DenseCore* inputs,
    size_t num_inputs,
    const char* device_code,
    size_t code_size,
    bool needs_broadcast,
    StreamHandle stream_handle
) {
    LOG_DEBUG("jit_elementwise_from_device_code: num_inputs=%zu out_numel=%lu code_size=%zu",
              num_inputs, out->num_elements, code_size);

    if (device_code == nullptr || code_size == 0) {
        LOG_ERROR("jit_elementwise_from_device_code: null or empty device code");
        return;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);

    /* Load module directly from device binary */
    hipModule_t module;
    hipError_t err = hipModuleLoadData(&module, device_code);
    if (err != hipSuccess) {
        LOG_ERROR("jit_elementwise_from_device_code: hipModuleLoadData failed: %s",
            hipGetErrorString(err));
        abort();
    }

    hipFunction_t function;
    err = hipModuleGetFunction(&function, module, "kernel");
    if (err != hipSuccess) {
        LOG_ERROR("jit_elementwise_from_device_code: hipModuleGetFunction failed: %s",
            hipGetErrorString(err));
        abort();
    }

    size_t n = out->num_elements;
    void* out_ptr = out->data;

    /* Build args array: out, in_0, ..., n [, _dim0, _dim1, ...] */
    void* args[34 + HIP_MAX_DIMS];
    size_t arg_idx = 0;
    args[arg_idx++] = &out_ptr;

    void* input_ptrs[32];
    for (size_t j = 0; j < num_inputs && j < 32; j++) {
        input_ptrs[j] = const_cast<void*>(inputs[j].data);
        args[arg_idx++] = &input_ptrs[j];
    }
    args[arg_idx++] = &n;

    /* Append output dimension sizes for broadcast kernels */
    size_t dim_sizes[HIP_MAX_DIMS];
    if (needs_broadcast) {
        for (size_t d = 0; d < out->shape.len && d < HIP_MAX_DIMS; d++) {
            dim_sizes[d] = out->shape.buffer[d];
            args[arg_idx++] = &dim_sizes[d];
        }
    }

    /* Launch */
    unsigned int block_size = 256;
    unsigned int grid_size = (n + block_size - 1) / block_size;
    HIP_ASSERT(hipModuleLaunchKernel(function, grid_size, 1, 1, block_size, 1, 1, 0, stream, args, nullptr));

    /* Cleanup */
    HIP_ASSERT(hipModuleUnload(module));
}

/* ============================================================================
 * Handle-based device code API for C3-driven codegen
 *
 * C3 handles: source generation, disk caching, arg array construction,
 *             in-memory handle caching (hash -> JitKernelHandle).
 * C++ handles: module load, kernel launch, module unload.
 * ============================================================================
 */

typedef struct {
    hipModule_t module;
    hipFunction_t function;
} JitKernel;

/*
 * Load device binary into a persistent kernel handle.
 * Returns heap-allocated JitKernel, or nullptr on failure.
 * Caller (C3) caches this and passes to jit_launch_kernel.
 */
extern "C" void* jit_load_device_code(
    const char* device_code,
    size_t code_size
) {
    if (device_code == nullptr || code_size == 0) {
        LOG_ERROR("jit_load_device_code: null or empty device code");
        return nullptr;
    }

    hipModule_t module;
    hipError_t err = hipModuleLoadData(&module, device_code);
    if (err != hipSuccess) {
        LOG_ERROR("jit_load_device_code: hipModuleLoadData failed: %s",
            hipGetErrorString(err));
        abort();
    }

    hipFunction_t function;
    err = hipModuleGetFunction(&function, module, "kernel");
    if (err != hipSuccess) {
        LOG_ERROR("jit_load_device_code: hipModuleGetFunction failed: %s",
            hipGetErrorString(err));
        abort();
    }

    JitKernel* k = (JitKernel*)malloc(sizeof(JitKernel));
    k->module = module;
    k->function = function;
    return k;
}

/*
 * Launch a previously loaded kernel.
 * args: void** array matching the kernel signature, built by C3.
 */
extern "C" void jit_launch_kernel(
    void* kernel_handle,
    void** args,
    size_t num_elements,
    StreamHandle stream_handle
) {
    JitKernel* k = (JitKernel*)kernel_handle;
    if (k == nullptr) return;

    hipStream_t stream = static_cast<hipStream_t>(stream_handle.ptr);
    unsigned int block_size = 256;
    unsigned int grid_size = (num_elements + block_size - 1) / block_size;
    HIP_ASSERT(hipModuleLaunchKernel(k->function, grid_size, 1, 1, block_size, 1, 1, 0, stream, args, nullptr));
}

/*
 * Unload a kernel module and free the handle.
 */
extern "C" void jit_unload_kernel(void* kernel_handle) {
    JitKernel* k = (JitKernel*)kernel_handle;
    if (k == nullptr) return;
    HIP_ASSERT(hipModuleUnload(k->module));
    free(k);
}

/* ============================================================================
 * Codegen-only API for C3 compile phase
 *
 * Generates kernel source at build time so C3 can cache via get_or_load_kernel.
 * ============================================================================
 */

/*
 * Generate fused chain kernel source into caller buffer.
 * Returns source length, or 0 on failure.
 *
 * dtype:       0=f32, 1=f64, 2=f16
 * ops/num_ops: ChainOp array from linearization
 * contiguous:  true if all inputs have same shape (no broadcast)
 * ndim:        number of dimensions of output tensor
 * buf/buf_max: caller-provided buffer for source
 */
extern "C" size_t jit_codegen_fused_chain_source(
    int dtype,
    const ChainOp *ops,
    size_t num_ops,
    bool contiguous,
    size_t ndim,
    char *buf,
    size_t buf_max
) {
    FusedChainKernel gen = codegen_fused_chain(
        static_cast<Dtype>(dtype), ops, num_ops, contiguous, static_cast<len_t>(ndim)
    );
    if (gen.source_len == 0 || gen.source_len >= buf_max) return 0;
    memcpy(buf, gen.source, gen.source_len);
    buf[gen.source_len] = '\0';
    return gen.source_len;
}

#endif /* __METAPHOR_JIT_ELEMENTWISE_H__ */
