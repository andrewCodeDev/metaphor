/*
 * jit/codegen.cpp - HIP kernel source code generation
 *
 * Generates HIP kernel source strings for elementwise operations.
 * Handles broadcasting via stride tricks (stride=0 for broadcast dims).
 */

#ifndef __METAPHOR_JIT_CODEGEN_H__
#define __METAPHOR_JIT_CODEGEN_H__

#include "../interop.h"
#include "../logging.h"
#include <cstdio>
#include <cstring>

#define FMT_HEADER_ONLY
#include <fmt/format.h>

/* ============================================================================
 * Operation String Tables
 * ============================================================================
 */

static const char *unary_op_expr(MapOp op) {
  switch (op) {
  case MAP_IDENTITY:
    return "x";
  case MAP_NEG:
    return "(-x)";
  case MAP_ABS:
    return "(x < 0 ? -x : x)";
  case MAP_SQR:
    return "(x * x)";
  case MAP_SQRT:
    return "sqrtf(x)";
  case MAP_RECIP:
    return "(1.0f / x)";
  case MAP_EXP:
    return "expf(x)";
  case MAP_LOG:
    return "logf(x)";
  case MAP_SIN:
    return "sinf(x)";
  case MAP_COS:
    return "cosf(x)";
  case MAP_TANH:
    return "tanhf(x)";
  case MAP_RELU:
    return "(x > 0 ? x : 0)";
  case MAP_HEAVISIDE:
    return "(x > 0 ? 1.0f : 0.0f)";
  case MAP_SIGMOID:
    return "(1.0f / (1.0f + expf(-x)))";
  case MAP_GELU:
    return "(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * "
           "x))))";
  case MAP_SILU:
    return "(x / (1.0f + expf(-x)))";
  case MAP_SOFTPLUS:
    // softplus(x) = log(1 + exp(x)), stable: max(x,0) + log(1 + exp(-|x|))
    return "(fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x))))";
  case MAP_NOT:
    return "(x == 0 ? 1.0f : 0.0f)";
  default:
    return "x";
  }
}

static const char *unary_op_expr_f64(MapOp op) {
  switch (op) {
  case MAP_IDENTITY:
    return "x";
  case MAP_NEG:
    return "(-x)";
  case MAP_ABS:
    return "(x < 0 ? -x : x)";
  case MAP_SQR:
    return "(x * x)";
  case MAP_SQRT:
    return "sqrt(x)";
  case MAP_RECIP:
    return "(1.0 / x)";
  case MAP_EXP:
    return "exp(x)";
  case MAP_LOG:
    return "log(x)";
  case MAP_SIN:
    return "sin(x)";
  case MAP_COS:
    return "cos(x)";
  case MAP_TANH:
    return "tanh(x)";
  case MAP_RELU:
    return "(x > 0 ? x : 0)";
  case MAP_HEAVISIDE:
    return "(x > 0 ? 1.0 : 0.0)";
  case MAP_SIGMOID:
    return "(1.0 / (1.0 + exp(-x)))";
  case MAP_GELU:
    return "(0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * "
           "x))))";
  case MAP_SILU:
    return "(x / (1.0 + exp(-x)))";
  case MAP_SOFTPLUS:
    return "(fmax(x, 0.0) + log(1.0 + exp(-fabs(x))))";
  case MAP_NOT:
    return "(x == 0 ? 1.0 : 0.0)";
  default:
    return "x";
  }
}

/* Binary op expression for f32 (default) */
static const char *binary_op_expr(BinOp op) {
  switch (op) {
  case BINOP_ADD:
    return "(a + b)";
  case BINOP_SUB:
    return "(a - b)";
  case BINOP_MUL:
    return "(a * b)";
  case BINOP_DIV:
    return "(a / b)";
  case BINOP_MAX:
    return "(a > b ? a : b)";
  case BINOP_MIN:
    return "(a < b ? a : b)";
  case BINOP_EQ:
    return "(a == b ? 1.0f : 0.0f)";
  case BINOP_NE:
    return "(a != b ? 1.0f : 0.0f)";
  case BINOP_GT:
    return "(a > b ? 1.0f : 0.0f)";
  case BINOP_LT:
    return "(a < b ? 1.0f : 0.0f)";
  case BINOP_GTE:
    return "(a >= b ? 1.0f : 0.0f)";
  case BINOP_LTE:
    return "(a <= b ? 1.0f : 0.0f)";
  case BINOP_AND:
    return "((a != 0 && b != 0) ? 1.0f : 0.0f)";
  case BINOP_OR:
    return "((a != 0 || b != 0) ? 1.0f : 0.0f)";
  case BINOP_XOR:
    return "(((a != 0) != (b != 0)) ? 1.0f : 0.0f)";
  default:
    return nullptr; /* Caller must handle unknown ops */
  }
}

/* F16 binary op expressions using HIP half intrinsics (SM 5.3+ required) */
static const char *binary_op_expr_f16(BinOp op) {
  switch (op) {
  case BINOP_ADD:
    return "(__hadd(a, b))";
  case BINOP_SUB:
    return "(__hsub(a, b))";
  case BINOP_MUL:
    return "(__hmul(a, b))";
  case BINOP_DIV:
    return "(__hdiv(a, b))";
  case BINOP_MAX:
    return "(__hgt(a, b) ? a : b)";
  case BINOP_MIN:
    return "(__hlt(a, b) ? a : b)";
  case BINOP_EQ:
    return "(__heq(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_NE:
    return "(__hne(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_GT:
    return "(__hgt(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_LT:
    return "(__hlt(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_GTE:
    return "(__hge(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_LTE:
    return "(__hle(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_AND:
    return "((__hne(a, __float2half(0.0f)) && __hne(b, __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_OR:
    return "((__hne(a, __float2half(0.0f)) || __hne(b, __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f))";
  case BINOP_XOR:
    return "(((__hne(a, __float2half(0.0f))) != (__hne(b, __float2half(0.0f)))) ? __float2half(1.0f) : __float2half(0.0f))";
  default:
    return nullptr;
  }
}

/* BF16 binary op expressions - use float conversion for compatibility */
static const char *binary_op_expr_bf16(BinOp op) {
  /*
   * CUSTOM KERNEL REQUIRED: BF16 native arithmetic intrinsics require SM 8.0+.
   * For now, convert to float, compute, convert back.
   */
  switch (op) {
  case BINOP_ADD:
    return "(hip_bfloat16((float)(a) + (float)(b)))";
  case BINOP_SUB:
    return "(hip_bfloat16((float)(a) - (float)(b)))";
  case BINOP_MUL:
    return "(hip_bfloat16((float)(a) * (float)(b)))";
  case BINOP_DIV:
    return "(hip_bfloat16((float)(a) / (float)(b)))";
  case BINOP_MAX:
    return "((float)(a) > (float)(b) ? a : b)";
  case BINOP_MIN:
    return "((float)(a) < (float)(b) ? a : b)";
  case BINOP_EQ:
    return "((float)(a) == (float)(b) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_NE:
    return "((float)(a) != (float)(b) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_GT:
    return "((float)(a) > (float)(b) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_LT:
    return "((float)(a) < (float)(b) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_GTE:
    return "((float)(a) >= (float)(b) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_LTE:
    return "((float)(a) <= (float)(b) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_AND:
    return "(((float)(a) != 0 && (float)(b) != 0) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_OR:
    return "(((float)(a) != 0 || (float)(b) != 0) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case BINOP_XOR:
    return "((((float)(a) != 0) != ((float)(b) != 0)) ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  default:
    return nullptr;
  }
}

/* Select the appropriate binary expression based on dtype */
static const char *binary_op_expr_for_dtype(Dtype dtype, BinOp op) {
  switch (dtype) {
  case DTYPE_F16:
    return binary_op_expr_f16(op);
  case DTYPE_BF16:
    return binary_op_expr_bf16(op);
  default:
    return binary_op_expr(op);  /* F32/F64 use same expressions */
  }
}

/* Binary op as function call (for inplace chains) - f32 */
static const char *binary_op_func_f32(BinOp op) {
  switch (op) {
  case BINOP_ADD:
    return nullptr; /* Use + operator */
  case BINOP_SUB:
    return nullptr; /* Use - operator */
  case BINOP_MUL:
    return nullptr; /* Use * operator */
  case BINOP_DIV:
    return nullptr; /* Use / operator */
  case BINOP_MAX:
    return "fmaxf";
  case BINOP_MIN:
    return "fminf";
  default:
    return nullptr;
  }
}

/* Binary op as function call (for inplace chains) - f64 */
static const char *binary_op_func_f64(BinOp op) {
  switch (op) {
  case BINOP_ADD:
    return nullptr; /* Use + operator */
  case BINOP_SUB:
    return nullptr; /* Use - operator */
  case BINOP_MUL:
    return nullptr; /* Use * operator */
  case BINOP_DIV:
    return nullptr; /* Use / operator */
  case BINOP_MAX:
    return "fmax";
  case BINOP_MIN:
    return "fmin";
  default:
    return nullptr;
  }
}

/* Binary op as infix operator */
static const char *binary_op_infix(BinOp op) {
  switch (op) {
  case BINOP_ADD:
    return "+";
  case BINOP_SUB:
    return "-";
  case BINOP_MUL:
    return "*";
  case BINOP_DIV:
    return "/";
  default:
    return nullptr;
  }
}

static const char *dtype_name(Dtype dtype) {
  switch (dtype) {
  case DTYPE_F32:
    return "float";
  case DTYPE_F64:
    return "double";
  case DTYPE_F16:
    return "__half";
  case DTYPE_BF16:
    return "hip_bfloat16";
  case DTYPE_U8:
    return "unsigned char";
  default:
    return "float";
  }
}

/*
 * NOTE: F16/BF16 CUSTOM KERNEL REQUIREMENTS
 *
 * HIP provides half-precision math intrinsics (hsqrt, hexp, hlog, hsin, hcos,
 * htanh, etc.) but they require SM 5.3+ and are device-only. The JIT compiler
 * can emit these intrinsics, but the generated kernels will fail on older GPUs.
 *
 * For broader compatibility, consider:
 * 1. Converting __half to float, computing, converting back (performance cost)
 * 2. Using vectorized __half2 operations for better throughput
 * 3. Writing specialized HIP kernels for critical operations
 *
 * Current implementation: Uses HIP half intrinsics directly, requires SM 5.3+.
 * BF16 intrinsics require SM 8.0+ (Ampere) for native support.
 */

/* F16 unary operations using HIP half intrinsics (SM 5.3+ required) */
static const char *unary_op_expr_f16(MapOp op) {
  switch (op) {
  case MAP_IDENTITY:
    return "x";
  case MAP_NEG:
    return "(__hneg(x))";
  case MAP_ABS:
    return "(__habs(x))";
  case MAP_SQR:
    return "(__hmul(x, x))";
  case MAP_SQRT:
    return "(hsqrt(x))";  /* SM 5.3+ */
  case MAP_RECIP:
    return "(__hdiv(__float2half(1.0f), x))";
  case MAP_EXP:
    return "(hexp(x))";   /* SM 5.3+ */
  case MAP_LOG:
    return "(hlog(x))";   /* SM 5.3+ */
  case MAP_SIN:
    return "(hsin(x))";   /* SM 5.3+ */
  case MAP_COS:
    return "(hcos(x))";   /* SM 5.3+ */
  case MAP_TANH:
    /* tanh = (exp(2x) - 1) / (exp(2x) + 1), or use __float2half(tanhf(__half2float(x))) */
    return "(__float2half(tanhf(__half2float(x))))";
  case MAP_RELU:
    return "(__hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f))";
  case MAP_HEAVISIDE:
    return "(__hgt(x, __float2half(0.0f)) ? __float2half(1.0f) : __float2half(0.0f))";
  case MAP_SIGMOID:
    /* sigmoid = 1 / (1 + exp(-x)) */
    return "(__hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x)))))";
  case MAP_GELU:
    /* GELU approximation - use float conversion for accuracy */
    return "(__float2half(0.5f * __half2float(x) * (1.0f + tanhf(0.7978845608f * (__half2float(x) + 0.044715f * __half2float(x) * __half2float(x) * __half2float(x))))))";
  case MAP_SILU:
    /* SiLU = x * sigmoid(x) */
    return "(__hmul(x, __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x))))))";
  case MAP_SOFTPLUS:
    /* softplus = log(1 + exp(x)), stable via float conversion */
    return "(__float2half(fmaxf(__half2float(x), 0.0f) + logf(1.0f + expf(-fabsf(__half2float(x))))))";
  case MAP_NOT:
    return "(__heq(x, __float2half(0.0f)) ? __float2half(1.0f) : __float2half(0.0f))";
  case MAP_SIGN:
    return "(__hgt(x, __float2half(0.0f)) ? __float2half(1.0f) : (__hlt(x, __float2half(0.0f)) ? __float2half(-1.0f) : __float2half(0.0f)))";
  default:
    return "x";
  }
}

/* BF16 unary operations - NOTE: requires SM 8.0+ for native intrinsics */
static const char *unary_op_expr_bf16(MapOp op) {
  /*
   * CUSTOM KERNEL REQUIRED: BF16 native intrinsics require SM 8.0+ (Ampere).
   * For now, convert to float, compute, convert back.
   * This is a placeholder - proper BF16 support needs architecture detection.
   */
  switch (op) {
  case MAP_IDENTITY:
    return "x";
  case MAP_NEG:
    return "(hip_bfloat16(-(float)(x)))";
  case MAP_ABS:
    return "(hip_bfloat16(fabsf((float)(x))))";
  case MAP_SQR:
    return "(hip_bfloat16((float)(x) * (float)(x)))";
  case MAP_SQRT:
    return "(hip_bfloat16(sqrtf((float)(x))))";
  case MAP_RECIP:
    return "(hip_bfloat16(1.0f / (float)(x)))";
  case MAP_EXP:
    return "(hip_bfloat16(expf((float)(x))))";
  case MAP_LOG:
    return "(hip_bfloat16(logf((float)(x))))";
  case MAP_SIN:
    return "(hip_bfloat16(sinf((float)(x))))";
  case MAP_COS:
    return "(hip_bfloat16(cosf((float)(x))))";
  case MAP_TANH:
    return "(hip_bfloat16(tanhf((float)(x))))";
  case MAP_RELU:
    return "((float)(x) > 0 ? x : hip_bfloat16(0.0f))";
  case MAP_HEAVISIDE:
    return "((float)(x) > 0 ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case MAP_SIGMOID:
    return "(hip_bfloat16(1.0f / (1.0f + expf(-(float)(x)))))";
  case MAP_GELU:
    return "(hip_bfloat16(0.5f * (float)(x) * (1.0f + tanhf(0.7978845608f * ((float)(x) + 0.044715f * (float)(x) * (float)(x) * (float)(x))))))";
  case MAP_SILU:
    return "(hip_bfloat16((float)(x) / (1.0f + expf(-(float)(x)))))";
  case MAP_SOFTPLUS:
    return "(hip_bfloat16(fmaxf((float)(x), 0.0f) + logf(1.0f + expf(-fabsf((float)(x))))))";
  case MAP_NOT:
    return "((float)(x) == 0 ? hip_bfloat16(1.0f) : hip_bfloat16(0.0f))";
  case MAP_SIGN:
    return "((float)(x) > 0 ? hip_bfloat16(1.0f) : ((float)(x) < 0 ? hip_bfloat16(-1.0f) : hip_bfloat16(0.0f)))";
  default:
    return "x";
  }
}

/* Returns true if this binary op produces boolean (u8) output */
static bool binop_produces_bool(BinOp op) {
  switch (op) {
  case BINOP_EQ:
  case BINOP_NE:
  case BINOP_GT:
  case BINOP_LT:
  case BINOP_GTE:
  case BINOP_LTE:
  case BINOP_AND:
  case BINOP_OR:
  case BINOP_XOR:
    return true;
  default:
    return false;
  }
}

/* Returns true if this is a comparison op (float input -> bool output) */
static bool binop_is_comparison(BinOp op) {
  switch (op) {
  case BINOP_EQ:
  case BINOP_NE:
  case BINOP_GT:
  case BINOP_LT:
  case BINOP_GTE:
  case BINOP_LTE:
    return true;
  default:
    return false;
  }
}

/* Returns true if this is a logical op (bool input -> bool output) */
static bool binop_is_logical(BinOp op) {
  switch (op) {
  case BINOP_AND:
  case BINOP_OR:
  case BINOP_XOR:
    return true;
  default:
    return false;
  }
}

/* Binary op expression that produces u8 (0 or 1) */
static const char *binary_op_expr_u8(BinOp op) {
  switch (op) {
  case BINOP_EQ:
    return "((unsigned char)(a == b))";
  case BINOP_NE:
    return "((unsigned char)(a != b))";
  case BINOP_GT:
    return "((unsigned char)(a > b))";
  case BINOP_LT:
    return "((unsigned char)(a < b))";
  case BINOP_GTE:
    return "((unsigned char)(a >= b))";
  case BINOP_LTE:
    return "((unsigned char)(a <= b))";
  case BINOP_AND:
    return "((unsigned char)(a != 0 && b != 0))";
  case BINOP_OR:
    return "((unsigned char)(a != 0 || b != 0))";
  case BINOP_XOR:
    return "((unsigned char)((a != 0) != (b != 0)))";
  default:
    return "((unsigned char)0)";
  }
}

/* Logical op expression for u8 input -> u8 output */
static const char *logical_op_expr_u8(BinOp op) {
  switch (op) {
  case BINOP_AND:
    return "((unsigned char)(a && b))";
  case BINOP_OR:
    return "((unsigned char)(a || b))";
  case BINOP_XOR:
    return "((unsigned char)(a != b))";
  default:
    return "((unsigned char)0)";
  }
}

/* Unary op expression for u8 NOT */
static const char *unary_op_expr_u8(MapOp op) {
  switch (op) {
  case MAP_NOT:
    return "((unsigned char)(x == 0))";
  default:
    return "x";
  }
}

/* ============================================================================
 * Kernel Generation
 * ============================================================================
 */

/* Buffer size for generated kernel source */
#define KERNEL_SOURCE_MAX 4096

/* Select the appropriate unary expression based on dtype */
static const char *unary_op_expr_for_dtype(Dtype dtype, MapOp op) {
  switch (dtype) {
  case DTYPE_F64:
    return unary_op_expr_f64(op);
  case DTYPE_F16:
    return unary_op_expr_f16(op);
  case DTYPE_BF16:
    return unary_op_expr_bf16(op);
  default:
    return unary_op_expr(op);  /* F32 or fallback */
  }
}

/* Header includes needed for F16/BF16 JIT kernels */
static const char *jit_header_for_dtype(Dtype dtype) {
  switch (dtype) {
  case DTYPE_F16:
    return "#include <hip/hip_fp16.h>\n";
  case DTYPE_BF16:
    return "#include <hip/hip_bfloat16.h>\n";
  default:
    return "";
  }
}

/*
 * Generate unary elementwise kernel (contiguous memory).
 *
 * Template:
 *   out[i] = op(in[i])
 */
static size_t gen_unary_contiguous(char *buf, size_t buf_size, Dtype dtype,
                                   MapOp op) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);
  const char *expr = unary_op_expr_for_dtype(dtype, op);

  return snprintf(buf, buf_size,
                  "%s"
                  "extern \"C\" __global__ void kernel(\n"
                  "    %s* __restrict__ out,\n"
                  "    const %s* __restrict__ in,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i < n) {\n"
                  "        %s x = in[i];\n"
                  "        out[i] = %s;\n"
                  "    }\n"
                  "}\n",
                  header, T, T, T, expr);
}

/*
 * Generate unary elementwise kernel with strides (for
 * non-contiguous/broadcast).
 *
 * Handles arbitrary shapes by computing multi-index from flat index.
 * Strides of 0 indicate broadcast dimensions.
 */
static size_t gen_unary_strided(char *buf, size_t buf_size, Dtype dtype,
                                MapOp op, len_t ndim) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);
  const char *expr = unary_op_expr_for_dtype(dtype, op);

  /* Use fixed-size arrays passed by value - graph capture safe, no device
   * malloc needed */
  return snprintf(buf, buf_size,
                  "%s"
                  "extern \"C\" __global__ void kernel(\n"
                  "    %s* __restrict__ out,\n"
                  "    const %s* __restrict__ in,\n"
                  "    size_t out_shape_0, size_t out_shape_1, size_t "
                  "out_shape_2, size_t out_shape_3,\n"
                  "    size_t out_shape_4, size_t out_shape_5, size_t "
                  "out_shape_6, size_t out_shape_7,\n"
                  "    size_t in_strides_0, size_t in_strides_1, size_t "
                  "in_strides_2, size_t in_strides_3,\n"
                  "    size_t in_strides_4, size_t in_strides_5, size_t "
                  "in_strides_6, size_t in_strides_7,\n"
                  "    size_t ndim,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    const size_t out_shape[8] = {out_shape_0, out_shape_1, "
                  "out_shape_2, out_shape_3,\n"
                  "                                  out_shape_4, out_shape_5, "
                  "out_shape_6, out_shape_7};\n"
                  "    const size_t in_strides[8] = {in_strides_0, "
                  "in_strides_1, in_strides_2, in_strides_3,\n"
                  "                                   in_strides_4, "
                  "in_strides_5, in_strides_6, in_strides_7};\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i >= n) return;\n"
                  "\n"
                  "    // Compute multi-index from flat output index\n"
                  "    size_t idx = i;\n"
                  "    size_t in_offset = 0;\n"
                  "    for (int d = %lu - 1; d >= 0; d--) {\n"
                  "        size_t coord = idx %% out_shape[d];\n"
                  "        idx /= out_shape[d];\n"
                  "        in_offset += coord * in_strides[d];\n"
                  "    }\n"
                  "\n"
                  "    %s x = in[in_offset];\n"
                  "    out[i] = %s;\n"
                  "}\n",
                  header, T, T, (unsigned long)ndim, T, expr);
}

/*
 * Generate binary elementwise kernel (contiguous memory, same shape).
 *
 * Template:
 *   out[i] = op(a[i], b[i])
 */
static size_t gen_binary_contiguous(char *buf, size_t buf_size, Dtype dtype,
                                    BinOp op) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);
  const char *expr = binary_op_expr_for_dtype(dtype, op);

  return snprintf(buf, buf_size,
                  "%s"
                  "extern \"C\" __global__ void kernel(\n"
                  "    %s* __restrict__ out,\n"
                  "    const %s* __restrict__ in_a,\n"
                  "    const %s* __restrict__ in_b,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i < n) {\n"
                  "        %s a = in_a[i];\n"
                  "        %s b = in_b[i];\n"
                  "        out[i] = %s;\n"
                  "    }\n"
                  "}\n",
                  header, T, T, T, T, T, expr);
}

/*
 * Generate binary elementwise kernel with strides (for broadcasting).
 *
 * Handles arbitrary broadcast patterns via stride=0 for broadcast dims.
 * Output shape is the broadcast result shape.
 */
static size_t gen_binary_strided(char *buf, size_t buf_size, Dtype dtype,
                                 BinOp op, len_t ndim) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);
  const char *expr = binary_op_expr_for_dtype(dtype, op);

  /* Use fixed-size arrays passed by value - graph capture safe, no device
   * malloc needed */
  return snprintf(
      buf, buf_size,
      "%s"
      "extern \"C\" __global__ void kernel(\n"
      "    %s* __restrict__ out,\n"
      "    const %s* __restrict__ in_a,\n"
      "    const %s* __restrict__ in_b,\n"
      "    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t "
      "out_shape_3,\n"
      "    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t "
      "out_shape_7,\n"
      "    size_t a_strides_0, size_t a_strides_1, size_t a_strides_2, size_t "
      "a_strides_3,\n"
      "    size_t a_strides_4, size_t a_strides_5, size_t a_strides_6, size_t "
      "a_strides_7,\n"
      "    size_t b_strides_0, size_t b_strides_1, size_t b_strides_2, size_t "
      "b_strides_3,\n"
      "    size_t b_strides_4, size_t b_strides_5, size_t b_strides_6, size_t "
      "b_strides_7,\n"
      "    size_t ndim,\n"
      "    size_t n\n"
      ") {\n"
      "    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, "
      "out_shape_3,\n"
      "                                  out_shape_4, out_shape_5, "
      "out_shape_6, out_shape_7};\n"
      "    const size_t a_strides[8] = {a_strides_0, a_strides_1, a_strides_2, "
      "a_strides_3,\n"
      "                                  a_strides_4, a_strides_5, "
      "a_strides_6, a_strides_7};\n"
      "    const size_t b_strides[8] = {b_strides_0, b_strides_1, b_strides_2, "
      "b_strides_3,\n"
      "                                  b_strides_4, b_strides_5, "
      "b_strides_6, b_strides_7};\n"
      "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "    if (i >= n) return;\n"
      "\n"
      "    // Compute multi-index from flat output index\n"
      "    size_t idx = i;\n"
      "    size_t a_offset = 0, b_offset = 0;\n"
      "    for (int d = %lu - 1; d >= 0; d--) {\n"
      "        size_t coord = idx %% out_shape[d];\n"
      "        idx /= out_shape[d];\n"
      "        a_offset += coord * a_strides[d];  // stride=0 broadcasts\n"
      "        b_offset += coord * b_strides[d];\n"
      "    }\n"
      "\n"
      "    %s a = in_a[a_offset];\n"
      "    %s b = in_b[b_offset];\n"
      "    out[i] = %s;\n"
      "}\n",
      header, T, T, T, (unsigned long)ndim, T, T, expr);
}

/* ============================================================================
 * Comparison Kernel Generation (float/double input -> u8 output)
 * ============================================================================
 */

/*
 * Generate comparison kernel (contiguous memory).
 * Input: float or double, Output: unsigned char (0 or 1)
 */
static size_t gen_comparison_contiguous(char *buf, size_t buf_size,
                                        Dtype input_dtype, BinOp op) {
  const char *T = dtype_name(input_dtype);
  const char *header = jit_header_for_dtype(input_dtype);
  const char *expr = binary_op_expr_u8(op);

  return snprintf(buf, buf_size,
                  "%s"
                  "extern \"C\" __global__ void kernel(\n"
                  "    unsigned char* __restrict__ out,\n"
                  "    const %s* __restrict__ in_a,\n"
                  "    const %s* __restrict__ in_b,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i < n) {\n"
                  "        %s a = in_a[i];\n"
                  "        %s b = in_b[i];\n"
                  "        out[i] = %s;\n"
                  "    }\n"
                  "}\n",
                  header, T, T, T, T, expr);
}

/*
 * Generate comparison kernel with strides (for broadcasting).
 * Input: float or double, Output: unsigned char (0 or 1)
 */
static size_t gen_comparison_strided(char *buf, size_t buf_size,
                                     Dtype input_dtype, BinOp op, len_t ndim) {
  const char *T = dtype_name(input_dtype);
  const char *header = jit_header_for_dtype(input_dtype);
  const char *expr = binary_op_expr_u8(op);

  /* Use fixed-size arrays passed by value - graph capture safe */
  return snprintf(buf, buf_size,
                  "%s"
                  "extern \"C\" __global__ void kernel(\n"
                  "    unsigned char* __restrict__ out,\n"
                  "    const %s* __restrict__ in_a,\n"
                  "    const %s* __restrict__ in_b,\n"
                  "    size_t out_shape_0, size_t out_shape_1, size_t "
                  "out_shape_2, size_t out_shape_3,\n"
                  "    size_t out_shape_4, size_t out_shape_5, size_t "
                  "out_shape_6, size_t out_shape_7,\n"
                  "    size_t a_strides_0, size_t a_strides_1, size_t "
                  "a_strides_2, size_t a_strides_3,\n"
                  "    size_t a_strides_4, size_t a_strides_5, size_t "
                  "a_strides_6, size_t a_strides_7,\n"
                  "    size_t b_strides_0, size_t b_strides_1, size_t "
                  "b_strides_2, size_t b_strides_3,\n"
                  "    size_t b_strides_4, size_t b_strides_5, size_t "
                  "b_strides_6, size_t b_strides_7,\n"
                  "    size_t ndim,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    const size_t out_shape[8] = {out_shape_0, out_shape_1, "
                  "out_shape_2, out_shape_3,\n"
                  "                                  out_shape_4, out_shape_5, "
                  "out_shape_6, out_shape_7};\n"
                  "    const size_t a_strides[8] = {a_strides_0, a_strides_1, "
                  "a_strides_2, a_strides_3,\n"
                  "                                  a_strides_4, a_strides_5, "
                  "a_strides_6, a_strides_7};\n"
                  "    const size_t b_strides[8] = {b_strides_0, b_strides_1, "
                  "b_strides_2, b_strides_3,\n"
                  "                                  b_strides_4, b_strides_5, "
                  "b_strides_6, b_strides_7};\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i >= n) return;\n"
                  "\n"
                  "    size_t idx = i;\n"
                  "    size_t a_offset = 0, b_offset = 0;\n"
                  "    for (int d = %lu - 1; d >= 0; d--) {\n"
                  "        size_t coord = idx %% out_shape[d];\n"
                  "        idx /= out_shape[d];\n"
                  "        a_offset += coord * a_strides[d];\n"
                  "        b_offset += coord * b_strides[d];\n"
                  "    }\n"
                  "\n"
                  "    %s a = in_a[a_offset];\n"
                  "    %s b = in_b[b_offset];\n"
                  "    out[i] = %s;\n"
                  "}\n",
                  header, T, T, (unsigned long)ndim, T, T, expr);
}

/* ============================================================================
 * Logical Kernel Generation (u8 input -> u8 output)
 * ============================================================================
 */

/*
 * Generate logical kernel (contiguous memory).
 * Input: unsigned char (0 or 1), Output: unsigned char (0 or 1)
 */
static size_t gen_logical_contiguous(char *buf, size_t buf_size, BinOp op) {
  const char *expr = logical_op_expr_u8(op);

  return snprintf(buf, buf_size,
                  "extern \"C\" __global__ void kernel(\n"
                  "    unsigned char* __restrict__ out,\n"
                  "    const unsigned char* __restrict__ in_a,\n"
                  "    const unsigned char* __restrict__ in_b,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i < n) {\n"
                  "        unsigned char a = in_a[i];\n"
                  "        unsigned char b = in_b[i];\n"
                  "        out[i] = %s;\n"
                  "    }\n"
                  "}\n",
                  expr);
}

/*
 * Generate logical kernel with strides (for broadcasting).
 * Input: unsigned char (0 or 1), Output: unsigned char (0 or 1)
 */
static size_t gen_logical_strided(char *buf, size_t buf_size, BinOp op,
                                  len_t ndim) {
  const char *expr = logical_op_expr_u8(op);

  /* Use fixed-size arrays passed by value - graph capture safe */
  return snprintf(buf, buf_size,
                  "extern \"C\" __global__ void kernel(\n"
                  "    unsigned char* __restrict__ out,\n"
                  "    const unsigned char* __restrict__ in_a,\n"
                  "    const unsigned char* __restrict__ in_b,\n"
                  "    size_t out_shape_0, size_t out_shape_1, size_t "
                  "out_shape_2, size_t out_shape_3,\n"
                  "    size_t out_shape_4, size_t out_shape_5, size_t "
                  "out_shape_6, size_t out_shape_7,\n"
                  "    size_t a_strides_0, size_t a_strides_1, size_t "
                  "a_strides_2, size_t a_strides_3,\n"
                  "    size_t a_strides_4, size_t a_strides_5, size_t "
                  "a_strides_6, size_t a_strides_7,\n"
                  "    size_t b_strides_0, size_t b_strides_1, size_t "
                  "b_strides_2, size_t b_strides_3,\n"
                  "    size_t b_strides_4, size_t b_strides_5, size_t "
                  "b_strides_6, size_t b_strides_7,\n"
                  "    size_t ndim,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    const size_t out_shape[8] = {out_shape_0, out_shape_1, "
                  "out_shape_2, out_shape_3,\n"
                  "                                  out_shape_4, out_shape_5, "
                  "out_shape_6, out_shape_7};\n"
                  "    const size_t a_strides[8] = {a_strides_0, a_strides_1, "
                  "a_strides_2, a_strides_3,\n"
                  "                                  a_strides_4, a_strides_5, "
                  "a_strides_6, a_strides_7};\n"
                  "    const size_t b_strides[8] = {b_strides_0, b_strides_1, "
                  "b_strides_2, b_strides_3,\n"
                  "                                  b_strides_4, b_strides_5, "
                  "b_strides_6, b_strides_7};\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i >= n) return;\n"
                  "\n"
                  "    size_t idx = i;\n"
                  "    size_t a_offset = 0, b_offset = 0;\n"
                  "    for (int d = %lu - 1; d >= 0; d--) {\n"
                  "        size_t coord = idx %% out_shape[d];\n"
                  "        idx /= out_shape[d];\n"
                  "        a_offset += coord * a_strides[d];\n"
                  "        b_offset += coord * b_strides[d];\n"
                  "    }\n"
                  "\n"
                  "    unsigned char a = in_a[a_offset];\n"
                  "    unsigned char b = in_b[b_offset];\n"
                  "    out[i] = %s;\n"
                  "}\n",
                  (unsigned long)ndim, expr);
}

/* ============================================================================
 * Select Kernel Generation (ternary: u8 mask, T true, T false -> T)
 * ============================================================================
 */

/*
 * Generate select kernel (contiguous memory).
 * mask: unsigned char (0 or 1)
 * true_vals, false_vals: float or double
 * out[i] = mask[i] ? true_vals[i] : false_vals[i]
 */
static size_t gen_select_contiguous(char *buf, size_t buf_size, Dtype dtype) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);

  return snprintf(buf, buf_size,
                  "%s"
                  "extern \"C\" __global__ void kernel(\n"
                  "    %s* __restrict__ out,\n"
                  "    const unsigned char* __restrict__ mask,\n"
                  "    const %s* __restrict__ true_vals,\n"
                  "    const %s* __restrict__ false_vals,\n"
                  "    size_t n\n"
                  ") {\n"
                  "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if (i < n) {\n"
                  "        out[i] = mask[i] ? true_vals[i] : false_vals[i];\n"
                  "    }\n"
                  "}\n",
                  header, T, T, T);
}

/*
 * Generate select kernel with strides (for broadcasting).
 */
static size_t gen_select_strided(char *buf, size_t buf_size, Dtype dtype,
                                 len_t ndim) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);

  /* Use fixed-size arrays passed by value - graph capture safe */
  /* 4 arrays * 8 dims = 32 size_t params */
  return snprintf(
      buf, buf_size,
      "%s"
      "extern \"C\" __global__ void kernel(\n"
      "    %s* __restrict__ out,\n"
      "    const unsigned char* __restrict__ mask,\n"
      "    const %s* __restrict__ true_vals,\n"
      "    const %s* __restrict__ false_vals,\n"
      "    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t "
      "out_shape_3,\n"
      "    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t "
      "out_shape_7,\n"
      "    size_t mask_strides_0, size_t mask_strides_1, size_t "
      "mask_strides_2, size_t mask_strides_3,\n"
      "    size_t mask_strides_4, size_t mask_strides_5, size_t "
      "mask_strides_6, size_t mask_strides_7,\n"
      "    size_t true_strides_0, size_t true_strides_1, size_t "
      "true_strides_2, size_t true_strides_3,\n"
      "    size_t true_strides_4, size_t true_strides_5, size_t "
      "true_strides_6, size_t true_strides_7,\n"
      "    size_t false_strides_0, size_t false_strides_1, size_t "
      "false_strides_2, size_t false_strides_3,\n"
      "    size_t false_strides_4, size_t false_strides_5, size_t "
      "false_strides_6, size_t false_strides_7,\n"
      "    size_t ndim,\n"
      "    size_t n\n"
      ") {\n"
      "    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, "
      "out_shape_3,\n"
      "                                  out_shape_4, out_shape_5, "
      "out_shape_6, out_shape_7};\n"
      "    const size_t mask_strides[8] = {mask_strides_0, mask_strides_1, "
      "mask_strides_2, mask_strides_3,\n"
      "                                     mask_strides_4, mask_strides_5, "
      "mask_strides_6, mask_strides_7};\n"
      "    const size_t true_strides[8] = {true_strides_0, true_strides_1, "
      "true_strides_2, true_strides_3,\n"
      "                                     true_strides_4, true_strides_5, "
      "true_strides_6, true_strides_7};\n"
      "    const size_t false_strides[8] = {false_strides_0, false_strides_1, "
      "false_strides_2, false_strides_3,\n"
      "                                      false_strides_4, false_strides_5, "
      "false_strides_6, false_strides_7};\n"
      "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "    if (i >= n) return;\n"
      "\n"
      "    size_t idx = i;\n"
      "    size_t mask_offset = 0, true_offset = 0, false_offset = 0;\n"
      "    for (int d = %lu - 1; d >= 0; d--) {\n"
      "        size_t coord = idx %% out_shape[d];\n"
      "        idx /= out_shape[d];\n"
      "        mask_offset += coord * mask_strides[d];\n"
      "        true_offset += coord * true_strides[d];\n"
      "        false_offset += coord * false_strides[d];\n"
      "    }\n"
      "\n"
      "    out[i] = mask[mask_offset] ? true_vals[true_offset] : "
      "false_vals[false_offset];\n"
      "}\n",
      header, T, T, T, (unsigned long)ndim);
}

/* ============================================================================
 * Inplace Chain Kernel Generation
 *
 * Generates fused kernels for chains of inplace operations like _clamp (_min
 * then _max). For N ops, we have N+1 inputs: in_0 (base tensor), in_1..in_N
 * (one child per op). Expression is built as nested function calls:
 * fmaxf(fminf(in_0, in_1), in_2)
 * ============================================================================
 */

#define MAX_INPLACE_CHAIN_OPS 8

/*
 * Build nested expression for inplace chain.
 * ops: array of binary ops [op_0, op_1, ...] where op_i uses in_i+1
 * num_ops: number of operations
 * is_f64: true for double precision
 * is_strided: true for strided access (offset_N), false for contiguous ([i])
 *
 * For [min, max] with contiguous access:
 *   fmaxf(fminf(in_0[i], in_1[i]), in_2[i])
 */
static std::string build_inplace_chain_expr(const BinOp *ops, size_t num_ops,
                                            bool is_f64, bool is_strided) {
  if (num_ops == 0 || num_ops > MAX_INPLACE_CHAIN_OPS) {
    return "ERROR_INVALID_CHAIN";
  }

  /* Build expression from inside out:
   * For ops = [min, max], result is: max(min(in_0, in_1), in_2)
   * Start with in_0, then wrap with each op and child */

  std::string expr = is_strided ? "in_0[offset_0]" : "in_0[i]";

  for (size_t j = 0; j < num_ops; j++) {
    BinOp op = ops[j];
    const char *func = is_f64 ? binary_op_func_f64(op) : binary_op_func_f32(op);
    const char *infix = binary_op_infix(op);

    std::string child = is_strided
                            ? fmt::format("in_{}[offset_{}]", j + 1, j + 1)
                            : fmt::format("in_{}[i]", j + 1);

    if (func != nullptr) {
      expr = fmt::format("{}({}, {})", func, expr, child);
    } else if (infix != nullptr) {
      expr = fmt::format("({} {} {})", expr, infix, child);
    } else {
      return fmt::format("ERROR_UNSUPPORTED_OP_{}", static_cast<int>(op));
    }
  }

  return expr;
}

/* ============================================================================
 * Fused Chain Code Generation (mixed unary/binary)
 *
 * Builds kernels that fuse sequences of unary and binary operations.
 * Example: exp(x) * y becomes: out[i] = expf(in_0[i]) * in_1[i]
 * ============================================================================
 */

/*
 * Build nested expression for fused chain with mixed unary/binary ops.
 * ops: array of ChainOp (either unary or binary)
 * num_ops: number of operations
 * dtype: data type for selecting proper intrinsics
 * is_strided: true for strided access
 *
 * For unary ops: apply to previous result, no new input needed
 * For binary ops: apply to previous result and next input
 *
 * Example: [MAP_EXP, BINOP_MUL] with 2 inputs (in_0, in_1):
 *   expf(in_0[i]) * in_1[i]
 */
static std::string build_fused_chain_expr(const ChainOp *ops, size_t num_ops,
                                          Dtype dtype, bool is_strided) {
  if (num_ops == 0 || num_ops > MAX_FUSED_CHAIN_OPS) {
    return "ERROR_INVALID_CHAIN";
  }

  /* Start with first input */
  std::string expr = is_strided ? "in_0[offset_0]" : "in_0[i]";
  size_t next_input = 1; /* Next child input index for binary ops */

  for (size_t j = 0; j < num_ops; j++) {
    if (ops[j].is_unary) {
      /* Unary op: wrap expression with unary function */
      MapOp op = ops[j].unary_op;
      const char *tmpl = unary_op_expr_for_dtype(dtype, op);

      /* Replace standalone 'x' with the current expression.
       * Must check that 'x' is not part of an identifier like 'hexp'. */
      std::string tmpl_str(tmpl);
      std::string result;
      for (size_t k = 0; k < tmpl_str.size(); k++) {
        char c = tmpl_str[k];
        bool prev_is_alnum = (k > 0) && (isalnum(tmpl_str[k-1]) || tmpl_str[k-1] == '_');
        bool next_is_alnum = (k + 1 < tmpl_str.size()) && (isalnum(tmpl_str[k+1]) || tmpl_str[k+1] == '_');

        if (c == 'x' && !prev_is_alnum && !next_is_alnum) {
          result += expr;
        } else {
          result += c;
        }
      }
      expr = result;
    } else {
      /* Binary op: combine with next child input */
      BinOp op = ops[j].binary_op;
      const char *bin_expr = binary_op_expr_for_dtype(dtype, op);

      std::string child = is_strided
                              ? fmt::format("in_{}[offset_{}]", next_input, next_input)
                              : fmt::format("in_{}[i]", next_input);
      next_input++;

      if (bin_expr != nullptr) {
        /* Replace standalone 'a' with expr, standalone 'b' with child.
         * Must check that 'a'/'b' are not part of identifiers. */
        std::string bin_str(bin_expr);
        std::string result;
        for (size_t k = 0; k < bin_str.size(); k++) {
          char c = bin_str[k];
          bool prev_is_alnum = (k > 0) && (isalnum(bin_str[k-1]) || bin_str[k-1] == '_');
          bool next_is_alnum = (k + 1 < bin_str.size()) && (isalnum(bin_str[k+1]) || bin_str[k+1] == '_');

          if (c == 'a' && !prev_is_alnum && !next_is_alnum) {
            result += expr;
          } else if (c == 'b' && !prev_is_alnum && !next_is_alnum) {
            result += child;
          } else {
            result += c;
          }
        }
        expr = result;
      } else {
        return fmt::format("ERROR_UNSUPPORTED_OP_{}", static_cast<int>(op));
      }
    }
  }

  return expr;
}

/*
 * Count number of binary ops in chain (determines number of child inputs needed).
 */
static size_t count_binary_ops(const ChainOp *ops, size_t num_ops) {
  size_t count = 0;
  for (size_t i = 0; i < num_ops; i++) {
    if (!ops[i].is_unary) count++;
  }
  return count;
}

/*
 * Generate fused chain kernel (contiguous memory).
 */
static std::string gen_fused_chain_contiguous(Dtype dtype, const ChainOp *ops,
                                              size_t num_ops) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);
  size_t num_binary = count_binary_ops(ops, num_ops);
  size_t num_inputs = num_binary + 1; /* base + one child per binary op */

  std::string src = header;
  src += fmt::format("extern \"C\" __global__ void kernel(\n"
                     "    {}* __restrict__ out,\n",
                     T);

  for (size_t j = 0; j < num_inputs; j++) {
    src += fmt::format("    const {}* __restrict__ in_{},\n", T, j);
  }

  src += "    size_t n\n"
         ") {\n"
         "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
         "    if (i < n) {\n"
         "        out[i] = ";

  src += build_fused_chain_expr(ops, num_ops, dtype, false);

  src += ";\n"
         "    }\n"
         "}\n";

  return src;
}

/*
 * Generate fused chain kernel with strides (for broadcasting).
 */
static std::string gen_fused_chain_strided(Dtype dtype, const ChainOp *ops,
                                           size_t num_ops, len_t ndim) {
  const char *T = dtype_name(dtype);
  const char *header = jit_header_for_dtype(dtype);
  size_t num_binary = count_binary_ops(ops, num_ops);
  size_t num_inputs = num_binary + 1;

  std::string src = header;
  src += fmt::format("extern \"C\" __global__ void kernel(\n"
                     "    {}* __restrict__ out,\n",
                     T);

  for (size_t j = 0; j < num_inputs; j++) {
    src += fmt::format("    const {}* __restrict__ in_{},\n", T, j);
  }

  /* Shape and stride values passed as individual scalar arguments */
  for (size_t d = 0; d < ndim; d++) {
    src += fmt::format("    size_t out_shape_{},\n", d);
  }

  for (size_t j = 0; j < num_inputs; j++) {
    for (size_t d = 0; d < ndim; d++) {
      src += fmt::format("    size_t in_{}_stride_{},\n", j, d);
    }
  }

  src += "    size_t n\n"
         ") {\n"
         "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
         "    if (i < n) {\n";

  /* Compute multi-index from flat index */
  src += "        size_t rem = i;\n";
  for (size_t d = 0; d < ndim; d++) {
    src += fmt::format("        size_t idx_{} = rem / (", d);
    src += "1";
    for (size_t dd = d + 1; dd < ndim; dd++) {
      src += fmt::format(" * out_shape_{}", dd);
    }
    src += ");\n";
    src += fmt::format("        rem = rem % (", d);
    src += "1";
    for (size_t dd = d + 1; dd < ndim; dd++) {
      src += fmt::format(" * out_shape_{}", dd);
    }
    src += ");\n";
  }

  /* Compute offsets for each input */
  for (size_t j = 0; j < num_inputs; j++) {
    src += fmt::format("        size_t offset_{} = ", j);
    for (size_t d = 0; d < ndim; d++) {
      if (d > 0) src += " + ";
      src += fmt::format("idx_{} * in_{}_stride_{}", d, j, d);
    }
    src += ";\n";
  }

  src += "        out[i] = ";
  src += build_fused_chain_expr(ops, num_ops, dtype, true);
  src += ";\n"
         "    }\n"
         "}\n";

  return src;
}

/* Kernel type for fused chain */
typedef enum {
  KERNEL_FUSED_CHAIN_CONTIGUOUS,
  KERNEL_FUSED_CHAIN_STRIDED,
} FusedChainKernelType;

/* Generated kernel info for fused chain */
typedef struct {
  FusedChainKernelType type;
  Dtype dtype;
  len_t ndim;
  size_t num_ops;
  ChainOp ops[MAX_FUSED_CHAIN_OPS];
  char source[KERNEL_SOURCE_MAX];
  size_t source_len;
} FusedChainKernel;

static FusedChainKernel codegen_fused_chain(Dtype dtype, const ChainOp *ops,
                                            size_t num_ops, bool contiguous,
                                            len_t ndim) {
  FusedChainKernel k = {};
  k.dtype = dtype;
  k.ndim = ndim;
  k.num_ops = num_ops;

  /* Copy ops array */
  size_t copy_count = num_ops < MAX_FUSED_CHAIN_OPS ? num_ops : MAX_FUSED_CHAIN_OPS;
  for (size_t i = 0; i < copy_count; i++) {
    k.ops[i] = ops[i];
  }

  std::string src;
  if (contiguous) {
    k.type = KERNEL_FUSED_CHAIN_CONTIGUOUS;
    src = gen_fused_chain_contiguous(dtype, ops, num_ops);
  } else {
    k.type = KERNEL_FUSED_CHAIN_STRIDED;
    src = gen_fused_chain_strided(dtype, ops, num_ops, ndim);
  }

  k.source_len = std::min(src.size(), static_cast<size_t>(KERNEL_SOURCE_MAX - 1));
  memcpy(k.source, src.c_str(), k.source_len);
  k.source[k.source_len] = '\0';

  LOG_DEBUG("codegen_fused_chain: type=%d dtype=%d num_ops=%zu ndim=%lu len=%zu",
            k.type, dtype, num_ops, ndim, k.source_len);

  return k;
}

/*
 * Generate inplace chain kernel (contiguous memory).
 * num_ops: number of binary operations in chain
 * ops: array of binary ops
 *
 * Template (for 2 ops [min, max]):
 *   out[i] = fmaxf(fminf(in_0[i], in_1[i]), in_2[i])
 */
static std::string gen_inplace_chain_contiguous(Dtype dtype, const BinOp *ops,
                                                size_t num_ops) {
  const char *T = dtype_name(dtype);
  bool is_f64 = (dtype == DTYPE_F64);
  size_t num_inputs = num_ops + 1;

  std::string src = fmt::format("extern \"C\" __global__ void kernel(\n"
                                "    {}* __restrict__ out,\n",
                                T);

  for (size_t j = 0; j < num_inputs; j++) {
    src += fmt::format("    const {}* __restrict__ in_{},\n", T, j);
  }

  src += "    size_t n\n"
         ") {\n"
         "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
         "    if (i < n) {\n"
         "        out[i] = ";

  src += build_inplace_chain_expr(ops, num_ops, is_f64, false);

  src += ";\n"
         "    }\n"
         "}\n";

  return src;
}

/*
 * Generate inplace chain kernel with strides (for broadcasting).
 * num_ops: number of binary operations in chain
 * ops: array of binary ops
 * ndim: number of dimensions
 */
static std::string gen_inplace_chain_strided(Dtype dtype, const BinOp *ops,
                                             size_t num_ops, len_t ndim) {
  const char *T = dtype_name(dtype);
  bool is_f64 = (dtype == DTYPE_F64);
  size_t num_inputs = num_ops + 1;

  std::string src = fmt::format("extern \"C\" __global__ void kernel(\n"
                                "    {}* __restrict__ out,\n",
                                T);

  for (size_t j = 0; j < num_inputs; j++) {
    src += fmt::format("    const {}* __restrict__ in_{},\n", T, j);
  }

  /* Shape and stride values passed as individual scalar arguments */
  for (size_t d = 0; d < ndim; d++) {
    src += fmt::format("    size_t out_shape_{},\n", d);
  }

  for (size_t j = 0; j < num_inputs; j++) {
    for (size_t d = 0; d < ndim; d++) {
      src += fmt::format("    size_t in_{}_stride_{},\n", j, d);
    }
  }

  src += fmt::format("    size_t n\n"
                     ") {{\n"
                     "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                     "    if (i >= n) return;\n"
                     "\n"
                     "    size_t idx = i;\n"
                     "    size_t offset_0 = 0");

  for (size_t j = 1; j < num_inputs; j++) {
    src += fmt::format(", offset_{} = 0", j);
  }
  src += ";\n";

  /* Unrolled dimension loop (was: for d = ndim-1 downto 0) */
  for (int d = (int)ndim - 1; d >= 0; d--) {
    src += fmt::format("    {{ size_t coord = idx % out_shape_{0};\n"
                       "      idx /= out_shape_{0};\n", d);
    for (size_t j = 0; j < num_inputs; j++) {
      src += fmt::format("      offset_{0} += coord * in_{0}_stride_{1};\n", j, d);
    }
    src += "    }\n";
  }

  src += "\n    out[i] = ";
  src += build_inplace_chain_expr(ops, num_ops, is_f64, true);
  src += ";\n}\n";

  return src;
}

/* ============================================================================
 * Public API
 * ============================================================================
 */

typedef enum {
  KERNEL_UNARY_CONTIGUOUS,
  KERNEL_UNARY_STRIDED,
  KERNEL_BINARY_CONTIGUOUS,
  KERNEL_BINARY_STRIDED,
  KERNEL_COMPARISON_CONTIGUOUS,
  KERNEL_COMPARISON_STRIDED,
  KERNEL_LOGICAL_CONTIGUOUS,
  KERNEL_LOGICAL_STRIDED,
  KERNEL_SELECT_CONTIGUOUS,
  KERNEL_SELECT_STRIDED,
  KERNEL_INPLACE_CHAIN_CONTIGUOUS,
  KERNEL_INPLACE_CHAIN_STRIDED,
} KernelType;

typedef struct {
  char source[KERNEL_SOURCE_MAX];
  size_t source_len;
  KernelType type;
  Dtype dtype;
  union {
    MapOp map_op;
    BinOp bin_op;
  };
  len_t ndim; /* For strided kernels */
} GeneratedKernel;

/*
 * Generate a unary elementwise kernel.
 */
static GeneratedKernel codegen_unary(Dtype dtype, MapOp op, bool contiguous,
                                     len_t ndim) {
  GeneratedKernel k = {};
  k.dtype = dtype;
  k.map_op = op;
  k.ndim = ndim;

  if (contiguous) {
    k.type = KERNEL_UNARY_CONTIGUOUS;
    k.source_len = gen_unary_contiguous(k.source, KERNEL_SOURCE_MAX, dtype, op);
  } else {
    k.type = KERNEL_UNARY_STRIDED;
    k.source_len =
        gen_unary_strided(k.source, KERNEL_SOURCE_MAX, dtype, op, ndim);
  }

  LOG_DEBUG("codegen_unary: type=%d op=%d dtype=%d ndim=%lu len=%zu", k.type,
            op, dtype, ndim, k.source_len);

  return k;
}

/*
 * Generate a binary elementwise kernel.
 */
static GeneratedKernel codegen_binary(Dtype dtype, BinOp op, bool contiguous,
                                      len_t ndim) {
  GeneratedKernel k = {};
  k.dtype = dtype;
  k.bin_op = op;
  k.ndim = ndim;

  if (contiguous) {
    k.type = KERNEL_BINARY_CONTIGUOUS;
    k.source_len =
        gen_binary_contiguous(k.source, KERNEL_SOURCE_MAX, dtype, op);
  } else {
    k.type = KERNEL_BINARY_STRIDED;
    k.source_len =
        gen_binary_strided(k.source, KERNEL_SOURCE_MAX, dtype, op, ndim);
  }

  LOG_DEBUG("codegen_binary: type=%d op=%d dtype=%d ndim=%lu len=%zu", k.type,
            op, dtype, ndim, k.source_len);

  return k;
}

/*
 * Generate a comparison kernel (float/double input -> u8 output).
 * input_dtype: dtype of the input tensors (f32 or f64)
 */
static GeneratedKernel codegen_comparison(Dtype input_dtype, BinOp op,
                                          bool contiguous, len_t ndim) {
  GeneratedKernel k = {};
  k.dtype = input_dtype; /* Store input dtype for cache key */
  k.bin_op = op;
  k.ndim = ndim;

  if (contiguous) {
    k.type = KERNEL_COMPARISON_CONTIGUOUS;
    k.source_len =
        gen_comparison_contiguous(k.source, KERNEL_SOURCE_MAX, input_dtype, op);
  } else {
    k.type = KERNEL_COMPARISON_STRIDED;
    k.source_len = gen_comparison_strided(k.source, KERNEL_SOURCE_MAX,
                                          input_dtype, op, ndim);
  }

  LOG_DEBUG("codegen_comparison: type=%d op=%d input_dtype=%d ndim=%lu len=%zu",
            k.type, op, input_dtype, ndim, k.source_len);

  return k;
}

/*
 * Generate a logical kernel (u8 input -> u8 output).
 */
static GeneratedKernel codegen_logical(BinOp op, bool contiguous, len_t ndim) {
  GeneratedKernel k = {};
  k.dtype = DTYPE_U8; /* Output is always u8 */
  k.bin_op = op;
  k.ndim = ndim;

  if (contiguous) {
    k.type = KERNEL_LOGICAL_CONTIGUOUS;
    k.source_len = gen_logical_contiguous(k.source, KERNEL_SOURCE_MAX, op);
  } else {
    k.type = KERNEL_LOGICAL_STRIDED;
    k.source_len = gen_logical_strided(k.source, KERNEL_SOURCE_MAX, op, ndim);
  }

  LOG_DEBUG("codegen_logical: type=%d op=%d ndim=%lu len=%zu", k.type, op, ndim,
            k.source_len);

  return k;
}

/*
 * Generate a select kernel (ternary: u8 mask, T true, T false -> T).
 */
static GeneratedKernel codegen_select(Dtype dtype, bool contiguous,
                                      len_t ndim) {
  GeneratedKernel k = {};
  k.dtype = dtype;
  k.ndim = ndim;

  if (contiguous) {
    k.type = KERNEL_SELECT_CONTIGUOUS;
    k.source_len = gen_select_contiguous(k.source, KERNEL_SOURCE_MAX, dtype);
  } else {
    k.type = KERNEL_SELECT_STRIDED;
    k.source_len = gen_select_strided(k.source, KERNEL_SOURCE_MAX, dtype, ndim);
  }

  LOG_DEBUG("codegen_select: type=%d dtype=%d ndim=%lu len=%zu", k.type, dtype,
            ndim, k.source_len);

  return k;
}

/*
 * Extended kernel struct for inplace chains (stores ops array).
 */
typedef struct {
  char source[KERNEL_SOURCE_MAX];
  size_t source_len;
  KernelType type;
  Dtype dtype;
  len_t ndim;
  size_t num_ops;
  BinOp ops[MAX_INPLACE_CHAIN_OPS];
} InplaceChainKernel;

/*
 * Generate an inplace chain kernel (fused sequence of binary ops).
 * ops: array of binary operations in execution order
 * num_ops: number of operations (must be >= 1)
 * contiguous: true if all inputs are contiguous
 * ndim: number of dimensions (for strided kernels)
 *
 * For clamp (min then max):
 *   ops = [BINOP_MIN, BINOP_MAX], num_ops = 2
 *   Generated: out[i] = fmaxf(fminf(in_0[i], in_1[i]), in_2[i])
 */
static InplaceChainKernel codegen_inplace_chain(Dtype dtype, const BinOp *ops,
                                                size_t num_ops, bool contiguous,
                                                len_t ndim) {
  InplaceChainKernel k = {};
  k.dtype = dtype;
  k.ndim = ndim;
  k.num_ops = num_ops;

  /* Copy ops array */
  size_t copy_count =
      num_ops < MAX_INPLACE_CHAIN_OPS ? num_ops : MAX_INPLACE_CHAIN_OPS;
  for (size_t i = 0; i < copy_count; i++) {
    k.ops[i] = ops[i];
  }

  std::string src;
  if (contiguous) {
    k.type = KERNEL_INPLACE_CHAIN_CONTIGUOUS;
    src = gen_inplace_chain_contiguous(dtype, ops, num_ops);
  } else {
    k.type = KERNEL_INPLACE_CHAIN_STRIDED;
    src = gen_inplace_chain_strided(dtype, ops, num_ops, ndim);
  }

  /* Copy to fixed buffer */
  k.source_len =
      std::min(src.size(), static_cast<size_t>(KERNEL_SOURCE_MAX - 1));
  memcpy(k.source, src.c_str(), k.source_len);
  k.source[k.source_len] = '\0';

  LOG_DEBUG(
      "codegen_inplace_chain: type=%d dtype=%d num_ops=%zu ndim=%lu len=%zu",
      k.type, dtype, num_ops, ndim, k.source_len);

  return k;
}

#endif /* __METAPHOR_JIT_CODEGEN_H__ */
