//! CUDA Kernel Source Code Generation (C3 port)
//!
//! Generates CUDA kernel source strings for elementwise operations.
//! Ported from src/cuda/jit/codegen.cu to enable C3-side disk caching
//! of compiled PTX for ALL elementwise kernel types.
//!
//! All functions write into a caller-provided char buffer and return
//! the number of bytes written (excluding null terminator).

module metaphor::device::cuda_device::codegen;
import metaphor::device::common::operations;
import metaphor::rtti;
import std::io;

// Constants

const usz KERNEL_SOURCE_MAX = 4096;

// Dtype Helpers

fn char[] cuda_dtype_name(rtti::Datatype dtype)
{
	switch (dtype)
	{
		case F32:  return "float";
		case F64:  return "double";
		case F16:  return "__half";
		case BF16: return "__nv_bfloat16";
		case BOOL: return "unsigned char";
		case U8:   return "unsigned char";
		default:   unreachable("cuda_dtype_name: unsupported dtype");
	}
}

fn char[] jit_header_for_dtype(rtti::Datatype dtype)
{
	switch (dtype)
	{
		case F16:  return "#include <cuda_fp16.h>\n";
		case BF16: return "#include <cuda_bf16.h>\n";
		default:   return "";
	}
}

// Unary Op Expression Tables

fn char[] unary_op_expr_f32(operations::UnaryOp op)
{
	switch (op)
	{
		case IDENTITY:  return "x";
		case NEG:       return "(-x)";
		case ABS:       return "(x < 0 ? -x : x)";
		case SQR:       return "(x * x)";
		case SQRT:      return "sqrtf(x)";
		case RECIP:     return "(1.0f / x)";
		case EXP:       return "expf(x)";
		case LOG:       return "logf(x)";
		case SIN:       return "sinf(x)";
		case COS:       return "cosf(x)";
		case TANH:      return "tanhf(x)";
		case RELU:      return "(x > 0 ? x : 0)";
		case HEAVISIDE: return "(x > 0 ? 1.0f : 0.0f)";
		case SIGMOID:   return "(1.0f / (1.0f + expf(-x)))";
		case GELU:      return "(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))";
		case SILU:      return "(x / (1.0f + expf(-x)))";
		case SOFTPLUS:  return "(fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x))))";
		case NOT:       return "(x == 0 ? 1.0f : 0.0f)";
		case SIGN:      return "(x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f))";
	}
}

fn char[] unary_op_expr_f64(operations::UnaryOp op)
{
	switch (op)
	{
		case IDENTITY:  return "x";
		case NEG:       return "(-x)";
		case ABS:       return "(x < 0 ? -x : x)";
		case SQR:       return "(x * x)";
		case SQRT:      return "sqrt(x)";
		case RECIP:     return "(1.0 / x)";
		case EXP:       return "exp(x)";
		case LOG:       return "log(x)";
		case SIN:       return "sin(x)";
		case COS:       return "cos(x)";
		case TANH:      return "tanh(x)";
		case RELU:      return "(x > 0 ? x : 0)";
		case HEAVISIDE: return "(x > 0 ? 1.0 : 0.0)";
		case SIGMOID:   return "(1.0 / (1.0 + exp(-x)))";
		case GELU:      return "(0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))))";
		case SILU:      return "(x / (1.0 + exp(-x)))";
		case SOFTPLUS:  return "(fmax(x, 0.0) + log(1.0 + exp(-fabs(x))))";
		case NOT:       return "(x == 0 ? 1.0 : 0.0)";
		case SIGN:      return "(x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0))";
	}
}

fn char[] unary_op_expr_f16(operations::UnaryOp op)
{
	switch (op)
	{
		case IDENTITY:  return "x";
		case NEG:       return "(__hneg(x))";
		case ABS:       return "(__habs(x))";
		case SQR:       return "(__hmul(x, x))";
		case SQRT:      return "(hsqrt(x))";
		case RECIP:     return "(__hdiv(__float2half(1.0f), x))";
		case EXP:       return "(hexp(x))";
		case LOG:       return "(hlog(x))";
		case SIN:       return "(hsin(x))";
		case COS:       return "(hcos(x))";
		case TANH:      return "(__float2half(tanhf(__half2float(x))))";
		case RELU:      return "(__hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f))";
		case HEAVISIDE: return "(__hgt(x, __float2half(0.0f)) ? __float2half(1.0f) : __float2half(0.0f))";
		case SIGMOID:   return "(__hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x)))))";
		case GELU:      return "(__float2half(0.5f * __half2float(x) * (1.0f + tanhf(0.7978845608f * (__half2float(x) + 0.044715f * __half2float(x) * __half2float(x) * __half2float(x))))))";
		case SILU:      return "(__hmul(x, __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x))))))";
		case SOFTPLUS:  return "(__float2half(fmaxf(__half2float(x), 0.0f) + logf(1.0f + expf(-fabsf(__half2float(x))))))";
		case NOT:       return "(__heq(x, __float2half(0.0f)) ? __float2half(1.0f) : __float2half(0.0f))";
		case SIGN:      return "(__hgt(x, __float2half(0.0f)) ? __float2half(1.0f) : (__hlt(x, __float2half(0.0f)) ? __float2half(-1.0f) : __float2half(0.0f)))";
	}
}

fn char[] unary_op_expr_bf16(operations::UnaryOp op)
{
	switch (op)
	{
		case IDENTITY:  return "x";
		case NEG:       return "(__float2bfloat16(-__bfloat162float(x)))";
		case ABS:       return "(__float2bfloat16(fabsf(__bfloat162float(x))))";
		case SQR:       return "(__float2bfloat16(__bfloat162float(x) * __bfloat162float(x)))";
		case SQRT:      return "(__float2bfloat16(sqrtf(__bfloat162float(x))))";
		case RECIP:     return "(__float2bfloat16(1.0f / __bfloat162float(x)))";
		case EXP:       return "(__float2bfloat16(expf(__bfloat162float(x))))";
		case LOG:       return "(__float2bfloat16(logf(__bfloat162float(x))))";
		case SIN:       return "(__float2bfloat16(sinf(__bfloat162float(x))))";
		case COS:       return "(__float2bfloat16(cosf(__bfloat162float(x))))";
		case TANH:      return "(__float2bfloat16(tanhf(__bfloat162float(x))))";
		case RELU:      return "(__bfloat162float(x) > 0 ? x : __float2bfloat16(0.0f))";
		case HEAVISIDE: return "(__bfloat162float(x) > 0 ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case SIGMOID:   return "(__float2bfloat16(1.0f / (1.0f + expf(-__bfloat162float(x)))))";
		case GELU:      return "(__float2bfloat16(0.5f * __bfloat162float(x) * (1.0f + tanhf(0.7978845608f * (__bfloat162float(x) + 0.044715f * __bfloat162float(x) * __bfloat162float(x) * __bfloat162float(x))))))";
		case SILU:      return "(__float2bfloat16(__bfloat162float(x) / (1.0f + expf(-__bfloat162float(x)))))";
		case SOFTPLUS:  return "(__float2bfloat16(fmaxf(__bfloat162float(x), 0.0f) + logf(1.0f + expf(-fabsf(__bfloat162float(x))))))";
		case NOT:       return "(__bfloat162float(x) == 0 ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case SIGN:      return "(__bfloat162float(x) > 0 ? __float2bfloat16(1.0f) : (__bfloat162float(x) < 0 ? __float2bfloat16(-1.0f) : __float2bfloat16(0.0f)))";
	}
}

fn char[] unary_op_expr_for_dtype(rtti::Datatype dtype, operations::UnaryOp op)
{
	switch (dtype)
	{
		case F64:  return unary_op_expr_f64(op);
		case F16:  return unary_op_expr_f16(op);
		case BF16: return unary_op_expr_bf16(op);
		default:   return unary_op_expr_f32(op);
	}
}

fn char[] unary_op_expr_u8(operations::UnaryOp op)
{
	switch (op)
	{
		case NOT: return "((unsigned char)(x == 0))";
		default:  return "x";
	}
}

// Binary Op Expression Tables

fn char[] binary_op_expr_f32(operations::BinaryOp op)
{
	switch (op)
	{
		case ADD: return "(a + b)";
		case SUB: return "(a - b)";
		case MUL: return "(a * b)";
		case DIV: return "(a / b)";
		case MAX: return "(a > b ? a : b)";
		case MIN: return "(a < b ? a : b)";
		case EQL: return "(a == b ? 1.0f : 0.0f)";
		case NEQ: return "(a != b ? 1.0f : 0.0f)";
		case GT:  return "(a > b ? 1.0f : 0.0f)";
		case LT:  return "(a < b ? 1.0f : 0.0f)";
		case GTE: return "(a >= b ? 1.0f : 0.0f)";
		case LTE: return "(a <= b ? 1.0f : 0.0f)";
		case AND: return "((a != 0 && b != 0) ? 1.0f : 0.0f)";
		case OR:  return "((a != 0 || b != 0) ? 1.0f : 0.0f)";
		case XOR: return "(((a != 0) != (b != 0)) ? 1.0f : 0.0f)";
	}
}

fn char[] binary_op_expr_f16(operations::BinaryOp op)
{
	switch (op)
	{
		case ADD: return "(__hadd(a, b))";
		case SUB: return "(__hsub(a, b))";
		case MUL: return "(__hmul(a, b))";
		case DIV: return "(__hdiv(a, b))";
		case MAX: return "(__hgt(a, b) ? a : b)";
		case MIN: return "(__hlt(a, b) ? a : b)";
		case EQL: return "(__heq(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
		case NEQ: return "(__hne(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
		case GT:  return "(__hgt(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
		case LT:  return "(__hlt(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
		case GTE: return "(__hge(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
		case LTE: return "(__hle(a, b) ? __float2half(1.0f) : __float2half(0.0f))";
		case AND: return "((__hne(a, __float2half(0.0f)) && __hne(b, __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f))";
		case OR:  return "((__hne(a, __float2half(0.0f)) || __hne(b, __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f))";
		case XOR: return "(((__hne(a, __float2half(0.0f))) != (__hne(b, __float2half(0.0f)))) ? __float2half(1.0f) : __float2half(0.0f))";
	}
}

fn char[] binary_op_expr_bf16(operations::BinaryOp op)
{
	switch (op)
	{
		case ADD: return "(__float2bfloat16(__bfloat162float(a) + __bfloat162float(b)))";
		case SUB: return "(__float2bfloat16(__bfloat162float(a) - __bfloat162float(b)))";
		case MUL: return "(__float2bfloat16(__bfloat162float(a) * __bfloat162float(b)))";
		case DIV: return "(__float2bfloat16(__bfloat162float(a) / __bfloat162float(b)))";
		case MAX: return "(__bfloat162float(a) > __bfloat162float(b) ? a : b)";
		case MIN: return "(__bfloat162float(a) < __bfloat162float(b) ? a : b)";
		case EQL: return "(__bfloat162float(a) == __bfloat162float(b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case NEQ: return "(__bfloat162float(a) != __bfloat162float(b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case GT:  return "(__bfloat162float(a) > __bfloat162float(b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case LT:  return "(__bfloat162float(a) < __bfloat162float(b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case GTE: return "(__bfloat162float(a) >= __bfloat162float(b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case LTE: return "(__bfloat162float(a) <= __bfloat162float(b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case AND: return "((__bfloat162float(a) != 0 && __bfloat162float(b) != 0) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case OR:  return "((__bfloat162float(a) != 0 || __bfloat162float(b) != 0) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
		case XOR: return "(((__bfloat162float(a) != 0) != (__bfloat162float(b) != 0)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f))";
	}
}

fn char[] binary_op_expr_for_dtype(rtti::Datatype dtype, operations::BinaryOp op)
{
	switch (dtype)
	{
		case F16:  return binary_op_expr_f16(op);
		case BF16: return binary_op_expr_bf16(op);
		default:   return binary_op_expr_f32(op);
	}
}

fn char[] binary_op_expr_u8(operations::BinaryOp op)
{
	switch (op)
	{
		case EQL: return "((unsigned char)(a == b))";
		case NEQ: return "((unsigned char)(a != b))";
		case GT:  return "((unsigned char)(a > b))";
		case LT:  return "((unsigned char)(a < b))";
		case GTE: return "((unsigned char)(a >= b))";
		case LTE: return "((unsigned char)(a <= b))";
		case AND: return "((unsigned char)(a != 0 && b != 0))";
		case OR:  return "((unsigned char)(a != 0 || b != 0))";
		case XOR: return "((unsigned char)((a != 0) != (b != 0)))";
		default:  return "((unsigned char)0)";
	}
}

fn char[] logical_op_expr_u8(operations::BinaryOp op)
{
	switch (op)
	{
		case AND: return "((unsigned char)(a && b))";
		case OR:  return "((unsigned char)(a || b))";
		case XOR: return "((unsigned char)(a != b))";
		default:  return "((unsigned char)0)";
	}
}

// String Buffer Writer — simple append helper

struct SourceBuf
{
	char* data;
	usz len;
	usz cap;
}

fn SourceBuf source_buf_init(char* buf, usz cap)
{
	return { .data = buf, .len = 0, .cap = cap };
}

fn void SourceBuf.append(&self, char[] s)
{
	for (usz i = 0; i < s.len && self.len < self.cap - 1; i++)
	{
		self.data[self.len] = s[i];
		self.len++;
	}
	self.data[self.len] = 0;
}

fn void SourceBuf.append_uint(&self, ulong val)
{
	char[20] tmp;
	usz pos = 0;
	if (val == 0)
	{
		self.append("0");
		return;
	}
	ulong v = val;
	while (v > 0 && pos < 20)
	{
		tmp[pos] = (char)('0' + (v % 10));
		pos++;
		v /= 10;
	}
	// Reverse
	for (usz i = 0; i < pos; i++)
	{
		if (self.len < self.cap - 1)
		{
			self.data[self.len] = tmp[pos - 1 - i];
			self.len++;
		}
	}
	self.data[self.len] = 0;
}

// Kernel Generators

/// Generate unary elementwise kernel (contiguous).
fn usz gen_unary_contiguous(char* buf, usz buf_size, rtti::Datatype dtype, operations::UnaryOp op)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(dtype);
	char[] header = jit_header_for_dtype(dtype);
	char[] expr = unary_op_expr_for_dtype(dtype, op);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    ");
	s.append(tname);
	s.append("* __restrict__ out,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in,\n    size_t n\n) {\n    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        ");
	s.append(tname);
	s.append(" x = in[i];\n        out[i] = ");
	s.append(expr);
	s.append(";\n    }\n}\n");

	return s.len;
}

/// Generate unary elementwise kernel (strided).
fn usz gen_unary_strided(char* buf, usz buf_size, rtti::Datatype dtype, operations::UnaryOp op, usz ndim)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(dtype);
	char[] header = jit_header_for_dtype(dtype);
	char[] expr = unary_op_expr_for_dtype(dtype, op);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    ");
	s.append(tname);
	s.append("* __restrict__ out,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in,\n");
	s.append("    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t out_shape_3,\n");
	s.append("    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t out_shape_7,\n");
	s.append("    size_t in_strides_0, size_t in_strides_1, size_t in_strides_2, size_t in_strides_3,\n");
	s.append("    size_t in_strides_4, size_t in_strides_5, size_t in_strides_6, size_t in_strides_7,\n");
	s.append("    size_t ndim,\n    size_t n\n) {\n");
	s.append("    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, out_shape_3,\n");
	s.append("                                  out_shape_4, out_shape_5, out_shape_6, out_shape_7};\n");
	s.append("    const size_t in_strides[8] = {in_strides_0, in_strides_1, in_strides_2, in_strides_3,\n");
	s.append("                                   in_strides_4, in_strides_5, in_strides_6, in_strides_7};\n");
	s.append("    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) return;\n\n");
	s.append("    size_t idx = i;\n    size_t in_offset = 0;\n    for (int d = ");
	s.append_uint(ndim);
	s.append(" - 1; d >= 0; d--) {\n        size_t coord = idx % out_shape[d];\n        idx /= out_shape[d];\n        in_offset += coord * in_strides[d];\n    }\n\n    ");
	s.append(tname);
	s.append(" x = in[in_offset];\n    out[i] = ");
	s.append(expr);
	s.append(";\n}\n");

	return s.len;
}

/// Generate binary elementwise kernel (contiguous).
fn usz gen_binary_contiguous(char* buf, usz buf_size, rtti::Datatype dtype, operations::BinaryOp op)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(dtype);
	char[] header = jit_header_for_dtype(dtype);
	char[] expr = binary_op_expr_for_dtype(dtype, op);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    ");
	s.append(tname);
	s.append("* __restrict__ out,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_a,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_b,\n    size_t n\n) {\n    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        ");
	s.append(tname);
	s.append(" a = in_a[i];\n        ");
	s.append(tname);
	s.append(" b = in_b[i];\n        out[i] = ");
	s.append(expr);
	s.append(";\n    }\n}\n");

	return s.len;
}

/// Generate binary elementwise kernel (strided/broadcast).
fn usz gen_binary_strided(char* buf, usz buf_size, rtti::Datatype dtype, operations::BinaryOp op, usz ndim)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(dtype);
	char[] header = jit_header_for_dtype(dtype);
	char[] expr = binary_op_expr_for_dtype(dtype, op);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    ");
	s.append(tname);
	s.append("* __restrict__ out,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_a,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_b,\n");
	s.append("    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t out_shape_3,\n");
	s.append("    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t out_shape_7,\n");
	s.append("    size_t a_strides_0, size_t a_strides_1, size_t a_strides_2, size_t a_strides_3,\n");
	s.append("    size_t a_strides_4, size_t a_strides_5, size_t a_strides_6, size_t a_strides_7,\n");
	s.append("    size_t b_strides_0, size_t b_strides_1, size_t b_strides_2, size_t b_strides_3,\n");
	s.append("    size_t b_strides_4, size_t b_strides_5, size_t b_strides_6, size_t b_strides_7,\n");
	s.append("    size_t ndim,\n    size_t n\n) {\n");
	s.append("    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, out_shape_3,\n");
	s.append("                                  out_shape_4, out_shape_5, out_shape_6, out_shape_7};\n");
	s.append("    const size_t a_strides[8] = {a_strides_0, a_strides_1, a_strides_2, a_strides_3,\n");
	s.append("                                  a_strides_4, a_strides_5, a_strides_6, a_strides_7};\n");
	s.append("    const size_t b_strides[8] = {b_strides_0, b_strides_1, b_strides_2, b_strides_3,\n");
	s.append("                                  b_strides_4, b_strides_5, b_strides_6, b_strides_7};\n");
	s.append("    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) return;\n\n");
	s.append("    size_t idx = i;\n    size_t a_offset = 0, b_offset = 0;\n    for (int d = ");
	s.append_uint(ndim);
	s.append(" - 1; d >= 0; d--) {\n        size_t coord = idx % out_shape[d];\n        idx /= out_shape[d];\n        a_offset += coord * a_strides[d];\n        b_offset += coord * b_strides[d];\n    }\n\n    ");
	s.append(tname);
	s.append(" a = in_a[a_offset];\n    ");
	s.append(tname);
	s.append(" b = in_b[b_offset];\n    out[i] = ");
	s.append(expr);
	s.append(";\n}\n");

	return s.len;
}

/// Generate comparison kernel (float input → u8 output, contiguous).
fn usz gen_comparison_contiguous(char* buf, usz buf_size, rtti::Datatype input_dtype, operations::BinaryOp op)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(input_dtype);
	char[] header = jit_header_for_dtype(input_dtype);
	char[] expr = binary_op_expr_u8(op);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    unsigned char* __restrict__ out,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_a,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_b,\n    size_t n\n) {\n    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        ");
	s.append(tname);
	s.append(" a = in_a[i];\n        ");
	s.append(tname);
	s.append(" b = in_b[i];\n        out[i] = ");
	s.append(expr);
	s.append(";\n    }\n}\n");

	return s.len;
}

/// Generate comparison kernel (strided).
fn usz gen_comparison_strided(char* buf, usz buf_size, rtti::Datatype input_dtype, operations::BinaryOp op, usz ndim)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(input_dtype);
	char[] header = jit_header_for_dtype(input_dtype);
	char[] expr = binary_op_expr_u8(op);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    unsigned char* __restrict__ out,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_a,\n    const ");
	s.append(tname);
	s.append("* __restrict__ in_b,\n");
	s.append("    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t out_shape_3,\n");
	s.append("    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t out_shape_7,\n");
	s.append("    size_t a_strides_0, size_t a_strides_1, size_t a_strides_2, size_t a_strides_3,\n");
	s.append("    size_t a_strides_4, size_t a_strides_5, size_t a_strides_6, size_t a_strides_7,\n");
	s.append("    size_t b_strides_0, size_t b_strides_1, size_t b_strides_2, size_t b_strides_3,\n");
	s.append("    size_t b_strides_4, size_t b_strides_5, size_t b_strides_6, size_t b_strides_7,\n");
	s.append("    size_t ndim,\n    size_t n\n) {\n");
	s.append("    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, out_shape_3,\n");
	s.append("                                  out_shape_4, out_shape_5, out_shape_6, out_shape_7};\n");
	s.append("    const size_t a_strides[8] = {a_strides_0, a_strides_1, a_strides_2, a_strides_3,\n");
	s.append("                                  a_strides_4, a_strides_5, a_strides_6, a_strides_7};\n");
	s.append("    const size_t b_strides[8] = {b_strides_0, b_strides_1, b_strides_2, b_strides_3,\n");
	s.append("                                  b_strides_4, b_strides_5, b_strides_6, b_strides_7};\n");
	s.append("    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) return;\n\n");
	s.append("    size_t idx = i;\n    size_t a_offset = 0, b_offset = 0;\n    for (int d = ");
	s.append_uint(ndim);
	s.append(" - 1; d >= 0; d--) {\n        size_t coord = idx % out_shape[d];\n        idx /= out_shape[d];\n        a_offset += coord * a_strides[d];\n        b_offset += coord * b_strides[d];\n    }\n\n    ");
	s.append(tname);
	s.append(" a = in_a[a_offset];\n    ");
	s.append(tname);
	s.append(" b = in_b[b_offset];\n    out[i] = ");
	s.append(expr);
	s.append(";\n}\n");

	return s.len;
}

/// Generate logical kernel (u8 input → u8 output, contiguous).
fn usz gen_logical_contiguous(char* buf, usz buf_size, operations::BinaryOp op)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] expr = logical_op_expr_u8(op);

	s.append("extern \"C\" __global__ void kernel(\n    unsigned char* __restrict__ out,\n    const unsigned char* __restrict__ in_a,\n    const unsigned char* __restrict__ in_b,\n    size_t n\n) {\n    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        unsigned char a = in_a[i];\n        unsigned char b = in_b[i];\n        out[i] = ");
	s.append(expr);
	s.append(";\n    }\n}\n");

	return s.len;
}

/// Generate logical kernel (strided).
fn usz gen_logical_strided(char* buf, usz buf_size, operations::BinaryOp op, usz ndim)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] expr = logical_op_expr_u8(op);

	s.append("extern \"C\" __global__ void kernel(\n    unsigned char* __restrict__ out,\n    const unsigned char* __restrict__ in_a,\n    const unsigned char* __restrict__ in_b,\n");
	s.append("    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t out_shape_3,\n");
	s.append("    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t out_shape_7,\n");
	s.append("    size_t a_strides_0, size_t a_strides_1, size_t a_strides_2, size_t a_strides_3,\n");
	s.append("    size_t a_strides_4, size_t a_strides_5, size_t a_strides_6, size_t a_strides_7,\n");
	s.append("    size_t b_strides_0, size_t b_strides_1, size_t b_strides_2, size_t b_strides_3,\n");
	s.append("    size_t b_strides_4, size_t b_strides_5, size_t b_strides_6, size_t b_strides_7,\n");
	s.append("    size_t ndim,\n    size_t n\n) {\n");
	s.append("    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, out_shape_3,\n");
	s.append("                                  out_shape_4, out_shape_5, out_shape_6, out_shape_7};\n");
	s.append("    const size_t a_strides[8] = {a_strides_0, a_strides_1, a_strides_2, a_strides_3,\n");
	s.append("                                  a_strides_4, a_strides_5, a_strides_6, a_strides_7};\n");
	s.append("    const size_t b_strides[8] = {b_strides_0, b_strides_1, b_strides_2, b_strides_3,\n");
	s.append("                                  b_strides_4, b_strides_5, b_strides_6, b_strides_7};\n");
	s.append("    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) return;\n\n");
	s.append("    size_t idx = i;\n    size_t a_offset = 0, b_offset = 0;\n    for (int d = ");
	s.append_uint(ndim);
	s.append(" - 1; d >= 0; d--) {\n        size_t coord = idx % out_shape[d];\n        idx /= out_shape[d];\n        a_offset += coord * a_strides[d];\n        b_offset += coord * b_strides[d];\n    }\n\n    unsigned char a = in_a[a_offset];\n    unsigned char b = in_b[b_offset];\n    out[i] = ");
	s.append(expr);
	s.append(";\n}\n");

	return s.len;
}

/// Generate select kernel (contiguous).
fn usz gen_select_contiguous(char* buf, usz buf_size, rtti::Datatype dtype)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(dtype);
	char[] header = jit_header_for_dtype(dtype);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    ");
	s.append(tname);
	s.append("* __restrict__ out,\n    const unsigned char* __restrict__ mask,\n    const ");
	s.append(tname);
	s.append("* __restrict__ true_vals,\n    const ");
	s.append(tname);
	s.append("* __restrict__ false_vals,\n    size_t n\n) {\n    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        out[i] = mask[i] ? true_vals[i] : false_vals[i];\n    }\n}\n");

	return s.len;
}

/// Generate select kernel (strided).
fn usz gen_select_strided(char* buf, usz buf_size, rtti::Datatype dtype, usz ndim)
{
	SourceBuf s = source_buf_init(buf, buf_size);
	char[] tname = cuda_dtype_name(dtype);
	char[] header = jit_header_for_dtype(dtype);

	s.append(header);
	s.append("extern \"C\" __global__ void kernel(\n    ");
	s.append(tname);
	s.append("* __restrict__ out,\n    const unsigned char* __restrict__ mask,\n    const ");
	s.append(tname);
	s.append("* __restrict__ true_vals,\n    const ");
	s.append(tname);
	s.append("* __restrict__ false_vals,\n");
	s.append("    size_t out_shape_0, size_t out_shape_1, size_t out_shape_2, size_t out_shape_3,\n");
	s.append("    size_t out_shape_4, size_t out_shape_5, size_t out_shape_6, size_t out_shape_7,\n");
	s.append("    size_t mask_strides_0, size_t mask_strides_1, size_t mask_strides_2, size_t mask_strides_3,\n");
	s.append("    size_t mask_strides_4, size_t mask_strides_5, size_t mask_strides_6, size_t mask_strides_7,\n");
	s.append("    size_t true_strides_0, size_t true_strides_1, size_t true_strides_2, size_t true_strides_3,\n");
	s.append("    size_t true_strides_4, size_t true_strides_5, size_t true_strides_6, size_t true_strides_7,\n");
	s.append("    size_t false_strides_0, size_t false_strides_1, size_t false_strides_2, size_t false_strides_3,\n");
	s.append("    size_t false_strides_4, size_t false_strides_5, size_t false_strides_6, size_t false_strides_7,\n");
	s.append("    size_t ndim,\n    size_t n\n) {\n");
	s.append("    const size_t out_shape[8] = {out_shape_0, out_shape_1, out_shape_2, out_shape_3,\n");
	s.append("                                  out_shape_4, out_shape_5, out_shape_6, out_shape_7};\n");
	s.append("    const size_t mask_strides[8] = {mask_strides_0, mask_strides_1, mask_strides_2, mask_strides_3,\n");
	s.append("                                     mask_strides_4, mask_strides_5, mask_strides_6, mask_strides_7};\n");
	s.append("    const size_t true_strides[8] = {true_strides_0, true_strides_1, true_strides_2, true_strides_3,\n");
	s.append("                                     true_strides_4, true_strides_5, true_strides_6, true_strides_7};\n");
	s.append("    const size_t false_strides[8] = {false_strides_0, false_strides_1, false_strides_2, false_strides_3,\n");
	s.append("                                      false_strides_4, false_strides_5, false_strides_6, false_strides_7};\n");
	s.append("    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) return;\n\n");
	s.append("    size_t idx = i;\n    size_t mask_offset = 0, true_offset = 0, false_offset = 0;\n    for (int d = ");
	s.append_uint(ndim);
	s.append(" - 1; d >= 0; d--) {\n        size_t coord = idx % out_shape[d];\n        idx /= out_shape[d];\n        mask_offset += coord * mask_strides[d];\n        true_offset += coord * true_strides[d];\n        false_offset += coord * false_strides[d];\n    }\n\n    out[i] = mask[mask_offset] ? true_vals[true_offset] : false_vals[false_offset];\n}\n");

	return s.len;
}

