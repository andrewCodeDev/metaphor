//! HIP Graph Compiler — JIT Source Code Generation
//!
//! Generates HIP C++ kernel source from CAS expression trees.
//! This is the fallback path when expressions can't be linearized
//! into fused chains — compiles to hipRTC JIT kernels instead.
//!
//! Part of the compile module.

module metaphor::device::hip_device::compile;

import metaphor::cas;
import metaphor::device::hip_device;
import metaphor::device::hip_device::codegen;
import metaphor::device::hip_device::executor;
import metaphor::device::hip_device::types;
import metaphor::device::common::codegen_context;
import metaphor::device::common::jit_cache;
import metaphor::logging;
import metaphor::rtti;
import metaphor::tensor::tensor_core;
import metaphor::utils::hash;
import std::core::dstring;
import std::collections::list;
import std::io;

// Specialize JitCache types for HIP's KernelHandle.
alias JitKey = jit_cache::JitKey{types::KernelHandle};

// Dynamic kernel parameters

/// Maximum number of runtime parameters a JIT kernel can have.
/// Each non-power-of-2 stride/size in broadcast accesses becomes one param.
const usz MAX_DYNAMIC_PARAMS = 16;

/// Accumulates runtime kernel parameters during JIT codegen.
/// Power-of-2 values are baked as literals (GPU compiler optimizes to shifts).
/// Non-power-of-2 values become kernel parameters passed at launch time.
struct DynamicParams
{
	ulong[MAX_DYNAMIC_PARAMS] values;
	usz count;
}

/// Emit a dimension-derived value into kernel source.
/// Power-of-2: emits literal constant (GPU compiler → shift/mask).
/// Non-power-of-2: emits parameter name `pN`, records value for launch.
fn void DynamicParams.emit(&self, DString* src, ulong value)
{
	if (value > 0 && (value & (value - 1)) == 0)
	{
		src.appendf("%dULL", value)!!;
	}
	else
	{
		assert(self.count < MAX_DYNAMIC_PARAMS, "JIT: too many dynamic params");
		src.appendf("p%d", self.count)!!;
		self.values[self.count] = value;
		self.count++;
	}
}

// Input collection for JIT kernels

/// Decomposition threshold: JIT kernels are limited to this many unique
/// tensor inputs. Expressions exceeding this are split into smaller kernels
/// with intermediate tensors.
const usz MAX_UNIQUE_INPUTS = 4;

alias CodegenContext = codegen_context::CodegenContext;

// JIT Kernel Generation

/// Generate a complete HIP kernel from a CAS expression tree.
/// Fills in the HipOp's jit_source, jit_num_inputs, cached_handle, etc.
///
/// Uses the device's JitCache to avoid redundant source generation and
/// compilation. On cache hit, only cached_handle + input pointers are set;
/// jit_source remains null and no codegen occurs.
fn void generate_hip_kernel_from_cas(
	hip_device::HipDevice* device,
	cas::Expr* expr,
	cas::IREntry* entry,
	void* output,
	types::HipOp* hip_op,
)
{
	CodegenContext imap;
	imap.tinit();
	codegen_context::collect_unique_inputs(expr, &imap);

	char num_inputs = (char)imap.count();

	// Output dtype — used for kernel return type and key header.
	tensor_core::TensorCore* out_tc = (tensor_core::TensorCore*)output;
	rtti::Datatype dtype = out_tc.dtype();

	// ---- Build JitKey ----
	// Canonical byte-string key: header(dtype) + CAS marker + tree + shapes.
	// Sentinelized dims mean different non-pow2 sizes produce the same key.
	JitKey key = {};
	defer key.free();

	// Convert CodegenContext tensors to void** for preamble and write_expr
	void*[types::MAX_JIT_INPUTS] input_ptrs_buf;
	for (usz i = 0; i < imap.count() && i < types::MAX_JIT_INPUTS; i++)
	{
		input_ptrs_buf[i] = imap.tensors[i];
	}
	key.write_cas_preamble(&input_ptrs_buf, imap.count(), dtype);
	key.write_expr(expr, &input_ptrs_buf, imap.count());

	// Input shape blocks (sentinelized)
	for (usz i = 0; i < imap.count(); i++)
	{
		tensor_core::TensorCore* in_tc = (tensor_core::TensorCore*)imap.tensors[i];
		ulong[] in_dims = in_tc.shape().buffer[:in_tc.shape().len];
		key.write_shape(in_dims);
	}

	// Output shape block (sentinelized)
	ulong[] out_dims = out_tc.shape().buffer[:out_tc.shape().len];
	key.write_shape(out_dims);

	// Set output indices for broadcast dimension matching.
	imap.output_indices = entry.output_indices;

	// ---- Cache lookup ----
	if (try handle = device.jit_kernel_cache.get(&key))
	{
		// Cache hit: compute dynamic param values from current shapes.
		// Re-render into throwaway buffer — the DynamicParams are a side effect.
		DynamicParams params;
		params.count = 0;
		DString throwaway;
		throwaway.tinit(256);
		render_cas_expr_hip(&throwaway, expr, &imap, dtype, out_tc, &params);

		fill_hip_op(hip_op, handle, num_inputs, &imap, &params, key.hash());

		logging::@debug(CODEGEN, "JIT cache hit: key_hash=%x num_inputs=%d dynamic_params=%d",
			key.hash(), (int)num_inputs, (int)params.count);
		return;
	}

	// ---- Cache miss: generate kernel source ----
	char[] tname = codegen::hip_dtype_name(dtype);
	char[] hdr = codegen::jit_header_for_dtype(dtype);
	bool is_half = dtype == F16 || dtype == BF16;

	// Phase 1: Pre-render expression body to discover dynamic params.
	DynamicParams params;
	params.count = 0;
	DString body;
	body.tinit(256);
	render_cas_expr_hip(&body, expr, &imap, dtype, out_tc, &params);

	// Phase 2: Assemble full kernel source with dynamic params in signature.
	DString src;
	src.tinit(512);

	if (hdr.len > 0)
	{
		src.append_bytes(hdr);
	}

	// Kernel signature
	src.append_string("extern \"C\" __global__ void kernel(\n    ");
	src.append_bytes(tname);
	src.append_string("* __restrict__ out,\n");

	for (usz i = 0; i < (usz)num_inputs; i++)
	{
		src.append_string("    const ");
		src.append_bytes(tname);
		src.append_string("* __restrict__ in_");
		src.appendf("%d", i)!!;
		src.append_string(",\n");
	}
	src.append_string("    size_t n");

	// Dynamic params (non-pow2 strides/sizes from broadcast access)
	for (usz i = 0; i < params.count; i++)
	{
		src.appendf(",\n    size_t p%d", i)!!;
	}

	src.append_string("\n) {\n");
	src.append_string("    size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n");
	src.append_string("    if (i < n) {\n");

	// Store expression with half-type wrappers
	if (is_half)
	{
		src.append_string("        out[i] = ");
		src.append_string(dtype == BF16 ? "hip_bfloat16(" : "__float2half(");
		src.append_bytes(body.str_view());
		src.append_string(");\n    }\n}\n");
	}
	else
	{
		src.append_string("        out[i] = ");
		src.append_bytes(body.str_view());
		src.append_string(";\n    }\n}\n");
	}

	// Compile and cache
	String str = src.str_view();

	types::KernelHandle handle = executor::get_or_load_kernel(str.ptr, str.len);
	device.jit_kernel_cache.put(&key, handle);

	fill_hip_op(hip_op, handle, num_inputs, &imap, &params, key.hash());

	logging::@debug(CODEGEN, "JIT cache miss — compiled: key_hash=%x num_inputs=%d source_len=%d dynamic_params=%d",
		hip_op.cas_expr.jit_hash, (int)num_inputs, (int)str.len, (int)params.count);
}

/// Fill HipOp fields from compiled kernel results. Shared by cache hit and miss paths.
fn void fill_hip_op(
	types::HipOp* hip_op,
	types::KernelHandle handle,
	char num_inputs,
	CodegenContext* imap,
	DynamicParams* params,
	ulong key_hash,
)
{
	hip_op.cached_handle = handle;
	hip_op.cas_expr.jit_source = null;
	hip_op.cas_expr.jit_source_len = 0;
	hip_op.cas_expr.jit_num_inputs = num_inputs;
	hip_op.cas_expr.jit_needs_broadcast = false;
	hip_op.cas_expr.jit_hash = key_hash;
	hip_op.cas_expr.jit_num_dynamic_params = (char)params.count;

	for (usz i = 0; i < params.count; i++)
	{
		hip_op.cas_expr.jit_dynamic_params[i] = params.values[i];
	}

	for (usz i = 0; i < imap.count() && i < types::MAX_JIT_INPUTS; i++)
	{
		hip_op.cas_expr.jit_inputs[i] = imap.tensors[i];
	}
}

// Expression Renderer

fn void render_cas_expr_hip(
	DString* src,
	cas::Expr* expr,
	CodegenContext* imap,
	rtti::Datatype dtype,
	tensor_core::TensorCore* output,
	DynamicParams* params,
)
{
	switch (expr.kind)
	{
		case ACCESS:
			render_access(src, expr, imap, dtype, output, params);

		case SCALAR:
			render_scalar(src, expr.scalar, dtype);

		case SUM:
			render_nary(src, expr.operands, imap, dtype, " + ", "0", output, params);

		case PRODUCT:
			render_nary(src, expr.operands, imap, dtype, " * ", "1", output, params);

		case NEGATE:
			src.append_string("(-");
			render_cas_expr_hip(src, expr.inner, imap, dtype, output, params);
			src.append_string(")");

		case UNARY:
			render_unary(src, expr, imap, dtype, output, params);

		case BINARY:
			render_binary(src, expr, imap, dtype, output, params);

		case COMPARE:
			render_compare(src, expr, imap, dtype, output, params);

		case SELECT:
			src.append_string("(");
			render_cas_expr_hip(src, expr.select_.cond, imap, dtype, output, params);
			src.append_string(" ? ");
			render_cas_expr_hip(src, expr.select_.then_expr, imap, dtype, output, params);
			src.append_string(" : ");
			render_cas_expr_hip(src, expr.select_.else_expr, imap, dtype, output, params);
			src.append_string(")");

		default:
			unreachable("unexpected CAS node in JIT codegen: %s", expr.kind);
	}
}

fn void render_access(DString* src, cas::Expr* expr, CodegenContext* imap, rtti::Datatype dtype, tensor_core::TensorCore* output, DynamicParams* params)
{
	usz idx = imap.get((void*)expr.access.tensor);
	tensor_core::TensorCore* tc = expr.access.tensor;
	bool is_scalar = tc.shape().len == 0;

	if (is_scalar || tc.shape().size() <= 1)
	{
		// Scalar/single-element tensor: always access element 0
		render_load(src, dtype, idx, "0");
	}
	else if (tc.shape().size() == output.shape().size())
	{
		// Same total elements: flat index (no broadcast needed)
		render_load(src, dtype, idx, "i");
	}
	else
	{
		// Broadcast: compute flat input index from output flat index.
		render_broadcast_access(src, expr, imap, dtype, output, params);
	}
}

/// Emit a load expression that converts half types to float for computation.
fn void render_load(DString* src, rtti::Datatype dtype, usz idx, String index_expr)
{
	switch (dtype)
	{
		case BF16:
			src.appendf("(float)(in_%d[%s])", idx, index_expr)!!;
		case F16:
			src.appendf("__half2float(in_%d[%s])", idx, index_expr)!!;
		case F32:
			src.appendf("in_%d[%s]", idx, index_expr)!!;
		case F64:
			src.appendf("in_%d[%s]", idx, index_expr)!!;
		default:
			unreachable("JIT render_load: unsupported dtype");
	}
}

/// Render a broadcast access expression.
/// For input with indices [e] mapping to output indices [b,e,l] with output shape [B,E,L]:
///   input_idx = (i / L) % E
/// General case: decompose i into output coords, pick the ones this input uses,
/// then recompose into the input's flat index.
fn void render_broadcast_access(
	DString* src,
	cas::Expr* expr,
	CodegenContext* imap,
	rtti::Datatype dtype,
	tensor_core::TensorCore* output,
	DynamicParams* params,
)
{
	usz idx = imap.get((void*)expr.access.tensor);
	tensor_core::TensorCore* tc = expr.access.tensor;
	cas::IndexId[] access_indices = expr.access.indices;
	usz input_ndim = access_indices.len;
	usz output_ndim = output.shape().len;
	bool is_half = dtype == F16 || dtype == BF16;

	// Build output strides (row-major)
	ulong[8] out_strides;
	out_strides[output_ndim - 1] = 1;
	for (usz d = output_ndim - 1; d > 0; d--)
	{
		out_strides[d - 1] = out_strides[d] * output.shape().get(d);
	}

	// Match input dims to output dims.
	usz[8] out_dim_for_input;
	bool[8] out_dim_used = {};

	if (input_ndim == output_ndim)
	{
		// Same rank: positional matching. Size-1 dims are broadcast dims.
		for (usz j = 0; j < input_ndim; j++)
		{
			out_dim_for_input[j] = j;
			out_dim_used[j] = true;
		}
	}
	else
	{
		// Different rank: match by IndexId from CAS expression indices.
		// Size-based matching is ambiguous when multiple output dims share the same size.
		if (imap.output_indices.len >= output_ndim)
		{
			for (usz j = 0; j < input_ndim; j++)
			{
				cas::IndexId input_idx_id = access_indices[j];
				bool found = false;
				for (usz d = 0; d < output_ndim; d++)
				{
					if (imap.output_indices[d] == input_idx_id)
					{
						out_dim_for_input[j] = d;
						found = true;
						break;
					}
				}
				if (!found)
				{
					// Fallback: can't determine mapping, use flat index
					render_load(src, dtype, idx, "i");
					return;
				}
			}
		}
		else
		{
			// No output indices available, fall back to size-based matching.
			for (usz j = 0; j < input_ndim; j++)
			{
				ulong input_size = tc.shape().get(j);
				bool found = false;
				for (usz d = 0; d < output_ndim; d++)
				{
					if (!out_dim_used[d] && output.shape().get(d) == input_size)
					{
						out_dim_for_input[j] = d;
						out_dim_used[d] = true;
						found = true;
						break;
					}
				}
				if (!found)
				{
					render_load(src, dtype, idx, "i");
					return;
				}
			}
		}
	}

	// Build input strides (row-major)
	ulong[8] in_strides;
	if (input_ndim > 0)
	{
		in_strides[input_ndim - 1] = 1;
		for (usz d = input_ndim - 1; d > 0; d--)
		{
			in_strides[d - 1] = in_strides[d] * tc.shape().get(d);
		}
	}

	// Generate: load(in_X[broadcast_index_expr])
	// Each stride/size value is either baked as a literal (pow2) or
	// emitted as a runtime parameter pN (non-pow2) via params.emit().
	if (is_half)
	{
		if (dtype == BF16)
		{
			src.appendf("(float)(in_%d[", idx)!!;
		}
		else
		{
			src.appendf("__half2float(in_%d[", idx)!!;
		}
	}
	else
	{
		src.appendf("in_%d[", idx)!!;
	}

	for (usz j = 0; j < input_ndim; j++)
	{
		if (j > 0) src.append_string(" + ");
		ulong out_stride = out_strides[out_dim_for_input[j]];
		ulong in_size = tc.shape().get(j);
		ulong in_stride = in_strides[j];

		src.append_string("(i / ");
		params.emit(src, out_stride);
		src.append_string(" % ");
		params.emit(src, in_size);
		src.append_string(")");

		if (in_stride != 1)
		{
			src.append_string(" * ");
			params.emit(src, in_stride);
		}
	}

	if (is_half) {
		src.append_string("])");
	} else {
		src.append_string("]");
	}
}

fn void render_scalar(DString* src, double val, rtti::Datatype dtype)
{
	switch (dtype)
	{
		case F64:
			src.appendf("%.6f", val)!!;
		case F32:
			src.appendf("%.6ff", val)!!;
		case F16:
			src.appendf("%.6ff", val)!!;
		case BF16:
			src.appendf("%.6ff", val)!!;
		default:
			unreachable("JIT render_scalar: unsupported dtype");
	}
}

fn void render_nary(
	DString* src,
	cas::Expr*[] terms,
	CodegenContext* imap,
	rtti::Datatype dtype,
	String op_str,
	String identity,
	tensor_core::TensorCore* output,
	DynamicParams* params,
)
{
	if (terms.len == 0) { src.append_string(identity); return; }
	if (terms.len == 1)
	{
		render_cas_expr_hip(src, terms[0], imap, dtype, output, params);
		return;
	}
	src.append_string("(");
	for (usz i = 0; i < terms.len; i++)
	{
		if (i > 0) src.append_string(op_str);
		render_cas_expr_hip(src, terms[i], imap, dtype, output, params);
	}
	src.append_string(")");
}

fn void render_unary(DString* src, cas::Expr* expr, CodegenContext* imap, rtti::Datatype dtype, tensor_core::TensorCore* output, DynamicParams* params)
{
	bool d = dtype == F64;  // double precision uses unsuffixed math functions
	switch (expr.unary.op)
	{
		case ABS:
			src.append_string(d ? "fabs(" : "fabsf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case SQRT:
			src.append_string(d ? "sqrt(" : "sqrtf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case SQR:
			src.append_string("(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(" * ");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case RECIP:
			src.append_string(d ? "(1.0 / " : "(1.0f / ");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case EXP:
			src.append_string(d ? "exp(" : "expf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case LOG:
			src.append_string(d ? "log(" : "logf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case SIN:
			src.append_string(d ? "sin(" : "sinf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case COS:
			src.append_string(d ? "cos(" : "cosf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case SIGN:
			src.append_string("(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " > 0.0 ? 1.0 : (" : " > 0.0f ? 1.0f : (");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " < 0.0 ? -1.0 : 0.0))" : " < 0.0f ? -1.0f : 0.0f))");
		case RELU:
			src.append_string(d ? "fmax(" : "fmaxf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? ", 0.0)" : ", 0.0f)");
		case SIGMOID:
			src.append_string(d ? "(1.0 / (1.0 + exp(-" : "(1.0f / (1.0f + expf(-");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")))");
		case TANH:
			src.append_string(d ? "tanh(" : "tanhf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")");
		case GELU:
			src.append_string(d ? "(0.5 * " : "(0.5f * ");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " * (1.0 + tanh(0.7978845608 * (" : " * (1.0f + tanhf(0.7978845608f * (");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " + 0.044715 * " : " + 0.044715f * ");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(" * ");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(" * ");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string("))))");
		case SILU:
			src.append_string("(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " / (1.0 + exp(-" : " / (1.0f + expf(-");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(")))");
		case SOFTPLUS:
			src.append_string(d ? "(fmax(" : "(fmaxf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? ", 0.0) + log(1.0 + exp(-fabs(" : ", 0.0f) + logf(1.0f + expf(-fabsf(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string("))))");
		case HEAVISIDE:
			src.append_string("(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " > 0.0 ? 1.0 : 0.0)" : " > 0.0f ? 1.0f : 0.0f)");
		case LOGIC_NOT:
			src.append_string("(");
			render_cas_expr_hip(src, expr.unary.child, imap, dtype, output, params);
			src.append_string(d ? " == 0.0 ? 1.0 : 0.0)" : " == 0.0f ? 1.0f : 0.0f)");
		default:
			unreachable("JIT render_unary: unsupported op");
	}
}

fn void render_binary(DString* src, cas::Expr* expr, CodegenContext* imap, rtti::Datatype dtype, tensor_core::TensorCore* output, DynamicParams* params)
{
	bool d = dtype == F64;
	switch (expr.binary.op)
	{
		case SUB:
			src.append_string("(");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(" - ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(")");
		case DIV:
			src.append_string("(");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(" / ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(")");
		case MAX:
			src.append_string(d ? "fmax(" : "fmaxf(");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(", ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(")");
		case MIN:
			src.append_string(d ? "fmin(" : "fminf(");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(", ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(")");
		case POW:
			src.append_string(d ? "pow(" : "powf(");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(", ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(")");
		case LOGIC_AND:
			src.append_string("((");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(" != 0 && ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(d ? " != 0) ? 1.0 : 0.0)" : " != 0) ? 1.0f : 0.0f)");
		case LOGIC_OR:
			src.append_string("((");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(" != 0 || ");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(d ? " != 0) ? 1.0 : 0.0)" : " != 0) ? 1.0f : 0.0f)");
		case LOGIC_XOR:
			src.append_string("((");
			render_cas_expr_hip(src, expr.binary.lhs, imap, dtype, output, params);
			src.append_string(" != 0 != (");
			render_cas_expr_hip(src, expr.binary.rhs, imap, dtype, output, params);
			src.append_string(d ? " != 0)) ? 1.0 : 0.0)" : " != 0)) ? 1.0f : 0.0f)");
	}
}

fn void render_compare(DString* src, cas::Expr* expr, CodegenContext* imap, rtti::Datatype dtype, tensor_core::TensorCore* output, DynamicParams* params)
{
	bool d = dtype == F64;
	String op_str;
	switch (expr.compare.op)
	{
		case EQ:  op_str = " == ";
		case NE:  op_str = " != ";
		case GT:  op_str = " > ";
		case LT:  op_str = " < ";
		case GTE: op_str = " >= ";
		case LTE: op_str = " <= ";
	}
	src.append_string("(");
	render_cas_expr_hip(src, expr.compare.lhs, imap, dtype, output, params);
	src.append_string(op_str);
	render_cas_expr_hip(src, expr.compare.rhs, imap, dtype, output, params);
	src.append_string(d ? " ? 1.0 : 0.0)" : " ? 1.0f : 0.0f)");
}
