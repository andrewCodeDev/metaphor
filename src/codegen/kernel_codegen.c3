//! Unified Kernel Code Generator
//!
//! Generates C3 source code from FusedKernel IR.
//! Handles all patterns uniformly:
//! - Element-wise: no bound indices, just free loops + expression
//! - Reduction: free loops + bound loops + accumulator + reduce_op
//! - Einsum: same as reduction, prologue_expr is typically mul
//!
//! Generated kernel signature (C3):
//!   fn void fk_{hash}(
//!       TensorCore** inputs, usz num_inputs,
//!       TensorCore* output, ulong* dims, double* params
//!   ) @export
//!
//! directly. The embedded struct definitions (via $embed in kernel_cache)
//! allow the JIT compiler to access fields at known offsets.
//!

module metaphor::codegen::kernel_codegen;
import metaphor::codegen::kernel_ir;
import metaphor::cas;
import metaphor::rtti;
import metaphor::tensor::tensor_core;
import std::io;

// Public API

/// Generate C3 source code for a single fused kernel.
/// Returns the source as a char[] slice. Caller owns the returned buffer.
fn char[] generate_kernel_source(FusedKernel* kernel, char[] buf)
{
	usz pos = 0;
	render_kernel(kernel, buf, &pos);
	return buf[:pos];
}

// Top-level kernel rendering

fn void render_kernel(FusedKernel* kernel, char[] buf, usz* pos)
{
	char[] dtype_str = dtype_to_c3_str(kernel.dtype);
	bool has_bound = kernel.num_bound > 0;

	// Module declaration (so dlsym finds "jit__fk_xxx")
	emit(buf, pos, "module jit;\n");
	emit(buf, pos, "import metaphor::tensor::tensor_core;\n");
	emit(buf, pos, "import metaphor::tensor::shape;\n");
	emit(buf, pos, "import metaphor::tensor::dense_core;\n");
	emit(buf, pos, "import metaphor::rtti;\n");
	emit(buf, pos, "import std::math;\n\n");

	// Function signature
	emit(buf, pos, "fn void ");
	emit(buf, pos, kernel.name());
	emit(buf, pos, "(\n");
	emit(buf, pos, "\tTensorCore** inputs,\n");
	emit(buf, pos, "\tusz num_inputs,\n");
	emit(buf, pos, "\tTensorCore* output,\n");
	emit(buf, pos, "\tulong* dims,\n");
	emit(buf, pos, "\tdouble* params,\n");
	emit(buf, pos, ") @export\n{\n");

	// Unpack input data pointers and strides
	for (usz i = 0; i < kernel.num_inputs; i++)
	{
		TensorBinding* binding = &kernel.inputs[i];
		if (!kernel_uses_input_data(kernel, i)) continue;

		char[] input_dtype = dtype_to_c3_str(binding.tensor.dtype());

		// Data pointer: cast state.dense.data (char[]) to typed pointer
		emit_fmt_input_data(buf, pos, i, input_dtype);

		// Strides (if input has multi-dimensional access)
		if (binding.indices_len > 0 || cas_expr_uses_strides(kernel.prologue_expr, binding.tensor))
		{
			emit_fmt_input_strides(buf, pos, i);
		}
	}

	// Output data pointer (skip for in-place — we write to in0)
	if (!kernel.is_inplace)
	{
		emit_fmt_output_data(buf, pos, dtype_str);
		if (kernel.num_free > 0)
		{
			emit(buf, pos, "\tulong* out_strides = output.state.dense.strides[0..];\n");
		}
	}

	// Unpack dimension sizes from dims array
	usz dim_idx = 0;
	for (usz i = 0; i < kernel.num_free; i++)
	{
		emit_fmt_dim(buf, pos, kernel.free_indices[i].name(), dim_idx);
		dim_idx++;
	}
	for (usz i = 0; i < kernel.num_bound; i++)
	{
		emit_fmt_dim(buf, pos, kernel.bound_indices[i].name(), dim_idx);
		dim_idx++;
	}

	// Unpack params
	for (usz i = 0; i < kernel.num_params; i++)
	{
		emit_fmt_param(buf, pos, i);
	}

	// Shape expressions
	for (usz i = 0; i < kernel.num_shape_exprs; i++)
	{
		emit_shape_expr(buf, pos, &kernel.shape_exprs[i], kernel, dtype_str, i);
	}

	// Open free index loops (output dimensions)
	for (usz i = 0; i < kernel.num_free; i++)
	{
		emit_fmt_for_loop(buf, pos, kernel.free_indices[i].name(), i + 1);
	}

	// Initialize accumulator if reduction
	if (has_bound)
	{
		emit_accumulator_init(buf, pos, dtype_str, kernel.has_reduce_op ? kernel.reduce_op : cas::ReduceOp.ADD);

		// Open bound index loops (contraction dimensions)
		for (usz i = 0; i < kernel.num_bound; i++)
		{
			char idx_char = kernel.bound_indices[i].name_buf[0];
			bool used = kernel.bound_index_is_used(idx_char);
			emit_fmt_for_loop_bound(buf, pos, kernel.bound_indices[i].name(), kernel.num_free + i + 1, used);
		}
	}

	// Handle expression and store based on kernel type
	if (has_bound)
	{
		// Standard reduction: nested loops with accumulator
		emit(buf, pos, "\t\t");
		emit_tabs(buf, pos, kernel.num_free + kernel.num_bound);
		emit(buf, pos, dtype_str);
		emit(buf, pos, " val = ");
		render_cas_expr(buf, pos, kernel.prologue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");

		// Accumulate
		emit_reduce_accumulate(buf, pos, kernel.has_reduce_op ? kernel.reduce_op : cas::ReduceOp.ADD,
			kernel.num_free + kernel.num_bound);

		// Close bound loops
		for (usz i = 0; i < kernel.num_bound; i++)
		{
			emit_tabs(buf, pos, kernel.num_free + kernel.num_bound - i);
			emit(buf, pos, "}\n");
		}

		// Store result
		if (kernel.is_inplace)
		{
			emit_store_inplace(buf, pos, kernel, dtype_str);
		}
		else
		{
			emit_store_output(buf, pos, kernel, dtype_str);
		}
	}
	else
	{
		// Pure element-wise — no accumulator
		if (kernel.is_inplace)
		{
			emit_elemwise_inplace(buf, pos, kernel, dtype_str);
		}
		else
		{
			emit_elemwise_output(buf, pos, kernel, dtype_str);
		}
	}

	// Close free loops
	for (usz i = 0; i < kernel.num_free; i++)
	{
		emit_tabs(buf, pos, kernel.num_free - i);
		emit(buf, pos, "}\n");
	}

	emit(buf, pos, "}\n");
}

// CAS Expression Rendering

fn void render_cas_expr(char[] buf, usz* pos, cas::Expr* expr, FusedKernel* kernel, char[] dtype_str)
{
	switch (expr.kind)
	{
		case ACCESS:
			render_access(buf, pos, &expr.access, kernel, dtype_str);

		case SCALAR:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")");
			emit_double(buf, pos, expr.scalar);

		case PRODUCT:
			cas::Expr*[] factors = expr.operands;
			if (factors.len == 0) { emit(buf, pos, "("); emit(buf, pos, dtype_str); emit(buf, pos, ")1"); return; }
			if (factors.len == 1) { render_cas_expr(buf, pos, factors[0], kernel, dtype_str); return; }
			emit(buf, pos, "(");
			for (usz i = 0; i < factors.len; i++)
			{
				if (i > 0) emit(buf, pos, " * ");
				render_cas_expr(buf, pos, factors[i], kernel, dtype_str);
			}
			emit(buf, pos, ")");

		case SUM:
			cas::Expr*[] terms = expr.operands;
			if (terms.len == 0) { emit(buf, pos, "("); emit(buf, pos, dtype_str); emit(buf, pos, ")0"); return; }
			if (terms.len == 1) { render_cas_expr(buf, pos, terms[0], kernel, dtype_str); return; }
			emit(buf, pos, "(");
			for (usz i = 0; i < terms.len; i++)
			{
				if (i > 0) emit(buf, pos, " + ");
				render_cas_expr(buf, pos, terms[i], kernel, dtype_str);
			}
			emit(buf, pos, ")");

		case CONTRACT:
			unreachable("contract should not appear in prologue expression");

		case NEGATE:
			emit(buf, pos, "(-");
			render_cas_expr(buf, pos, expr.inner, kernel, dtype_str);
			emit(buf, pos, ")");

		case UNARY:
			render_unary(buf, pos, &expr.unary, kernel, dtype_str);

		case BINARY:
			render_binary(buf, pos, &expr.binary, kernel, dtype_str);

		case COMPARE:
			render_compare(buf, pos, &expr.compare, kernel, dtype_str);

		case SELECT:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, expr.select_.cond, kernel, dtype_str);
			emit(buf, pos, " ? ");
			render_cas_expr(buf, pos, expr.select_.then_expr, kernel, dtype_str);
			emit(buf, pos, " : ");
			render_cas_expr(buf, pos, expr.select_.else_expr, kernel, dtype_str);
			emit(buf, pos, ")");

		case PERMUTE:
			render_permute(buf, pos, &expr.permute, kernel, dtype_str);

		case BUILTIN:
			unreachable("builtin functions not yet supported in kernel codegen");
	}
}

fn void render_access(char[] buf, usz* pos, cas::AccessData* a, FusedKernel* kernel, char[] dtype_str)
{
	// Find which input this tensor corresponds to
	usz input_idx = find_input_idx(kernel, a.tensor);
	TensorBinding* binding = &kernel.inputs[input_idx];
	rtti::Datatype input_dtype = binding.tensor.dtype();
	bool needs_cast = input_dtype != kernel.dtype;
	bool is_scalar = binding.indices_len == 0;

	if (needs_cast)
	{
		emit(buf, pos, "(");
		emit(buf, pos, dtype_str);
		emit(buf, pos, ")");
	}

	emit(buf, pos, "in");
	emit_uint(buf, pos, input_idx);
	emit(buf, pos, "[");

	if (is_scalar)
	{
		emit(buf, pos, "0");
	}
	else
	{
		render_load_index(buf, pos, binding.indices(), input_idx, kernel);
	}

	emit(buf, pos, "]");
}

fn void render_unary(char[] buf, usz* pos, cas::UnaryData* u, FusedKernel* kernel, char[] dtype_str)
{
	switch (u.op)
	{
		case ABS:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " < 0 ? -");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " : ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case SQRT:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::sqrt(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case SQR:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " * ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case RECIP:
			emit(buf, pos, "(1 / ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case EXP:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::exp(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case LOG:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::ln(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case SIN:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::sin(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case COS:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::cos(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case SIGN:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " > 0 ? (");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")1 : (");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " < 0 ? (");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")-1 : (");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")0))");

		case FLOOR:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::floor(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case CEIL:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::ceil(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case ROUND:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::round(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case RELU:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " > 0 ? ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " : 0)");

		case SIGMOID:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")(1 / (1 + math::exp(-");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")))");

		case TANH:
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::tanh(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");

		case GELU:
			// gelu(x) = 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")((");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")0.5 * ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " * ((");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")1.0 + math::tanh((");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")0.7978845608 * (");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " + (");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")0.044715 * ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " * ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " * ");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, "))))");

		case SILU:
			// silu(x) = x / (1 + exp(-x))
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " / ((");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")1.0 + math::exp(-");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")))");

		case SOFTPLUS:
			// softplus(x) = ln(1 + exp(x))
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")math::ln((");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")1.0 + math::exp(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, "))");

		case HEAVISIDE:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, " >= 0 ? (");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")1 : (");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")0)");

		case LOGIC_NOT:
			emit(buf, pos, "(!");
			render_cas_expr(buf, pos, u.child, kernel, dtype_str);
			emit(buf, pos, ")");
	}
}

fn void render_binary(char[] buf, usz* pos, cas::BinaryData* b, FusedKernel* kernel, char[] dtype_str)
{
	switch (b.op)
	{
		case DIV:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " / ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, ")");

		case SUB:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " - ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, ")");

		case POW:
			emit(buf, pos, "math::pow(");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, ", ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, ")");

		case MAX:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " > ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, " ? ");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " : ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, ")");

		case MIN:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " < ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, " ? ");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " : ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, ")");

		case LOGIC_AND:
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " && ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);

		case LOGIC_OR:
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " || ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);

		case LOGIC_XOR:
			emit(buf, pos, "(");
			render_cas_expr(buf, pos, b.lhs, kernel, dtype_str);
			emit(buf, pos, " != ");
			render_cas_expr(buf, pos, b.rhs, kernel, dtype_str);
			emit(buf, pos, ")");
	}
}

fn void render_compare(char[] buf, usz* pos, cas::CompareData* c, FusedKernel* kernel, char[] dtype_str)
{
	render_cas_expr(buf, pos, c.lhs, kernel, dtype_str);
	switch (c.op)
	{
		case EQ:  emit(buf, pos, " == ");
		case NE:  emit(buf, pos, " != ");
		case GT:  emit(buf, pos, " > ");
		case LT:  emit(buf, pos, " < ");
		case GTE: emit(buf, pos, " >= ");
		case LTE: emit(buf, pos, " <= ");
	}
	render_cas_expr(buf, pos, c.rhs, kernel, dtype_str);
}

fn void render_permute(char[] buf, usz* pos, cas::PermuteData* p, FusedKernel* kernel, char[] dtype_str)
{
	// Permute reorders indices when accessing the child.
	if (p.child.kind == cas::ExprKind.ACCESS)
	{
		cas::AccessData* a = &p.child.access;
		usz input_idx = find_input_idx(kernel, a.tensor);
		TensorBinding* binding = &kernel.inputs[input_idx];
		rtti::Datatype input_dtype = binding.tensor.dtype();
		bool needs_cast = input_dtype != kernel.dtype;

		if (needs_cast)
		{
			emit(buf, pos, "(");
			emit(buf, pos, dtype_str);
			emit(buf, pos, ")");
		}

		emit(buf, pos, "in");
		emit_uint(buf, pos, input_idx);
		emit(buf, pos, "[");

		// Permuted index computation
		usz num_dims = p.mapping.len;
		if (num_dims == 0)
		{
			emit(buf, pos, "0");
		}
		else
		{
			for (usz out_dim = 0; out_dim < num_dims; out_dim++)
			{
				if (out_dim > 0) emit(buf, pos, " + ");
				char input_dim = p.mapping[out_dim];
				char idx_char = (char)('i' + out_dim);
				char[] idx_name = kernel.find_index_name(idx_char);
				emit(buf, pos, idx_name);
				emit(buf, pos, "_i * in");
				emit_uint(buf, pos, input_idx);
				emit(buf, pos, "_strides[");
				emit_uint(buf, pos, (usz)input_dim);
				emit(buf, pos, "]");
			}
		}

		emit(buf, pos, "]");
	}
	else
	{
		// For non-access children, just render the child
		render_cas_expr(buf, pos, p.child, kernel, dtype_str);
	}
}

// Index computation helpers

/// Render linear index computation from an index pattern.
/// E.g., "ij" -> i_i * strides[0] + j_i * strides[1]
fn void render_linear_index(char[] buf, usz* pos, char[] indices, char[] stride_name, FusedKernel* kernel)
{
	if (indices.len == 0)
	{
		emit(buf, pos, "0");
		return;
	}

	for (usz dim = 0; dim < indices.len; dim++)
	{
		if (dim > 0) emit(buf, pos, " + ");
		char[] idx_name = kernel.find_index_name(indices[dim]);
		emit(buf, pos, idx_name);
		emit(buf, pos, "_i * ");
		emit(buf, pos, stride_name);
		emit(buf, pos, "[");
		emit_uint(buf, pos, dim);
		emit(buf, pos, "]");
	}
}

/// Render index computation for a normal load (non-permuted).
fn void render_load_index(char[] buf, usz* pos, char[] indices, usz input_idx, FusedKernel* kernel)
{
	if (indices.len == 0)
	{
		emit(buf, pos, "0");
		return;
	}

	for (usz dim = 0; dim < indices.len; dim++)
	{
		if (dim > 0) emit(buf, pos, " + ");
		char[] idx_name = kernel.find_index_name(indices[dim]);
		emit(buf, pos, idx_name);
		emit(buf, pos, "_i * in");
		emit_uint(buf, pos, input_idx);
		emit(buf, pos, "_strides[");
		emit_uint(buf, pos, dim);
		emit(buf, pos, "]");
	}
}

// CAS expression queries

/// Check if a tensor is accessed through a permute expression in the CAS tree.
fn bool cas_expr_uses_strides(cas::Expr* expr, TensorCore* tensor)
{
	switch (expr.kind)
	{
		case ACCESS: return false;
		case SCALAR: return false;
		case PRODUCT:
			foreach (f : expr.operands)
			{
				if (cas_expr_uses_strides(f, tensor)) return true;
			}
			return false;
		case SUM:
			foreach (t : expr.operands)
			{
				if (cas_expr_uses_strides(t, tensor)) return true;
			}
			return false;
		case CONTRACT: return cas_expr_uses_strides(expr.contract.body, tensor);
		case NEGATE: return cas_expr_uses_strides(expr.inner, tensor);
		case UNARY: return cas_expr_uses_strides(expr.unary.child, tensor);
		case BINARY: return cas_expr_uses_strides(expr.binary.lhs, tensor) || cas_expr_uses_strides(expr.binary.rhs, tensor);
		case COMPARE: return cas_expr_uses_strides(expr.compare.lhs, tensor) || cas_expr_uses_strides(expr.compare.rhs, tensor);
		case SELECT: return cas_expr_uses_strides(expr.select_.cond, tensor) || cas_expr_uses_strides(expr.select_.then_expr, tensor) || cas_expr_uses_strides(expr.select_.else_expr, tensor);
		case PERMUTE:
			if (expr.permute.child.kind == cas::ExprKind.ACCESS && expr.permute.child.access.tensor == tensor)
			{
				return true;
			}
			return cas_expr_uses_strides(expr.permute.child, tensor);
		case BUILTIN:
			foreach (arg : expr.builtin.args)
			{
				if (cas_expr_uses_strides(arg, tensor)) return true;
			}
			return false;
	}
}

/// Check if a tensor is referenced in the CAS expression tree.
fn bool cas_expr_uses_tensor(cas::Expr* expr, TensorCore* tensor)
{
	switch (expr.kind)
	{
		case ACCESS: return expr.access.tensor == tensor;
		case SCALAR: return false;
		case PRODUCT:
			foreach (f : expr.operands)
			{
				if (cas_expr_uses_tensor(f, tensor)) return true;
			}
			return false;
		case SUM:
			foreach (t : expr.operands)
			{
				if (cas_expr_uses_tensor(t, tensor)) return true;
			}
			return false;
		case CONTRACT: return cas_expr_uses_tensor(expr.contract.body, tensor);
		case NEGATE: return cas_expr_uses_tensor(expr.inner, tensor);
		case UNARY: return cas_expr_uses_tensor(expr.unary.child, tensor);
		case BINARY: return cas_expr_uses_tensor(expr.binary.lhs, tensor) || cas_expr_uses_tensor(expr.binary.rhs, tensor);
		case COMPARE: return cas_expr_uses_tensor(expr.compare.lhs, tensor) || cas_expr_uses_tensor(expr.compare.rhs, tensor);
		case SELECT: return cas_expr_uses_tensor(expr.select_.cond, tensor) || cas_expr_uses_tensor(expr.select_.then_expr, tensor) || cas_expr_uses_tensor(expr.select_.else_expr, tensor);
		case PERMUTE: return cas_expr_uses_tensor(expr.permute.child, tensor);
		case BUILTIN:
			foreach (arg : expr.builtin.args)
			{
				if (cas_expr_uses_tensor(arg, tensor)) return true;
			}
			return false;
	}
}

/// Check if input at given index is used for data access.
fn bool kernel_uses_input_data(FusedKernel* kernel, usz input_idx)
{
	if (input_idx >= kernel.num_inputs) return false;
	TensorCore* tensor = kernel.inputs[input_idx].tensor;
	if (cas_expr_uses_tensor(kernel.prologue_expr, tensor)) return true;
	if (kernel.epilogue_expr != null)
	{
		if (cas_expr_uses_tensor(kernel.epilogue_expr, tensor)) return true;
	}
	return false;
}

// Lookup helpers

fn usz find_input_idx(FusedKernel* kernel, TensorCore* tensor)
{
	for (usz i = 0; i < kernel.num_inputs; i++)
	{
		if (kernel.inputs[i].tensor == tensor) return i;
	}
	unreachable("CAS access references tensor not in kernel inputs");
}

// Dtype to C3 type string

fn char[] dtype_to_c3_str(rtti::Datatype dtype)
{
	switch (dtype)
	{
		case BOOL: return "bool";
		case U8:   return "char";
		case U64:  return "ulong";
		case F16:  return "half";
		case BF16: return "bfloat";
		case F32:  return "float";
		case F64:  return "double";
	}
}

// Low-level buffer emission helpers

fn void emit(char[] buf, usz* pos, char[] s)
{
	for (usz i = 0; i < s.len && *pos < buf.len; i++)
	{
		buf[*pos] = s[i];
		(*pos)++;
	}
}

fn void emit_char(char[] buf, usz* pos, char c)
{
	if (*pos < buf.len)
	{
		buf[*pos] = c;
		(*pos)++;
	}
}

fn void emit_uint(char[] buf, usz* pos, usz val)
{
	char[20] tmp;
	usz len = 0;
	if (val == 0)
	{
		emit_char(buf, pos, '0');
		return;
	}
	usz v = val;
	while (v > 0)
	{
		tmp[len] = (char)('0' + v % 10);
		len++;
		v /= 10;
	}
	for (usz i = 0; i < len; i++)
	{
		emit_char(buf, pos, tmp[len - 1 - i]);
	}
}

fn void emit_double(char[] buf, usz* pos, double val)
{
	// Simple representation: integer part or scientific notation
	if (val == 0.0)
	{
		emit(buf, pos, "0.0");
		return;
	}
	if (val < 0)
	{
		emit_char(buf, pos, '-');
		val = -val;
	}
	if (val == 1.0) { emit(buf, pos, "1.0"); return; }
	if (val == 2.0) { emit(buf, pos, "2.0"); return; }
	if (val == 0.5) { emit(buf, pos, "0.5"); return; }

	// For integer values, emit as X.0
	if (val == (double)(long)val && val < 1e15)
	{
		emit_uint(buf, pos, (usz)(long)val);
		emit(buf, pos, ".0");
		return;
	}

	// General case: use a simple fixed-point representation
	usz integer_part = (usz)val;
	emit_uint(buf, pos, integer_part);
	emit_char(buf, pos, '.');
	double frac = val - (double)integer_part;
	for (int d = 0; d < 12; d++)
	{
		frac *= 10.0;
		usz digit = (usz)frac;
		emit_char(buf, pos, (char)('0' + digit));
		frac -= (double)digit;
	}
}

fn void emit_tabs(char[] buf, usz* pos, usz depth)
{
	for (usz i = 0; i < depth; i++)
	{
		emit_char(buf, pos, '\t');
	}
}

// Formatted emission helpers

fn void emit_fmt_input_data(char[] buf, usz* pos, usz i, char[] dtype_str)
{
	// e.g.: float* in0 = (float*)inputs[0].state.dense.data.ptr;
	emit(buf, pos, "\t");
	emit(buf, pos, dtype_str);
	emit(buf, pos, "* in");
	emit_uint(buf, pos, i);
	emit(buf, pos, " = (");
	emit(buf, pos, dtype_str);
	emit(buf, pos, "*)inputs[");
	emit_uint(buf, pos, i);
	emit(buf, pos, "].state.dense.data;\n");
}

fn void emit_fmt_input_strides(char[] buf, usz* pos, usz i)
{
	// e.g.: ulong* in0_strides = inputs[0].state.dense.strides.buffer[0..];
	emit(buf, pos, "\tulong* in");
	emit_uint(buf, pos, i);
	emit(buf, pos, "_strides = inputs[");
	emit_uint(buf, pos, i);
	emit(buf, pos, "].state.dense.strides[0..];\n");
}

fn void emit_fmt_output_data(char[] buf, usz* pos, char[] dtype_str)
{
	emit(buf, pos, "\t");
	emit(buf, pos, dtype_str);
	emit(buf, pos, "* out = (");
	emit(buf, pos, dtype_str);
	emit(buf, pos, "*)output.state.dense.data;\n");
}

fn void emit_fmt_dim(char[] buf, usz* pos, char[] name, usz dim_idx)
{
	// e.g.: ulong i = dims[0];
	emit(buf, pos, "\tulong ");
	emit(buf, pos, name);
	emit(buf, pos, " = dims[");
	emit_uint(buf, pos, dim_idx);
	emit(buf, pos, "];\n");
}

fn void emit_fmt_param(char[] buf, usz* pos, usz i)
{
	emit(buf, pos, "\tdouble p");
	emit_uint(buf, pos, i);
	emit(buf, pos, " = params[");
	emit_uint(buf, pos, i);
	emit(buf, pos, "];\n");
}

fn void emit_fmt_for_loop(char[] buf, usz* pos, char[] name, usz depth)
{
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "for (ulong ");
	emit(buf, pos, name);
	emit(buf, pos, "_i = 0; ");
	emit(buf, pos, name);
	emit(buf, pos, "_i < ");
	emit(buf, pos, name);
	emit(buf, pos, "; ");
	emit(buf, pos, name);
	emit(buf, pos, "_i++)\n");
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "{\n");
}

fn void emit_fmt_for_loop_bound(char[] buf, usz* pos, char[] name, usz depth, bool used)
{
	emit_tabs(buf, pos, depth);
	if (used)
	{
		emit(buf, pos, "for (ulong ");
		emit(buf, pos, name);
		emit(buf, pos, "_i = 0; ");
		emit(buf, pos, name);
		emit(buf, pos, "_i < ");
		emit(buf, pos, name);
		emit(buf, pos, "; ");
		emit(buf, pos, name);
		emit(buf, pos, "_i++)\n");
	}
	else
	{
		emit(buf, pos, "for (ulong _unused = 0; _unused < ");
		emit(buf, pos, name);
		emit(buf, pos, "; _unused++)\n");
	}
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "{\n");
}

fn void emit_accumulator_init(char[] buf, usz* pos, char[] dtype_str, cas::ReduceOp op)
{
	emit(buf, pos, "\t");
	emit(buf, pos, dtype_str);
	emit(buf, pos, " acc = ");
	bool is_f32 = dtype_str.len == 5 && dtype_str[0] == 'f'; // "float"
	switch (op)
	{
		case ADD: emit(buf, pos, "0");
		case MUL: emit(buf, pos, "1");
		case MAX:
			if (is_f32) { emit(buf, pos, "(float)-3.4028235e+38"); }
			else { emit(buf, pos, "-1.7976931348623157e+308"); }
		case MIN:
			if (is_f32) { emit(buf, pos, "(float)3.4028235e+38"); }
			else { emit(buf, pos, "1.7976931348623157e+308"); }
	}
	emit(buf, pos, ";\n");
}

fn void emit_reduce_accumulate(char[] buf, usz* pos, cas::ReduceOp op, usz depth)
{
	emit(buf, pos, "\t\t");
	emit_tabs(buf, pos, depth);
	switch (op)
	{
		case ADD: emit(buf, pos, "acc += val;\n");
		case MUL: emit(buf, pos, "acc *= val;\n");
		case MAX: emit(buf, pos, "acc = (acc > val ? acc : val);\n");
		case MIN: emit(buf, pos, "acc = (acc < val ? acc : val);\n");
	}
}

fn void emit_store_output(char[] buf, usz* pos, FusedKernel* kernel, char[] dtype_str)
{
	usz depth = kernel.num_free + 1;
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "ulong out_idx = ");
	render_linear_index(buf, pos, kernel.output.indices(), "out_strides", kernel);
	emit(buf, pos, ";\n");

	emit_tabs(buf, pos, depth);
	if (kernel.epilogue_expr != null)
	{
		emit(buf, pos, "out[out_idx] = acc + ");
		render_cas_expr(buf, pos, kernel.epilogue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
	}
	else
	{
		emit(buf, pos, "out[out_idx] = acc;\n");
	}
}

fn void emit_store_inplace(char[] buf, usz* pos, FusedKernel* kernel, char[] dtype_str)
{
	usz depth = kernel.num_free + 1;
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "ulong in0_idx = ");
	render_linear_index(buf, pos, kernel.output.indices(), "in0_strides", kernel);
	emit(buf, pos, ";\n");

	emit_tabs(buf, pos, depth);
	if (kernel.epilogue_expr != null)
	{
		emit(buf, pos, "in0[in0_idx] = acc + ");
		render_cas_expr(buf, pos, kernel.epilogue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
	}
	else
	{
		emit(buf, pos, "in0[in0_idx] = acc;\n");
	}
}

fn void emit_elemwise_output(char[] buf, usz* pos, FusedKernel* kernel, char[] dtype_str)
{
	usz depth = kernel.num_free + 1;
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "ulong out_idx = ");
	render_linear_index(buf, pos, kernel.output.indices(), "out_strides", kernel);
	emit(buf, pos, ";\n");

	emit_tabs(buf, pos, depth);
	if (kernel.epilogue_expr != null)
	{
		emit(buf, pos, dtype_str);
		emit(buf, pos, " acc = ");
		render_cas_expr(buf, pos, kernel.prologue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
		emit_tabs(buf, pos, depth);
		emit(buf, pos, "out[out_idx] = ");
		render_cas_expr(buf, pos, kernel.epilogue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
	}
	else
	{
		emit(buf, pos, "out[out_idx] = ");
		render_cas_expr(buf, pos, kernel.prologue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
	}
}

fn void emit_elemwise_inplace(char[] buf, usz* pos, FusedKernel* kernel, char[] dtype_str)
{
	usz depth = kernel.num_free + 1;
	emit_tabs(buf, pos, depth);
	emit(buf, pos, "ulong in0_idx = ");
	render_linear_index(buf, pos, kernel.inputs[0].indices(), "in0_strides", kernel);
	emit(buf, pos, ";\n");

	emit_tabs(buf, pos, depth);
	if (kernel.epilogue_expr != null)
	{
		emit(buf, pos, dtype_str);
		emit(buf, pos, " acc = ");
		render_cas_expr(buf, pos, kernel.prologue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
		emit_tabs(buf, pos, depth);
		emit(buf, pos, "in0[in0_idx] = ");
		render_cas_expr(buf, pos, kernel.epilogue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
	}
	else
	{
		emit(buf, pos, "in0[in0_idx] = ");
		render_cas_expr(buf, pos, kernel.prologue_expr, kernel, dtype_str);
		emit(buf, pos, ";\n");
	}
}

fn void emit_shape_expr(char[] buf, usz* pos, ShapeExpr* se, FusedKernel* kernel, char[] dtype_str, usz shape_idx)
{
	// e.g.: float shape0 = (float)inputs[0].state.dense.num_elements;
	emit(buf, pos, "\t");
	emit(buf, pos, dtype_str);
	emit(buf, pos, " shape");
	emit_uint(buf, pos, shape_idx);
	emit(buf, pos, " = (");
	emit(buf, pos, dtype_str);
	emit(buf, pos, ")");

	switch (se.kind)
	{
		case NUMEL:
			emit(buf, pos, "inputs[");
			emit_uint(buf, pos, se.input);
			emit(buf, pos, "].state.dense.num_elements");

		case DIM:
			emit(buf, pos, "inputs[");
			emit_uint(buf, pos, se.input);
			emit(buf, pos, "].state.dense.shape.get(");
			emit_uint(buf, pos, (usz)se.axis);
			emit(buf, pos, ")");
	}

	emit(buf, pos, ";\n");
}
