//! Codegen Context
//!
//! Shared context for JIT kernel code generation. Tracks unique tensor inputs
//! and carries output metadata (output indices for broadcast dimension matching).
//! Used by both HIP and CUDA JIT compilers.

module metaphor::device::common::codegen_context;
import metaphor::cas;
import std::collections::list;

alias PtrList = list::List{void*};

/// JIT codegen context: tracks unique tensor inputs and output metadata.
/// Backed by a dynamic list so there is no fixed capacity — expressions
/// of any fan-in can be counted before decomposition decides how to
/// partition them.
struct CodegenContext
{
	PtrList tensors;
	cas::IndexId[] output_indices;
}

fn void CodegenContext.tinit(&self)
{
	self.tensors.tinit();
}

fn usz CodegenContext.count(&self) @inline
{
	return self.tensors.len();
}

fn usz CodegenContext.get_or_add(&self, void* tensor)
{
	for (usz i = 0; i < self.tensors.len(); i++)
	{
		if (self.tensors[i] == tensor) return i;
	}
	self.tensors.push(tensor);
	return self.tensors.len() - 1;
}

fn usz CodegenContext.get(&self, void* tensor)
{
	for (usz i = 0; i < self.tensors.len(); i++)
	{
		if (self.tensors[i] == tensor) return i;
	}
	unreachable("CodegenContext.get: tensor not found");
}

/// Recursively collect unique tensor inputs from a CAS expression tree.
fn void collect_unique_inputs(cas::Expr* expr, CodegenContext* map)
{
	switch (expr.kind)
	{
		case ACCESS:
			map.get_or_add((void*)expr.access.tensor);
		case SCALAR: {}
		case PRODUCT:
			foreach (f : expr.operands) collect_unique_inputs(f, map);
		case SUM:
			foreach (t : expr.operands) collect_unique_inputs(t, map);
		case CONTRACT:
			collect_unique_inputs(expr.contract.body, map);
		case NEGATE:
			collect_unique_inputs(expr.inner, map);
		case UNARY:
			collect_unique_inputs(expr.unary.child, map);
		case BINARY:
			collect_unique_inputs(expr.binary.lhs, map);
			collect_unique_inputs(expr.binary.rhs, map);
		case COMPARE:
			collect_unique_inputs(expr.compare.lhs, map);
			collect_unique_inputs(expr.compare.rhs, map);
		case SELECT:
			collect_unique_inputs(expr.select_.cond, map);
			collect_unique_inputs(expr.select_.then_expr, map);
			collect_unique_inputs(expr.select_.else_expr, map);
		case PERMUTE:
			collect_unique_inputs(expr.permute.child, map);
		case BUILTIN:
			foreach (arg : expr.builtin.args) collect_unique_inputs(arg, map);
	}
}
