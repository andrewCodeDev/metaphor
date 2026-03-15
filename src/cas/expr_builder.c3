//! ExprBuilder — construct CAS expressions
//!
//! All expressions are arena-allocated and immutable after creation.
//! The arena is freed when the compilation pass completes.
//!

module metaphor::cas;
import metaphor::tensor::tensor_core;
import std::core::mem::allocator;

struct ExprBuilder
{
	Allocator arena;
	IndexId next_index;
}

fn ExprBuilder expr_builder_init(Allocator arena)
{
	ExprBuilder b;
	b.arena = arena;
	b.next_index = 0;
	return b;
}

// Create a new unique index with given size.
fn Index ExprBuilder.index(&self, ulong size)
{
	IndexId id = self.next_index;
	self.next_index++;
	Index idx;
	idx.id = id;
	idx.size = size;
	return idx;
}

// Create a tensor access expression.
fn Expr*? ExprBuilder.access(&self, TensorCore* tensor, IndexId[] indices, bool is_stable)
{
	IndexId[] indices_copy = allocator::alloc_array_try(self.arena, IndexId, indices.len)!;
	indices_copy[..] = indices[..];

	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = {
		.kind = ACCESS,
		.access = {
			.tensor = tensor,
			.indices = indices_copy,
			.is_stable = is_stable,
		},
	};
	return expr;
}

// Create a scalar constant.
fn Expr*? ExprBuilder.scalar(&self, double value)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = SCALAR, .scalar = value };
	return expr;
}

// Create a product of expressions.
fn Expr*? ExprBuilder.product(&self, Expr*[] factors)
{
	Expr*[] factors_copy = allocator::alloc_array_try(self.arena, Expr*, factors.len)!;
	factors_copy[..] = factors[..];

	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = PRODUCT, .operands = factors_copy };
	return expr;
}

// Create a sum of expressions.
fn Expr*? ExprBuilder.sum(&self, Expr*[] terms)
{
	Expr*[] terms_copy = allocator::alloc_array_try(self.arena, Expr*, terms.len)!;
	terms_copy[..] = terms[..];

	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = SUM, .operands = terms_copy };
	return expr;
}

// Create a contraction over an index with the specified reduce operation.
fn Expr*? ExprBuilder.contract(&self, Index idx, Expr* body, ReduceOp op)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = {
		.kind = CONTRACT,
		.contract = { .index = idx, .body = body, .op = op },
	};
	return expr;
}

// Create negation.
fn Expr*? ExprBuilder.create_negate(&self, Expr* e)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = NEGATE, .inner = e };
	return expr;
}

// Create a unary function application.
fn Expr*? ExprBuilder.create_unary(&self, UnaryOp op, Expr* child)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = UNARY, .unary = { .op = op, .child = child } };
	return expr;
}

// Create a binary operation.
fn Expr*? ExprBuilder.create_binary(&self, BinaryOp op, Expr* lhs, Expr* rhs)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = BINARY, .binary = { .op = op, .lhs = lhs, .rhs = rhs } };
	return expr;
}

// Create a comparison.
fn Expr*? ExprBuilder.create_compare(&self, CompareOp op, Expr* lhs, Expr* rhs)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = { .kind = COMPARE, .compare = { .op = op, .lhs = lhs, .rhs = rhs } };
	return expr;
}

// Create a ternary select.
fn Expr*? ExprBuilder.create_select(&self, Expr* cond, Expr* then_expr, Expr* else_expr)
{
	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = {
		.kind = SELECT,
		.select_ = { .cond = cond, .then_expr = then_expr, .else_expr = else_expr },
	};
	return expr;
}

// Create permutation (index reordering).
// mapping[i] gives the input position that goes to output position i.
fn Expr*? ExprBuilder.permute(&self, Expr* child, char[] mapping)
{
	char[] mapping_copy = allocator::alloc_array_try(self.arena, char, mapping.len)!;
	mapping_copy[..] = mapping[..];

	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = {
		.kind = PERMUTE,
		.permute = { .child = child, .mapping = mapping_copy },
	};
	return expr;
}

// Create a built-in function call (NN ops, etc.)
fn Expr*? ExprBuilder.builtin(&self, BuiltinFn func, Expr*[] args, IndexId[] output_indices)
{
	Expr*[] args_copy = allocator::alloc_array_try(self.arena, Expr*, args.len)!;
	args_copy[..] = args[..];

	IndexId[] indices_copy = allocator::alloc_array_try(self.arena, IndexId, output_indices.len)!;
	indices_copy[..] = output_indices[..];

	Expr* expr = allocator::alloc_try(self.arena, Expr)!;
	*expr = {
		.kind = BUILTIN,
		.builtin = { .func = func, .args = args_copy, .output_indices = indices_copy },
	};
	return expr;
}

// Create a contraction from einsum-like pattern.
// Example: einsum(A, B, contracted_indices) creates Σ_j(A * B) for each contracted index j.
fn Expr*? ExprBuilder.einsum2(&self, Expr* a, Expr* b, Index[] contracted_indices)
{
	Expr*[2] pair = { a, b };
	Expr* result = self.product(&pair)!;

	foreach (idx : contracted_indices)
	{
		result = self.contract(idx, result, ADD)!;
	}

	return result;
}
