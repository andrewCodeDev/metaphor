module metaphor::testing::cas::cas_test;
import metaphor::cas;
import std::core::mem::allocator;

// ============================================================================
// Expression construction tests
// ============================================================================

fn void test_basic_expr_construction() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 64);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);
	Index i = builder.index(4);
	Index j = builder.index(4);
	Index k = builder.index(4);

	// A[i,j] * B[j,k] contracted over j (summation) => matmul
	IndexId[2] a_indices = { i.id, j.id };
	IndexId[2] b_indices = { j.id, k.id };

	Expr* a_expr = builder.access(null, &a_indices, false)!!;
	Expr* b_expr = builder.access(null, &b_indices, false)!!;

	Expr*[2] factors = { a_expr, b_expr };
	Expr* prod = builder.product(&factors)!!;

	Expr* matmul = builder.contract(j, prod, ADD)!!;

	assert(matmul.kind == CONTRACT);
	assert(matmul.contract.index.id == j.id);
	assert(matmul.contract.body.kind == PRODUCT);
	assert(matmul.contract.body.operands.len == 2);
}

fn void test_scalar_construction() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 16);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);
	Expr* s = builder.scalar(3.14)!!;

	assert(s.kind == SCALAR);
	assert(s.scalar == 3.14);
}

fn void test_einsum2() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 64);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);
	Index i = builder.index(4);
	Index j = builder.index(4);
	Index k = builder.index(4);

	IndexId[2] a_indices = { i.id, j.id };
	IndexId[2] b_indices = { j.id, k.id };

	Expr* a_expr = builder.access(null, &a_indices, false)!!;
	Expr* b_expr = builder.access(null, &b_indices, false)!!;

	// einsum2 with one contracted index => Σ_j(A[i,j] * B[j,k])
	Index[1] contracted = { j };
	Expr* matmul = builder.einsum2(a_expr, b_expr, &contracted)!!;

	assert(matmul.kind == CONTRACT);
	assert(matmul.contract.index.id == j.id);
	assert(matmul.contract.body.kind == PRODUCT);
}

// ============================================================================
// Expr query tests
// ============================================================================

fn void test_depends_on() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 64);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);
	Index i = builder.index(4);
	Index j = builder.index(4);
	Index k = builder.index(4);

	IndexId[2] a_indices = { i.id, j.id };
	Expr* a_expr = builder.access(null, &a_indices, false)!!;

	assert(a_expr.depends_on(i.id), "A[i,j] depends on i");
	assert(a_expr.depends_on(j.id), "A[i,j] depends on j");
	assert(!a_expr.depends_on(k.id), "A[i,j] does not depend on k");

	// Scalar doesn't depend on anything
	Expr* s = builder.scalar(1.0)!!;
	assert(!s.depends_on(i.id));

	// Contraction hides the contracted index
	Expr*[1] body = { a_expr };
	Expr* prod = builder.product(&body)!!;
	Expr* contracted = builder.contract(j, prod, ADD)!!;
	assert(contracted.depends_on(i.id), "Σ_j(A[i,j]) depends on i");
	assert(!contracted.depends_on(j.id), "Σ_j(A[i,j]) does NOT depend on j (bound)");
}

fn void test_has_stable() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 64);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);
	Index i = builder.index(4);

	IndexId[1] idx = { i.id };
	Expr* stable = builder.access(null, &idx, true)!!;
	Expr* unstable = builder.access(null, &idx, false)!!;

	assert(stable.has_stable());
	assert(!unstable.has_stable());

	// Product containing a stable access
	Expr*[2] factors = { stable, unstable };
	Expr* prod = builder.product(&factors)!!;
	assert(prod.has_stable());

	// Negate wrapping stable
	Expr* neg = builder.create_negate(stable)!!;
	assert(neg.has_stable());
}

fn void test_expr_equal() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 64);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);

	// Same scalar value
	Expr* s1 = builder.scalar(2.0)!!;
	Expr* s2 = builder.scalar(2.0)!!;
	assert(cas::expr_equal(s1, s2));

	// Different scalar value
	Expr* s3 = builder.scalar(3.0)!!;
	assert(!cas::expr_equal(s1, s3));

	// Pointer equality shortcut
	assert(cas::expr_equal(s1, s1));
}

fn void test_prev() @test
{
	DynamicArenaAllocator arena;
	arena.init(mem, 1024 * 64);
	defer arena.free();
	

	ExprBuilder builder = cas::expr_builder_init(&arena);

	Expr* s = builder.scalar(1.0)!!;
	Expr* neg = builder.create_negate(s)!!;
	Expr* u = builder.create_unary(ABS, s)!!;

	assert(neg.prev() == s, "negate.prev() is inner");
	assert(u.prev() == s, "unary.prev() is child");
	assert(s.prev() == null, "scalar.prev() is null");
}

