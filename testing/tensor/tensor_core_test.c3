module metaphor::testing::tensor::tensor_core_test;
import metaphor::tensor::tensor_core;
import metaphor::tensor::dense_core;
import metaphor::tensor::shape;
import metaphor::rtti;

// ============================================================================
// Persistence tests
// ============================================================================

fn void test_persistence_ordering() @test
{
	Persistence none = NONE;
	Persistence unstable = UNSTABLE;
	Persistence stable = STABLE;
	Persistence leaf = LEAF;

	assert(none.lt(unstable));
	assert(none.lt(leaf));
	assert(!leaf.lt(none));

	assert(unstable.lte(unstable));
	assert(unstable.lte(stable));

	assert(leaf.gt(stable));
	assert(!none.gt(unstable));

	assert(stable.gte(stable));
	assert(stable.gte(unstable));
	assert(!none.gte(unstable));
}

// ============================================================================
// TensorCoreFlags tests
// ============================================================================

fn void test_default_flags() @test
{
	TensorCoreFlags f = tensor_core::DEFAULT_FLAGS;
	assert(f.alive);
	assert(f.attached);
	assert(!f.grad_operand);
	assert(!f._requires_grad);
	assert(f.persistence == Persistence.NONE);
}

// ============================================================================
// CoreState tests
// ============================================================================

fn void test_core_state_dense() @test
{
	ulong[2] dims = { 3, 4 };
	DenseCore dc = dense_core::dense_core_empty(&dims, F32, dense_core::DEFAULT_DENSE_CONFIG);
	CoreState state = tensor_core::core_state_dense(dc);

	assert(state.class_ == CoreClass.DENSE);
	assert(state.dense.num_elements == 12);
	assert(state.dense.data == null, "no data allocated yet");
}

// ============================================================================
// TensorCore lifecycle tests
// ============================================================================

fn void test_tensor_core_has_data() @test
{
	DenseCore dc = dense_core::dense_core_empty({ 2, 3 }, F32, dense_core::DEFAULT_DENSE_CONFIG);
	tensor_core::TensorCore tc;
	tc.state = tensor_core::core_state_dense(dc);
	tc.flags = tensor_core::DEFAULT_FLAGS;
	tc.label = {};
	tc.consumers = 0;
	tc.version = 0;
	tc.creation_context = 0;

	assert(!tc.has_data(), "no data before alloc");
}

fn void test_tensor_core_compilation_boundary() @test
{
	tensor_core::TensorCore tc;
	tc.flags = tensor_core::DEFAULT_FLAGS;

	// Default (NONE): not a compilation boundary
	assert(!tc.is_compilation_boundary());
	assert(tc.needs_compilation());

	// UNSTABLE makes it a boundary
	tc.flags.persistence = Persistence.UNSTABLE;
	assert(tc.is_compilation_boundary());

	// STABLE is also a boundary
	tc.flags.persistence = Persistence.STABLE;
	assert(tc.is_compilation_boundary());

	// Leaf is a boundary but doesn't need compilation
	tc.flags.persistence = Persistence.LEAF;
	assert(tc.is_compilation_boundary());
	assert(!tc.needs_compilation());
}

fn void test_tensor_core_increment_version() @test
{
	tensor_core::TensorCore tc;
	tc.version = 0;
	tc.increment_version();
	assert(tc.version == 1);
	tc.increment_version();
	assert(tc.version == 2);
}
