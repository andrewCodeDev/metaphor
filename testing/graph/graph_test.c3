module metaphor::testing::graph::graph_test;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::dense_core;
import metaphor::tensor::shape;
import metaphor::device::device_reference;
import metaphor::rtti;

// ============================================================================
// Mock device for graph tests
// ============================================================================

struct MockDevice (Device)
{
	int alloc_calls;
	int free_calls;
}

fn ZString MockDevice.name(&self) @dynamic { return "host"; }
fn char[] MockDevice.mem_alloc(&self, Datatype dtype, usz n) @dynamic
{
	self.alloc_calls++;
	return {};
}
fn void MockDevice.mem_free(&self, char[] slice) @dynamic { self.free_calls++; }
fn void MockDevice.mem_copy(&self, char[] src, char[] dst) @dynamic {}
fn void MockDevice.mem_transfer(&self, char[] src, char[] dst, TransferDirection dir) @dynamic {}
fn void MockDevice.sync(&self) @dynamic {}
fn usz MockDevice.mem_used(&self) @dynamic { return 0; }
fn usz MockDevice.mem_reserved(&self) @dynamic { return 0; }
fn usz MockDevice.mem_peak(&self) @dynamic { return 0; }
fn void MockDevice.execute(&self, any operation, device_reference::TensorCorePtr[] sources, device_reference::TensorCorePtr target) @dynamic {}
fn void MockDevice.prepare_execute(&self, device_reference::TensorCorePtr core) @dynamic {}
fn void MockDevice.finalize_execute(&self, device_reference::TensorCorePtr core) @dynamic {}
fn void MockDevice.mem_random(&self, Datatype dtype, char[] slice, device_reference::RandType rand_type, device_reference::Scale scale = device_reference::DEFAULT_SCALE) @dynamic {}
fn void MockDevice.mem_fill(&self, Datatype dtype, char[] slice, double value) @dynamic {}
fn char[] MockDevice.mem_dupe(&self, Datatype dtype, char[] src) @dynamic { return {}; }
fn void MockDevice.mem_sequence(&self, Datatype dtype, char[] dst, double initial, double step) @dynamic {}
fn void MockDevice.mem_cast(&self, Datatype src_dtype, Datatype dst_dtype, char[] src, char[] dst) @dynamic {}
fn void MockDevice.compile(&self, void* graph, void* entry) @dynamic {}
fn uint MockDevice.sample(&self, Datatype dtype, char[] logits, usz vocab_size, char[] context, usz context_len, void* config) @dynamic { return 0; }
fn uint MockDevice.argmax(&self, Datatype dtype, char[] logits, usz vocab_size) @dynamic { return 0; }

// ============================================================================
// Global graph lifecycle tests
// ============================================================================

fn void test_global_graph_init_deinit() @test
{

	graph::Graph* g = graph::get();
	assert(g != null, "graph::get() should return non-null after init");
}

fn void test_global_graph_create_tensor_core() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[2] dims = { 3, 4 };

	graph::Graph* g = graph::get();
	uint before = g.counts.tensors;
	TensorCore* tc = g.create_tensor_core(&dims, F32, dev);

	assert(tc != null, "create_tensor_core should return non-null");
	assert(tc.dtype() ==F32);
	assert(tc.shape().len == 2, "should have 2 dimensions");
	assert(tc.flags.alive, "should be alive");
	assert(tc.flags.attached, "should be attached");
	assert(!tc.flags._requires_grad, "should not require grad by default");
	assert(tc.flags.persistence == tensor_core::Persistence.NONE, "should start at NONE persistence");
	assert(tc.graph == g, "graph backpointer should match");
	assert(g.counts.tensors == before + 1, "graph should track tensor count");
}

fn void test_global_graph_create_tensor_core_with_options() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[1] dims = { 10 };

	graph::Graph* g = graph::get();
	TensorCore* tc = g.create_tensor_core(&dims, F64, dev,
		{ .requires_grad = true, .label = "weights" });

	assert(tc.flags._requires_grad, "should require grad when requested");
	assert(tc.dtype() ==F64);
}

fn void test_global_graph_multiple_tensors() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[1] dims = { 5 };

	graph::Graph* g = graph::get();
	uint before = g.counts.tensors;
	TensorCore* tc1 = g.create_tensor_core(&dims, F32, dev);
	TensorCore* tc2 = g.create_tensor_core(&dims, F32, dev);
	TensorCore* tc3 = g.create_tensor_core(&dims, F32, dev);

	assert(tc1 != tc2, "tensors should be distinct");
	assert(tc2 != tc3, "tensors should be distinct");
	assert(g.counts.tensors == before + 3, "should have 3 more tensors");
}

// ============================================================================
// @subgraph() tests
// ============================================================================

fn void test_subgraph_isolation() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[1] dims = { 5 };

	graph::Graph* outer = graph::get();
	uint before = outer.counts.tensors;
	TensorCore* outer_tc = outer.create_tensor_core(&dims, F32, dev);

	graph::@subgraph()
	{
		graph::Graph* inner = graph::get();
		assert(inner != outer, "subgraph should be a different graph");

		TensorCore* inner_tc = inner.create_tensor_core(&dims, F32, dev);
		assert(inner_tc.graph == inner, "inner tensor should belong to subgraph");
		assert(inner.counts.tensors == 1, "subgraph should have 1 tensor");
	};

	// After subgraph exits, we should be back on the outer graph
	graph::Graph* restored = graph::get();
	assert(restored == outer, "should restore outer graph after subgraph");
	assert(outer.counts.tensors == before + 1, "outer graph tensor count unchanged");
}

fn void test_subgraph_nested() @test
{

	graph::Graph* g0 = graph::get();

	graph::@subgraph()
	{
		graph::Graph* g1 = graph::get();
		assert(g1 != g0);

		graph::@subgraph()
		{
			graph::Graph* g2 = graph::get();
			assert(g2 != g1);
			assert(g2 != g0);
		};

		// Back to g1
		assert(graph::get() == g1);
	};

	// Back to g0
	assert(graph::get() == g0);
}

fn void test_subgraph_without_parent() @test
{
	// @subgraph works even when no global graph is set — parent will be null
	graph::@subgraph()
	{
		graph::Graph* g = graph::get();
		assert(g != null, "subgraph should provide a valid graph");
	};

	// After subgraph exits, current_graph should be null again (restored to parent=null)
	// We can't call graph::get() here because it asserts non-null, so test passes
	// if we reach this point without crashing
}

// ============================================================================
// Tensor factory function tests
// ============================================================================

fn void test_tensor_empty() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[2] dims = { 3, 4 };

	Tensor t = tensor::empty(F32, dev, &dims)!!;

	assert(t.core() != null, "tensor should have a core");
	assert(t.dtype() == F32);
	assert(t.core().shape().len == 2);
	assert(t.core().flags.persistence == tensor_core::Persistence.LEAF,
		"empty tensor should be marked as leaf");
	assert(mock.alloc_calls == 1, "should have allocated data");
}

fn void test_tensor_zeros() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[1] dims = { 10 };

	Tensor t = tensor::zeros(F32, dev, &dims)!!;

	assert(t.core() != null);
	assert(t.dtype() == F32);
	assert(t.core().flags.persistence == tensor_core::Persistence.LEAF);
}

fn void test_tensor_ones() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[3] dims = { 2, 3, 4 };

	Tensor t = tensor::ones(F64, dev, &dims)!!;

	assert(t.core() != null);
	assert(t.dtype() == F64);
	assert(t.core().shape().len == 3);
}

fn void test_tensor_factory_in_subgraph() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[1] dims = { 5 };

	graph::Graph* outer = graph::get();
	Tensor outer_t = tensor::empty(F32, dev, &dims)!!;

	graph::@subgraph()
	{
		graph::Graph* inner = graph::get();
		Tensor inner_t = tensor::empty(F32, dev, &dims)!!;

		assert(inner_t.core().graph == inner,
			"tensor created in subgraph belongs to subgraph");
		assert(outer_t.core().graph == outer,
			"outer tensor still belongs to outer graph");
	};
}

// ============================================================================
// export tests
// ============================================================================

fn void test_export_from_subgraph() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };
	ulong[2] dims = { 3, 4 };

	graph::Graph* global = graph::get();
	uint initial_count = global.counts.tensors;

	Tensor exported;
	graph::@subgraph()
	{
		graph::Graph* sub = graph::get();
		assert(sub != global);

		// Create a tensor in the subgraph
		TensorCore* tc = sub.create_tensor_core(&dims, F32, dev);
		Tensor sub_tensor = tensor::tensor_from_core(tc);

		// Export it to the global graph (steals data)
		exported = graph::export(sub_tensor);

		// Source tensor's data should be zeroed (stolen)
		assert(!sub_tensor.has_data(), "source data should be stolen");

		// Exported tensor belongs to the global graph
		assert(exported.core().graph == global,
			"exported tensor should belong to global graph");
	};

	// After subgraph exit, exported tensor is still valid
	assert(exported.core().graph == global);
	assert(exported.dtype() == F32);
	assert(exported.core().shape().len == 2);
	assert(exported.core().flags.persistence == tensor_core::Persistence.LEAF,
		"exported tensor should be a leaf");
	assert(global.counts.tensors == initial_count + 1,
		"global graph should have one more tensor");
}

fn void test_export_preserves_properties() @test
{

	MockDevice mock;
	DeviceReference dev = { .device = &mock };

	Tensor exported;
	graph::@subgraph()
	{
		TensorCore* tc = graph::get().create_tensor_core(
			{ 2, 3, 4 }, F64, dev,
			{ .requires_grad = false, .label = "tmp" });

		exported = graph::export(tensor::tensor_from_core(tc));
	};

	assert(exported.dtype() == F64, "dtype should be preserved");
	assert(exported.core().shape().len == 3, "shape rank should be preserved");
}

