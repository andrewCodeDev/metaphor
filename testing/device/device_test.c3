module metaphor::device::device_test;
import metaphor::rtti;
import metaphor::device::device_reference;
import metaphor::device::sync;
import std::io;

// ============================================================================
// Datatype tests
// ============================================================================


// ============================================================================
// OpTag tests
// ============================================================================

fn void test_optag_classification() @test
{
	rtti::OpTag add = rtti::OpTag.ADD;
	rtti::OpTag cmp = rtti::OpTag.CMP_EQ;
	rtti::OpTag land = rtti::OpTag.LOGIC_AND;
	rtti::OpTag conv = rtti::OpTag.LOGIC_DENSE;
	rtti::OpTag cmp_gt = rtti::OpTag.CMP_GT;

	// Comparison
	assert(!add.is_comparison());
	assert(cmp.is_comparison());
	assert(cmp_gt.is_comparison());

	// Ordering comparison (excludes CMP_EQ)
	assert(!cmp.is_ordering_comparison());
	assert(cmp_gt.is_ordering_comparison());

	// Logical
	assert(land.is_logical());
	assert(!add.is_logical());

	// Gradient
	assert(add.has_gradient());
	assert(!cmp.has_gradient());
	assert(!land.has_gradient());

	// Bool inputs
	assert(land.expects_bool_inputs());
	assert(conv.expects_bool_inputs());
	assert(!add.expects_bool_inputs());

	// Bool output
	assert(cmp.produces_bool());
	assert(land.produces_bool());
	assert(!add.produces_bool());

	// Converter
	assert(conv.is_converter());
	assert(!add.is_converter());
}

// ============================================================================
// HostSync tests
// ============================================================================

fn void test_host_sync_empty() @test
{
	HostSync hs;
	hs.init_empty();

	assert(hs.is_ready(), "empty sync should be ready");

	// begin/complete on null counter should be safe
	hs.begin();
	hs.complete();
	assert(hs.is_ready(), "still ready with null counter");
}

fn void test_host_sync_lifecycle() @test
{
	HostSync hs;
	hs.init_with_counter(mem);

	assert(hs.is_ready(), "counter starts at 0");

	hs.begin();
	assert(!hs.is_ready(), "pending after begin");

	hs.complete();
	assert(hs.is_ready(), "ready after complete");

	// Multiple begins
	hs.begin();
	hs.begin();
	assert(!hs.is_ready());
	hs.complete();
	assert(!hs.is_ready(), "one complete is not enough");
	hs.complete();
	assert(hs.is_ready(), "two completes balance two begins");

	hs.free_counter(mem);
}

fn void test_host_sync_reset() @test
{
	HostSync hs;
	hs.init_with_counter(mem);

	hs.begin();
	hs.begin();
	hs.begin();
	assert(!hs.is_ready());

	hs.reset();
	assert(hs.is_ready(), "reset clears pending count");

	hs.free_counter(mem);
}

// ============================================================================
// Mock device for DeviceReference tests
// ============================================================================

struct MockDevice (Device)
{
	int alloc_calls;
	int free_calls;
	int exec_calls;
}

fn ZString MockDevice.name(&self) @dynamic
{
	return "host";
}

fn char[] MockDevice.mem_alloc(&self, Datatype dtype, usz n) @dynamic
{
	self.alloc_calls++;
	// Return empty slice — we're not actually allocating
	return {};
}

fn void MockDevice.mem_free(&self, char[] slice) @dynamic
{
	self.free_calls++;
}

fn void MockDevice.mem_copy(&self, char[] src, char[] dst) @dynamic {}
fn void MockDevice.mem_transfer(&self, char[] src, char[] dst, TransferDirection dir) @dynamic {}
fn void MockDevice.sync(&self) @dynamic {}
fn usz MockDevice.mem_used(&self) @dynamic { return 0; }
fn usz MockDevice.mem_reserved(&self) @dynamic { return 0; }
fn usz MockDevice.mem_peak(&self) @dynamic { return 0; }

fn void MockDevice.execute(&self, any operation, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	self.exec_calls++;
}

fn void MockDevice.prepare_execute(&self, TensorCorePtr core) @dynamic {}
fn void MockDevice.finalize_execute(&self, TensorCorePtr core) @dynamic {}
fn void MockDevice.mem_random(&self, Datatype dtype, char[] slice, device_reference::RandType rand_type, device_reference::Scale scale = device_reference::DEFAULT_SCALE) @dynamic {}
fn void MockDevice.mem_fill(&self, Datatype dtype, char[] slice, double value) @dynamic {}
fn char[] MockDevice.mem_dupe(&self, Datatype dtype, char[] src) @dynamic { return {}; }
fn void MockDevice.mem_sequence(&self, Datatype dtype, char[] dst, double initial, double step) @dynamic {}
fn void MockDevice.mem_cast(&self, Datatype src_dtype, Datatype dst_dtype, char[] src, char[] dst) @dynamic {}
fn void MockDevice.compile(&self, void* graph, void* entry) @dynamic {}
fn uint MockDevice.sample(&self, Datatype dtype, char[] logits, usz vocab_size, char[] context, usz context_len, void* config) @dynamic { return 0; }
fn uint MockDevice.argmax(&self, Datatype dtype, char[] logits, usz vocab_size) @dynamic { return 0; }

// ============================================================================
// DeviceReference tests
// ============================================================================

fn void test_device_reference_dispatch() @test
{
	MockDevice mock = { .alloc_calls = 0, .free_calls = 0, .exec_calls = 0 };
	DeviceReference ref = { .device = &mock };

	device_reference::reset_mem_stats();

	assert(ref.name() == "host");

	ref.mem_alloc(F32, 100);
	assert(mock.alloc_calls == 1);

	char[] dummy = {};
	ref.mem_free(dummy);
	assert(mock.free_calls == 1);

	TensorCorePtr[] no_sources = {};
	ref.execute(null, no_sources, null);
	assert(mock.exec_calls == 1);
}

fn void test_device_reference_is_host() @test
{
	MockDevice mock = { .alloc_calls = 0, .free_calls = 0, .exec_calls = 0 };
	DeviceReference ref = { .device = &mock };

	assert(ref.is_host(), "mock device named 'host' should be host");
}

fn void test_device_reference_compatibility() @test
{
	MockDevice mock1 = { .alloc_calls = 0, .free_calls = 0, .exec_calls = 0 };
	MockDevice mock2 = { .alloc_calls = 0, .free_calls = 0, .exec_calls = 0 };
	DeviceReference ref1 = { .device = &mock1 };
	DeviceReference ref2 = { .device = &mock2 };

	assert(ref1.is_compatible(ref2), "same device type should be compatible");
}
