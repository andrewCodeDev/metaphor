//! Host Device integration tests
//!
//! Tests the full pipeline: HostDevice → DeviceReference → Graph → Tensor
//! These tests exercise real memory allocation via the caching allocator.

module metaphor::testing::device::host_device::host_device_test;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::rtti;

// ============================================================================
// HostDevice lifecycle
// ============================================================================

fn void test_host_device_init_deinit() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();

	DeviceReference dev = cpu.reference();
	assert(dev.name() == "host", "device name should be 'host'");
}

fn void test_host_device_mem_alloc_free() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();

	DeviceReference dev = cpu.reference();

	// Allocate 100 f32 elements = 400 bytes
	char[] buf = dev.mem_alloc(F32, 100);
	assert(buf.len >= 400, "should allocate at least 400 bytes");

	// Memory should be tracked
	assert(cpu.mem_used() > 0, "should report non-zero current memory");

	// Free
	dev.mem_free(buf);
}

fn void test_host_device_mem_copy() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();

	DeviceReference dev = cpu.reference();

	char[] src = dev.mem_alloc(F32, 4);
	char[] dst = dev.mem_alloc(F32, 4);
	defer { dev.mem_free(src); dev.mem_free(dst); }

	// Write known pattern to source via pointer cast
	float* src_f = (float*)src.ptr;
	src_f[0] = 1.0;
	src_f[1] = 2.0;
	src_f[2] = 3.0;
	src_f[3] = 4.0;

	// Copy
	dev.mem_copy(src, dst);

	// Verify via pointer cast
	float* dst_f = (float*)dst.ptr;
	assert(dst_f[0] == 1.0);
	assert(dst_f[1] == 2.0);
	assert(dst_f[2] == 3.0);
	assert(dst_f[3] == 4.0);
}

// ============================================================================
// Full pipeline: HostDevice → Graph → Tensor
// ============================================================================

fn void test_tensor_on_host_device() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();

	DeviceReference dev = cpu.reference();

	ulong[2] dims = { 3, 4 };
	Tensor t = tensor::empty(F32, dev, &dims)!!;

	// Verify tensor properties
	assert(t.core() != null);
	assert(t.dtype() == F32);
	assert(t.shape().len == 2);
	assert(t.shape().get(0) == 3);
	assert(t.shape().get(1) == 4);
	assert(t.has_data(), "tensor should have allocated data on host device");

	// Verify device is host
	assert(t.device().is_host(), "tensor should be on host device");
}

fn void test_tensor_data_accessible() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();

	DeviceReference dev = cpu.reference();

	Tensor t = tensor::empty(F32, dev, { 4 })!!;

	// Host device memory is directly accessible — write and read back
	TensorCore* tc = t.core();
	assert(tc.state.class_ == tensor_core::CoreClass.DENSE);
	void* data_ptr = tc.state.dense.data;
	assert(data_ptr != null, "data pointer should be non-null");

	// Cast to float pointer and write values
	float* fp = (float*)data_ptr;
	fp[0] = 10.0;
	fp[1] = 20.0;
	fp[2] = 30.0;
	fp[3] = 40.0;

	// Read back and verify
	assert(fp[0] == 10.0);
	assert(fp[1] == 20.0);
	assert(fp[2] == 30.0);
	assert(fp[3] == 40.0);
}

fn void test_multiple_tensors_on_host() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();

	DeviceReference dev = cpu.reference();

	Tensor t1 = tensor::empty(F32, dev, { 10 })!!;
	Tensor t2 = tensor::empty(F64, dev, { 3, 4 })!!;
	Tensor t3 = tensor::empty(F32, dev, { 2, 3, 5 })!!;

	// All should have data
	assert(t1.has_data());
	assert(t2.has_data());
	assert(t3.has_data());

	// All on host
	assert(t1.device().is_host());
	assert(t2.device().is_host());
	assert(t3.device().is_host());

	// Distinct cores
	assert(t1.core() != t2.core());
	assert(t2.core() != t3.core());

	// Memory should be tracked
	assert(cpu.mem_used() > 0);
}
