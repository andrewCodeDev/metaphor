//! HIP Device Infrastructure Tests
//!
//! Tests HIP device types, alignment helpers, and configuration
//! without requiring GPU hardware. These verify compile-time
//! infrastructure used by the HIP backend.

module metaphor::testing::device::hip_device::hip_device_test;
import metaphor::device::hip_device;

// ============================================================================
// Alignment Helpers
// ============================================================================

fn void test_align_size_basic() @test
{
	// Exact alignment should remain unchanged
	assert(hip_device::align_size(256, 256) == 256);
	assert(hip_device::align_size(512, 256) == 512);

	// Slightly over should round up
	assert(hip_device::align_size(257, 256) == 512);
	assert(hip_device::align_size(300, 256) == 512);

	// Zero should stay zero
	assert(hip_device::align_size(0, 256) == 0);

	// One byte should round up to alignment
	assert(hip_device::align_size(1, 256) == 256);
}

fn void test_align_size_power_of_two() @test
{
	// Powers of two
	assert(hip_device::align_size(128, 256) == 256);
	assert(hip_device::align_size(64, 256) == 256);

	// Exactly 2x alignment
	assert(hip_device::align_size(512, 256) == 512);
}

fn void test_aligned_stride() @test
{
	// batch_size=1, features=64, elem_size=4 (f32)
	// raw = 1*64*4 = 256 => aligned to 256
	usz s1 = hip_device::aligned_stride(1, 64, 4);
	assert(s1 == 256, "64 floats = 256 bytes, already aligned");

	// batch_size=1, features=65, elem_size=4
	// raw = 1*65*4 = 260 => aligned to 512
	usz s2 = hip_device::aligned_stride(1, 65, 4);
	assert(s2 == 512, "65 floats = 260 bytes, rounds up to 512");
}

fn void test_padding_elements() @test
{
	// Already aligned: 64 floats = 256 bytes
	usz p1 = hip_device::padding_elements(1, 64, 4);
	assert(p1 == 0, "no padding when already aligned");

	// 65 floats = 260 bytes, aligned=512, padding=(512-260)/4=63
	usz p2 = hip_device::padding_elements(1, 65, 4);
	assert(p2 == 63, "63 padding elements to reach alignment");
}

fn void test_aligned_buffer_size() @test
{
	// 4 batches of 64 floats, each batch aligned
	usz buf = hip_device::aligned_buffer_size(4, 1, 64, 4);
	assert(buf == 4 * 256, "4 batches * 256 bytes stride");

	// 2 batches of 65 floats
	usz buf2 = hip_device::aligned_buffer_size(2, 1, 65, 4);
	assert(buf2 == 2 * 512, "2 batches * 512 bytes stride");
}

// ============================================================================
// Default Options
// ============================================================================

fn void test_hip_device_default_options() @test
{
	hip_device::HipDeviceOptions opts = hip_device::DEFAULT_HIP_OPTIONS;
	assert(opts.device_id == 0, "default device_id should be 0");
	assert(opts.graph_capture_mode == hip_device::GraphCaptureMode.DISABLED, "default capture mode should be DISABLED");
}

// ============================================================================
// Constants
// ============================================================================

fn void test_alignment_constant() @test
{
	assert(hip_device::ALIGNMENT == 256, "HIP alignment must be 256 bytes");
}
