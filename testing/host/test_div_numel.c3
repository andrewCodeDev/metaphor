//! Div-Numel Test
//!
//! Tests division of f64 scalar (sum) by u64 scalar (element count)
//! to verify correct type handling and numeric result (mean calculation).
//!
//!
//! NOTE: Not @test yet — tensor operations are stubs. Mark @test
//! when tensor_ops pipeline is ported.

module metaphor::testing::host::test_div_numel;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

/// Test: divide f64 sum by u64 count to compute mean.
/// data = [1, 2, 3, 4, 5, 6] (shape 2x3, f64)
/// sum  = 21.0 (f64 scalar)
/// count = 6 (u64 scalar)
/// mean = sum / count = 3.5
fn void? div_numel_mean()
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Create tensor from data
	double[6] data = { 1, 2, 3, 4, 5, 6 };
	Tensor x = tensor::@from_data(dev, { 2, 3 }, &data)!!;

	// Sum = 21
	Tensor sum = x.reduce("ij->")!;
	sum.collect()!;

	// Numel = 6 (u64)
	Tensor count = x.numel()!;
	count.collect()!;

	// Mean = sum / count = 3.5
	Tensor mean = sum.div(count)!;
	mean.collect()!;

	char[] mean_bytes = mean.value_bytes();
	assert(mean_bytes.len >= double.sizeof, "mean should have data");
	double mean_val = ((double*)mean_bytes.ptr)[0];

	assert(math::abs(mean_val - 3.5) < 0.01, "mean should be 3.5");
}
