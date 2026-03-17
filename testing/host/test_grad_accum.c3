//! Gradient Accumulation Test
//!
//! Tests that gradients accumulate correctly when a single tensor
//! is used in multiple computational paths (diamond DAG).
//!
//!
module metaphor::testing::host::test_grad_accum;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

/// Complex case: tensor used in two paths via add and reduce.
/// Forward:
///   a = constant(2.0), b = constant(1.0)
///   c = a + b               [3, 3]
///   d = reduce(a, "i->")    scalar = 4
///   e = c * expand(d, "->i") = [3,3] * [4,4] = [12, 12]
///
/// Backward:
///   de/da via path c: de/dc * dc/da = expand(d) * 1 = [4, 4]
///   de/da via path d: de/dd * dd/da = sum(c) * 1 = 6
///   Total: [4+6, 4+6] = [10, 10]
fn void grad_accum_diamond() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };

	// Create tensors
	Tensor a = tensor::constant(F32, dev, &dims, 2.0)!!.enable_grad();
	Tensor b = tensor::constant(F32, dev, &dims, 1.0)!!;

	// Path 1: c = a + b
	Tensor c = a.add(b)!!;

	// Path 2: d = sum(a)
	Tensor d = a.reduce("i->")!!;

	// Merge: e = c * expand(d)
	Tensor expanded_d = d.expand("->i", { 2 })!!;
	Tensor e = c.mul(expanded_d)!!.stable();

	// Forward
	e.collect()!!;

	// Backward
	e.backward();

	// Get gradient of a
	Tensor grad_a = a.grad();
	grad_a.collect()!!;

	// Verify: gradient should be [10, 10]
	char[] grad_bytes = grad_a.value_bytes();
	float* grad_data = (float*)grad_bytes.ptr;
	float expected = 10.0;
	float tol = 1e-5;

	assert(math::abs(grad_data[0] - expected) < tol, "grad_accum: wrong gradient[0]");
	assert(math::abs(grad_data[1] - expected) < tol, "grad_accum: wrong gradient[1]");
}
