//! Expand Bug Test
//!
//! Tests that expand() (broadcast) combined with reduce()
//! correctly backpropagates gradients. When a 1D tensor is expanded
//! to 2D and then summed, the gradient should accumulate properly.
//!
//!
module metaphor::testing::host::test_expand_bug;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;

/// Forward:
///   x = [1, 2, 3, 4] (shape [4], requires_grad)
///   y = expand(x, "i->ji", sizes=[3]) -> 3x4 matrix (each row = x)
///   z = reduce(y, "ji->") -> scalar = 3 * (1+2+3+4) = 30
///
/// Backward:
///   dz/dx_i = 3 (each x_i contributes to 3 rows)
///   grad(x) should be [3, 3, 3, 3]
fn void expand_reduce_gradient() @test
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	// Create input
	float[4] data = { 1, 2, 3, 4 };
	Tensor x = tensor::@from_data(dev, { 4 }, &data)!!.enable_grad();

	// Expand to 3x4, then reduce to scalar
	Tensor y = x.expand("i->ji", { 3 })!!;
	Tensor z = y.reduce("ji->")!!.stable();

	// Forward
	z.collect()!!;

	// Backward
	z.backward();

	// Check gradient
	Tensor grad_x = x.grad();
	grad_x.collect()!!;

	char[] grad_bytes = grad_x.value_bytes();
	float* grad_data = (float*)grad_bytes.ptr;

	float expected = 3.0;
	float tol = 1e-5;

	for (usz i = 0; i < 4; i++)
	{
		assert(math::abs(grad_data[i] - expected) < tol, "expand_bug: wrong gradient");
	}
}
