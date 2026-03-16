//! Dropout Kernel
//!
//! Inverted dropout forward: generates a random U8 mask (0 or 1) and
//! applies y = x * mask / (1 - p).  Uses the C3 stdlib RNG (auto-seeded
//! from entropy on first call).

module metaphor::device::host_device::kernels::dropout;
import std::math::random;

/// Inverted dropout forward — generic over element type.
/// Writes a random U8 mask and the scaled output.
fn void dropout_fwd(Type* x, Type* y, char* mask, usz n, float p) <Type>
{
	Type scale = (Type)(1.0 / (1.0 - (double)p));
	for (usz i = 0; i < n; i++)
	{
		char keep = rnd() >= p ? 1 : 0;
		mask[i] = keep;
		y[i] = keep != 0 ? x[i] * scale : 0;
	}
}
