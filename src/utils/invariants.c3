//! Invariants
//!
//! Compile-time-gated structural assertions. These express properties that
//! must ALWAYS hold — violations are bugs, not edge cases.
//!
//! In safe mode or testing: compiled as asserts (panics with message).
//! In release: compiled out entirely (zero cost).
//!
//! Usage:
//!   invariants::@check(x > 0, "x must be positive");
//!   invariants::@shape_buffer_match(tc);
//!   invariants::@broadcast_compatible(a, b);
//!   invariants::@dims_match_output(kernel_elems, out_tc);

module metaphor::utils::invariants;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;

const bool ENABLED = env::COMPILER_SAFE_MODE || env::TESTING;

// General invariant

/// Compile-time-gated assert. Zero cost in release.
macro void @check(#cond, $msg)
{
	$if ENABLED:
		assert(#cond, $msg);
	$endif
}

// Tensor invariants

/// The tensor's buffer must be large enough for its shape.
/// byte_size >= num_elements * dtype_size.
macro void @shape_buffer_match(tensor_core::TensorCore* tc)
{
	$if ENABLED:
		ulong needed = tc.state.dense.num_elements * (ulong)tc.dtype().size_of();
		assert(tc.state.dense.byte_size >= needed,
			"invariant: output buffer too small for derived shape");
	$endif
}

/// Two dimension values sharing the same index must be broadcast-compatible:
/// equal, or one of them is 1.
macro void @broadcast_compatible(ulong a, ulong b)
{
	$if ENABLED:
		assert(a == b || a == 1 || b == 1,
			"invariant: incompatible broadcast dimensions");
	$endif
}

/// Kernel dims product must equal output tensor's num_elements.
/// If these disagree, the kernel iterates over the wrong element count.
macro void @dims_match_output(ulong kernel_elems, tensor_core::TensorCore* out_tc)
{
	$if ENABLED:
		assert(kernel_elems == out_tc.state.dense.num_elements,
			"invariant: kernel dims product != output num_elements");
	$endif
}
