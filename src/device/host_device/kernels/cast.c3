//! Cast Kernel
//!
//! Converts tensor data between dtypes. Element-wise copy with type conversion.
//! This is a boundary operation that cannot be fused with other kernels.
//!

module metaphor::device::host_device::kernels::cast;
import metaphor::rtti;

// Public API

/// Cast `len` elements from `src` (of `src_dtype`) to `dst` (of `dst_dtype`).
/// Both pointers must be valid for at least `len * dtype.size_of()` bytes.
fn void cast_elements(void* src, void* dst, usz len, rtti::Datatype src_dtype, rtti::Datatype dst_dtype)
{
	if (len == 0) { return; }
	if (src_dtype == dst_dtype)
	{
		// Same type: just memcpy
		mem::copy(dst, src, len * src_dtype.size_of());
		return;
	}

	switch (src_dtype)
	{
		case F16:  cast_from_f16(src, dst, len, dst_dtype);
		case BF16: unreachable("bf16 cast not yet implemented");
		case F32:  cast_from_float(src, dst, len, dst_dtype);
		case F64:  cast_from_double(src, dst, len, dst_dtype);
		case U8:   cast_from_u8(src, dst, len, dst_dtype);
		case U64:  cast_from_u64(src, dst, len, dst_dtype);
		case rtti::Datatype.BOOL: cast_from_u8(src, dst, len, dst_dtype); // bool stored as u8
	}
}

// Typed cast dispatch — from a known source type

fn void cast_from_float(void* src, void* dst, usz len, rtti::Datatype dst_dtype)
{
	float* s = (float*)src;
	switch (dst_dtype)
	{
		case F64:
			double* d = (double*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (double)s[i]; }
		case F16:
			// f16 not natively supported in C3; store as-is for now
			// TODO: implement float->f16 conversion
			unreachable("float -> f16 cast not yet implemented");
		case U8:
			char* d = (char*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (char)(int)s[i]; }
		case U64:
			ulong* d = (ulong*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (ulong)(long)s[i]; }
		default:
			unreachable("unsupported cast destination dtype");
	}
}

fn void cast_from_double(void* src, void* dst, usz len, rtti::Datatype dst_dtype)
{
	double* s = (double*)src;
	switch (dst_dtype)
	{
		case F32:
			float* d = (float*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (float)s[i]; }
		case F16:
			unreachable("double -> f16 cast not yet implemented");
		case U8:
			char* d = (char*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (char)(int)s[i]; }
		case U64:
			ulong* d = (ulong*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (ulong)(long)s[i]; }
		default:
			unreachable("unsupported cast destination dtype");
	}
}

fn void cast_from_f16(void* src, void* dst, usz len, rtti::Datatype dst_dtype)
{
	// f16 not natively supported; stub for now
	unreachable("f16 source cast not yet implemented");
}

fn void cast_from_u8(void* src, void* dst, usz len, rtti::Datatype dst_dtype)
{
	char* s = (char*)src;
	switch (dst_dtype)
	{
		case F32:
			float* d = (float*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (float)s[i]; }
		case F64:
			double* d = (double*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (double)s[i]; }
		case U64:
			ulong* d = (ulong*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (ulong)s[i]; }
		default:
			unreachable("unsupported cast destination dtype");
	}
}

fn void cast_from_u64(void* src, void* dst, usz len, rtti::Datatype dst_dtype)
{
	ulong* s = (ulong*)src;
	switch (dst_dtype)
	{
		case F32:
			float* d = (float*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (float)s[i]; }
		case F64:
			double* d = (double*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (double)s[i]; }
		case U8:
			char* d = (char*)dst;
			for (usz i = 0; i < len; i++) { d[i] = (char)s[i]; }
		default:
			unreachable("unsupported cast destination dtype");
	}
}
