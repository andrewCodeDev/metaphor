//! Tensor Display
//!
//! Numpy-style pretty-printing for tensors, supporting up to 4 dimensions.
//! Handles f32, f64, and u64 dtypes. For device (GPU) tensors, data is
//! copied to host before printing using the global allocator.
//!

module metaphor::tensor::display;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::rtti;
import metaphor::device::device_reference;
import std::io;

// Public API

/// Print tensor in numpy-style format to stderr.
/// Shows header (label, dtype, shape) followed by formatted data.
/// For device tensors, allocates a temporary host buffer and copies data over.
fn void print(tensor_core::TensorCore* c)
{
	// Print header
	ZString label = c.label_or_null();
	ZString display_label = label != null ? label : (ZString)"(unlabeled)";
	io::eprintf("Tensor(\"%s\", %s, shape=[", display_label, dtype_name(c.dtype()));
	ulong[] sh = c.shape().slice();
	for (usz i = 0; i < sh.len; i++)
	{
		if (i > 0) io::eprintf(", ");
		io::eprintf("%d", sh[i]);
	}
	io::eprintfn("])");

	if (!c.has_data())
	{
		io::eprintfn("  (no data)");
		return;
	}

	usz ndim = sh.len;
	if (ndim > 4)
	{
		io::eprintfn("  (display not supported for rank > 4)");
		return;
	}

	// Get source data pointer
	void* src_data = c.state.dense.data;
	if (src_data == null) { io::eprintfn("  (null data)"); return; }

	rtti::Datatype dt = c.dtype();
	usz num_elements = (usz)c.shape().size();
	usz byte_size = num_elements * dt.size_of();

	// If tensor is on a device (GPU), copy to host first
	void* print_data;
	bool did_alloc = false;

	if (!c.device.is_host())
	{
		// Allocate host buffer and transfer
		char* host_buf = (char*)std::core::mem::malloc(byte_size);
		if (host_buf == null) { io::eprintfn("  (alloc failed)"); return; }
		char[] src_slice = ((char*)src_data)[:byte_size];
		char[] dst_slice = host_buf[:byte_size];
		c.device.mem_transfer(src_slice, dst_slice, device_reference::TransferDirection.DTOH);
		c.device.sync();
		print_data = (void*)host_buf;
		did_alloc = true;
	}
	else
	{
		print_data = src_data;
	}

	// Print data
	if (ndim == 0)
	{
		print_scalar_value(print_data, dt);
	}
	else
	{
		switch (ndim)
		{
			case 1: print_1d(print_data, (usz)sh[0], dt);
			case 2: print_2d(print_data, sh, dt);
			case 3: print_3d(print_data, sh, dt);
			case 4: print_4d(print_data, sh, dt);
			default:
		}
	}
	io::eprintfn("");

	// Free temporary host buffer if we allocated one
	if (did_alloc)
	{
		std::core::mem::free(print_data);
	}
}

// Internal

fn void print_scalar_value(void* data, rtti::Datatype dt)
{
	switch (dt)
	{
		case F32: print_float_value(((float*)data)[0]);
		case F64: print_float_value((float)((double*)data)[0]);
		case U64: io::eprintf("%d", ((ulong*)data)[0]);
		default: io::eprintf("?");
	}
}

fn void print_float_value(float v)
{
	if (v < 0)
	{
		io::eprintf("%.4f", v);
	}
	else
	{
		io::eprintf(" %.4f", v);
	}
}

fn void print_element(void* data, usz idx, rtti::Datatype dt)
{
	switch (dt)
	{
		case F32: print_float_value(((float*)data)[idx]);
		case F64: print_float_value((float)((double*)data)[idx]);
		case BF16: print_float_value((float)((bfloat*)data)[idx]);
		case F16: print_float_value((float)((float16*)data)[idx]);
		case U64: io::eprintf("%d", ((ulong*)data)[idx]);
		default: unreachable("print_element: unsupported dtype");
	}
}

fn void print_1d(void* data, usz size, rtti::Datatype dt)
{
	io::eprintf("[");
	for (usz i = 0; i < size; i++)
	{
		if (i > 0) io::eprintf(", ");
		print_element(data, i, dt);
	}
	io::eprintf("]");
}

fn void print_2d(void* data, ulong[] sh, rtti::Datatype dt)
{
	usz rows = (usz)sh[0];
	usz cols = (usz)sh[1];

	io::eprintf("[");
	for (usz i = 0; i < rows; i++)
	{
		if (i > 0) io::eprintf(",\n ");
		io::eprintf("[");
		for (usz j = 0; j < cols; j++)
		{
			if (j > 0) io::eprintf(", ");
			print_element(data, i * cols + j, dt);
		}
		io::eprintf("]");
	}
	io::eprintf("]");
}

fn void print_3d(void* data, ulong[] sh, rtti::Datatype dt)
{
	usz d0 = (usz)sh[0];
	usz d1 = (usz)sh[1];
	usz d2 = (usz)sh[2];
	usz stride0 = d1 * d2;

	io::eprintf("[");
	for (usz i = 0; i < d0; i++)
	{
		if (i > 0) io::eprintf(",\n\n ");
		io::eprintf("[");
		for (usz j = 0; j < d1; j++)
		{
			if (j > 0) io::eprintf(",\n  ");
			io::eprintf("[");
			for (usz k = 0; k < d2; k++)
			{
				if (k > 0) io::eprintf(", ");
				print_element(data, i * stride0 + j * d2 + k, dt);
			}
			io::eprintf("]");
		}
		io::eprintf("]");
	}
	io::eprintf("]");
}

fn void print_4d(void* data, ulong[] sh, rtti::Datatype dt)
{
	usz d0 = (usz)sh[0];
	usz d1 = (usz)sh[1];
	usz d2 = (usz)sh[2];
	usz d3 = (usz)sh[3];
	usz stride0 = d1 * d2 * d3;
	usz stride1 = d2 * d3;

	io::eprintf("[");
	for (usz i = 0; i < d0; i++)
	{
		if (i > 0) io::eprintf(",\n\n\n ");
		io::eprintf("[");
		for (usz j = 0; j < d1; j++)
		{
			if (j > 0) io::eprintf(",\n\n  ");
			io::eprintf("[");
			for (usz k = 0; k < d2; k++)
			{
				if (k > 0) io::eprintf(",\n   ");
				io::eprintf("[");
				for (usz m = 0; m < d3; m++)
				{
					if (m > 0) io::eprintf(", ");
					print_element(data, i * stride0 + j * stride1 + k * d3 + m, dt);
				}
				io::eprintf("]");
			}
			io::eprintf("]");
		}
		io::eprintf("]");
	}
	io::eprintf("]");
}

fn char[] dtype_name(rtti::Datatype dt)
{
	switch (dt)
	{
		case F32: return "f32";
		case F64: return "f64";
		case BF16: return "bf16";
		case F16: return "f16";
		case U64: return "u64";
		case U8:  return "u8";
		case rtti::Datatype.BOOL: return "bool";
		default: unreachable("dtype_name: unsupported dtype");
	}
}
