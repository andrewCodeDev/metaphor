//! IDX File Format Reader
//!
//! Reads the IDX binary format (used by MNIST datasets).
//! Specializes for float output — ubytes are normalized to 0–1 range.
//!
//! IDX format:
//!   - 4 bytes: magic number (0x0000 + dtype_byte + ndims) big-endian
//!   - 4 bytes per dimension: size (big-endian)
//!   - raw data bytes
//!

module metaphor::utils::idx;
import std::core::bitorder;
import std::io;
import std::core::mem;

faultdef IDX_INVALID_FORMAT;
faultdef IDX_INVALID_DIMENSIONS;
faultdef IDX_INSUFFICIENT_DATA;
faultdef IDX_UNSUPPORTED_DTYPE;

struct IdxData
{
	uint[] dims;
	float[] data;
}

fn void IdxData.free(&self)
{
	if (self.dims.len > 0) mem::free(self.dims.ptr);
	if (self.data.len > 0) mem::free(self.data.ptr);
}

fn usz IdxData.count(&self)
{
	return self.dims.len > 0 ? (usz)self.dims[0] : 0;
}

fn usz IdxData.sample_size(&self)
{
	usz size = 1;
	for (usz i = 1; i < self.dims.len; i++)
	{
		size *= (usz)self.dims[i];
	}
	return size;
}

fn float[] IdxData.get(&self, usz index)
{
	usz size = self.sample_size();
	usz start = index * size;
	usz end = start + size - 1;
	return self.data[start..end];
}

fn IdxData? parse(char[] bytes)
{
	if (bytes.len < 4) return IDX_INVALID_FORMAT~;

	uint magic = bitorder::read(bytes[0:4], bitorder::UIntBE);
	char dtype_byte = (char)((magic >> 8) & 0xFF);
	char ndims = (char)(magic & 0xFF);

	if (ndims == 0 || ndims > 8) return IDX_INVALID_DIMENSIONS~;
	if (bytes.len < 4 + (usz)ndims * 4) return IDX_INVALID_FORMAT~;

	// Allocate dims
	uint* dims_ptr = (uint*)mem::malloc((usz)ndims * uint.sizeof);
	assert(dims_ptr != null, "idx parse: dims allocation failed");
	uint[] dims = dims_ptr[:(usz)ndims];

	usz total_elements = 1;
	for (usz i = 0; i < (usz)ndims; i++)
	{
		usz offset = 4 + i * 4;
		dims[i] = bitorder::read(bytes[offset:4], bitorder::UIntBE);
		total_elements *= (usz)dims[i];
	}

	usz header_size = 4 + (usz)ndims * 4;
	char[] data_bytes = bytes[header_size..];

	// Allocate output data
	float* data_ptr = (float*)mem::malloc(total_elements * float.sizeof);
	assert(data_ptr != null, "idx parse: data allocation failed");
	float[] data = data_ptr[:total_elements];

	switch (dtype_byte)
	{
		case 0x08: // unsigned byte → float (normalized 0–1)
			if (data_bytes.len < total_elements) { mem::free(data_ptr); mem::free(dims_ptr); return IDX_INSUFFICIENT_DATA~; }
			for (usz i = 0; i < total_elements; i++)
			{
				data[i] = (float)data_bytes[i] / 255.0;
			}
		case 0x09: // signed byte → float
			if (data_bytes.len < total_elements) { mem::free(data_ptr); mem::free(dims_ptr); return IDX_INSUFFICIENT_DATA~; }
			for (usz i = 0; i < total_elements; i++)
			{
				ichar val = (ichar)data_bytes[i];
				data[i] = (float)val;
			}
		case 0x0B: // i16 big-endian → float
			if (data_bytes.len < total_elements * 2) { mem::free(data_ptr); mem::free(dims_ptr); return IDX_INSUFFICIENT_DATA~; }
			for (usz i = 0; i < total_elements; i++)
			{
				usz off = header_size + i * 2;
				short val = bitorder::read(bytes[off:2], bitorder::ShortBE);
				data[i] = (float)val;
			}
		case 0x0C: // i32 big-endian → float
			if (data_bytes.len < total_elements * 4) { mem::free(data_ptr); mem::free(dims_ptr); return IDX_INSUFFICIENT_DATA~; }
			for (usz i = 0; i < total_elements; i++)
			{
				usz off = header_size + i * 4;
				int val = bitorder::read(bytes[off:4], bitorder::IntBE);
				data[i] = (float)val;
			}
		case 0x0D: // f32 big-endian → float
			if (data_bytes.len < total_elements * 4) { mem::free(data_ptr); mem::free(dims_ptr); return IDX_INSUFFICIENT_DATA~; }
			for (usz i = 0; i < total_elements; i++)
			{
				usz off = header_size + i * 4;
				uint bits = bitorder::read(bytes[off:4], bitorder::UIntBE);
				data[i] = bitcast(bits, float);
			}
		case 0x0E: // f64 big-endian → float (truncate)
			if (data_bytes.len < total_elements * 8) { mem::free(data_ptr); mem::free(dims_ptr); return IDX_INSUFFICIENT_DATA~; }
			for (usz i = 0; i < total_elements; i++)
			{
				usz off = header_size + i * 8;
				ulong bits = bitorder::read(bytes[off:8], bitorder::ULongBE);
				double val = bitcast(bits, double);
				data[i] = (float)val;
			}
		default:
			mem::free(data_ptr);
			mem::free(dims_ptr);
			return IDX_UNSUPPORTED_DTYPE~;
	}

	return { .dims = dims, .data = data };
}

/// Read an IDX file from disk and parse it.
/// The path must be null-terminated (ZString).
fn IdxData? read_file(String path)
{
	char[] buf = io::file::load(mem, path)!;
	defer mem::free(buf.ptr);
	return parse(buf);
}
