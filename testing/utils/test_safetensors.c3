module metaphor::testing::utils::test_safetensors;
import metaphor::utils::safetensors;

fn usz build_test_buffer(char[] buf, char[] header, usz data_size)
{
	usz header_len = header.len;
	usz padded_len = (header_len + 7) / 8 * 8;
	usz total = 8 + padded_len + data_size;
	assert(total <= buf.len);

	// Write header size as LE u64
	ulong hs = (ulong)padded_len;
	buf[0] = (char)(hs & 0xFF);
	buf[1] = (char)((hs >> 8) & 0xFF);
	buf[2] = (char)((hs >> 16) & 0xFF);
	buf[3] = (char)((hs >> 24) & 0xFF);
	buf[4] = 0; buf[5] = 0; buf[6] = 0; buf[7] = 0;

	// Copy header
	for (usz i = 0; i < header_len; i++) { buf[8 + i] = header[i]; }
	// Pad with spaces
	for (usz i = header_len; i < padded_len; i++) { buf[8 + i] = ' '; }
	// Zero tensor data
	for (usz i = 0; i < data_size; i++) { buf[8 + padded_len + i] = 0; }

	return total;
}

// ============================================================================
// Existing tests (regression)
// ============================================================================

fn void test_deserialize_single_tensor() @test
{
	char[256] buf;
	char[] header = `{"weights":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}`;
	usz total = build_test_buffer(&buf, header, 16);

	SafeTensorsFile sf = safetensors::deserialize(buf[:total])!!;
	defer sf.free();

	assert(sf.len() == 1, "expected 1 tensor");
	assert(sf.tensors[0].name == "weights", "expected name 'weights'");
	assert(sf.tensors[0].dtype == Dtype.F32, "expected dtype F32");
	assert(sf.tensors[0].shape.len == 2, "expected 2 shape dims");
	assert(sf.tensors[0].shape[0] == 2, "expected shape[0] == 2");
	assert(sf.tensors[0].shape[1] == 2, "expected shape[1] == 2");
}

fn void test_lookup_found() @test
{
	char[256] buf;
	char[] header = `{"weights":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}`;
	usz total = build_test_buffer(&buf, header, 16);

	SafeTensorsFile sf = safetensors::deserialize(buf[:total])!!;
	defer sf.free();

	TensorView view = sf.get("weights")!!;
	assert(view.info.name == "weights", "expected name 'weights'");
	assert(view.data.len == 16, "expected 16 bytes of data");
}

fn void test_lookup_not_found() @test
{
	char[256] buf;
	char[] header = `{"weights":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}`;
	usz total = build_test_buffer(&buf, header, 16);

	SafeTensorsFile sf = safetensors::deserialize(buf[:total])!!;
	defer sf.free();

	TensorView? result = sf.get("nonexistent");
	assert(@catch(result) == safetensors::ST_TENSOR_NOT_FOUND, "expected ST_TENSOR_NOT_FOUND fault");
}

fn void test_dtype_methods() @test
{
	// byte_size
	assert(Dtype.F32.byte_size() == 4, "F32 should be 4 bytes");
	assert(Dtype.F64.byte_size() == 8, "F64 should be 8 bytes");
	assert(Dtype.U8.byte_size() == 1, "U8 should be 1 byte");
	assert(Dtype.F16.byte_size() == 2, "F16 should be 2 bytes");

	// to_string
	assert(Dtype.F32.to_string() == "F32", "F32 string mismatch");
	assert(Dtype.BF16.to_string() == "BF16", "BF16 string mismatch");

	// from_string round-trip
	Dtype d = safetensors::from_string("F32")!!;
	assert(d == Dtype.F32, "from_string F32 mismatch");
	d = safetensors::from_string("BF16")!!;
	assert(d == Dtype.BF16, "from_string BF16 mismatch");
	d = safetensors::from_string("U64")!!;
	assert(d == Dtype.U64, "from_string U64 mismatch");

	// unsupported
	Dtype? bad = safetensors::from_string("INVALID");
	assert(@catch(bad) == safetensors::ST_UNSUPPORTED_DTYPE, "expected ST_UNSUPPORTED_DTYPE fault");
}

// ============================================================================
// Data view length test (validates off-by-one fix)
// ============================================================================

fn void test_data_view_exact_length() @test
{
	char[512] buf;
	// 3 floats = 12 bytes
	char[] header = `{"vec":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}`;
	usz total = build_test_buffer(&buf, header, 12);

	// Write known data pattern: bytes 0xAA repeated
	usz padded_hdr = (header.len + 7) / 8 * 8;
	// We wrote padded_len not header.len to the LE header, so recalculate:
	usz data_start = 8 + padded_hdr;
	for (usz i = 0; i < 12; i++) { buf[data_start + i] = (char)0xAA; }

	SafeTensorsFile sf = safetensors::deserialize(buf[:total])!!;
	defer sf.free();

	TensorView view = sf.get("vec")!!;
	assert(view.data.len == 12, "data view should be exactly 12 bytes");

	// Verify every byte
	for (usz i = 0; i < 12; i++)
	{
		assert(view.data[i] == (char)0xAA, "data byte mismatch");
	}
}

// ============================================================================
// Metadata skip test
// ============================================================================

fn void test_metadata_skip() @test
{
	char[512] buf;
	char[] header = `{"__metadata__":{"format":"pt"},"bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`;
	usz total = build_test_buffer(&buf, header, 16);

	SafeTensorsFile sf = safetensors::deserialize(buf[:total])!!;
	defer sf.free();

	assert(sf.len() == 1, "expected 1 tensor after skipping metadata");
	assert(sf.tensors[0].name == "bias", "expected tensor name 'bias'");
}

// ============================================================================
// Error handling tests
// ============================================================================

fn void test_error_truncated_buffer() @test
{
	char[4] tiny_buf;
	SafeTensorsFile? result = safetensors::deserialize(tiny_buf[:4]);
	assert(@catch(result) == safetensors::ST_BUFFER_TOO_SMALL, "expected ST_BUFFER_TOO_SMALL for 4-byte buffer");
}

fn void test_error_empty_buffer() @test
{
	char[] empty;
	SafeTensorsFile? result = safetensors::deserialize(empty);
	assert(@catch(result) == safetensors::ST_BUFFER_TOO_SMALL, "expected ST_BUFFER_TOO_SMALL for empty buffer");
}

// ============================================================================
// Multi-tensor deserialization test
// ============================================================================

fn void test_deserialize_multi_tensor() @test
{
	char[512] buf;
	// Two tensors: a (2 floats = 8 bytes), b (1 double = 8 bytes)
	char[] header = `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F64","shape":[1],"data_offsets":[8,16]}}`;
	usz total = build_test_buffer(&buf, header, 16);

	SafeTensorsFile sf = safetensors::deserialize(buf[:total])!!;
	defer sf.free();

	assert(sf.len() == 2, "expected 2 tensors");

	// Find tensor "a"
	TensorView va = sf.get("a")!!;
	assert(va.info.dtype == Dtype.F32, "tensor 'a' should be F32");
	assert(va.data.len == 8, "tensor 'a' data should be 8 bytes");

	// Find tensor "b"
	TensorView vb = sf.get("b")!!;
	assert(vb.info.dtype == Dtype.F64, "tensor 'b' should be F64");
	assert(vb.data.len == 8, "tensor 'b' data should be 8 bytes");
}

// ============================================================================
// Serialization round-trip tests
// ============================================================================

fn void test_round_trip_single() @test
{
	// Create a single F32 tensor with known data
	char[16] data;
	for (usz i = 0; i < 16; i++) { data[i] = (char)(i + 1); }
	ulong[2] shape = { 2, 2 };

	TensorDescriptor[1] descs = {
		{
			.name = "weights",
			.dtype = Dtype.F32,
			.shape = &shape,
			.data = &data,
		}
	};

	char[] buf = safetensors::serialize_tensors(&descs)!!;
	defer safetensors::free_serialized(buf);

	// Deserialize and verify
	SafeTensorsFile sf = safetensors::deserialize(buf)!!;
	defer sf.free();

	assert(sf.len() == 1, "round-trip: expected 1 tensor");
	assert(sf.tensors[0].name == "weights", "round-trip: name mismatch");
	assert(sf.tensors[0].dtype == Dtype.F32, "round-trip: dtype mismatch");
	assert(sf.tensors[0].shape.len == 2, "round-trip: shape dims mismatch");
	assert(sf.tensors[0].shape[0] == 2, "round-trip: shape[0] mismatch");
	assert(sf.tensors[0].shape[1] == 2, "round-trip: shape[1] mismatch");

	TensorView view = sf.get("weights")!!;
	assert(view.data.len == 16, "round-trip: data length mismatch");

	for (usz i = 0; i < 16; i++)
	{
		assert(view.data[i] == (char)(i + 1), "round-trip: data byte mismatch");
	}
}

fn void test_round_trip_multi() @test
{
	// Tensor A: 2 x F32 = 8 bytes
	char[8] data_a;
	for (usz i = 0; i < 8; i++) { data_a[i] = (char)(0x10 + i); }

	// Tensor B: 4 x U8 = 4 bytes
	char[4] data_b;
	for (usz i = 0; i < 4; i++) { data_b[i] = (char)(0x20 + i); }

	// Tensor C: 1 x F64 = 8 bytes
	char[8] data_c;
	for (usz i = 0; i < 8; i++) { data_c[i] = (char)(0x30 + i); }

	TensorDescriptor[3] descs = {
		{
			.name = "alpha",
			.dtype = Dtype.F32,
			.shape = { 2 },
			.data = &data_a,
		},
		{
			.name = "beta",
			.dtype = Dtype.U8,
			.shape = { 4 },
			.data = &data_b,
		},
		{
			.name = "gamma",
			.dtype = Dtype.F64,
			.shape = { 1 },
			.data = &data_c,
		}
	};

	char[] buf = safetensors::serialize_tensors(&descs)!!;
	defer safetensors::free_serialized(buf);

	SafeTensorsFile sf = safetensors::deserialize(buf)!!;
	defer sf.free();

	assert(sf.len() == 3, "multi round-trip: expected 3 tensors");

	// Verify tensor alpha
	TensorView va = sf.get("alpha")!!;
	assert(va.info.dtype == Dtype.F32, "alpha: dtype mismatch");
	assert(va.data.len == 8, "alpha: data length mismatch");
	for (usz i = 0; i < 8; i++)
	{
		assert(va.data[i] == (char)(0x10 + i), "alpha: data byte mismatch");
	}

	// Verify tensor beta
	TensorView vb = sf.get("beta")!!;
	assert(vb.info.dtype == Dtype.U8, "beta: dtype mismatch");
	assert(vb.data.len == 4, "beta: data length mismatch");
	for (usz i = 0; i < 4; i++)
	{
		assert(vb.data[i] == (char)(0x20 + i), "beta: data byte mismatch");
	}

	// Verify tensor gamma
	TensorView vc = sf.get("gamma")!!;
	assert(vc.info.dtype == Dtype.F64, "gamma: dtype mismatch");
	assert(vc.data.len == 8, "gamma: data length mismatch");
	for (usz i = 0; i < 8; i++)
	{
		assert(vc.data[i] == (char)(0x30 + i), "gamma: data byte mismatch");
	}
}

fn void test_round_trip_empty_shape() @test
{
	// Scalar tensor: empty shape, 4 bytes of F32 data
	char[4] data;
	data[0] = 0x00; data[1] = 0x00; data[2] = (char)0x80; data[3] = 0x3F; // 1.0f in LE

	TensorDescriptor[1] descs = {
		{
			.name = "scalar",
			.dtype = Dtype.F32,
			.shape = {},
			.data = &data,
		}
	};

	char[] buf = safetensors::serialize_tensors(&descs)!!;
	defer safetensors::free_serialized(buf);

	SafeTensorsFile sf = safetensors::deserialize(buf)!!;
	defer sf.free();

	assert(sf.len() == 1, "scalar: expected 1 tensor");
	assert(sf.tensors[0].shape.len == 0, "scalar: expected empty shape");

	TensorView view = sf.get("scalar")!!;
	assert(view.data.len == 4, "scalar: data length mismatch");
	assert(view.data[2] == (char)0x80, "scalar: data byte mismatch");
}

fn void test_round_trip_data_integrity() @test
{
	// Create a 4-element F32 tensor with specific float values
	// Using known IEEE 754 representations
	char[16] data;
	// 1.0f = 0x3F800000 LE: 00 00 80 3F
	data[0] = 0x00; data[1] = 0x00; data[2] = (char)0x80; data[3] = 0x3F;
	// 2.0f = 0x40000000 LE: 00 00 00 40
	data[4] = 0x00; data[5] = 0x00; data[6] = 0x00; data[7] = 0x40;
	// -1.0f = 0xBF800000 LE: 00 00 80 BF
	data[8] = 0x00; data[9] = 0x00; data[10] = (char)0x80; data[11] = (char)0xBF;
	// 0.5f = 0x3F000000 LE: 00 00 00 3F
	data[12] = 0x00; data[13] = 0x00; data[14] = 0x00; data[15] = 0x3F;

	TensorDescriptor[1] descs = {
		{
			.name = "values",
			.dtype = Dtype.F32,
			.shape = { 4 },
			.data = &data,
		}
	};

	char[] buf = safetensors::serialize_tensors(&descs)!!;
	defer safetensors::free_serialized(buf);

	SafeTensorsFile sf = safetensors::deserialize(buf)!!;
	defer sf.free();

	TensorView view = sf.get("values")!!;
	assert(view.data.len == 16, "integrity: expected 16 bytes");

	// Byte-exact comparison
	for (usz i = 0; i < 16; i++)
	{
		assert(view.data[i] == data[i], "integrity: byte mismatch");
	}
}
