module metaphor::testing::utils::test_idx;
import metaphor::utils::idx;

fn void test_parse_ubyte_matrix() @test
{
	// Synthetic 2×3 ubyte matrix
	// Magic: 0x00000802 (ubyte, 2 dims), Dims: 2, 3, Data: 1..6
	char[18] test_data = {
		0x00, 0x00, 0x08, 0x02, // magic
		0x00, 0x00, 0x00, 0x02, // dim 0 = 2
		0x00, 0x00, 0x00, 0x03, // dim 1 = 3
		1, 2, 3, 4, 5, 6,       // data
	};

	IdxData result = idx::parse(&test_data)!!;
	defer result.free();

	assert(result.dims.len == 2);
	assert(result.dims[0] == 2);
	assert(result.dims[1] == 3);
	assert(result.data.len == 6);

	// ubyte 1 → 1/255.0
	float expected_first = 1.0 / 255.0;
	float expected_last  = 6.0 / 255.0;
	float eps = 0.0001;
	assert(result.data[0] - expected_first < eps && expected_first - result.data[0] < eps);
	assert(result.data[5] - expected_last < eps && expected_last - result.data[5] < eps);
}

fn void test_accessors() @test
{
	char[18] test_data = {
		0x00, 0x00, 0x08, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x03,
		1, 2, 3, 4, 5, 6,
	};

	IdxData result = idx::parse(&test_data)!!;
	defer result.free();

	assert(result.count() == 2);
	assert(result.sample_size() == 3);

	float[] sample0 = result.get(0);
	assert(sample0.len == 3);
	float[] sample1 = result.get(1);
	assert(sample1.len == 3);

	// sample1[0] should be data[3] = 4/255.0
	float expected = 4.0 / 255.0;
	float eps = 0.0001;
	float diff = sample1[0] - expected;
	assert(diff < eps && diff > -eps);
}

fn void test_reject_short_buffer() @test
{
	char[3] short_buf = { 0x00, 0x00, 0x08 };
	IdxData? result = idx::parse(&short_buf);
	assert(@catch(result) == idx::IDX_INVALID_FORMAT);
}
