module metaphor::testing::tensor::shape_test;
import metaphor::tensor::shape;

// ============================================================================
// Shape initialization tests
// ============================================================================

fn void test_shape_init() @test
{
	ulong[3] dims = { 3, 4, 5 };
	Shape s = shape::shape_init(&dims);
	assert(s.len == 3);
	assert(s.get(0) == 3);
	assert(s.get(1) == 4);
	assert(s.get(2) == 5);
}

fn void test_shape_size() @test
{
	ulong[3] dims = { 3, 4, 5 };
	Shape s = shape::shape_init(&dims);
	assert(s.size() == 60);

	// Empty shape (scalar)
	Shape scalar = shape::shape_init({});
	assert(scalar.size() == 1, "scalar shape has size 1 (all 1s product)");
}

fn void test_shape_empty_shape() @test
{
	Shape empty = shape::EMPTY_SHAPE;
	assert(empty.len == 0);
	assert(empty.size() == 1, "empty shape is all 1s -> product=1");
	assert(empty.buffer[0] == 1);
	assert(empty.buffer[7] == 1);
}

// ============================================================================
// Stride computation
// ============================================================================

fn void test_strides_contiguous() @test
{
	// Shape (3, 4, 5) -> strides (20, 5, 1)
	ulong[3] dims = { 3, 4, 5 };
	Shape s = shape::shape_init(&dims);
	Strides st = s.strides();

	assert(st.len == 3);
	assert(st.get(0) == 20, "stride[0] = 4*5 = 20");
	assert(st.get(1) == 5, "stride[1] = 5");
	assert(st.get(2) == 1, "stride[2] = 1");
}

fn void test_strides_scalar() @test
{
	// Scalar -> stride 0
	Shape s = shape::shape_init({});
	Strides st = s.strides();
	assert(st.buffer[0] == 0);
}

fn void test_strides_1d() @test
{
	ulong[1] dims = { 10 };
	Shape s = shape::shape_init(&dims);
	Strides st = s.strides();
	assert(st.len == 1);
	assert(st.get(0) == 1);
}

fn void test_strides_pos_to_offset() @test
{
	// Shape (3, 4, 5), strides (20, 5, 1)
	// Position (1, 2, 3) -> offset 1*20 + 2*5 + 3*1 = 33
	ulong[3] dims = { 3, 4, 5 };
	Shape s = shape::shape_init(&dims);
	Strides st = s.strides();

	ulong[3] pos = { 1, 2, 3 };
	ulong offset = st.pos_to_offset(&pos);
	assert(offset == 33, "1*20 + 2*5 + 3*1 = 33");
}

fn void test_strides_offset_to_pos() @test
{
	ulong[3] dims = { 3, 4, 5 };
	Shape s = shape::shape_init(&dims);
	Strides st = s.strides();

	Indices pos = st.offset_to_pos(33);
	assert(pos.get(0) == 1);
	assert(pos.get(1) == 2);
	assert(pos.get(2) == 3);
}

// ============================================================================
// Shape operations
// ============================================================================

fn void test_shape_equal() @test
{
	ulong[3] dims = { 3, 4, 5 };
	Shape a = shape::shape_init(&dims);
	Shape b = shape::shape_init(&dims);
	assert(a.equal(&b));

	Shape c = shape::shape_init({ 3, 4, 6 });
	assert(!a.equal(&c));
}

fn void test_shape_broadcast() @test
{
	ulong[4] d1 = { 5, 3, 4, 2 };
	ulong[2] d2 = { 4, 2 };
	Shape a = shape::shape_init(&d1);
	Shape b = shape::shape_init(&d2);
	Shape result;

	assert(a.broadcast(&b, &result));
	assert(result.equal(&a), "broadcasting (5,3,4,2) and (4,2) -> (5,3,4,2)");

	// Non-broadcastable
	ulong[2] d3 = { 4, 3 };
	Shape c = shape::shape_init(&d3);
	assert(!a.broadcast(&c, &result), "(5,3,4,2) and (4,3) not broadcastable");
}

fn void test_shape_squeeze() @test
{
	ulong[4] d = { 1, 3, 1, 4 };
	Shape s = shape::shape_init(&d);
	Shape squeezed = s.squeeze();
	assert(squeezed.len == 2);
	assert(squeezed.get(0) == 3);
	assert(squeezed.get(1) == 4);
}

fn void test_shape_unsqueeze() @test
{
	ulong[2] d = { 3, 4 };
	Shape s = shape::shape_init(&d);
	Shape unsqueezed = s.unsqueeze();
	assert(unsqueezed.len == 3);
	assert(unsqueezed.get(0) == 1);
	assert(unsqueezed.get(1) == 3);
	assert(unsqueezed.get(2) == 4);
}

fn void test_shape_remove() @test
{
	ulong[4] d = { 5, 4, 3, 2 };
	Shape s = shape::shape_init(&d);
	s.remove(3);
	assert(s.size() == 60);
	s.remove(0);
	assert(s.size() == 12);
	assert(s.get(0) == 4);
	s.remove(1);
	assert(s.size() == 4);
	s.remove(0);
	assert(s.size() == 1);
	assert(s.len == 0);
}

fn void test_shape_append() @test
{
	ulong[1] d = { 3 };
	Shape s = shape::shape_init(&d);
	s.append(4);
	assert(s.len == 2);
	assert(s.get(0) == 3);
	assert(s.get(1) == 4);
}

fn void test_shape_compatible() @test
{
	ulong[4] d1 = { 5, 3, 4, 2 };
	ulong[2] d2 = { 4, 2 };
	Shape a = shape::shape_init(&d1);
	Shape b = shape::shape_init(&d2);
	assert(a.compatible(b));

	ulong[2] d3 = { 4, 3 };
	Shape c = shape::shape_init(&d3);
	assert(!a.compatible(c));
}

fn void test_shape_mismatch() @test
{
	// (5,5,5) vs (1,5,5,5) -> no mismatch (same after alignment)
	Shape a = shape::shape_init({ 5, 5, 5 });
	Shape b = shape::shape_init({ 1, 5, 5, 5 });
	MismatchResult mm;
	assert(!a.mismatch(b, &mm), "same tensors after alignment");

	// (5,5,5,6) vs (5,5,5) -> mismatch at a_pos=3, b_pos=2
	Shape c = shape::shape_init({ 5, 5, 5, 6 });
	Shape d = shape::shape_init({ 5, 5, 5 });
	assert(c.mismatch(d, &mm));
	assert(mm.a_pos == 3);
	assert(mm.b_pos == 2);
}
