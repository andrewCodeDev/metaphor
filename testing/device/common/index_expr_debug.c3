module metaphor::testing::device::common::index_expr_debug;
import metaphor::device::common::index_expr;

fn void test_string_length() @test
{
	char[] s = "ij,jk->ik";
	assert(s.len == 9, "expected length 9");

	// Find arrow
	usz arrow_pos = 0;
	bool found = false;
	for (usz i = 0; i + 1 < s.len; i++)
	{
		if (s[i] == '-' && s[i+1] == '>')
		{
			arrow_pos = i;
			found = true;
			break;
		}
	}
	assert(found, "should find arrow");
	assert(arrow_pos == 5, "arrow should be at 5");

	// C3 slicing is inclusive: s[a..b] = elements a through b inclusive
	char[] inputs_part = s[0..arrow_pos - 1];
	assert(inputs_part.len == 5, "inputs_part should be 5");
	assert(inputs_part[0] == 'i');
	assert(inputs_part[4] == 'k');

	char[] output_part = s[arrow_pos + 2..];
	assert(output_part.len == 2, "output_part should be 2");
	assert(output_part[0] == 'i');
	assert(output_part[1] == 'k');
}

fn void test_slice_inclusive() @test
{
	char[5] arr = { 'a', 'b', 'c', 'd', 'e' };
	char[] sl = arr[1..3];
	// If inclusive: b, c, d = 3 elements
	// If exclusive: b, c = 2 elements
	assert(sl.len == 3, "C3 slices are inclusive");
}
