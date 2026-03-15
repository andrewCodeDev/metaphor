module metaphor::testing::device::common::index_expr_test;
import metaphor::device::common::index_expr;

fn void test_parse_basic() @test
{
	IndexExpr expr = index_expr::index_expr_parse("ij,jk->ik")!!;
	assert(expr.num_inputs == 2);
	assert(expr.get_input(0).len == 2);
	assert(expr.get_input(0)[0] == 'i');
	assert(expr.get_input(0)[1] == 'j');
	assert(expr.get_input(1)[0] == 'j');
	assert(expr.get_input(1)[1] == 'k');
	assert(expr.get_output()[0] == 'i');
	assert(expr.get_output()[1] == 'k');
	assert(!expr.has_ellipsis);
}

fn void test_parse_reduce() @test
{
	IndexExpr expr = index_expr::index_expr_parse("ijk->ik")!!;
	assert(expr.num_inputs == 1);
	assert(expr.get_input(0).len == 3);
	assert(expr.get_output().len == 2);

	char[16] buf;
	char[] bound = expr.get_bound(&buf);
	assert(bound.len == 1);
	assert(bound[0] == 'j');
}

fn void test_parse_ellipsis() @test
{
	IndexExpr expr = index_expr::index_expr_parse("...ij,...jk->...ik")!!;
	assert(expr.has_ellipsis);
	assert(expr.get_input(0)[0] == 'i');
	assert(expr.get_input(0)[1] == 'j');
	assert(expr.get_input(1)[0] == 'j');
	assert(expr.get_input(1)[1] == 'k');
	assert(expr.get_output()[0] == 'i');
	assert(expr.get_output()[1] == 'k');
}

fn void test_bound_indices() @test
{
	IndexExpr expr = index_expr::index_expr_parse("ij,kj->ik")!!;
	char[16] buf;
	char[] bound = expr.get_bound(&buf);
	assert(bound.len == 1);
	assert(bound[0] == 'j');
}

fn void test_index_in_pattern() @test
{
	char[3] pattern = { 'i', 'j', 'k' };
	assert(index_expr::index_in_pattern('j', &pattern)!! == 1);
	assert(@catch(index_expr::index_in_pattern('z', &pattern)));
}

fn void test_parse_single_input() @test
{
	IndexExpr expr = index_expr::index_expr_parse("i->i")!!;
	assert(expr.num_inputs == 1);
	assert(expr.get_input(0).len == 1);
	assert(expr.get_output().len == 1);
}
