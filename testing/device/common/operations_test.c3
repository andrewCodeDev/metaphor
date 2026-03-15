module metaphor::testing::device::common::operations_test;
import metaphor::device::common::operations;
import metaphor::device::common::op_payload;
import metaphor::device::common::index_expr;

fn void test_binary_op_classification() @test
{
	BinaryOp add = BinaryOp.ADD;
	BinaryOp eql = BinaryOp.EQL;
	BinaryOp and_op = BinaryOp.AND;

	assert(!add.is_comparison());
	assert(!add.is_logical());

	assert(eql.is_comparison());
	assert(!eql.is_logical());

	assert(!and_op.is_comparison());
	assert(and_op.is_logical());
}

fn void test_unary_op_backward_needs() @test
{
	// Input-only backward
	assert(UnaryOp.NEG.backward_needs_input_only());
	assert(UnaryOp.RELU.backward_needs_input_only());
	assert(UnaryOp.EXP.backward_needs_input_only());

	// Needs output backward
	assert(!UnaryOp.SQRT.backward_needs_input_only());
	assert(!UnaryOp.TANH.backward_needs_input_only());
	assert(!UnaryOp.SIGMOID.backward_needs_input_only());

	// Not differentiable
	assert(!UnaryOp.SIGN.backward_needs_input_only());
	assert(!UnaryOp.NOT.backward_needs_input_only());
}

fn void test_op_payload_einsum() @test
{
	IndexExpr expr = index_expr::index_expr_parse("ij,jk->ik")!!;
	OpPayload p = op_payload::op_payload_einsum(expr, BinaryOp.MUL, BinaryOp.ADD);
	assert(p.tag() == OpTag.EINSUM);
	assert(p.einsum.inner == BinaryOp.MUL);
	assert(p.einsum.outer == BinaryOp.ADD);
	assert(p.einsum.alpha == 1.0);
	assert(p.einsum.beta == 0.0);
	assert(p.einsum.hash == 0);
}

fn void test_op_payload_map() @test
{
	OpPayload p = op_payload::op_payload_map(UnaryOp.RELU);
	assert(p.tag() == OpTag.MAP);
	assert(p.map_payload.unary == UnaryOp.RELU);
	assert(p.map_payload.beta == 0.0);
}

fn void test_op_payload_select() @test
{
	OpPayload p = op_payload::op_payload_select();
	assert(p.tag() == OpTag.SELECT);
	assert(p.select.hash == 0);
}

fn void test_einsum_payload_fields() @test
{
	EinsumPayload cp;
	cp.inner = BinaryOp.ADD;
	cp.outer = BinaryOp.MUL;
	cp.alpha = 2.0;
	cp.beta = 0.5;
	cp.hash = 0;

	assert(cp.inner == BinaryOp.ADD);
	assert(cp.outer == BinaryOp.MUL);
	assert(cp.alpha == 2.0);
	assert(cp.beta == 0.5);
}

fn void test_slice_payload() @test
{
	SlicePayload sp;
	sp.offsets[0] = 1;
	sp.offsets[1] = 2;
	sp.steps[0] = 1;
	sp.steps[1] = 2;
	sp.ndim = 2;
	sp.is_scatter = false;
	sp.hash = 0;

	assert(sp.offsets[0] == 1);
	assert(sp.steps[1] == 2);
	assert(sp.ndim == 2);
	assert(!sp.is_scatter);
}

fn void test_tag_round_trip_all_constructors() @test
{
	// Verify every constructor sets the correct tag
	IndexExpr expr = index_expr::index_expr_parse("ij,jk->ik")!!;

	OpPayload einsum = op_payload::op_payload_einsum(expr, BinaryOp.MUL, BinaryOp.ADD);
	assert(einsum.tag() == OpTag.EINSUM);

	OpPayload reduce = op_payload::op_payload_reduce(expr, BinaryOp.ADD);
	assert(reduce.tag() == OpTag.REDUCE);

	OpPayload map_p = op_payload::op_payload_map(UnaryOp.RELU);
	assert(map_p.tag() == OpTag.MAP);

	OpPayload sel = op_payload::op_payload_select();
	assert(sel.tag() == OpTag.SELECT);

	OpPayload perm = op_payload::op_payload_permute(expr);
	assert(perm.tag() == OpTag.PERMUTE);

	OpPayload conv = op_payload::op_payload_conv(operations::DEFAULT_CONV_PARAMS);
	assert(conv.tag() == OpTag.CONV);

	OpPayload gather = op_payload::op_payload_gather({ .dim = 1 });
	assert(gather.tag() == OpTag.GATHER);

	OpPayload cast = op_payload::op_payload_cast(0, 1);
	assert(cast.tag() == OpTag.CAST);

	OpPayload softmax = op_payload::op_payload_softmax(false, false);
	assert(softmax.tag() == OpTag.SOFTMAX);

	OpPayload rms = op_payload::op_payload_rms_norm(1e-5, RmsOpType.FORWARD);
	assert(rms.tag() == OpTag.RMS_NORM);

	OpPayload scatter = op_payload::op_payload_scatter_add({ .dim = 0 });
	assert(scatter.tag() == OpTag.SCATTER_ADD);

	OpPayload rmb = op_payload::op_payload_reduce_maxmin_backward(4, 8);
	assert(rmb.tag() == OpTag.REDUCE_MAXMIN_BACKWARD);
}

fn void test_hash_set_get_round_trip() @test
{
	OpPayload p = op_payload::op_payload_map(UnaryOp.EXP);
	assert(p.get_hash() == 0);

	p.set_hash(0xDEADBEEF);
	assert(p.get_hash() == 0xDEADBEEF);
	assert(p.tag() == OpTag.MAP); // tag preserved after set_hash
}
