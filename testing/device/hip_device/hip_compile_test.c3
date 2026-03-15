//! HIP Compile Pipeline Tests
//!
//! Tests the CAS-to-Operations enum mapping functions and HipOp
//! construction helpers without requiring GPU hardware.
//!
//! These verify the compile-time translation layer that converts
//! CAS IR expressions into HIP operation descriptors.

module metaphor::testing::device::hip_device::hip_compile_test;
import metaphor::device::hip_device::compile;
import metaphor::device::hip_device::types;
import metaphor::device::common::operations;
import metaphor::cas;

// ============================================================================
// CAS UnaryOp → operations::UnaryOp mapping
// ============================================================================

fn void test_cas_unary_to_op_math() @test
{
	// Math functions should map directly
	assert(compile::cas_unary_to_op(cas::UnaryOp.ABS) == operations::UnaryOp.ABS);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SQRT) == operations::UnaryOp.SQRT);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SQR) == operations::UnaryOp.SQR);
	assert(compile::cas_unary_to_op(cas::UnaryOp.RECIP) == operations::UnaryOp.RECIP);
	assert(compile::cas_unary_to_op(cas::UnaryOp.EXP) == operations::UnaryOp.EXP);
	assert(compile::cas_unary_to_op(cas::UnaryOp.LOG) == operations::UnaryOp.LOG);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SIN) == operations::UnaryOp.SIN);
	assert(compile::cas_unary_to_op(cas::UnaryOp.COS) == operations::UnaryOp.COS);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SIGN) == operations::UnaryOp.SIGN);
}

fn void test_cas_unary_to_op_activations() @test
{
	assert(compile::cas_unary_to_op(cas::UnaryOp.RELU) == operations::UnaryOp.RELU);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SIGMOID) == operations::UnaryOp.SIGMOID);
	assert(compile::cas_unary_to_op(cas::UnaryOp.TANH) == operations::UnaryOp.TANH);
	assert(compile::cas_unary_to_op(cas::UnaryOp.GELU) == operations::UnaryOp.GELU);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SILU) == operations::UnaryOp.SILU);
	assert(compile::cas_unary_to_op(cas::UnaryOp.SOFTPLUS) == operations::UnaryOp.SOFTPLUS);
	assert(compile::cas_unary_to_op(cas::UnaryOp.HEAVISIDE) == operations::UnaryOp.HEAVISIDE);
}

fn void test_cas_unary_to_op_logical() @test
{
	assert(compile::cas_unary_to_op(cas::UnaryOp.LOGIC_NOT) == operations::UnaryOp.NOT);
}

fn void test_cas_unary_to_op_unmapped_defaults_to_identity() @test
{
	// CAS ops without explicit mapping (e.g. FLOOR, CEIL, ROUND) fall through
	// to IDENTITY via the default branch
	assert(compile::cas_unary_to_op(cas::UnaryOp.FLOOR) == operations::UnaryOp.IDENTITY);
	assert(compile::cas_unary_to_op(cas::UnaryOp.CEIL) == operations::UnaryOp.IDENTITY);
	assert(compile::cas_unary_to_op(cas::UnaryOp.ROUND) == operations::UnaryOp.IDENTITY);
}

// ============================================================================
// CAS BinaryOp → operations::BinaryOp mapping
// ============================================================================

fn void test_cas_binary_to_op() @test
{
	assert(compile::cas_binary_to_op(cas::BinaryOp.DIV) == operations::BinaryOp.DIV);
	assert(compile::cas_binary_to_op(cas::BinaryOp.SUB) == operations::BinaryOp.SUB);
	assert(compile::cas_binary_to_op(cas::BinaryOp.MAX) == operations::BinaryOp.MAX);
	assert(compile::cas_binary_to_op(cas::BinaryOp.MIN) == operations::BinaryOp.MIN);
	assert(compile::cas_binary_to_op(cas::BinaryOp.LOGIC_AND) == operations::BinaryOp.AND);
	assert(compile::cas_binary_to_op(cas::BinaryOp.LOGIC_OR) == operations::BinaryOp.OR);
}

fn void test_cas_binary_to_op_unmapped_defaults_to_add() @test
{
	// POW has no explicit case and falls through to ADD
	assert(compile::cas_binary_to_op(cas::BinaryOp.POW) == operations::BinaryOp.ADD);
}

// ============================================================================
// CAS ReduceOp → operations::BinaryOp mapping
// ============================================================================

fn void test_cas_reduce_to_binary() @test
{
	assert(compile::cas_reduce_to_binary(cas::ReduceOp.ADD) == operations::BinaryOp.ADD);
	assert(compile::cas_reduce_to_binary(cas::ReduceOp.MUL) == operations::BinaryOp.MUL);
	assert(compile::cas_reduce_to_binary(cas::ReduceOp.MAX) == operations::BinaryOp.MAX);
	assert(compile::cas_reduce_to_binary(cas::ReduceOp.MIN) == operations::BinaryOp.MIN);
}

// ============================================================================
// HipOp Types and Constants
// ============================================================================

fn void test_op_type_enum_distinct() @test
{
	// Verify key op types are distinct
	types::OpType contraction = types::OpType.CONTRACTION;
	types::OpType reduction = types::OpType.REDUCTION;
	types::OpType permutation = types::OpType.PERMUTATION;
	types::OpType transfer = types::OpType.TRANSFER;
	types::OpType softmax = types::OpType.SOFTMAX;
	types::OpType gather = types::OpType.GATHER;
	types::OpType scatter = types::OpType.SCATTER_ADD;

	assert(contraction != reduction);
	assert(reduction != permutation);
	assert(permutation != transfer);
	assert(transfer != softmax);
	assert(softmax != gather);
	assert(gather != scatter);
}

fn void test_type_constants() @test
{
	assert(types::MAX_INPUTS == 3, "max inputs should be 3");
	assert(types::MAX_MULTI_INPUTS == 8, "max multi-inputs should be 8");
	assert(types::MAX_FUSED_CHAIN == 8);
	assert(types::MAX_JIT_INPUTS == 32);
	assert(types::MAX_PATTERN_LEN == 16);
	assert(types::MAX_SLICE_DIMS == 8);
}

// ============================================================================
// HipOp Pattern Methods
// ============================================================================

fn void test_hip_op_set_get_input_pattern() @test
{
	types::HipOp op;

	// Set input pattern for index 0
	char[] pat0 = "ij";
	op.set_input_pattern(0, pat0);

	char[] got0 = op.get_input_pattern(0);
	assert(got0.len == 2, "input pattern 0 should have length 2");
	assert(got0[0] == 'i');
	assert(got0[1] == 'j');

	// Set input pattern for index 1
	char[] pat1 = "jk";
	op.set_input_pattern(1, pat1);

	char[] got1 = op.get_input_pattern(1);
	assert(got1.len == 2);
	assert(got1[0] == 'j');
	assert(got1[1] == 'k');
}

fn void test_hip_op_set_get_output_pattern() @test
{
	types::HipOp op;

	char[] pat = "ik";
	op.set_output_pattern(pat);

	char[] got = op.get_output_pattern();
	assert(got.len == 2, "output pattern should have length 2");
	assert(got[0] == 'i');
	assert(got[1] == 'k');
}


// ============================================================================
// HipOp Construction
// ============================================================================

fn void test_hip_op_contraction_setup() @test
{
	types::HipOp op;
	op.op_type = types::OpType.CONTRACTION;

	int dummy_a;
	int dummy_b;
	int dummy_c;
	op.inputs[0] = &dummy_a;
	op.inputs[1] = &dummy_b;
	op.input_count = 2;
	op.result = &dummy_c;

	op.set_input_pattern(0, "ij");
	op.set_input_pattern(1, "jk");
	op.set_output_pattern("ik");

	assert(op.op_type == types::OpType.CONTRACTION);
	assert(op.input_count == 2);
	assert(op.result == &dummy_c);

	char[] p0 = op.get_input_pattern(0);
	assert(p0.len == 2 && p0[0] == 'i' && p0[1] == 'j');

	char[] p1 = op.get_input_pattern(1);
	assert(p1.len == 2 && p1[0] == 'j' && p1[1] == 'k');

	char[] po = op.get_output_pattern();
	assert(po.len == 2 && po[0] == 'i' && po[1] == 'k');
}

fn void test_hip_op_transfer_setup() @test
{
	types::HipOp op;
	op.op_type = types::OpType.TRANSFER;

	int dummy_src;
	int dummy_dst;
	op.inputs[0] = &dummy_src;
	op.input_count = 1;
	op.result = &dummy_dst;
	op.transfer.direction = DTOH;

	assert(op.op_type == types::OpType.TRANSFER);
	assert(op.transfer.direction == DTOH);
}

fn void test_hip_op_gather_setup() @test
{
	types::HipOp op;
	op.op_type = types::OpType.GATHER;

	int dummy_input;
	int dummy_idx;
	int dummy_out;
	op.inputs[0] = &dummy_input;
	op.input_count = 1;
	op.gather.indices_tensor = &dummy_idx;
	op.result = &dummy_out;
	op.gather.dim = 1;

	assert(op.op_type == types::OpType.GATHER);
	assert(op.gather.dim == 1);
	assert(op.gather.indices_tensor == &dummy_idx);
}

fn void test_hip_op_rms_norm_setup() @test
{
	types::HipOp op;
	op.op_type = types::OpType.RMS_NORM;
	op.rms_norm.eps = 1e-5;

	assert(op.op_type == RMS_NORM);
	assert(op.rms_norm.eps > 0.0);
}

fn void test_hip_op_reduce_maxmin_bwd_setup() @test
{
	types::HipOp op;
	op.op_type = types::OpType.REDUCE_MAXMIN_BACKWARD;
	op.reduce_maxmin_bwd.row = 32;
	op.reduce_maxmin_bwd.col = 64;
	op.reduce_maxmin_bwd.reduce_type = ROWS;
	op.reduce_maxmin_bwd.is_min = false;

	assert(op.reduce_maxmin_bwd.row == 32);
	assert(op.reduce_maxmin_bwd.col == 64);
	assert(op.reduce_maxmin_bwd.reduce_type == ROWS);
	assert(!op.reduce_maxmin_bwd.is_min);
}

// ============================================================================
// ChainOp construction
// ============================================================================

fn void test_chain_op_unary() @test
{
	types::ChainOp cop;
	cop.is_unary = true;
	cop.unary = (CInt)operations::UnaryOp.RELU.ordinal;

	assert(cop.is_unary);
	assert(cop.unary == (CInt)operations::UnaryOp.RELU.ordinal);
}

fn void test_chain_op_binary() @test
{
	types::ChainOp cop;
	cop.is_unary = false;
	cop.binary = (CInt)operations::BinaryOp.ADD.ordinal;

	assert(!cop.is_unary);
	assert(cop.binary == (CInt)operations::BinaryOp.ADD.ordinal);
}
