//! Logical Tensor Interface
//!
//! Typed wrapper around boolean dense tensors providing logical operations.
//! Backend storage is a regular tensor with Datatype.BOOL.
//!

module metaphor::tensor::logical_tensor;
import metaphor::tensor;
import metaphor::tensor::tensor_ops;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::device::common::operations;
import metaphor::device::common::index_expr;

// LogicalTensor — typed wrapper around a boolean Tensor

struct LogicalTensor
{
	tensor::Tensor tensor;
}

fn LogicalTensor from_tensor(tensor::Tensor t)
{
	return { .tensor = t };
}

fn tensor::Tensor LogicalTensor.as_tensor(&self)
{
	return self.tensor;
}

fn tensor_core::TensorCore* LogicalTensor.core(&self)
{
	return self.tensor.core();
}

fn tensor::Tensor? LogicalTensor.collect(&self, bool sync = true)
{
	return self.tensor.collect(sync);
}

// Logical Operations

fn LogicalTensor? LogicalTensor.and(&self, LogicalTensor other)
{
	return from_tensor(tensor_ops::logical_and(self.tensor, other.tensor)!);
}

fn LogicalTensor? LogicalTensor.or(&self, LogicalTensor other)
{
	return from_tensor(tensor_ops::logical_or(self.tensor, other.tensor)!);
}

fn LogicalTensor? LogicalTensor.xor(&self, LogicalTensor other)
{
	return from_tensor(tensor_ops::logical_xor(self.tensor, other.tensor)!);
}

fn LogicalTensor? LogicalTensor.nor(&self, LogicalTensor other)
{
	tensor::Tensor or_result = tensor_ops::logical_or(self.tensor, other.tensor)!;
	return from_tensor(tensor_ops::logical_not(or_result)!);
}

fn LogicalTensor? LogicalTensor.nand(&self, LogicalTensor other)
{
	tensor::Tensor and_result = tensor_ops::logical_and(self.tensor, other.tensor)!;
	return from_tensor(tensor_ops::logical_not(and_result)!);
}

fn LogicalTensor? LogicalTensor.not(&self)
{
	return from_tensor(tensor_ops::logical_not(self.tensor)!);
}

// Reduction Operations

const usz ALL_AXES = usz.max;

fn LogicalTensor? LogicalTensor.all_true(&self, usz axis = ALL_AXES)
{
	char[] expr;
	if (axis == ALL_AXES)
	{
		expr = build_reduce_all_expr(self.tensor.shape().len);
	}
	else
	{
		expr = build_reduce_expr(self.tensor.shape().len, axis);
	}
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	tensor::Tensor result = tensor_ops::reduce(
		self.tensor, &parsed,
		{ .op = operations::BinaryOp.AND, .sink = null })!;
	return from_tensor(result);
}

fn LogicalTensor? LogicalTensor.any_true(&self, usz axis = ALL_AXES)
{
	char[] expr;
	if (axis == ALL_AXES)
	{
		expr = build_reduce_all_expr(self.tensor.shape().len);
	}
	else
	{
		expr = build_reduce_expr(self.tensor.shape().len, axis);
	}
	index_expr::IndexExpr parsed = index_expr::index_expr_parse(expr)!;
	tensor::Tensor result = tensor_ops::reduce(
		self.tensor, &parsed,
		{ .op = operations::BinaryOp.OR, .sink = null })!;
	return from_tensor(result);
}

// Masking / Selection

fn tensor::Tensor? LogicalTensor.where(&self, tensor::Tensor if_true, tensor::Tensor if_false)
{
	return tensor_ops::select(self.tensor, if_true, if_false);
}

// Comparison functions — produce LogicalTensors from regular Tensors

fn LogicalTensor? eq(tensor::Tensor a, tensor::Tensor b)
{
	return from_tensor(tensor_ops::cmp_eq(a, b)!);
}

fn LogicalTensor? ne(tensor::Tensor a, tensor::Tensor b)
{
	return from_tensor(tensor_ops::cmp_ne(a, b)!);
}

fn LogicalTensor? lt(tensor::Tensor a, tensor::Tensor b)
{
	return from_tensor(tensor_ops::cmp_lt(a, b)!);
}

fn LogicalTensor? le(tensor::Tensor a, tensor::Tensor b)
{
	return from_tensor(tensor_ops::cmp_le(a, b)!);
}

fn LogicalTensor? gt(tensor::Tensor a, tensor::Tensor b)
{
	return from_tensor(tensor_ops::cmp_gt(a, b)!);
}

fn LogicalTensor? ge(tensor::Tensor a, tensor::Tensor b)
{
	return from_tensor(tensor_ops::cmp_ge(a, b)!);
}

// Helper functions — build einsum reduction expressions

fn char[] build_reduce_expr(usz rank, usz axis)
{
	assert(rank <= 8 && rank > 0, "build_reduce_expr: unsupported rank");
	assert(axis < rank, "build_reduce_expr: axis out of range");

	// Build "ijk...->ij..." dropping the axis-th char
	// Input pattern: first rank chars of "ijklmnop"
	// Output pattern: same but without the axis-th char
	static char[20] buf;
	usz pos = 0;
	// Input side
	for (usz i = 0; i < rank; i++) { buf[pos++] = (char)('i' + i); }
	buf[pos++] = '-'; buf[pos++] = '>';
	// Output side: all indices except axis
	for (usz i = 0; i < rank; i++)
	{
		if (i != axis) { buf[pos++] = (char)('i' + i); }
	}
	return buf[:pos];
}

fn char[] build_reduce_all_expr(usz rank)
{
	assert(rank <= 8, "build_reduce_all_expr: unsupported rank");

	// Build "ijk...->" (all axes reduced)
	static char[12] buf;
	usz pos = 0;
	for (usz i = 0; i < rank; i++) { buf[pos++] = (char)('i' + i); }
	buf[pos++] = '-'; buf[pos++] = '>';
	return buf[:pos];
}

