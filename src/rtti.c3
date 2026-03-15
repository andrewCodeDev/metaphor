//! Runtime Type Information
//!
//! Datatype enum for tensor element types and OpTag for operation classification.

module metaphor::rtti;

// Datatype — tensor element type

enum Datatype : char
{
	BOOL,
	U8,
	U64,
	F16,
	BF16,
	F32,
	F64,
}

fn usz Datatype.size_of(&self)
{
	switch (*self)
	{
		case BOOL: return 1;
		case U8:   return 1;
		case U64:  return 8;
		case F16:  return 2;
		case BF16: return 2;
		case F32:  return 4;
		case F64:  return 8;
	}
}

/// Map a C3 type to a Datatype at compile time.
macro Datatype from_type($Type)
{
	$switch ($Type):
	$case bool:    return Datatype.BOOL;
	$case char:    return Datatype.U8;
	$case ulong:   return Datatype.U64;
	$case float16: return Datatype.F16;
	$case bfloat:  return Datatype.BF16;
	$case float:   return Datatype.F32;
	$case double:  return Datatype.F64;
	$default:      $error("Unsupported type for Datatype mapping");
	$endswitch
}

fn bool Datatype.is_float(&self)
{
	switch (*self)
	{
		case F16:  return true;
		case BF16: return true;
		case F32:  return true;
		case F64:  return true;
		default:   return false;
	}
}

fn bool Datatype.is_int(&self)
{
	switch (*self)
	{
		case U8:  return true;
		case U64: return true;
		default:  return false;
	}
}

fn String Datatype.name(&self)
{
	return self.nameof;
}

// OpTag — operation classification

enum OpTag : char
{
	// Element-wise binary
	ADD,
	SUB,
	MUL,
	DIV,

	// Element-wise unary
	NEG,
	SIN,
	COS,
	RELU,
	HEAVISIDE,

	// Comparison (produces bool tensor)
	CMP_EQ,
	CMP_GT,
	CMP_LT,
	CMP_GTE,
	CMP_LTE,

	// Logical (bool -> bool)
	LOGIC_AND,
	LOGIC_OR,
	LOGIC_NOT,
	LOGIC_XOR,
	LOGIC_XNOR,

	// Converters (bool -> non-bool)
	LOGIC_DENSE,
	LOGIC_INDEX,

	// Axis operations (symbolic)
	AXIS_COMPARE,

	// Softmax operations (fused forward/backward)
	SOFTMAX,
	LOG_SOFTMAX,
	SOFTMAX_BACKWARD,
	LOG_SOFTMAX_BACKWARD,

	// BLAS
	GEMM,
	GEMV,
	AXPY,
	SUM,
	NRM2,

	// Einsum (JIT compiled contractions)
	EINSUM,

	// Compiled (fused kernel)
	COMPILED,

	UNKNOWN,
}

fn bool OpTag.is_comparison(&self)
{
	switch (*self)
	{
		case CMP_EQ:  return true;
		case CMP_GT:  return true;
		case CMP_LT:  return true;
		case CMP_GTE: return true;
		case CMP_LTE: return true;
		default:      return false;
	}
}

fn bool OpTag.is_ordering_comparison(&self)
{
	switch (*self)
	{
		case CMP_GT:  return true;
		case CMP_LT:  return true;
		case CMP_GTE: return true;
		case CMP_LTE: return true;
		default:      return false;
	}
}

fn bool OpTag.is_logical(&self)
{
	switch (*self)
	{
		case LOGIC_AND:  return true;
		case LOGIC_OR:   return true;
		case LOGIC_NOT:  return true;
		case LOGIC_XOR:  return true;
		case LOGIC_XNOR: return true;
		default:         return false;
	}
}

fn bool OpTag.has_gradient(&self)
{
	return !(self.is_comparison() || self.is_logical());
}

fn bool OpTag.expects_bool_inputs(&self)
{
	switch (*self)
	{
		case LOGIC_AND:   return true;
		case LOGIC_OR:    return true;
		case LOGIC_NOT:   return true;
		case LOGIC_XOR:   return true;
		case LOGIC_XNOR:  return true;
		case LOGIC_DENSE: return true;
		case LOGIC_INDEX: return true;
		default:          return false;
	}
}

fn bool OpTag.produces_bool(&self)
{
	return self.is_comparison() || self.is_logical();
}

fn bool OpTag.is_converter(&self)
{
	switch (*self)
	{
		case LOGIC_DENSE: return true;
		case LOGIC_INDEX: return true;
		default:          return false;
	}
}
