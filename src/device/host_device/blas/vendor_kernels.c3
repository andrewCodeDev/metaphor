//! Vendor Kernel Registry
//!
//! Provides a unified interface to vendor-optimized kernels (BLAS).
//! Normalizes einsum patterns to canonical forms and dispatches to
//! appropriate CBLAS callbacks.
//!
//! Usage:
//!   if (vendor_kernels::lookup(pattern, children, result, inner, outer, dtype)) |vkr| {
//!       g.add_execution_unit(vkr.operation, ...);
//!   }
//!

module metaphor::device::host_device::blas::vendor_kernels;
import metaphor::device::host_device::blas::blas_callbacks;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::rtti;
import metaphor::device::common::operations;

// Pattern Analysis Types

enum TransformKind : char
{
	NONE,
	GEMM,
	BATCH_GEMM,
}

struct TransformInfo
{
	TransformKind kind;
	bool transpose_a;
	bool transpose_b;
	char batch_dims;
}

struct NormalizationResult
{
	char[64] pattern_buf;
	usz pattern_len;
	bool swap_operands;
	TransformInfo transform;
}

fn char[] NormalizationResult.pattern(&self)
{
	return self.pattern_buf[:self.pattern_len];
}

struct ParsedPattern
{
	char[] op1;
	char[] op2;
	char[] output;
	bool has_op2;
}

// Public Lookup Entry Point

/// Look up a vendor kernel for an einsum operation.
/// Returns a VendorKernelResult if a BLAS kernel matches, null otherwise.
fn VendorKernelResult? lookup(
	char[] pattern,
	TensorCore*[2] children,
	TensorCore* result,
	BinaryOp inner,
	BinaryOp outer,
	rtti::Datatype dtype)
{
	NormalizationResult norm = analyze_pattern(pattern);

	// Reorder children if pattern analysis says to swap
	TensorCore*[2] reordered;
	if (norm.swap_operands)
	{
		reordered[0] = children[1];
		reordered[1] = children[0];
	}
	else
	{
		reordered[0] = children[0];
		reordered[1] = children[1];
	}

	bool transpose_a;
	bool transpose_b;
	switch (norm.transform.kind)
	{
		case GEMM:
		case BATCH_GEMM:
			transpose_a = norm.transform.transpose_a;
			transpose_b = norm.transform.transpose_b;
		case NONE:
			transpose_a = false;
			transpose_b = false;
	}

	// Match canonical pattern to factory function
	char[] pat = norm.pattern();

	if (str_eq(pat, "ij,jk->ik"))
	{
		return blas_callbacks::make_gemm(reordered, result, inner, outer, dtype, transpose_a, transpose_b);
	}
	if (str_eq(pat, "bij,bjk->bik"))
	{
		return blas_callbacks::make_batch_gemm(reordered, result, inner, outer, dtype, transpose_a, transpose_b);
	}
	if (str_eq(pat, "ij,j->i"))
	{
		return blas_callbacks::make_gemv(reordered, result, inner, outer, dtype);
	}
	if (str_eq(pat, "i,i->"))
	{
		return blas_callbacks::make_dot(reordered, result, inner, outer, dtype);
	}
	if (str_eq(pat, "i,j->ij"))
	{
		return blas_callbacks::make_ger(reordered, result, inner, dtype);
	}

	return blas_callbacks::NO_VENDOR_KERNEL~;
}

// Pattern Analysis

/// Analyze and normalize an einsum pattern to canonical form.
fn NormalizationResult analyze_pattern(char[] pattern)
{
	// Strip ellipsis
	char[64] temp;
	usz temp_len = 0;
	usz i = 0;
	while (i < pattern.len)
	{
		if (i + 2 < pattern.len && pattern[i] == '.' && pattern[i + 1] == '.' && pattern[i + 2] == '.')
		{
			i += 3;
			continue;
		}
		if (temp_len < 64) { temp[temp_len] = pattern[i]; temp_len++; }
		i++;
	}
	char[] stripped = temp[:temp_len];

	// Parse into operands
	ParsedPattern parsed;
	if (!parse_pattern(stripped, &parsed))
	{
		return fallback_result(stripped);
	}

	// Try batch GEMM first (more specific)
	NormalizationResult result;
	if (analyze_batch_gemm(parsed, &result)) { return result; }
	if (analyze_gemm(parsed, &result)) { return result; }
	if (normalize_gemv(parsed, &result)) { return result; }
	if (normalize_dot(parsed, &result)) { return result; }
	if (normalize_outer(parsed, &result)) { return result; }

	return fallback_result(stripped);
}

fn NormalizationResult fallback_result(char[] pattern)
{
	NormalizationResult r;
	r.swap_operands = false;
	r.transform = { .kind = NONE };
	r.pattern_len = 0;

	// First-appearance normalization
	char[26] index_map = { [0..25] = 0 };
	char next_canonical = 'i';
	for (usz k = 0; k < pattern.len; k++)
	{
		char c = pattern[k];
		if (c >= 'a' && c <= 'z' && index_map[c - 'a'] == 0)
		{
			index_map[c - 'a'] = next_canonical;
			next_canonical++;
		}
	}

	for (usz k = 0; k < pattern.len && r.pattern_len < 64; k++)
	{
		char c = pattern[k];
		if (c >= 'a' && c <= 'z')
		{
			r.pattern_buf[r.pattern_len] = index_map[c - 'a'];
		}
		else
		{
			r.pattern_buf[r.pattern_len] = c;
		}
		r.pattern_len++;
	}
	return r;
}

fn bool parse_pattern(char[] pattern, ParsedPattern* result)
{
	// Find "->"
	usz arrow_pos = 0;
	bool found = false;
	for (usz k = 0; k + 1 < pattern.len; k++)
	{
		if (pattern[k] == '-' && pattern[k + 1] == '>')
		{
			arrow_pos = k;
			found = true;
			break;
		}
	}
	if (!found) return false;

	char[] inputs = pattern[:arrow_pos];
	char[] output = pattern[arrow_pos + 2..];

	// Find comma
	bool has_comma = false;
	usz comma_pos = 0;
	for (usz k = 0; k < inputs.len; k++)
	{
		if (inputs[k] == ',')
		{
			comma_pos = k;
			has_comma = true;
			break;
		}
	}

	result.output = output;
	if (has_comma)
	{
		result.op1 = inputs[:comma_pos];
		result.op2 = inputs[comma_pos + 1..];
		result.has_op2 = true;
	}
	else
	{
		result.op1 = inputs;
		result.op2 = {};
		result.has_op2 = false;
	}
	return true;
}

// GEMM Analysis: "ij,jk->ik" with transpose/swap detection

fn bool analyze_gemm(ParsedPattern parsed, NormalizationResult* result)
{
	if (!parsed.has_op2) return false;
	char[] op1 = parsed.op1;
	char[] op2 = parsed.op2;
	char[] output = parsed.output;

	if (op1.len != 2 || op2.len != 2 || output.len != 2) return false;

	// Find contracted index (in both operands, not in output)
	char con = 0;
	for (usz a = 0; a < 2; a++)
	{
		for (usz b = 0; b < 2; b++)
		{
			if (op1[a] == op2[b])
			{
				bool in_output = (op1[a] == output[0] || op1[a] == output[1]);
				if (!in_output) { con = op1[a]; }
			}
		}
	}
	if (con == 0) return false;

	char op1_free = op1[0] == con ? op1[1] : op1[0];
	char op2_free = op2[0] == con ? op2[1] : op2[0];

	// Verify both free indices are in output
	bool op1_in_out = (op1_free == output[0] || op1_free == output[1]);
	bool op2_in_out = (op2_free == output[0] || op2_free == output[1]);
	if (!op1_in_out || !op2_in_out) return false;

	bool op1_contributes_row = (op1_free == output[0]);

	bool swap;
	bool ta;
	bool tb;

	if (op1_contributes_row)
	{
		swap = false;
		ta = (op1[0] == con);
		tb = (op2[0] != con);
	}
	else
	{
		swap = true;
		ta = (op2[0] == con);
		tb = (op1[0] != con);
	}

	char[] canonical = "ij,jk->ik";
	for (usz k = 0; k < canonical.len; k++) { result.pattern_buf[k] = canonical[k]; }
	result.pattern_len = canonical.len;
	result.swap_operands = swap;
	result.transform = { .kind = GEMM, .transpose_a = ta, .transpose_b = tb };
	return true;
}

// Batch GEMM: "bij,bjk->bik"

fn bool analyze_batch_gemm(ParsedPattern parsed, NormalizationResult* result)
{
	if (!parsed.has_op2) return false;
	char[] op1 = parsed.op1;
	char[] op2 = parsed.op2;
	char[] output = parsed.output;

	if (op1.len < 3 || op2.len < 3 || output.len < 3) return false;

	usz batch_dims_1 = op1.len - 2;
	usz batch_dims_2 = op2.len - 2;
	usz batch_dims_out = output.len - 2;
	if (batch_dims_1 != batch_dims_2 || batch_dims_1 != batch_dims_out) return false;

	// Verify batch indices match
	for (usz bi = 0; bi < batch_dims_1; bi++)
	{
		if (op1[bi] != op2[bi] || op1[bi] != output[bi]) return false;
	}

	// Extract matrix indices (last 2)
	char[] mat1 = op1[batch_dims_1..];
	char[] mat2 = op2[batch_dims_1..];
	char[] mat_out = output[batch_dims_1..];
	if (mat1.len != 2 || mat2.len != 2 || mat_out.len != 2) return false;

	// Find contracted
	char con = 0;
	for (usz a = 0; a < 2; a++)
	{
		for (usz b = 0; b < 2; b++)
		{
			if (mat1[a] == mat2[b])
			{
				bool in_output = (mat1[a] == mat_out[0] || mat1[a] == mat_out[1]);
				if (!in_output) { con = mat1[a]; }
			}
		}
	}
	if (con == 0) return false;

	char op1_free = mat1[0] == con ? mat1[1] : mat1[0];
	char op2_free = mat2[0] == con ? mat2[1] : mat2[0];

	bool op1_in_out = (op1_free == mat_out[0] || op1_free == mat_out[1]);
	bool op2_in_out = (op2_free == mat_out[0] || op2_free == mat_out[1]);
	if (!op1_in_out || !op2_in_out) return false;

	bool op1_contributes_row = (op1_free == mat_out[0]);
	bool swap;
	bool ta;
	bool tb;

	if (op1_contributes_row)
	{
		swap = false;
		ta = (mat1[0] == con);
		tb = (mat2[0] != con);
	}
	else
	{
		swap = true;
		ta = (mat2[0] == con);
		tb = (mat1[0] != con);
	}

	char[] canonical = "bij,bjk->bik";
	for (usz k = 0; k < canonical.len; k++) { result.pattern_buf[k] = canonical[k]; }
	result.pattern_len = canonical.len;
	result.swap_operands = swap;
	result.transform = { .kind = BATCH_GEMM, .transpose_a = ta, .transpose_b = tb, .batch_dims = (char)batch_dims_1 };
	return true;
}

// GEMV: "ij,j->i"

fn bool normalize_gemv(ParsedPattern parsed, NormalizationResult* result)
{
	if (!parsed.has_op2) return false;
	char[] op1 = parsed.op1;
	char[] op2 = parsed.op2;
	char[] output = parsed.output;
	if (output.len != 1) return false;

	if (op1.len == 2 && op2.len == 1)
	{
		char vec_idx = op2[0];
		bool mat_has_vec = false;
		char mat_free = 0;
		for (usz k = 0; k < 2; k++)
		{
			if (op1[k] == vec_idx) { mat_has_vec = true; }
			else { mat_free = op1[k]; }
		}
		if (!mat_has_vec || mat_free != output[0]) return false;
	}
	else if (op1.len == 1 && op2.len == 2)
	{
		char vec_idx = op1[0];
		bool mat_has_vec = false;
		char mat_free = 0;
		for (usz k = 0; k < 2; k++)
		{
			if (op2[k] == vec_idx) { mat_has_vec = true; }
			else { mat_free = op2[k]; }
		}
		if (!mat_has_vec || mat_free != output[0]) return false;
	}
	else { return false; }

	char[] canonical = "ij,j->i";
	for (usz k = 0; k < canonical.len; k++) { result.pattern_buf[k] = canonical[k]; }
	result.pattern_len = canonical.len;
	result.swap_operands = false;
	result.transform = { .kind = NONE };
	return true;
}

// DOT: "i,i->"

fn bool normalize_dot(ParsedPattern parsed, NormalizationResult* result)
{
	if (!parsed.has_op2) return false;
	if (parsed.op1.len != 1 || parsed.op2.len != 1 || parsed.output.len != 0) return false;
	if (parsed.op1[0] != parsed.op2[0]) return false;

	char[] canonical = "i,i->";
	for (usz k = 0; k < canonical.len; k++) { result.pattern_buf[k] = canonical[k]; }
	result.pattern_len = canonical.len;
	result.swap_operands = false;
	result.transform = { .kind = NONE };
	return true;
}

// Outer Product: "i,j->ij"

fn bool normalize_outer(ParsedPattern parsed, NormalizationResult* result)
{
	if (!parsed.has_op2) return false;
	if (parsed.op1.len != 1 || parsed.op2.len != 1 || parsed.output.len != 2) return false;

	char idx1 = parsed.op1[0];
	char idx2 = parsed.op2[0];
	if (idx1 == idx2) return false;

	bool idx1_in_out = (idx1 == parsed.output[0] || idx1 == parsed.output[1]);
	bool idx2_in_out = (idx2 == parsed.output[0] || idx2 == parsed.output[1]);
	if (!idx1_in_out || !idx2_in_out) return false;

	char[] canonical = "i,j->ij";
	for (usz k = 0; k < canonical.len; k++) { result.pattern_buf[k] = canonical[k]; }
	result.pattern_len = canonical.len;
	result.swap_operands = false;
	result.transform = { .kind = NONE };
	return true;
}

// Helpers

fn bool str_eq(char[] a, char[] b)
{
	if (a.len != b.len) return false;
	for (usz k = 0; k < a.len; k++)
	{
		if (a[k] != b[k]) return false;
	}
	return true;
}
