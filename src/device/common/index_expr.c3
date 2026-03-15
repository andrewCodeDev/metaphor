//! Parsed Index Expression
//!
//! Handles einsum-style expressions like "ij,jk->ik" for matrix multiply,
//! "ijk->ik" for reduction, "...ij,...jk->...ik" for batched operations.
//!

module metaphor::device::common::index_expr;

faultdef PARSE_ERROR;

// Parsed Index Expression

/// Parsed index expression from a string like "ij,kj->ik".
struct IndexExpr
{
	/// Input index patterns stored inline (max 2 inputs, max 8 indices each).
	char[8] input0;
	char[8] input1;
	char[2] input_lens;
	char num_inputs;

	/// Output index pattern.
	char[8] output;
	char output_len;

	/// Whether ellipsis (...) is present.
	bool has_ellipsis;
}

/// Parse an expression string like "ij,jk->ik".
fn IndexExpr? index_expr_parse(char[] expr)
{
	IndexExpr result = {
		.input0 = { [0..7] = 0 },
		.input1 = { [0..7] = 0 },
		.input_lens = { 0, 0 },
		.num_inputs = 0,
		.output = { [0..7] = 0 },
		.output_len = 0,
		.has_ellipsis = false,
	};

	// Check for ellipsis
	for (usz i = 0; i + 2 < expr.len; i++)
	{
		if (expr[i] == '.' && expr[i+1] == '.' && expr[i+2] == '.')
		{
			result.has_ellipsis = true;
			break;
		}
	}

	// Find "->" separator
	usz arrow_pos = 0;
	bool found_arrow = false;
	for (usz i = 0; i + 1 < expr.len; i++)
	{
		if (expr[i] == '-' && expr[i+1] == '>')
		{
			arrow_pos = i;
			found_arrow = true;
			break;
		}
	}
	if (!found_arrow) return PARSE_ERROR~;

	char[] inputs_part = expr[0..arrow_pos - 1];
	char[] output_part = expr[arrow_pos + 2..];

	// Parse output (strip ellipsis)
	char out_idx = 0;
	usz i = 0;
	while (i < output_part.len)
	{
		if (i + 2 < output_part.len && output_part[i] == '.' && output_part[i+1] == '.' && output_part[i+2] == '.')
		{
			i += 3;
			continue;
		}
		if (out_idx < 8)
		{
			result.output[out_idx] = output_part[i];
			out_idx++;
		}
		i++;
	}
	result.output_len = out_idx;

	// Split inputs by comma and parse each
	usz start = 0;
	for (usz p = 0; p <= inputs_part.len; p++)
	{
		if (p == inputs_part.len || inputs_part[p] == ',')
		{
			if (result.num_inputs >= 2) return PARSE_ERROR~;
			char[] input = inputs_part[start..p - 1];

			char in_idx = 0;
			usz j = 0;
			while (j < input.len)
			{
				if (j + 2 < input.len && input[j] == '.' && input[j+1] == '.' && input[j+2] == '.')
				{
					j += 3;
					continue;
				}
				if (in_idx < 8)
				{
					if (result.num_inputs == 0) {
						result.input0[in_idx] = input[j];
					} else {
						result.input1[in_idx] = input[j];
					}
					in_idx++;
				}
				j++;
			}
			result.input_lens[result.num_inputs] = in_idx;
			result.num_inputs++;

			start = p + 1;
		}
	}

	return result;
}

/// Get input indices as slice.
fn char[] IndexExpr.get_input(&self, usz idx)
{
	if (idx >= self.num_inputs) return {};
	if (idx == 0) {
		return self.input0[0..self.input_lens[0] - 1];
	} else {
		return self.input1[0..self.input_lens[1] - 1];
	}
}

/// Get output indices as slice.
fn char[] IndexExpr.get_output(&self)
{
	return self.output[0..self.output_len - 1];
}

/// Get bound indices (appear in inputs but not output).
fn char[] IndexExpr.get_bound(&self, char[] buf)
{
	usz count = 0;
	char[] free = self.get_output();

	for (usz i = 0; i < self.num_inputs; i++)
	{
		char[] inp = self.get_input(i);
		foreach (c : inp)
		{
			// Skip if in output (free index)
			bool in_free = false;
			foreach (f : free)
			{
				if (f == c) { in_free = true; break; }
			}
			if (in_free) continue;

			// Skip if already in buf
			bool in_buf = false;
			for (usz k = 0; k < count; k++)
			{
				if (buf[k] == c) { in_buf = true; break; }
			}
			if (in_buf) continue;

			if (count < buf.len)
			{
				buf[count] = c;
				count++;
			}
		}
	}
	return buf[0..count - 1];
}

/// Find position of an index in a pattern.
fn usz? index_in_pattern(char idx, char[] pattern)
{
	for (usz i = 0; i < pattern.len; i++)
	{
		if (pattern[i] == idx) return i;
	}
	return PARSE_ERROR~;
}
