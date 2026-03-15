//! Boundary Callbacks
//!
//! Callbacks for non-fusable boundary operations: convolution,
//! reduce max/min backward, gather/scatter, softmax, rms_norm,
//! slice, concat, stack, cast, dim, numel, transfer.
//!
//! These operations are fusion boundaries and use specialized
//! callbacks rather than JIT-compiled kernels.
//!

module metaphor::device::host_device::boundary_callbacks;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::rtti;
import metaphor::cas;
import metaphor::device::host_device::executable_callback;
import metaphor::device::host_device::kernel_cache;
import metaphor::device::common::op_payload;
import metaphor::device::common::operations;
import metaphor::device::host_device::kernels::cast;
import metaphor::device::host_device::kernels::softmax;
import metaphor::device::host_device::kernels::reduce;
import metaphor::device::host_device::kernels::scaling;
import std::core::mem::allocator;
import std::io;

alias TensorCorePtr = execution_sequence::TensorCorePtr;

// Slice Operations

/// Compile a slice operation.
/// Forward: copy from source using strides + offsets.
/// Backward (is_scatter): scatter gradient to output.
fn void compile_slice(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	SlicePayload* sp = &payload.slice;

	assert(entry.inputs.len >= 1, "boundary callback: missing input");

	TensorCore* source = entry.inputs[0];

	if (sp.is_scatter)
	{
		SliceScatterCallback* cb = allocator::new(g.allocator, SliceScatterCallback);
		*cb = {
			.payload = *sp,
		};

		TensorCorePtr[8] source_cores;
		source_cores[0] = source;
		usz num_sources = 1;
		g.add_execution_unit(
			(any)cb,
			source_cores[:num_sources],
			output,
		);
	}
	else
	{
		SliceCopyCallback* cb = allocator::new(g.allocator, SliceCopyCallback);
		*cb = {
			.payload = *sp,
		};

		TensorCorePtr[8] source_cores;
		source_cores[0] = source;
		usz num_sources = 1;
		g.add_execution_unit(
			(any)cb,
			source_cores[:num_sources],
			output,
		);
	}
}

/// Callback for slice scatter (backward pass).
struct SliceScatterCallback (Executable)
{
	SlicePayload payload;
}

fn void SliceScatterCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* grad = sources[0];
	TensorCore* tgt = target;

	assert(grad.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* grad_data = grad.state.dense.data;
	void* target_data = tgt.state.dense.data;
	if (grad_data == null || target_data == null) { return; }

	usz ndim = (usz)self.payload.ndim;
	usz num_elements = grad.shape().size();
	usz elem_size = grad.dtype().size_of();

	// Zero target first
	usz target_bytes = tgt.shape().size() * elem_size;
	mem::set(target_data, 0, target_bytes);

	// Scatter elements
	usz grad_idx = 0;
	ulong[8] indices;
	for (usz i = 0; i < 8; i++) indices[i] = 0;

	shape::Strides target_strides = tgt.shape().strides();

	while (grad_idx < num_elements)
	{
		ulong target_linear = 0;
		for (usz d = 0; d < ndim; d++)
		{
			ulong target_dim_idx = self.payload.offsets[d] + indices[d] * (ulong)self.payload.steps[d];
			target_linear += target_dim_idx * target_strides.get(d);
		}

		char* src = (char*)grad_data + grad_idx * elem_size;
		char* dst = (char*)target_data + target_linear * elem_size;
		mem::copy(dst, src, elem_size);

		grad_idx++;
		usz d = ndim;
		while (d > 0)
		{
			d--;
			indices[d]++;
			if (indices[d] < grad.shape().get(d)) { break; }
			indices[d] = 0;
		}
	}
}

fn void SliceScatterCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

/// Callback for forward slice copy.
struct SliceCopyCallback (Executable)
{
	SlicePayload payload;
}

fn void SliceCopyCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* tgt = target;

	assert(source.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* source_data = source.state.dense.data;
	void* target_data = tgt.state.dense.data;
	if (source_data == null || target_data == null) { return; }

	usz ndim = (usz)self.payload.ndim;
	usz num_elements = tgt.shape().size();
	usz elem_size = tgt.dtype().size_of();

	shape::Strides source_strides = source.shape().strides();

	usz target_idx = 0;
	ulong[8] indices;
	for (usz i = 0; i < 8; i++) indices[i] = 0;

	while (target_idx < num_elements)
	{
		// Compute source linear index using strides (not row-major accumulation)
		ulong source_linear = 0;
		for (usz d = 0; d < ndim; d++)
		{
			ulong src_idx = self.payload.offsets[d] + indices[d] * (ulong)self.payload.steps[d];
			source_linear += src_idx * source_strides.get(d);
		}

		char* src = (char*)source_data + source_linear * elem_size;
		char* dst = (char*)target_data + target_idx * elem_size;
		mem::copy(dst, src, elem_size);

		target_idx++;
		usz d = ndim;
		while (d > 0)
		{
			d--;
			indices[d]++;
			if (indices[d] < tgt.shape().get(d)) { break; }
			indices[d] = 0;
		}
	}
}

fn void SliceCopyCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Concat Operations

fn void compile_concat(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	ConcatPayload* cp = &payload.concat;

	ConcatCallback* cb = allocator::new(g.allocator, ConcatCallback);
	*cb = {
		.num_sources = cp.num_inputs,
		.dim = cp.dim,
		.offsets = cp.offsets,
	};

	usz num = (usz)cp.num_inputs;
	if (num > entry.inputs.len) num = entry.inputs.len;

	g.add_execution_unit(
		(any)cb,
		entry.inputs[:num],
		output,
	);
}

struct ConcatCallback (Executable)
{
	char num_sources;
	char dim;
	ulong[8] offsets;
}

fn void ConcatCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* tgt = target;
	assert(tgt.state.class_ == CoreClass.DENSE);
	void* target_data = tgt.state.dense.data;
	if (target_data == null) { return; }

	usz ndim = tgt.shape().len;
	usz concat_dim = (usz)self.dim;

	for (usz src_idx = 0; src_idx < (usz)self.num_sources && src_idx < sources.len; src_idx++)
	{
		TensorCore* source = sources[src_idx];
		assert(source.state.class_ == CoreClass.DENSE);
		void* source_data = source.state.dense.data;
		if (source_data == null) { continue; }

		usz num_elements = source.shape().size();
		usz elem_size = source.dtype().size_of();

		usz src_linear = 0;
		ulong[8] indices;
		for (usz i = 0; i < 8; i++) indices[i] = 0;

		while (src_linear < num_elements)
		{
			ulong target_linear = 0;
			for (usz d = 0; d < ndim; d++)
			{
				ulong idx = indices[d];
				if (d == concat_dim) idx += self.offsets[src_idx];
				target_linear = target_linear * tgt.shape().get(d) + idx;
			}

			char* src = (char*)source_data + src_linear * elem_size;
			char* dst = (char*)target_data + target_linear * elem_size;
			mem::copy(dst, src, elem_size);

			src_linear++;
			usz d = ndim;
			while (d > 0)
			{
				d--;
				indices[d]++;
				if (indices[d] < source.shape().get(d)) { break; }
				indices[d] = 0;
			}
		}
	}
}

fn void ConcatCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Stack Operations

fn void compile_stack(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	StackPayload* sp = &payload.stack;

	StackCallback* cb = allocator::new(g.allocator, StackCallback);
	*cb = {
		.num_sources = sp.num_inputs,
		.dim = sp.dim,
	};

	usz num = (usz)sp.num_inputs;
	if (num > entry.inputs.len) num = entry.inputs.len;

	g.add_execution_unit(
		(any)cb,
		entry.inputs[:num],
		output,
	);
}

struct StackCallback (Executable)
{
	char num_sources;
	char dim;
}

fn void StackCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* tgt = target;
	assert(tgt.state.class_ == CoreClass.DENSE);
	void* target_data = tgt.state.dense.data;
	if (target_data == null) { return; }

	usz target_ndim = tgt.shape().len;
	usz stack_dim = (usz)self.dim;

	for (usz src_idx = 0; src_idx < (usz)self.num_sources && src_idx < sources.len; src_idx++)
	{
		TensorCore* source = sources[src_idx];
		assert(source.state.class_ == CoreClass.DENSE);
		void* source_data = source.state.dense.data;
		if (source_data == null) { continue; }

		usz source_ndim = source.shape().len;
		usz num_elements = source.shape().size();
		usz elem_size = source.dtype().size_of();

		usz src_linear = 0;
		ulong[8] src_indices;
		for (usz i = 0; i < 8; i++) src_indices[i] = 0;

		while (src_linear < num_elements)
		{
			ulong target_linear = 0;
			usz src_d = 0;
			for (usz d = 0; d < target_ndim; d++)
			{
				ulong idx;
				if (d == stack_dim)
				{
					idx = (ulong)src_idx;
				}
				else
				{
					idx = src_indices[src_d];
					src_d++;
				}
				target_linear = target_linear * tgt.shape().get(d) + idx;
			}

			char* src = (char*)source_data + src_linear * elem_size;
			char* dst = (char*)target_data + target_linear * elem_size;
			mem::copy(dst, src, elem_size);

			src_linear++;
			usz d = source_ndim;
			while (d > 0)
			{
				d--;
				src_indices[d]++;
				if (src_indices[d] < source.shape().get(d)) { break; }
				src_indices[d] = 0;
			}
		}
	}
}

fn void StackCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Cast Operation

fn void compile_cast(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	CastPayload* cp = &payload.cast;

	assert(entry.inputs.len >= 1, "boundary callback: missing input");
	TensorCore* source = entry.inputs[0];

	CastCallback* cb = allocator::new(g.allocator, CastCallback);
	*cb = {
		.source_dtype = (rtti::Datatype)cp.source_dtype,
		.target_dtype = (rtti::Datatype)cp.target_dtype,
	};

	TensorCorePtr[8] source_cores;
	source_cores[0] = source;
	usz num_sources = 1;
	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct CastCallback (Executable)
{
	rtti::Datatype source_dtype;
	rtti::Datatype target_dtype;
}

fn void CastCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* tgt = target;

	assert(source.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* src_data = source.state.dense.data;
	void* dst_data = tgt.state.dense.data;
	if (src_data == null || dst_data == null) { return; }

	usz len = tgt.shape().size();
	cast::cast_elements(src_data, dst_data, len, self.source_dtype, self.target_dtype);
}

fn void CastCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Softmax Operation

fn void compile_softmax(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	SoftmaxPayload* sp = &payload.softmax;

	assert(entry.inputs.len >= 1, "boundary callback: missing input");
	TensorCore* source = entry.inputs[0];

	SoftmaxCallback* cb = allocator::new(g.allocator, SoftmaxCallback);
	*cb = {
		.axis = sp.axis,
		.is_log = sp.is_log,
		.is_backward = sp.is_backward,
	};

	TensorCorePtr[8] source_cores;
	source_cores[0] = source;
	usz num_sources = 1;
	// Backward has two inputs: grad_y and y
	if (sp.is_backward && entry.inputs.len >= 2)
	{
		source_cores[1] = entry.inputs[1];
		num_sources = 2;
	}

	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct SoftmaxCallback (Executable)
{
	int axis;
	bool is_log;
	bool is_backward;
}

fn void SoftmaxCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* tgt = target;

	assert(source.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* src_data = source.state.dense.data;
	void* dst_data = tgt.state.dense.data;
	if (src_data == null || dst_data == null) { return; }

	// Resolve axis: negative values index from end
	usz ndim = source.shape().len;
	usz resolved_axis;
	if (self.axis < 0)
	{
		resolved_axis = (usz)((int)ndim + self.axis);
	}
	else
	{
		resolved_axis = (usz)self.axis;
	}

	// Compute batch (product of dims before axis) and dim (size at axis)
	usz batch = 1;
	for (usz d = 0; d < resolved_axis; d++)
	{
		batch *= (usz)source.shape().get(d);
	}
	usz dim = (usz)source.shape().get(resolved_axis);

	if (self.is_backward)
	{
		// sources[0] = y (forward output), sources[1] = grad_y (upstream gradient)
		assert(sources.len >= 2, "softmax backward requires 2 inputs");
		TensorCore* grad_y_tc = sources[1];
		assert(grad_y_tc.state.class_ == CoreClass.DENSE);
		void* grad_y_data = grad_y_tc.state.dense.data;
		if (grad_y_data == null) { return; }

		// src_data = sources[0] = y (forward softmax output)
		if (self.is_log)
		{
			softmax::log_softmax_rows_backward_float(
				(float*)grad_y_data, (float*)src_data, (float*)dst_data, batch, dim);
		}
		else
		{
			softmax::softmax_rows_backward_float(
				(float*)grad_y_data, (float*)src_data, (float*)dst_data, batch, dim);
		}
	}
	else
	{
		if (self.is_log)
		{
			softmax::log_softmax_rows_float((float*)src_data, (float*)dst_data, batch, dim);
		}
		else
		{
			softmax::softmax_rows_float((float*)src_data, (float*)dst_data, batch, dim);
		}
	}
}

fn void SoftmaxCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// RMS Norm Operation

fn void compile_rms_norm(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	RmsNormPayload* rp = &payload.rms_norm;
	operations::RmsOpType op_type = rp.op_type;

	RmsNormCallback* cb = allocator::new(g.allocator, RmsNormCallback);
	*cb = {
		.eps = (float)rp.eps,
		.op_type = op_type,
	};

	TensorCorePtr[8] source_cores;
	usz num_sources;

	switch (op_type)
	{
		case FORWARD:
			// inputs: [x, weight]
			assert(entry.inputs.len >= 2, "rms_norm forward requires 2 inputs (x, weight)");
			source_cores[0] = entry.inputs[0];
			source_cores[1] = entry.inputs[1];
			num_sources = 2;

		case BACKWARD_X:
			// inputs: [grad_y, x, weight]
			assert(entry.inputs.len >= 3, "rms_norm backward_x requires 3 inputs (grad, x, weight)");
			source_cores[0] = entry.inputs[0];
			source_cores[1] = entry.inputs[1];
			source_cores[2] = entry.inputs[2];
			num_sources = 3;

		case BACKWARD_WEIGHT:
			// inputs: [grad_y, x]
			assert(entry.inputs.len >= 2, "rms_norm backward_weight requires 2 inputs (grad, x)");
			source_cores[0] = entry.inputs[0];
			source_cores[1] = entry.inputs[1];
			num_sources = 2;
	}

	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct RmsNormCallback (Executable)
{
	float eps;
	operations::RmsOpType op_type;
}

fn void RmsNormCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* tgt = target;
	assert(tgt.state.class_ == CoreClass.DENSE);
	void* dst_data = tgt.state.dense.data;
	if (dst_data == null) { return; }

	switch (self.op_type)
	{
		case FORWARD:
		{
			TensorCore* x_tc = sources[0];
			TensorCore* w_tc = sources[1];
			assert(x_tc.state.class_ == CoreClass.DENSE);
			assert(w_tc.state.class_ == CoreClass.DENSE);

			void* x_data = x_tc.state.dense.data;
			void* w_data = w_tc.state.dense.data;
			if (x_data == null || w_data == null) { return; }

			// Flatten N-D to 2D: batch = product of all dims except last, dim = last
			usz dim = x_tc.shape().get(x_tc.shape().len - 1);
			usz batch = 1;
			for (usz i = 0; i < x_tc.shape().len - 1; i++) { batch *= x_tc.shape().get(i); }

			scaling::rms_norm_weighted_rows_float(
				(float*)x_data, (float*)w_data, (float*)dst_data,
				batch, dim, self.eps);
		}

		case BACKWARD_X:
		{
			TensorCore* gy_tc = sources[0];
			TensorCore* x_tc = sources[1];
			TensorCore* w_tc = sources[2];
			assert(gy_tc.state.class_ == CoreClass.DENSE);
			assert(x_tc.state.class_ == CoreClass.DENSE);
			assert(w_tc.state.class_ == CoreClass.DENSE);

			void* gy_data = gy_tc.state.dense.data;
			void* x_data = x_tc.state.dense.data;
			void* w_data = w_tc.state.dense.data;
			if (gy_data == null || x_data == null || w_data == null) { return; }

			usz dim = x_tc.shape().get(x_tc.shape().len - 1);
			usz batch = 1;
			for (usz i = 0; i < x_tc.shape().len - 1; i++) { batch *= x_tc.shape().get(i); }

			scaling::rms_norm_backward_x_rows_float(
				(float*)gy_data, (float*)x_data, (float*)w_data, (float*)dst_data,
				batch, dim, self.eps);
		}

		case BACKWARD_WEIGHT:
		{
			TensorCore* gy_tc = sources[0];
			TensorCore* x_tc = sources[1];
			assert(gy_tc.state.class_ == CoreClass.DENSE);
			assert(x_tc.state.class_ == CoreClass.DENSE);

			void* gy_data = gy_tc.state.dense.data;
			void* x_data = x_tc.state.dense.data;
			if (gy_data == null || x_data == null) { return; }

			usz dim = x_tc.shape().get(x_tc.shape().len - 1);
			usz batch = 1;
			for (usz i = 0; i < x_tc.shape().len - 1; i++) { batch *= x_tc.shape().get(i); }

			// Zero the output before accumulating
			usz bytes = dim * float.sizeof;
			char[] dst_bytes = ((char*)dst_data)[:bytes];
			for (usz i = 0; i < bytes; i++) { dst_bytes[i] = 0; }

			scaling::rms_norm_backward_weight_rows_float(
				(float*)gy_data, (float*)x_data, (float*)dst_data,
				batch, dim, self.eps);
		}
	}
}

fn void RmsNormCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Conv Stubs (require dimensional extraction — kept as stub)

fn void compile_conv(graph::Graph* g, cas::IREntry* entry)
{
	unreachable("boundary op CONV not yet implemented for host device");
}

fn void compile_conv_transpose(graph::Graph* g, cas::IREntry* entry)
{
	unreachable("boundary op CONV_TRANSPOSE not yet implemented for host device");
}

// Gather Operation

fn void compile_gather(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	operations::GatherPayload* gp = &payload.gather;

	assert(entry.inputs.len >= 2, "boundary callback: missing inputs");
	TensorCore* source = entry.inputs[0];
	TensorCore* indices = entry.inputs[1];

	GatherCallback* cb = allocator::new(&graph::graph_alloc, GatherCallback);
	*cb = {
		.dim = gp.params.dim,
	};

	g.add_execution_unit(
		(any)cb,
		(TensorCorePtr[]){ source, indices },
		output,
	);
}

struct GatherCallback (Executable)
{
	char dim;
}

fn void GatherCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* indices = sources[1];
	TensorCore* tgt = target;

	assert(source.state.class_ == CoreClass.DENSE);
	assert(indices.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* src_data = source.state.dense.data;
	void* idx_data = indices.state.dense.data;
	void* dst_data = tgt.state.dense.data;
	if (src_data == null || idx_data == null || dst_data == null) { return; }

	usz gather_dim = (usz)self.dim;
	usz output_ndim = tgt.shape().len;
	usz input_ndim = source.shape().len;
	usz indices_ndim = indices.shape().len;
	usz elem_size = tgt.dtype().size_of();

	// Compute strides for all three tensors
	shape::Strides src_strides = source.shape().strides();
	shape::Strides dst_strides = tgt.shape().strides();
	shape::Strides idx_strides = indices.shape().strides();

	// Iterate over all output elements
	usz total = tgt.shape().size();
	for (usz flat = 0; flat < total; flat++)
	{
		// Decompose flat output index into multi-dim output coords
		shape::Indices pos = dst_strides.offset_to_pos(flat);

		// Compute flat index into indices tensor.
		// Indices coords are output coords [gather_dim .. gather_dim + indices_ndim)
		usz idx_flat = 0;
		for (usz i = 0; i < indices_ndim; i++)
		{
			idx_flat += pos.get(gather_dim + i) * idx_strides.get(i);
		}

		// Read the gather index value
		ulong gather_idx = ((ulong*)idx_data)[idx_flat];

		// Compute flat index into source tensor.
		// Before gather_dim: use output coords directly
		// At gather_dim: use the looked-up index value
		// After gather_dim: use output coords shifted by (indices_ndim - 1)
		usz src_flat = 0;
		for (usz d = 0; d < input_ndim; d++)
		{
			if (d < gather_dim)
			{
				src_flat += pos.get(d) * src_strides.get(d);
			}
			else if (d == gather_dim)
			{
				src_flat += gather_idx * src_strides.get(d);
			}
			else
			{
				// Output coord index for this input dim
				usz out_coord = d + (indices_ndim - 1);
				src_flat += pos.get(out_coord) * src_strides.get(d);
			}
		}

		// Copy element
		char* src = (char*)src_data + src_flat * elem_size;
		char* dst = (char*)dst_data + flat * elem_size;
		mem::copy(dst, src, elem_size);
	}
}

fn void GatherCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Scatter-Add Operation (backward for gather)

fn void compile_scatter_add(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	operations::ScatterAddPayload* sp = &payload.scatter_add;

	assert(entry.inputs.len >= 2, "boundary callback: missing inputs");
	TensorCore* grad = entry.inputs[0];
	TensorCore* indices = entry.inputs[1];

	ScatterAddCallback* cb = allocator::new(&graph::graph_alloc, ScatterAddCallback);
	*cb = {
		.dim = sp.params.dim,
	};

	g.add_execution_unit(
		(any)cb,
		(TensorCorePtr[]){ grad, indices },
		output,
	);
}

struct ScatterAddCallback (Executable)
{
	char dim;
}

fn void ScatterAddCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* grad = sources[0];
	TensorCore* indices = sources[1];
	TensorCore* tgt = target;

	assert(grad.state.class_ == CoreClass.DENSE);
	assert(indices.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* grad_data = grad.state.dense.data;
	void* idx_data = indices.state.dense.data;
	void* dst_data = tgt.state.dense.data;
	if (grad_data == null || idx_data == null || dst_data == null) { return; }

	usz scatter_dim = (usz)self.dim;
	usz grad_ndim = grad.shape().len;
	usz output_ndim = tgt.shape().len;
	usz indices_ndim = indices.shape().len;
	usz elem_size = tgt.dtype().size_of();

	// Zero the output first
	usz out_bytes = tgt.shape().size() * elem_size;
	char[] out_slice = ((char*)dst_data)[:out_bytes];
	for (usz i = 0; i < out_slice.len; i++) { out_slice[i] = 0; }

	// Compute strides
	shape::Strides grad_strides = grad.shape().strides();
	shape::Strides dst_strides = tgt.shape().strides();
	shape::Strides idx_strides = indices.shape().strides();

	// Iterate over all grad elements and scatter-add to output
	usz total = grad.shape().size();
	for (usz flat = 0; flat < total; flat++)
	{
		// Decompose flat grad index into multi-dim grad coords
		shape::Indices pos = grad_strides.offset_to_pos(flat);

		// Compute flat index into indices tensor.
		// Indices coords are grad coords [scatter_dim .. scatter_dim + indices_ndim)
		usz idx_flat = 0;
		for (usz i = 0; i < indices_ndim; i++)
		{
			idx_flat += pos.get(scatter_dim + i) * idx_strides.get(i);
		}

		// Read the scatter index value
		ulong scatter_idx = ((ulong*)idx_data)[idx_flat];

		// Compute flat index into output tensor.
		// Before scatter_dim: use grad coords directly
		// At scatter_dim: use the looked-up index value
		// After scatter_dim: use grad coords shifted by (indices_ndim - 1)
		usz dst_offset = 0;
		for (usz d = 0; d < output_ndim; d++)
		{
			if (d < scatter_dim)
			{
				dst_offset += pos.get(d) * dst_strides.get(d);
			}
			else if (d == scatter_dim)
			{
				dst_offset += scatter_idx * dst_strides.get(d);
			}
			else
			{
				usz grad_coord = d + (indices_ndim - 1);
				dst_offset += pos.get(grad_coord) * dst_strides.get(d);
			}
		}

		// Add element
		if (elem_size == 4) // F32
		{
			((float*)dst_data)[dst_offset] += ((float*)grad_data)[flat];
		}
		else if (elem_size == 8) // F64
		{
			((double*)dst_data)[dst_offset] += ((double*)grad_data)[flat];
		}
	}
}

fn void ScatterAddCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

fn void compile_transfer(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	assert(entry.inputs.len >= 1, "boundary callback: missing input");
	TensorCore* source = entry.inputs[0];

	TransferCallback* cb = allocator::new(g.allocator, TransferCallback);
	*cb = {};

	TensorCorePtr[8] source_cores;
	source_cores[0] = source;
	usz num_sources = 1;
	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct TransferCallback (Executable)
{
	char _pad;
}

fn void TransferCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* tgt = target;

	void* src_data = source.state.dense.data;
	void* dst_data = tgt.state.dense.data;
	if (src_data == null || dst_data == null) { return; }

	usz nbytes = source.state.dense.num_elements * source.dtype().size_of();
	mem::copy(dst_data, src_data, nbytes);
}

fn void TransferCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Reduce Max/Min Backward

fn void compile_reduce_maxmin_bwd(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	ReduceMaxMinBackwardPayload* rp = &payload.reduce_maxmin_backward;

	assert(entry.inputs.len >= 2, "boundary callback: missing inputs");
	TensorCore* original = entry.inputs[0];
	TensorCore* upstream = entry.inputs[1];

	ReduceMaxMinBwdCallback* cb = allocator::new(g.allocator, ReduceMaxMinBwdCallback);
	*cb = {
		.row = (usz)rp.row,
		.col = (usz)rp.col,
	};

	TensorCorePtr[8] source_cores;
	source_cores[0] = original;
	source_cores[1] = upstream;
	usz num_sources = 2;
	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct ReduceMaxMinBwdCallback (Executable)
{
	usz row;
	usz col;
}

fn void ReduceMaxMinBwdCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* original = sources[0];
	TensorCore* upstream = sources[1];
	TensorCore* tgt = target;

	assert(original.state.class_ == CoreClass.DENSE);
	assert(upstream.state.class_ == CoreClass.DENSE);
	assert(tgt.state.class_ == CoreClass.DENSE);

	void* orig_data = original.state.dense.data;
	void* up_data = upstream.state.dense.data;
	void* out_data = tgt.state.dense.data;
	assert(orig_data != null && up_data != null && out_data != null, "ReduceMaxMinBwdCallback: null data pointer — upstream allocation failed");

	usz orig_len = original.shape().size();
	usz up_len = upstream.shape().size();
	usz out_len = tgt.shape().size();

	// Determine reduce_type from shape comparison
	usz reduce_type;
	if (self.row > 0 && self.col > 0)
	{
		// rows: (M,N) -> (1,N), cols: (M,N) -> (M,1)
		if (up_len == self.col) { reduce_type = 0; } // rows reduce
		else if (up_len == self.row) { reduce_type = 1; } // cols reduce
		else { reduce_type = 2; } // flat
	}
	else { reduce_type = 2; }

	// Determine is_min from hash encoding (MSB of hash indicates min)
	// For now assume max (is_min = 0) — this can be refined
	usz is_min = 0;

	reduce::reduce_maxmin_backward_dispatch_float(
		(float*)orig_data, orig_len,
		(float*)up_data, up_len,
		(float*)out_data, out_len,
		self.row, self.col,
		reduce_type, is_min);
}

fn void ReduceMaxMinBwdCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

fn void compile_numel(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	assert(entry.inputs.len >= 1, "boundary callback: missing input");
	TensorCore* source = entry.inputs[0];

	NumelCallback* cb = allocator::new(g.allocator, NumelCallback);
	*cb = {};

	TensorCorePtr[8] source_cores;
	source_cores[0] = source;
	usz num_sources = 1;
	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct NumelCallback (Executable) { char _pad; }

fn void NumelCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* tgt = target;
	usz numel = source.shape().size();

	void* out_data = tgt.state.dense.data;
	assert(out_data != null, "NumelCallback: output data is null — upstream allocation failed");

	// Store as ulong (u64)
	*(ulong*)out_data = (ulong)numel;
}

fn void NumelCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

fn void compile_dim(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	assert(entry.inputs.len >= 1, "boundary callback: missing input");
	TensorCore* source = entry.inputs[0];

	usz axis = entry.payload.dim.axis;

	DimCallback* cb = allocator::new(g.allocator, DimCallback);
	*cb = {
		.axis = axis,
	};

	TensorCorePtr[8] source_cores;
	source_cores[0] = source;
	usz num_sources = 1;
	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct DimCallback (Executable)
{
	usz axis;
}

fn void DimCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* source = sources[0];
	TensorCore* tgt = target;
	shape::Shape* s = source.shape();
	assert(self.axis < (usz)s.len, "DimCallback: axis out of bounds for tensor shape");

	ulong dim_size = s.get(self.axis);

	void* out_data = tgt.state.dense.data;
	assert(out_data != null, "DimCallback: output data is null — upstream allocation failed");

	*(ulong*)out_data = dim_size;
}

fn void DimCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Public dispatch entry point

/// Dispatch a boundary operation by payload tag.
/// Returns true if the op was handled, false if it should fall through to JIT.
// Reshape — zero-copy shape reinterpretation

struct ReshapeCallback (execution_sequence::Executable)
{
	char _pad;
}

fn void ReshapeCallback.execute(&self, TensorCorePtr[] sources,
	TensorCorePtr target) @dynamic
{
	tensor_core::TensorCore* src = sources[0];
	tensor_core::TensorCore* tgt = target;
	src.share_data(tgt);
}

fn void ReshapeCallback.destroy(&self) @dynamic {}

fn void compile_reshape(graph::Graph* g, cas::IREntry* entry)
{
	tensor_core::TensorCore* output = entry.output;
	assert(entry.inputs.len >= 1, "boundary callback: missing input");
	tensor_core::TensorCore* source = entry.inputs[0];

	ReshapeCallback* cb = allocator::new(g.allocator, ReshapeCallback);
	*cb = {};

	TensorCorePtr[1] source_cores = { source };
	g.add_execution_unit(
		(any)cb,
		&source_cores,
		output,
	);
}

// Dropout Operation

fn void compile_dropout(graph::Graph* g, cas::IREntry* entry)
{
	TensorCore* output = entry.output;
	OpPayload* payload = entry.payload;
	operations::DropoutPayload* dp = &payload.dropout;

	DropoutCallback* cb = allocator::new(g.allocator, DropoutCallback);
	*cb = {
		.p = dp.p,
		.seed = dp.seed,
		.is_backward = dp.is_backward,
	};

	TensorCorePtr[8] source_cores;
	source_cores[0] = entry.inputs[0];
	usz num_sources = 1;

	g.add_execution_unit(
		(any)cb,
		source_cores[:num_sources],
		output,
	);
}

struct DropoutCallback (Executable)
{
	float p;
	ulong seed;
	bool is_backward;
}

/// Simple splitmix64 PRNG for deterministic mask generation from seed.
macro ulong splitmix64(ulong* state)
{
	*state += 0x9e3779b97f4a7c15;
	ulong z = *state;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

fn void DropoutCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* tgt = target;
	assert(tgt.state.class_ == CoreClass.DENSE);
	void* dst_data = tgt.state.dense.data;
	if (dst_data == null) { return; }

	usz n = tgt.shape().size();
	float scale = 1.0f / (1.0f - self.p);

	if (self.is_backward)
	{
		// Backward: regenerate mask from same seed, apply to gradient
		TensorCore* grad_tc = sources[0];
		assert(grad_tc.state.class_ == CoreClass.DENSE);

		void* grad_data = grad_tc.state.dense.data;
		if (grad_data == null) { return; }

		// Regenerate mask using same seed (deterministic)
		ulong rng_state = self.seed;
		for (usz i = 0; i < n; i++)
		{
			ulong r = splitmix64(&rng_state);
			// Use upper bits for uniform [0,1)
			float u = (float)(r >> 40) / (float)(1 << 24);
			bool keep = u >= self.p;
			((float*)dst_data)[i] = keep ? ((float*)grad_data)[i] * scale : 0.0f;
		}
	}
	else
	{
		// Forward: generate mask, apply inverted dropout
		TensorCore* src_tc = sources[0];
		assert(src_tc.state.class_ == CoreClass.DENSE);

		void* src_data = src_tc.state.dense.data;
		if (src_data == null) { return; }

		ulong rng_state = self.seed;
		for (usz i = 0; i < n; i++)
		{
			ulong r = splitmix64(&rng_state);
			float u = (float)(r >> 40) / (float)(1 << 24);
			bool keep = u >= self.p;
			((float*)dst_data)[i] = keep ? ((float*)src_data)[i] * scale : 0.0f;
		}
	}
}

fn void DropoutCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

fn bool dispatch_boundary_op(graph::Graph* g, cas::IREntry* entry)
{
	if (entry.payload == null) { return false; }

	switch (entry.payload.tag())
	{
		case SLICE:       compile_slice(g, entry);
		case CONCAT:      compile_concat(g, entry);
		case STACK:       compile_stack(g, entry);
		case CAST:        compile_cast(g, entry);
		case RESHAPE:     compile_reshape(g, entry);
		case SOFTMAX:     compile_softmax(g, entry);
		case RMS_NORM:    compile_rms_norm(g, entry);
		case CONV:        compile_conv(g, entry);
		case CONV_TRANSPOSE: compile_conv_transpose(g, entry);
		case GATHER:      compile_gather(g, entry);
		case SCATTER_ADD: compile_scatter_add(g, entry);
		case TRANSFER:    compile_transfer(g, entry);
		case REDUCE_MAXMIN_BACKWARD: compile_reduce_maxmin_bwd(g, entry);
		case SELECTIVE_SCAN: unreachable("selective_scan requires CUDA device");
		case CAUSAL_CONV1D: unreachable("causal_conv1d requires CUDA device");
		case DROPOUT:     compile_dropout(g, entry);
		case NUMEL:       compile_numel(g, entry);
		case DIM:         compile_dim(g, entry);
		default:
			return false;
	}
	return true;
}
