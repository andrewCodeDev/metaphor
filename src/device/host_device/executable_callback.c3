//! Executable Callback
//!
//! Self-contained callback for executing compiled kernels. This is the payload
//! stored in execution units for compiled operations.
//!
//! Supports rebinding: tensor pointers are updated before each execution via
//! the rebind() method, allowing kernel reuse across different tensor instances
//! with the same shape/operation pattern.
//!
//! C3 design: No CTensor. Kernel functions receive TensorCore pointers directly.
//! C3 struct layout is C-ABI compatible, so JIT kernels access fields at known
//! offsets without any translation layer.
//!

module metaphor::device::host_device::executable_callback;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::device::host_device::kernel_cache;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import std::core::mem::allocator;

// Binding — maps kernel argument slots to execution unit sources/targets

enum BindingSource : char
{
	INPUT,
	OUTPUT,
}

struct Binding
{
	char slot;
	BindingSource source;
	char source_idx;
}

// ExecutableCallback

const usz MAX_INPUTS = 64;
const usz MAX_DIMS = 16;
const usz MAX_PARAMS = 8;
const usz MAX_PATTERN_LEN = 8;

struct ExecutableCallback (Executable)
{
	/// Bindings describe how to populate kernel arguments from sources/targets.
	Binding[MAX_INPUTS + 1] bindings_buf;
	usz bindings_count;

	/// Number of input bindings.
	char input_count;

	/// Compiled kernel hash for cache lookup.
	ulong hash;

	/// Dimension information for kernel dispatch.
	ulong[MAX_DIMS] dims;
	char num_dims;

	/// Runtime parameters (scale factors, epsilon, etc.).
	double[MAX_PARAMS] params;
	char num_params;

	/// Kernel function (set directly for static kernels, or resolved from cache).
	FusedKernelFn kernel_fn;

	/// True if this is an inplace kernel (sink must be prepended to inputs).
	bool is_inplace;

	// Runtime state populated by rebind()
	TensorCore*[MAX_INPUTS] inputs;
	TensorCore* output;

	/// Index pattern per input: e.g., "ij", "jk".
	/// Used to rederive dims from input shapes when batch size changes.
	char[MAX_PATTERN_LEN][MAX_INPUTS] input_patterns;
	char[MAX_INPUTS] input_pattern_lens;

	/// Dim names in order: dim_names[0..num_free_dims] are free, rest are bound.
	char[MAX_DIMS] dim_names;
	char num_free_dims;
}

// Executable interface

/// Execute the compiled kernel. Binds tensor pointers from sources/target,
/// then dispatches to the kernel function.
fn void ExecutableCallback.execute(&self, execution_sequence::TensorCorePtr[] sources, execution_sequence::TensorCorePtr target) @dynamic
{
	// Bind tensor pointers from sources/target.
	foreach (binding : self.bindings_buf[:self.bindings_count])
	{
		switch (binding.source)
		{
			case INPUT:
				if ((usz)binding.source_idx < sources.len)
				{
					self.inputs[binding.slot] = sources[binding.source_idx];
				}
			case OUTPUT:
				self.output = target;
		}
	}

	// If index patterns are stored, resolve dims from current input shapes.
	if (self.num_free_dims > 0)
	{
		self.resolve_dims_from_patterns();
	}

	// Resolve kernel from cache.
	// kernel_fn is only used for statically-set kernels (hash=0 boundary ops).
	FusedKernelFn func;
	if (self.hash == 0 && self.kernel_fn != null)
	{
		func = self.kernel_fn;
	}
	else
	{
		CachedKernel cached = kernel_cache::lookup_cached_kernel(self.hash) ??
			unreachable("ExecutableCallback: kernel not found in cache");
		func = cached.func;
	}

	// Pass TensorCore pointers directly — no CTensor translation.
	// The kernel accesses shape, strides, data at fixed offsets.
	func(
		&self.inputs,
		(usz)self.input_count,
		self.output,
		&self.dims,
		&self.params,
	);
}

fn void ExecutableCallback.destroy(&self) @dynamic
{
	allocator::free(&graph::graph_alloc, self);
}

// Builder helpers

fn void ExecutableCallback.add_dim(&self, ulong size)
{
	assert(self.num_dims < MAX_DIMS, "too many dims");
	self.dims[self.num_dims] = size;
	self.num_dims++;
}

fn void ExecutableCallback.add_param(&self, double value)
{
	assert(self.num_params < MAX_PARAMS, "too many params");
	self.params[self.num_params] = value;
	self.num_params++;
}

fn void ExecutableCallback.set_index_pattern(&self, char input_idx, char[] pattern)
{
	assert(pattern.len <= MAX_PATTERN_LEN);
	for (usz i = 0; i < pattern.len; i++)
	{
		self.input_patterns[input_idx][i] = pattern[i];
	}
	self.input_pattern_lens[input_idx] = (char)pattern.len;
}

fn void ExecutableCallback.set_dim_name(&self, char dim_idx, char name)
{
	self.dim_names[dim_idx] = name;
}

// Internal — resolve dims from index patterns

/// Resolve dims from input shapes using stored index patterns.
/// Called during rebind when shapes may have changed.
/// NOTE: This only updates self.dims for kernel dispatch. It does NOT
/// mutate the output TensorCore's shape — shapes are set correctly at
/// graph construction time and must not be overwritten.
fn void ExecutableCallback.resolve_dims_from_patterns(&self)
{
	// For each dim, find its value from an input's shape.
	for (usz d = 0; d < (usz)self.num_dims; d++)
	{
		char dim_name = self.dim_names[d];
		ulong resolved = 0;
		bool found = false;

		// Scan input patterns for this dim name.
		for (usz inp = 0; inp < (usz)self.input_count; inp++)
		{
			usz pat_len = (usz)self.input_pattern_lens[inp];
			for (usz pos = 0; pos < pat_len; pos++)
			{
				if (self.input_patterns[inp][pos] == dim_name)
				{
					TensorCore* input_core = self.inputs[inp];
					assert(pos < input_core.shape().len);
					ulong dim_size = input_core.shape().get(pos);
					// Prefer non-broadcast (size > 1) values.
					if (dim_size > 1 || !found)
					{
						resolved = dim_size;
						found = true;
					}
					if (dim_size > 1) { break; }
				}
			}
			if (found && resolved > 1) { break; }
		}

		if (found)
		{
			self.dims[d] = resolved;
		}
	}
}
