//! Device Reference
//!
//! Uniform interface for device-agnostic memory management and operation dispatch.
//!
//! Tensors store a DeviceReference. Runtime device selection enables mixed-device graphs.
//!

module metaphor::device::device_reference;
import metaphor::rtti;
import metaphor::device::sync;
import metaphor::tensor::tensor_core;
import std::io;
import std::math;

alias TensorCorePtr = tensor_core::TensorCore*;

// TransferDirection — cross-device memory transfer direction

enum TransferDirection : char
{
	HTOH,
	HTOD,
	DTOH,
	DTOD,
}

// EqualMode — tensor equality comparison mode

enum EqualMode : char
{
	EXACT,
	APPROX,
}

// Random init types — used by Device.mem_random interface

/// Distribution type for random tensor initialization.
enum RandType : char
{
	/// Uniform distribution in [0, 1).
	UNIFORM,
	/// Standard normal distribution (mean=0, std=1).
	NORMAL,
}

/// Scale mode for random initialization.
enum ScaleMode : char
{
	/// Use value directly as the scale factor.
	EXACT,
	/// Kaiming He init: compute sqrt(2 / value) where value = fan_in.
	KAIMING,
}

/// Scale parameter for random initialization.
struct Scale
{
	ScaleMode mode;
	double value;
}

const Scale DEFAULT_SCALE = { .mode = ScaleMode.EXACT, .value = 1.0 };

fn Scale exact_scale(double s)
{
	return { .mode = ScaleMode.EXACT, .value = s };
}

fn Scale kaiming_scale(usz fan_in)
{
	return { .mode = ScaleMode.KAIMING, .value = (double)fan_in };
}

/// Resolve a Scale to a concrete multiplier.
fn double Scale.resolve(&self)
{
	switch (self.mode)
	{
		case EXACT: return self.value;
		case KAIMING: return math::sqrt(2.0 / self.value);
	}
}

// Device interface — the method contract that all device backends implement

<*
 Core device interface. Every device backend (host, CUDA, etc.) implements this.

 C3 interfaces provide the same polymorphism without comptime machinery.
*>
interface Device
{
	/// Returns the device name (e.g., "host", "cuda_graph").
	fn ZString name();

	/// Allocate memory for `n` elements of the given datatype.
	/// Returns a byte slice of the allocated memory.
	fn char[] mem_alloc(Datatype dtype, usz n);

	/// Free previously allocated memory.
	fn void mem_free(char[] slice);

	/// Copy memory within the same device.
	fn void mem_copy(char[] src, char[] dst);

	/// Transfer memory across devices with explicit direction.
	fn void mem_transfer(char[] src, char[] dst, TransferDirection direction);

	/// Synchronize — block until all pending operations complete.
	fn void sync();

	/// Get memory currently in use from device pool.
	fn usz mem_used();

	/// Get total memory reserved by device pool.
	fn usz mem_reserved();

	/// Get peak memory usage (high-water mark).
	fn usz mem_peak();

	/// Execute an operation on this device.
	fn void execute(any operation, TensorCorePtr[] sources, TensorCorePtr target);

	/// Prepare a tensor for execution (device-specific sync init).
	fn void prepare_execute(TensorCorePtr core);

	/// Finalize execution for a tensor (record completion event).
	fn void finalize_execute(TensorCorePtr core);

	/// Fill device memory with random values.
	/// Pass kaiming_scale(fan_in) for Kaiming He init.
	/// Uses the global RNG from sampling::get_rng().
	fn void mem_random(Datatype dtype, char[] slice, RandType rand_type, Scale scale = DEFAULT_SCALE);

	/// Fill device memory with a constant value.
	fn void mem_fill(Datatype dtype, char[] slice, double value);

	/// Duplicate a memory buffer on the device (alloc + copy).
	fn char[] mem_dupe(Datatype dtype, char[] src);

	/// Fill device memory with an arithmetic sequence: initial, initial+step, initial+2*step, ...
	fn void mem_sequence(Datatype dtype, char[] dst, double initial, double step);

	/// Cast elements from one dtype to another.
	fn void mem_cast(Datatype src_dtype, Datatype dst_dtype, char[] src, char[] dst);

	/// Compile an IR entry to an execution unit.
	/// graph and entry are void* to avoid circular imports (cast in implementations).
	fn void compile(void* graph, void* entry);

	/// Sample a token from logits using sampling configuration.
	fn uint sample(Datatype dtype, char[] logits, usz vocab_size, char[] context, usz context_len, void* config);

	/// Greedy decoding — return index of maximum logit.
	fn uint argmax(Datatype dtype, char[] logits, usz vocab_size);
}

// DeviceReference — wraps a Device interface with memory stats tracking

<*
 DeviceReference wraps a Device interface value and adds memory statistics tracking.

 Every tensor stores a DeviceReference. This provides:
 - Uniform dispatch to any device backend via the Device interface
 - Global memory allocation/free statistics
 - Compatibility checks between devices
*>
struct DeviceReference
{
	Device device;
}

// Memory statistics (module-level)

ulong total_freed_bytes = 0;
ulong free_count = 0;
ulong total_alloc_bytes = 0;
ulong alloc_count = 0;

fn void reset_mem_stats()
{
	total_freed_bytes = 0;
	free_count = 0;
	total_alloc_bytes = 0;
	alloc_count = 0;
}

// DeviceReference methods — delegate to Device interface + track stats

fn ZString DeviceReference.name(&self)
{
	return self.device.name();
}

fn char[] DeviceReference.mem_alloc(&self, Datatype dtype, usz n)
{
	char[] result = self.device.mem_alloc(dtype, n);
	total_alloc_bytes += (ulong)result.len;
	alloc_count++;
	return result;
}

fn void DeviceReference.mem_free(&self, char[] slice)
{
	total_freed_bytes += (ulong)slice.len;
	free_count++;
	self.device.mem_free(slice);
}

fn void DeviceReference.mem_copy(&self, char[] src, char[] dst)
{
	self.device.mem_copy(src, dst);
}

fn void DeviceReference.mem_transfer(&self, char[] src, char[] dst, TransferDirection direction)
{
	self.device.mem_transfer(src, dst, direction);
}

fn void DeviceReference.sync(&self)
{
	self.device.sync();
}

fn usz DeviceReference.mem_used(&self)
{
	return self.device.mem_used();
}

fn usz DeviceReference.mem_reserved(&self)
{
	return self.device.mem_reserved();
}

fn usz DeviceReference.mem_peak(&self)
{
	return self.device.mem_peak();
}

fn void DeviceReference.execute(&self, any operation, TensorCorePtr[] sources, TensorCorePtr target)
{
	self.device.execute(operation, sources, target);
}

fn void DeviceReference.prepare_execute(&self, TensorCorePtr core)
{
	self.device.prepare_execute(core);
}

fn void DeviceReference.finalize_execute(&self, TensorCorePtr core)
{
	self.device.finalize_execute(core);
}

fn void DeviceReference.mem_random(&self, Datatype dtype, char[] slice, RandType rand_type, Scale scale = DEFAULT_SCALE)
{
	self.device.mem_random(dtype, slice, rand_type, scale);
}

fn void DeviceReference.mem_fill(&self, Datatype dtype, char[] slice, double value)
{
	self.device.mem_fill(dtype, slice, value);
}

fn char[] DeviceReference.mem_dupe(&self, Datatype dtype, char[] src)
{
	char[] result = self.device.mem_dupe(dtype, src);
	total_alloc_bytes += (ulong)result.len;
	alloc_count++;
	return result;
}

fn void DeviceReference.mem_sequence(&self, Datatype dtype, char[] dst, double initial, double step)
{
	self.device.mem_sequence(dtype, dst, initial, step);
}

fn void DeviceReference.mem_cast(&self, Datatype src_dtype, Datatype dst_dtype, char[] src, char[] dst)
{
	self.device.mem_cast(src_dtype, dst_dtype, src, dst);
}

fn void DeviceReference.compile(&self, void* graph, void* entry)
{
	self.device.compile(graph, entry);
}

fn uint DeviceReference.sample(&self, Datatype dtype, char[] logits, usz vocab_size, char[] context, usz context_len, void* config)
{
	return self.device.sample(dtype, logits, vocab_size, context, context_len, config);
}

fn uint DeviceReference.argmax(&self, Datatype dtype, char[] logits, usz vocab_size)
{
	return self.device.argmax(dtype, logits, vocab_size);
}

/// Compare two memory buffers for equality (exact or approximate).
/// This is an inline helper, not dispatched through the device interface.
fn bool DeviceReference.mem_equal(&self, Datatype dtype, char[] a, char[] b, EqualMode mode)
{
	if (a.len != b.len) { return false; }

	switch (mode)
	{
		case EXACT:
			for (usz i = 0; i < a.len; i++)
			{
				if (a[i] != b[i]) { return false; }
			}
			return true;
		case APPROX:
			switch (dtype)
			{
				case Datatype.F32: return approx_equal_f32(a, b);
				case Datatype.F64: return approx_equal_f64(a, b);
				default:
					for (usz i = 0; i < a.len; i++)
					{
						if (a[i] != b[i]) { return false; }
					}
					return true;
			}
	}
}

fn bool approx_equal_f32(char[] a, char[] b) @private
{
	float[] a_vals = ((float*)a.ptr)[:a.len / float.sizeof];
	float[] b_vals = ((float*)b.ptr)[:b.len / float.sizeof];

	const float RTOL = 1e-5;
	const float ATOL = 1e-8;

	for (usz i = 0; i < a_vals.len; i++)
	{
		float diff = math::abs(a_vals[i] - b_vals[i]);
		float threshold = ATOL + RTOL * math::abs(b_vals[i]);
		if (diff > threshold) { return false; }
	}
	return true;
}

fn bool approx_equal_f64(char[] a, char[] b) @private
{
	double[] a_vals = ((double*)a.ptr)[:a.len / double.sizeof];
	double[] b_vals = ((double*)b.ptr)[:b.len / double.sizeof];

	const double RTOL = 1e-5;
	const double ATOL = 1e-8;

	for (usz i = 0; i < a_vals.len; i++)
	{
		double diff = math::abs(a_vals[i] - b_vals[i]);
		double threshold = ATOL + RTOL * math::abs(b_vals[i]);
		if (diff > threshold) { return false; }
	}
	return true;
}

/// Check if two device references are compatible (can share memory).
/// Host and host_stream are compatible (shared memory pool).
fn bool DeviceReference.is_compatible(&self, DeviceReference other)
{
	// For now, compare device name strings.
	// When more devices are added, expand this logic.
	ZString self_name = self.device.name();
	ZString other_name = other.device.name();

	// Host variants are always compatible with each other
	bool self_host = self_name == "host" || self_name == "host_stream";
	bool other_host = other_name == "host" || other_name == "host_stream";
	if (self_host && other_host) { return true; }

	return self_name == other_name;
}

fn bool DeviceReference.is_host(&self)
{
	ZString n = self.device.name();
	return n == "host" || n == "host_stream";
}

// Typed memory helpers — macros over the char[]-based device interface

/// Reinterpret a typed slice as a raw byte slice.
macro char[] as_bytes(slice) @local
{
	return ((char*)slice.ptr)[:slice.len * $typeof(slice[0]).sizeof];
}

/// Map a C3 type to a Datatype enum value at compile time.
macro Datatype dtype_of($Type) @local
{
	return rtti::from_type($Type);
}

<*
 Allocate typed memory on a device.
   float[] buf = dev.@alloc(float, 512);

 @require n > 0 : "Cannot allocate zero-length buffer"
*>
macro DeviceReference.@alloc(&self, $Type, usz n)
{
	char[] bytes = self.mem_alloc(dtype_of($Type), n);
	return (($Type*)bytes.ptr)[:n];
}

/// Free typed device memory.
macro void DeviceReference.@free(&self, slice)
{
	self.mem_free(as_bytes(slice));
}

/// Copy typed device memory (same device).
macro void DeviceReference.@copy(&self, src, dst)
{
	self.mem_copy(as_bytes(src), as_bytes(dst));
}

<*
 Transfer a typed host slice to a device byte buffer (HTOD).
   dev.@transfer_to_device(host_floats, tensor.value_bytes());

 @require as_bytes(src).len == dst.len : "Source and destination size mismatch"
*>
macro void DeviceReference.@transfer_to_device(&self, src, char[] dst)
{
	self.mem_transfer(as_bytes(src), dst, TransferDirection.HTOD);
}

<*
 Transfer a device byte buffer to a typed host slice (DTOH).
   dev.@transfer_to_host(tensor.value_bytes(), host_floats);

 @require src.len == as_bytes(dst).len : "Source and destination size mismatch"
*>
macro void DeviceReference.@transfer_to_host(&self, char[] src, dst)
{
	self.mem_transfer(src, as_bytes(dst), TransferDirection.DTOH);
}

/// Duplicate typed device memory (alloc + copy).
macro DeviceReference.@dupe(&self, $Type, src)
{
	char[] bytes = self.mem_dupe(dtype_of($Type), as_bytes(src));
	return (($Type*)bytes.ptr)[:src.len];
}

/// Fill typed device memory with a constant value.
macro void DeviceReference.@fill(&self, $Type, slice, double value)
{
	self.mem_fill(dtype_of($Type), as_bytes(slice), value);
}

/// Fill typed device memory with an arithmetic sequence.
macro void DeviceReference.@sequence(&self, $Type, dst, double initial, double step)
{
	self.mem_sequence(dtype_of($Type), as_bytes(dst), initial, step);
}

/// Fill typed device memory with random values.
macro void DeviceReference.@random(&self, $Type, slice, RandType rand_type, Scale scale = DEFAULT_SCALE)
{
	self.mem_random(dtype_of($Type), as_bytes(slice), rand_type, scale);
}
