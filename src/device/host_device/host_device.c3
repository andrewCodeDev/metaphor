//! Host Device
//!
//! CPU-based device implementation. Provides memory management,
//! operation dispatch, and scheduling for host-accessible memory.
//!
//! Memory allocated by HostDevice is safe to dereference directly
//! from user code, unlike GPU devices where memory resides in
//! separate address spaces.
//!

module metaphor::device::host_device;
import metaphor::device::host_device::caching_allocator;
import metaphor::device::host_device::compile;
import metaphor::device::host_device::kernel_cache;
import metaphor::device::common::jit_cache;
import metaphor::device::device_reference;
import metaphor::device::sync;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::cas;
import metaphor::rtti;
import metaphor::utils::sampling;
import std::math;
import std::core::mem;

// HostDevice

struct HostDevice (Device)
{
	CachingAllocator cache;
	jit_cache::JitCache{kernel_cache::CachedKernel} jit_kernel_cache;
}

// Lifecycle

// TODO: Make CachingAllocator lazily initializable so create() doesn't need to assert.
fn HostDevice create(usz pool_size = 1024 * 1024 * 1024) {
	HostDevice dev;
	dev.cache.init(pool_size)!!;
	dev.jit_kernel_cache.init();
	return dev;
}

fn void HostDevice.deinit(&self)
{
	self.jit_kernel_cache.free();
	self.cache.deinit();
}

/// Get a DeviceReference for this device.
fn DeviceReference HostDevice.reference(&self)
{
	return { .device = self };
}

// Device interface implementation

fn ZString HostDevice.name(&self) @dynamic
{
	return "host";
}

fn char[] HostDevice.mem_alloc(&self, Datatype dtype, usz n) @dynamic
{
	return self.cache.alloc(dtype, n)!!;
}

fn void HostDevice.mem_free(&self, char[] slice) @dynamic
{
	self.cache.free(slice);
}

fn void HostDevice.mem_copy(&self, char[] src, char[] dst) @dynamic
{
	dst[..] = src[..];
}

fn void HostDevice.mem_transfer(&self, char[] src, char[] dst, TransferDirection dir) @dynamic
{
	// Host device: all transfers are just memcpy
	dst[..] = src[..];
}

fn void HostDevice.sync(&self) @dynamic
{
	// Synchronous device — nothing to wait for
}

fn usz HostDevice.mem_used(&self) @dynamic
{
	return self.cache.mem_used();
}

fn usz HostDevice.mem_reserved(&self) @dynamic
{
	return self.cache.mem_reserved();
}

fn usz HostDevice.mem_peak(&self) @dynamic
{
	return self.cache.mem_peak();
}

fn void HostDevice.execute(&self, any operation, device_reference::TensorCorePtr[] sources, device_reference::TensorCorePtr target) @dynamic
{
	Executable e = (Executable)operation;
	e.execute(sources, target);
}

fn void HostDevice.prepare_execute(&self, device_reference::TensorCorePtr core) @dynamic
{
	// TODO: allocate sync counter from arena when sync is fully wired
}

fn void HostDevice.finalize_execute(&self, device_reference::TensorCorePtr core) @dynamic
{
	// No-op for synchronous host device
}

fn void HostDevice.mem_random(&self, Datatype dtype, char[] slice, device_reference::RandType rand_type, device_reference::Scale scale = device_reference::DEFAULT_SCALE) @dynamic
{
	sampling::Prng* rng = sampling::get_rng();
	switch (dtype)
	{
		case Datatype.F32:
			float[] data = ((float*)slice.ptr)[:slice.len / float.sizeof];
			sampling::fill_random(data, rand_type, rng, scale);
		case Datatype.F64:
			double[] data = ((double*)slice.ptr)[:slice.len / double.sizeof];
			float s = (float)scale.resolve();
			switch (rand_type)
			{
				case RandType.UNIFORM:
					for (usz i = 0; i < data.len; i++)
					{
						data[i] = (double)(rng.float01() * s);
					}
				case RandType.NORMAL:
					usz i = 0;
					while (i < data.len)
					{
						float u1 = rng.float01();
						float u2 = rng.float01();
						if (u1 < 1e-10) u1 = 1e-10;
						float r = math::sqrt(-2.0f * math::ln(u1));
						float theta = 2.0f * (float)math::PI * u2;
						data[i] = (double)(r * math::cos(theta) * s);
						if (i + 1 < data.len)
						{
							data[i + 1] = (double)(r * math::sin(theta) * s);
						}
						i += 2;
					}
			}
		default:
			unreachable("mem_random: only float types supported");
	}
}

fn void HostDevice.mem_fill(&self, Datatype dtype, char[] slice, double value) @dynamic
{
	self.mem_sequence(dtype, slice, value, 0.0);
}

fn char[] HostDevice.mem_dupe(&self, Datatype dtype, char[] src) @dynamic
{
	if (src.len == 0) { return {}; }
	char[] dst = self.cache.alloc(dtype, src.len / dtype.size_of())!!;
	dst[..] = src[..];
	return dst;
}

fn void HostDevice.mem_sequence(&self, Datatype dtype, char[] dst, double initial, double step) @dynamic
{
	switch (dtype)
	{
		case Datatype.F32:
			float[] data = ((float*)dst.ptr)[:dst.len / float.sizeof];
			float init_f = (float)initial;
			float step_f = (float)step;
			for (usz i = 0; i < data.len; i++)
			{
				data[i] = init_f + (float)i * step_f;
			}
		case Datatype.F64:
			double[] data = ((double*)dst.ptr)[:dst.len / double.sizeof];
			for (usz i = 0; i < data.len; i++)
			{
				data[i] = initial + (double)i * step;
			}
		default:
			unreachable("mem_sequence: unsupported dtype");
	}
}

fn void HostDevice.mem_cast(&self, Datatype src_dtype, Datatype dst_dtype, char[] src, char[] dst) @dynamic
{
	switch (src_dtype)
	{
		case Datatype.F32: cast_from_f32(dst_dtype, src, dst);
		case Datatype.F64: cast_from_f64(dst_dtype, src, dst);
		default: unreachable("mem_cast: unsupported source dtype");
	}
}

fn void HostDevice.compile(&self, void* graph_ptr, void* entry_ptr) @dynamic
{
	graph::Graph* g = (graph::Graph*)graph_ptr;
	cas::IREntry* entry = (cas::IREntry*)entry_ptr;
	compile::compile(self, g, entry);
}

fn uint HostDevice.sample(&self, Datatype dtype, char[] logits, usz vocab_size, char[] context, usz context_len, void* config_ptr) @dynamic
{
	sampling::SamplingConfig* config = (sampling::SamplingConfig*)config_ptr;
	switch (dtype)
	{
		case Datatype.F32:
			float[] flogits = ((float*)logits.ptr)[:vocab_size];
			uint[] ctx;
			if (context.len > 0 && context_len > 0)
			{
				ctx = ((uint*)context.ptr)[:context_len];
			}
			sampling::Sampler sampler;
			sampler.init((uint)vocab_size, *config);
			uint result = sampler.sample_with_context(flogits, ctx);
			sampler.deinit();
			return result;
		default:
			unreachable("sample: only f32 logits supported on host");
	}
}

fn uint HostDevice.argmax(&self, Datatype dtype, char[] logits, usz vocab_size) @dynamic
{
	switch (dtype)
	{
		case Datatype.F32:
			float[] flogits = ((float*)logits.ptr)[:vocab_size];
			return sampling::argmax(flogits);
		case Datatype.F64:
			// Argmax directly on f64
			double[] dlogits = ((double*)logits.ptr)[:vocab_size];
			uint max_idx = 0;
			double max_val = dlogits[0];
			for (usz i = 1; i < vocab_size; i++)
			{
				if (dlogits[i] > max_val)
				{
					max_val = dlogits[i];
					max_idx = (uint)i;
				}
			}
			return max_idx;
		default:
			unreachable("argmax: only f32/f64 logits supported");
	}
}

// Cast helpers

fn void cast_from_f32(Datatype dst_dtype, char[] src, char[] dst) @private
{
	float[] src_vals = ((float*)src.ptr)[:src.len / float.sizeof];
	switch (dst_dtype)
	{
		case Datatype.F64:
			double[] dst_vals = ((double*)dst.ptr)[:dst.len / double.sizeof];
			for (usz i = 0; i < src_vals.len; i++) { dst_vals[i] = (double)src_vals[i]; }
		case Datatype.F32:
			dst[..] = src[..];
		default:
			unreachable("cast_from_f32: unsupported target dtype");
	}
}

fn void cast_from_f64(Datatype dst_dtype, char[] src, char[] dst) @private
{
	double[] src_vals = ((double*)src.ptr)[:src.len / double.sizeof];
	switch (dst_dtype)
	{
		case Datatype.F32:
			float[] dst_vals = ((float*)dst.ptr)[:dst.len / float.sizeof];
			for (usz i = 0; i < src_vals.len; i++) { dst_vals[i] = (float)src_vals[i]; }
		case Datatype.F64:
			dst[..] = src[..];
		default:
			unreachable("cast_from_f64: unsupported target dtype");
	}
}
