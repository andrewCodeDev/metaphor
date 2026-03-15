//! Kernel Cache
//!
//! Global cache mapping kernel hash → compiled shared library function pointer.
//! Each kernel is JIT-compiled into its own .so file. Once compiled, function
//! pointers are stable for the lifetime of the process.
//!
//! C3 design: No CTensor translation layer. C3 structs use C-compatible layout
//! by default (fields in declaration order, standard alignment), so TensorCore
//! pointers are passed directly to JIT kernels. The generated kernel code
//! accesses fields at known compile-time offsets.
//!
//! (CTensor eliminated — see ffi_layout_test for proof of C3 layout stability)

module metaphor::device::host_device::kernel_cache;
import metaphor::tensor::tensor_core;
import metaphor::utils::hash_map;
import std::core::mem::allocator;
import std::thread;
import std::io;
import std::os::process;
import std::os::posix;
import libc;

// Kernel function signature

/// JIT-compiled kernel function pointer type.
///
/// TensorCore pointers directly. C3 struct layout is C-ABI compatible,
/// so generated kernel code can access fields at fixed offsets.
///
/// Arguments:
///   inputs     — array of pointers to input TensorCores
///   num_inputs — number of valid entries in inputs
///   output     — pointer to the output TensorCore
///   dims       — dimension sizes for loop bounds
///   params     — runtime parameters (scale factors, epsilon, etc.)
alias FusedKernelFn = fn void(
	TensorCore** inputs,
	usz num_inputs,
	TensorCore* output,
	ulong* dims,
	double* params,
);

// CachedKernel — a compiled kernel entry

struct CachedKernel
{
	FusedKernelFn func;
	void* lib_handle;
}

// Constants

const ZString CACHE_DIR = "/tmp/metaphor";

/// Embedded source for the real struct definitions. Written into generated
/// kernel files so the JIT compiler can dereference TensorCore pointers.
/// These read the actual source files at compile time — no separate copy.
const char[] SHAPE_SOURCE = $embed("../../tensor/shape.c3");
const char[] DENSE_CORE_SOURCE = $embed("../../tensor/dense_core.c3");
const char[] SPARSE_CORE_SOURCE = $embed("../../tensor/sparse_core.c3");
const char[] TENSOR_CORE_SOURCE = $embed("../../tensor/tensor_core.c3");
const char[] RTTI_SOURCE = $embed("../../rtti.c3");
const char[] DEVICE_REF_SOURCE = $embed("../device_reference.c3");
const char[] SYNC_SOURCE = $embed("../enabled_sync.c3");

// Global state

alias KernelCacheMap = hash_map::HashMap{ulong, CachedKernel};

allocator::LibcAllocator libc_alloc;
Mutex cache_mutex;
KernelCacheMap kernel_cache;
ulong cache_hits = 0;
ulong cache_misses = 0;
ulong cache_compiles = 0;
bool cache_initialized = false;

// Public API

faultdef CACHE_NOT_INITIALIZED, KERNEL_NOT_FOUND;

fn void init_cache_mutex() @init @local
{
	cache_mutex.init()!!;
}

fn void initialize_kernel_cache()
{
	cache_mutex.lock()!!;
	defer cache_mutex.unlock();
	if (!cache_initialized)
	{
		kernel_cache.init(&libc_alloc);
		posix::mkdir(CACHE_DIR, 0o755);
		cache_initialized = true;
	}
}

fn CachedKernel? lookup_cached_kernel(ulong hash)
{
	cache_mutex.lock()!!;
	defer cache_mutex.unlock();
	if (try cached = kernel_cache.get(hash))
	{
		cache_hits++;
		return cached;
	}
	cache_misses++;
	return KERNEL_NOT_FOUND~;
}

/// Compile kernel source into a shared library and cache it.
/// Uses double-checked locking for in-process safety and exclusive
/// file creation for cross-process safety.
fn CachedKernel? compile_and_cache(ulong hash, ZString func_name, char[] source)
{
	// Check cache under lock first (double-check pattern).
	{
		cache_mutex.lock()!!;
		defer cache_mutex.unlock();
		if (try existing = kernel_cache.get(hash))
		{
			return existing;
		}
	}

	// All compilation work happens outside the lock.
	// File I/O, subprocess calls, and dlopen can panic/fail safely
	// without leaving the mutex locked.

	// Build file paths.
	char[128] c3_path_buf;
	String c3_path = string::bformat(&c3_path_buf, "%s/%x.c3", CACHE_DIR, hash);

	char[128] so_path_buf;
	String so_path = string::bformat(&so_path_buf, "%s/%x.so", CACHE_DIR, hash);

	char[128] out_base_buf;
	String out_base = string::bformat(&out_base_buf, "%s/%x", CACHE_DIR, hash);

	// Write kernel source file.
	io::File? maybe_file = io::file::open(c3_path, "wx");
	bool we_own_compilation;
	if (try file = maybe_file)
	{
		defer (void)file.close();
		file.write((char[])"// JIT Kernel (auto-generated)\n")!!;
		file.write(SHAPE_SOURCE)!!;
		file.write((char[])"\n")!!;
		file.write(DENSE_CORE_SOURCE)!!;
		file.write((char[])"\n")!!;
		file.write(SPARSE_CORE_SOURCE)!!;
		file.write((char[])"\n")!!;
		file.write(RTTI_SOURCE)!!;
		file.write((char[])"\n")!!;
		file.write(DEVICE_REF_SOURCE)!!;
		file.write((char[])"\n")!!;
		file.write(SYNC_SOURCE)!!;
		file.write((char[])"\n")!!;
		file.write(TENSOR_CORE_SOURCE)!!;
		file.write((char[])"\n\n")!!;
		file.write(source)!!;
		we_own_compilation = true;
	}
	else
	{
		we_own_compilation = false;
	}

	if (we_own_compilation)
	{
		io::eprintfn("JIT compiling kernel %x ...", hash);
		SubProcess proc = process::create(
			{ "c3c", "dynamic-lib", "-O3", "-o", out_base, c3_path },
			{ .inherit_environment = true },
		)!;

		CInt exit_code = proc.join()!;
		proc.destroy();

		if (exit_code != 0)
		{
			io::eprintfn("JIT compile failed (exit %d): c3c dynamic-lib -O3 -o %s %s",
				exit_code, out_base, c3_path);
			unreachable("Kernel compilation failed");
		}
	}
	else
	{
		bool found = false;
		for (int attempts = 0; attempts < 300; attempts++)
		{
			if (io::file::exists(so_path)) { found = true; break; }
			thread::sleep_ms(100);
		}
		assert(found, "Kernel compilation timed out waiting for .so from another process");
	}

	// dlopen the .so
	char[128] so_z_buf2;
	ZString so_z2 = (ZString)string::bformat(&so_z_buf2, "%s/%x.so", CACHE_DIR, hash).ptr;
	void* lib_handle = libc::dlopen(so_z2, libc::RTLD_NOW);
	if (lib_handle == null)
	{
		unreachable("dlopen failed for kernel .so");
	}

	// dlsym — C3 mangles as "module__func_name"
	char[48] mangled_buf;
	mangled_buf[0] = 'j';
	mangled_buf[1] = 'i';
	mangled_buf[2] = 't';
	mangled_buf[3] = '_';
	mangled_buf[4] = '_';
	usz mpos = 5;
	usz fi = 0;
	while (func_name[fi] != 0 && mpos < mangled_buf.len - 1)
	{
		mangled_buf[mpos] = func_name[fi];
		mpos++;
		fi++;
	}
	mangled_buf[mpos] = 0;
	ZString mangled_name = (ZString)&mangled_buf;

	void* sym = libc::dlsym(lib_handle, mangled_name);
	if (sym == null)
	{
		libc::dlclose(lib_handle);
		unreachable("dlsym failed for kernel function");
	}

	CachedKernel cached = {
		.func = (FusedKernelFn)sym,
		.lib_handle = lib_handle,
	};

	// Insert into cache under lock.
	cache_mutex.lock()!!;
	kernel_cache.set(hash, cached);
	cache_compiles++;
	cache_mutex.unlock();

	return cached;
}

fn void record_kernel_compilation()
{
	cache_mutex.lock()!!;
	defer cache_mutex.unlock();
	cache_compiles++;
}

fn void shutdown_kernel_cache() @finalizer
{
	if (!cache_initialized) return;
	kernel_cache.free();
	cache_initialized = false;
	// Don't destroy mutex at process exit — other finalizers may still
	// be running and it may be in an undefined state. OS reclaims it.
}

// Cache statistics

struct CacheStats
{
	ulong hits;
	ulong misses;
	ulong compiles;
	usz entries;
}

fn CacheStats get_cache_statistics()
{
	cache_mutex.lock()!!;
	defer cache_mutex.unlock();
	return {
		.hits = cache_hits,
		.misses = cache_misses,
		.compiles = cache_compiles,
		.entries = kernel_cache.len(),
	};
}

fn void reset_cache_statistics()
{
	cache_mutex.lock()!!;
	defer cache_mutex.unlock();
	cache_hits = 0;
	cache_misses = 0;
	cache_compiles = 0;
}
