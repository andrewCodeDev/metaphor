//! Operation Cache Unit Tests
//!
//! Tests the OpCache data structure directly: lookup, register, clear,
//! hit/miss tracking, and key discrimination (different ops, inputs, params).
//!

module metaphor::testing::graph::test_op_cache;
import metaphor::graph::op_cache;
import metaphor::tensor::tensor_core;
import metaphor::device::common::operations;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::tensor;
import metaphor::rtti;
import std::core::mem::allocator;

// ============================================================================
// Category 1: Basic Cache Operations
// ============================================================================

fn void test_cache_miss_returns_null() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Lookup should miss (nothing registered)
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0);

	assert(result == null, "expected cache miss");
}

fn void test_cache_hit_returns_tensor() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Register (a + b) -> c
	cache.register(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0, c.core());

	// Lookup should hit
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0);

	assert(result == c.core(), "expected cache hit returning registered tensor");
}

fn void test_same_inputs_different_op_is_miss() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Register as EINSUM
	cache.register(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0, c.core());

	// Lookup as MAP — different tag, should miss
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core(), b.core() }, operations::OpTag.MAP, 0);

	assert(result == null, "expected cache miss for different op tag");
}

fn void test_same_op_different_inputs_is_miss() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;
	Tensor d = tensor::constant(F32, dev, &dims, 4.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Register (a + b) -> d
	cache.register(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0, d.core());

	// Lookup (a + c) — different input, should miss
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core(), c.core() }, operations::OpTag.EINSUM, 0);

	assert(result == null, "expected cache miss for different inputs");
}

// ============================================================================
// Category 2: Input Identity
// ============================================================================

fn void test_different_input_order_is_miss() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Register (a, b)
	cache.register(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0, c.core());

	// Lookup (b, a) — swapped order, should miss (non-commutative)
	tensor_core::TensorCore* result = cache.lookup(
		{ b.core(), a.core() }, operations::OpTag.EINSUM, 0);

	assert(result == null, "expected cache miss for swapped input order");
}

fn void test_same_tensor_twice_is_hit() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 2.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Register (a, a)
	cache.register(
		{ a.core(), a.core() }, operations::OpTag.EINSUM, 0, c.core());

	// Lookup (a, a) — same tensor twice, should hit
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core(), a.core() }, operations::OpTag.EINSUM, 0);

	assert(result == c.core(), "expected cache hit for (a, a)");
}

// ============================================================================
// Category 3: Operation Parameters
// ============================================================================

fn void test_different_params_is_miss() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[2] dims = { 2, 3 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 2.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	ulong axis0_hash = 0;
	ulong axis1_hash = 1;

	// Register reduce(a, axis=0)
	cache.register(
		{ a.core() }, operations::OpTag.REDUCE, axis0_hash, c.core());

	// Lookup reduce(a, axis=1) — different params, should miss
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core() }, operations::OpTag.REDUCE, axis1_hash);

	assert(result == null, "expected cache miss for different params");
}

fn void test_same_params_is_hit() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[2] dims = { 2, 3 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 2.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	ulong axis0_hash = 42;

	// Register reduce(a, axis=0)
	cache.register(
		{ a.core() }, operations::OpTag.REDUCE, axis0_hash, c.core());

	// Lookup reduce(a, axis=0) — same params, should hit
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core() }, operations::OpTag.REDUCE, axis0_hash);

	assert(result == c.core(), "expected cache hit for same params");
}

// ============================================================================
// Category 4: Statistics
// ============================================================================

fn void test_stats_tracks_hits_misses() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Miss
	cache.lookup({ a.core(), b.core() }, operations::OpTag.EINSUM, 0);

	// Register
	cache.register(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0, c.core());

	// Two hits
	cache.lookup({ a.core(), b.core() }, operations::OpTag.EINSUM, 0);
	cache.lookup({ a.core(), b.core() }, operations::OpTag.EINSUM, 0);

	assert(cache.hits == 2, "expected 2 hits");
	assert(cache.misses == 1, "expected 1 miss");
}

// ============================================================================
// Category 5: Clear
// ============================================================================

fn void test_clear_resets_cache() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	ulong[1] dims = { 2 };
	Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
	Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
	Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

	allocator::LibcAllocator alloc;
	op_cache::OpCache cache;
	cache.init(&alloc);
	defer cache.free();

	// Register and verify hit
	cache.register(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0, c.core());
	tensor_core::TensorCore* result = cache.lookup(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0);
	assert(result == c.core(), "expected hit before clear");

	// Clear cache
	cache.clear();

	// Should miss after clear
	tensor_core::TensorCore* result2 = cache.lookup(
		{ a.core(), b.core() }, operations::OpTag.EINSUM, 0);
	assert(result2 == null, "expected miss after clear");
	assert(cache.hits == 0, "expected hits reset after clear");
	assert(cache.misses == 1, "expected 1 miss after clear");
}
