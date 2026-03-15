module metaphor::testing::device::host_device::block_pool_test;
import metaphor::device::host_device::block_pool;
import metaphor::device::host_device::caching_allocator;
import metaphor::rtti;

// ============================================================================
// BlockPool tests
// ============================================================================

fn void test_basic_alloc_free() @test
{
	BlockPool pool;
	pool.init(8192)!!;
	defer pool.deinit();

	char[] slice = pool.alloc(64)!!;
	assert(slice.len >= 64);
	pool.free(slice);
}

fn void test_small_alloc_scalar() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	// 8 bytes = single f64, should work at MIN_ORDER
	char[] slice = pool.alloc(8)!!;
	assert(slice.len >= 8);
	pool.free(slice);
}

fn void test_multiple_alloc() @test
{
	BlockPool pool;
	pool.init(8192)!!;
	defer pool.deinit();

	char[] a = pool.alloc(64)!!;
	char[] b = pool.alloc(128)!!;
	char[] c = pool.alloc(256)!!;

	assert(a.len >= 64);
	assert(b.len >= 128);
	assert(c.len >= 256);

	pool.free(a);
	pool.free(b);
	pool.free(c);
}

fn void test_alloc_returns_sufficient_size() @test
{
	BlockPool pool;
	pool.init(8192)!!;
	defer pool.deinit();

	// Odd sizes should round up to min_order multiples
	char[] s1 = pool.alloc(13)!!;
	assert(s1.len >= 13);

	char[] s2 = pool.alloc(100)!!;
	assert(s2.len >= 100);

	pool.free(s1);
	pool.free(s2);
}

fn void test_block_splitting() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	// Alloc small from large pool — forces splitting
	char[] a = pool.alloc(8)!!;
	assert(a.len >= 8);

	// Alloc another — verifies splitting left free blocks
	char[] b = pool.alloc(8)!!;
	assert(b.len >= 8);

	pool.free(a);
	pool.free(b);
}

fn void test_block_fusion() @test
{
	// Use a pool of 1024 bytes
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	// Alloc two blocks that together fill significant space
	char[] a = pool.alloc(1024)!!;
	char[] b = pool.alloc(1024)!!;

	pool.free(a);
	pool.free(b);

	// After fusion, should be able to alloc the combined size
	char[] c = pool.alloc(2048)!!;
	assert(c.len >= 2048);
	pool.free(c);
}

fn void test_free_order_independence() @test
{
	BlockPool pool;
	pool.init(8192)!!;
	defer pool.deinit();

	char[] a = pool.alloc(64)!!;
	char[] b = pool.alloc(64)!!;
	char[] c = pool.alloc(64)!!;

	// Free in non-sequential order
	pool.free(b);
	pool.free(a);
	pool.free(c);
}

fn void test_memory_reuse() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	char[] a = pool.alloc(128)!!;
	usz after_alloc = pool.mem_used();
	pool.free(a);

	char[] b = pool.alloc(128)!!;
	usz after_realloc = pool.mem_used();
	pool.free(b);

	// Memory usage should be bounded — reuse freed blocks
	assert(after_realloc <= after_alloc);
}

fn void test_zero_size_alloc() @test
{
	CachingAllocator ca;
	ca.init(4096)!!;
	defer ca.deinit();

	// Zero-size alloc via CachingAllocator returns empty slice
	char[] slice = ca.alloc(F32, 0)!!;
	assert(slice.len == 0);
}

fn void test_reset() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	char[] a = pool.alloc(256)!!;
	assert(pool.mem_used() > 0);

	pool.reset();
	assert(pool.mem_used() == 0);

	// After reset, should be able to alloc again
	char[] b = pool.alloc(256)!!;
	assert(b.len >= 256);
	pool.free(b);
}

fn void test_mem_tracking() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	assert(pool.mem_reserved() == 4096);
	assert(pool.mem_used() == 0);

	char[] a = pool.alloc(512)!!;
	assert(pool.mem_used() > 0);
	usz used = pool.mem_used();

	pool.free(a);
	assert(pool.mem_used() == 0);
}

fn void test_overflow_error() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	// Request more than pool size — should fail
	if (try pool.alloc(8192))
	{
		assert(false, "Expected overflow fault");
	}
}

fn void test_pool_exhaustion() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	// Fill the pool
	char[] a = pool.alloc(2048)!!;
	char[] b = pool.alloc(1024)!!;
	char[] c = pool.alloc(512)!!;
	char[] d = pool.alloc(256)!!;

	// Next alloc should fail (only ~256 bytes left, fragmented)
	if (try pool.alloc(512))
	{
		assert(false, "Expected exhaustion fault");
	}

	pool.free(a);
	pool.free(b);
	pool.free(c);
	pool.free(d);
}

fn void test_fusion_chain() @test
{
	BlockPool pool;
	pool.init(4096)!!;
	defer pool.deinit();

	// Alloc 4 blocks
	char[] a = pool.alloc(512)!!;
	char[] b = pool.alloc(512)!!;
	char[] c = pool.alloc(512)!!;
	char[] d = pool.alloc(512)!!;

	// Free all — should fuse back into large block
	pool.free(a);
	pool.free(b);
	pool.free(c);
	pool.free(d);

	assert(pool.mem_used() == 0);

	// Should be able to alloc nearly the full pool
	char[] big = pool.alloc(2048)!!;
	assert(big.len >= 2048);
	pool.free(big);
}

// ============================================================================
// CachingAllocator tests
// ============================================================================

fn void test_caching_allocator_basic() @test
{
	CachingAllocator ca;
	ca.init(8192)!!;
	defer ca.deinit();

	char[] slice = ca.alloc(F32, 16)!!;
	assert(slice.len >= 64); // 16 * 4 bytes
	ca.free(slice);
}

fn void test_caching_allocator_mem_tracking() @test
{
	CachingAllocator ca;
	ca.init(4096)!!;
	defer ca.deinit();

	assert(ca.mem_reserved() == 4096);
	assert(ca.mem_used() == 0);

	char[] slice = ca.alloc(F64, 8)!!; // 64 bytes
	assert(ca.mem_used() > 0);

	ca.free(slice);
	assert(ca.mem_used() == 0);
}

fn void test_caching_allocator_reset() @test
{
	CachingAllocator ca;
	ca.init(4096)!!;
	defer ca.deinit();

	char[] a = ca.alloc(F32, 32)!!;
	ca.reset();
	assert(ca.mem_used() == 0);

	char[] b = ca.alloc(F32, 32)!!;
	assert(b.len >= 128);
	ca.free(b);
}
