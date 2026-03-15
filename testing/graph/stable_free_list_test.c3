module metaphor::testing::graph::stable_free_list_test;
import metaphor::mem::stable_free_list;
import std::core::test;
import std::core::mem;

alias IntPool = stable_free_list::StableFreeList {int};

struct BigStruct
{
	int[16] data;
}

alias BigPool = stable_free_list::StableFreeList {BigStruct};

// ============================================================================
// Tests
// ============================================================================

fn void test_create_and_use() @test
{
	IntPool pool;
	pool.init(mem);
	defer pool.free();

	int* a = pool.create()!!;
	*a = 42;
	test::eq(*a, 42);
}

fn void test_destroy_and_reuse() @test
{
	IntPool pool;
	pool.init(mem);
	defer pool.free();

	int* a = pool.create()!!;
	*a = 42;
	pool.destroy(a);

	// Reuse: should get same address back
	int* b = pool.create()!!;
	test::eq(a, b);

	// Data survives destroy (stable property)
	test::eq(*b, 42);
}

fn void test_multiple_create_destroy() @test
{
	IntPool pool;
	pool.init(mem);
	defer pool.free();

	int* a = pool.create()!!;
	int* b = pool.create()!!;
	int* c = pool.create()!!;

	*a = 1;
	*b = 2;
	*c = 3;

	// Destroy in reverse
	pool.destroy(c);
	pool.destroy(b);

	// LIFO reuse: b address first, then c address
	int* d = pool.create()!!;
	int* e = pool.create()!!;

	test::eq(d, b);
	test::eq(e, c);
}

fn void test_big_struct() @test
{
	BigPool pool;
	pool.init(mem);
	defer pool.free();

	BigStruct* a = pool.create()!!;
	a.data[0] = 99;
	a.data[15] = 77;

	pool.destroy(a);

	BigStruct* b = pool.create()!!;
	test::eq(a, b);
	test::eq(b.data[0], 99);
	test::eq(b.data[15], 77);
}

fn void test_reset() @test
{
	IntPool pool;
	pool.init(mem);
	defer pool.free();

	int* a = pool.create()!!;
	*a = 123;

	pool.reset();

	// After reset, pool is empty — new allocation from fresh arena
	int* b = pool.create()!!;
	// b may or may not equal a (arena was reset), just verify it works
	*b = 456;
	test::eq(*b, 456);
}
