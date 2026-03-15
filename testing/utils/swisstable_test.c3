module metaphor::testing::utils::swisstable_test;
import metaphor::utils::swisstable;
import metaphor::utils::hash_map;
import metaphor::utils::hash_set;
import std::core::test;
import std::core::mem;

// ============================================================================
// Test key type
// ============================================================================

struct TestKey
{
	int value;
}

fn uint TestKey.hash(&self)
{
	uint h = (uint)self.value;
	h = (h ^ (h >> 16)) * 0x45d9f3b;
	h = (h ^ (h >> 16)) * 0x45d9f3b;
	h = h ^ (h >> 16);
	return h;
}

fn bool TestKey.equals(self, TestKey other) @operator(==)
{
	return self.value == other.value;
}

// ============================================================================
// Type aliases
// ============================================================================

alias Table = swisstable::SwissTable {TestKey, int};
alias Map = hash_map::HashMap {TestKey, int};
alias Set = hash_set::HashSet {TestKey};

fn TestKey key(int v) @inline => { .value = v };

// ============================================================================
// SwissTable: basic operations
// ============================================================================

fn void test_empty_table() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	test::eq(t.len(), (usz)0);
	test::@check(t.is_empty());
	test::@error(t.get_ref(key(42)));
}

fn void test_single_insert_and_get() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	bool was_update = t.set(key(1), 100);
	test::@check(!was_update);
	test::eq(t.len(), (usz)1);
	test::eq(t.get(key(1))!!, 100);
}

fn void test_update_existing_key() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(1), 100);
	bool was_update = t.set(key(1), 200);
	test::@check(was_update);
	test::eq(t.len(), (usz)1);
	test::eq(t.get(key(1))!!, 200);
}

fn void test_get_missing_key() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(1), 100);
	test::@error(t.get_ref(key(2)));
	test::@error(t.get(key(999)));
}

fn void test_remove() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(10), 1000);
	t.set(key(20), 2000);
	test::eq(t.len(), (usz)2);

	t.remove(key(10))!!;
	test::eq(t.len(), (usz)1);
	test::@check(!t.has_key(key(10)));
	test::@check(t.has_key(key(20)));
}

fn void test_remove_missing() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(1), 100);
	test::@error(t.remove(key(999)));
	test::eq(t.len(), (usz)1);
}

fn void test_has_key() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(5), 50);
	test::@check(t.has_key(key(5)));
	test::@check(!t.has_key(key(6)));
}

fn void test_clear() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 20; i++)
	{
		t.set(key(i), i * 10);
	}
	test::eq(t.len(), (usz)20);

	t.clear();
	test::eq(t.len(), (usz)0);
	test::@check(t.is_empty());

	t.set(key(42), 420);
	test::eq(t.len(), (usz)1);
	test::eq(t.get(key(42))!!, 420);
}

// ============================================================================
// Growth and rehashing
// ============================================================================

fn void test_grow_from_empty() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 100; i++)
	{
		t.set(key(i), i);
	}
	test::eq(t.len(), (usz)100);

	for (int i = 0; i < 100; i++)
	{
		test::eq(t.get(key(i))!!, i);
	}
}

fn void test_grow_preserves_all_entries() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 1000; i++)
	{
		t.set(key(i), i * 7);
	}
	test::eq(t.len(), (usz)1000);

	for (int i = 0; i < 1000; i++)
	{
		test::eq(t.get(key(i))!!, i * 7);
	}

	for (int i = 1000; i < 1100; i++)
	{
		test::@check(!t.has_key(key(i)));
	}
}

fn void test_preallocated_capacity() @test
{
	Table t;
	t.init(mem, 256);
	defer t.free();

	test::@check(t.num_buckets() >= 256);

	for (int i = 0; i < 200; i++)
	{
		t.set(key(i), i);
	}
	test::eq(t.len(), (usz)200);
}

// ============================================================================
// Collision handling
// ============================================================================

fn void test_collision_handling() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 500; i++)
	{
		t.set(key(i), i);
	}

	for (int i = 0; i < 500; i++)
	{
		test::eq(t.get(key(i))!!, i);
	}
}

// ============================================================================
// Tombstone handling
// ============================================================================

fn void test_tombstone_reuse() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 50; i++)
	{
		t.set(key(i), i);
	}

	for (int i = 0; i < 50; i += 2)
	{
		t.remove(key(i))!!;
	}
	test::eq(t.len(), (usz)25);

	for (int i = 1; i < 50; i += 2)
	{
		test::eq(t.get(key(i))!!, i);
	}

	for (int i = 0; i < 50; i += 2)
	{
		t.set(key(i), i + 1000);
	}
	test::eq(t.len(), (usz)50);

	for (int i = 0; i < 50; i++)
	{
		int expected = (i % 2 == 0) ? i + 1000 : i;
		test::eq(t.get(key(i))!!, expected);
	}
}

fn void test_remove_all_then_reinsert() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 30; i++)
	{
		t.set(key(i), i);
	}

	for (int i = 0; i < 30; i++)
	{
		t.remove(key(i))!!;
	}
	test::eq(t.len(), (usz)0);

	for (int i = 0; i < 30; i++)
	{
		t.set(key(i), i * 2);
	}
	test::eq(t.len(), (usz)30);

	for (int i = 0; i < 30; i++)
	{
		test::eq(t.get(key(i))!!, i * 2);
	}
}

// ============================================================================
// Iterator
// ============================================================================

fn void test_iterator() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	for (int i = 0; i < 10; i++)
	{
		t.set(key(i), i * 10);
	}

	usz count = 0;
	int sum = 0;
	swisstable::SwissTableIterator {TestKey, int} iter = t.iter();
	while (try entry = iter.next())
	{
		count++;
		sum += entry.value;
	}
	test::eq(count, (usz)10);
	test::eq(sum, 450);
}

fn void test_iterator_empty_table() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	swisstable::SwissTableIterator {TestKey, int} iter = t.iter();
	test::@error(iter.next());
}

// ============================================================================
// HashMap wrapper
// ============================================================================

fn void test_hashmap_basic() @test
{
	Map m;
	m.init(mem);
	defer m.free();

	m.set(key(1), 10);
	m.set(key(2), 20);
	m.set(key(3), 30);

	test::eq(m.len(), (usz)3);
	test::eq(m.get(key(1))!!, 10);
	test::eq(m.get(key(2))!!, 20);
	test::eq(m.get(key(3))!!, 30);
	test::@check(!m.has_key(key(4)));
}

fn void test_hashmap_update() @test
{
	Map m;
	m.init(mem);
	defer m.free();

	test::@check(!m.set(key(1), 10));
	test::@check(m.set(key(1), 99));
	test::eq(m.get(key(1))!!, 99);
}

fn void test_hashmap_remove() @test
{
	Map m;
	m.init(mem);
	defer m.free();

	m.set(key(5), 50);
	m.set(key(6), 60);

	m.remove(key(5))!!;
	test::eq(m.len(), (usz)1);
	test::@error(m.get(key(5)));
}

fn void test_hashmap_get_ref() @test
{
	Map m;
	m.init(mem);
	defer m.free();

	m.set(key(42), 100);
	int* ptr = m.get_ref(key(42))!!;
	test::eq(*ptr, 100);

	*ptr = 200;
	test::eq(m.get(key(42))!!, 200);
}

fn void test_hashmap_many_entries() @test
{
	Map m;
	m.init(mem);
	defer m.free();

	for (int i = 0; i < 10000; i++)
	{
		m.set(key(i), i);
	}
	test::eq(m.len(), (usz)10000);

	for (int i = 0; i < 10000; i++)
	{
		test::eq(m.get(key(i))!!, i);
	}
}

// ============================================================================
// HashSet wrapper
// ============================================================================

fn void test_hashset_basic() @test
{
	Set s;
	s.init(mem);
	defer s.free();

	test::@check(!s.add(key(1)));
	test::@check(!s.add(key(2)));
	test::@check(!s.add(key(3)));
	test::eq(s.len(), (usz)3);

	test::@check(s.contains(key(1)));
	test::@check(s.contains(key(2)));
	test::@check(s.contains(key(3)));
	test::@check(!s.contains(key(4)));
}

fn void test_hashset_add_duplicate() @test
{
	Set s;
	s.init(mem);
	defer s.free();

	test::@check(!s.add(key(1)));
	test::@check(s.add(key(1)));
	test::eq(s.len(), (usz)1);
}

fn void test_hashset_remove() @test
{
	Set s;
	s.init(mem);
	defer s.free();

	s.add(key(10));
	s.add(key(20));

	test::@check(s.remove(key(10)));
	test::eq(s.len(), (usz)1);
	test::@check(!s.contains(key(10)));
	test::@check(s.contains(key(20)));
}

fn void test_hashset_remove_missing() @test
{
	Set s;
	s.init(mem);
	defer s.free();

	s.add(key(1));
	test::@check(!s.remove(key(999)));
	test::eq(s.len(), (usz)1);
}

fn void test_hashset_add_all() @test
{
	Set a;
	a.init(mem);
	defer a.free();

	Set b;
	b.init(mem);
	defer b.free();

	a.add(key(1));
	a.add(key(2));

	b.add(key(2));
	b.add(key(3));
	b.add(key(4));

	a.add_all(&b);

	test::eq(a.len(), (usz)4);
	test::@check(a.contains(key(1)));
	test::@check(a.contains(key(2)));
	test::@check(a.contains(key(3)));
	test::@check(a.contains(key(4)));
}

// ============================================================================
// Edge cases
// ============================================================================

fn void test_zero_value() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(0), 0);
	test::eq(t.get(key(0))!!, 0);
	test::@check(t.has_key(key(0)));
}

fn void test_negative_keys() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(-1), -100);
	t.set(key(-999), -9990);

	test::eq(t.get(key(-1))!!, -100);
	test::eq(t.get(key(-999))!!, -9990);
}

fn void test_large_values() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	t.set(key(int.max), int.max);
	t.set(key(int.min), int.min);

	test::eq(t.get(key(int.max))!!, int.max);
	test::eq(t.get(key(int.min))!!, int.min);
}

fn void test_init_free_init() @test
{
	Table t;
	t.init(mem);
	t.set(key(1), 10);
	t.free();

	t.init(mem);
	defer t.free();
	test::@check(t.is_empty());
	t.set(key(2), 20);
	test::eq(t.get(key(2))!!, 20);
}

// ============================================================================
// SwissTable vs std HashMap — grueling 1000-entry comparison
// ============================================================================

import std::collections::map;

alias StdMap = map::HashMap {TestKey, int};

/// Simple LCG pseudo-random number generator (deterministic, no imports needed).
/// Returns next state; use lower bits for the value.
fn uint lcg_next(uint state) @inline => state * 1664525 + 1013904223;

fn void test_swisstable_vs_std_1000_entries() @test
{
	Map swiss;
	swiss.init(mem);
	defer swiss.free();

	StdMap std_map;
	std_map.init(mem);
	defer std_map.free();

	// Generate 1000 pseudo-random keys and 1000 pseudo-random values
	const usz N = 1000;
	int[N] keys;
	int[N] values;

	uint rng = 0xDEADBEEF;  // seed
	for (usz i = 0; i < N; i++)
	{
		rng = lcg_next(rng);
		keys[i] = (int)rng;
	}
	for (usz i = 0; i < N; i++)
	{
		rng = lcg_next(rng);
		values[i] = (int)rng;
	}

	// Load both maps with the same key-value pairs
	for (usz i = 0; i < N; i++)
	{
		swiss.set(key(keys[i]), values[i]);
		std_map.set(key(keys[i]), values[i]);
	}

	// Verify lengths match
	test::eq(swiss.len(), std_map.len());

	// Verify every key returns the same value from both maps
	for (usz i = 0; i < N; i++)
	{
		TestKey k = key(keys[i]);
		int swiss_val = swiss.get(k)!!;
		int std_val = std_map.get(k)!!;
		test::eq(swiss_val, std_val);
	}

	// Verify both maps agree on keys that were NOT inserted
	for (int i = 0; i < 100; i++)
	{
		rng = lcg_next(rng);
		TestKey probe = key((int)rng);

		bool swiss_has = swiss.has_key(probe);
		bool std_has = std_map.has_key(probe);
		test::eq(swiss_has, std_has);
	}
}

fn void test_swisstable_vs_std_overwrite_consistency() @test
{
	Map swiss;
	swiss.init(mem);
	defer swiss.free();

	StdMap std_map;
	std_map.init(mem);
	defer std_map.free();

	const usz N = 1000;
	int[N] keys;
	int[N] values_v1;
	int[N] values_v2;

	uint rng = 0xCAFEBABE;
	for (usz i = 0; i < N; i++)
	{
		rng = lcg_next(rng);
		keys[i] = (int)(rng % 500);  // Only 500 unique keys → forces overwrites
	}
	for (usz i = 0; i < N; i++)
	{
		rng = lcg_next(rng);
		values_v1[i] = (int)rng;
	}
	for (usz i = 0; i < N; i++)
	{
		rng = lcg_next(rng);
		values_v2[i] = (int)rng;
	}

	// Insert first batch
	for (usz i = 0; i < N; i++)
	{
		swiss.set(key(keys[i]), values_v1[i]);
		std_map.set(key(keys[i]), values_v1[i]);
	}
	test::eq(swiss.len(), std_map.len());

	// Overwrite with second batch
	for (usz i = 0; i < N; i++)
	{
		swiss.set(key(keys[i]), values_v2[i]);
		std_map.set(key(keys[i]), values_v2[i]);
	}
	test::eq(swiss.len(), std_map.len());

	// Verify all values match after overwrites
	for (usz i = 0; i < N; i++)
	{
		TestKey k = key(keys[i]);
		test::eq(swiss.get(k)!!, std_map.get(k)!!);
	}
}

fn void test_swisstable_vs_std_remove_and_reinsert() @test
{
	Map swiss;
	swiss.init(mem);
	defer swiss.free();

	StdMap std_map;
	std_map.init(mem);
	defer std_map.free();

	const usz N = 1000;
	int[N] keys;

	uint rng = 0x12345678;
	for (usz i = 0; i < N; i++)
	{
		rng = lcg_next(rng);
		keys[i] = (int)rng;
	}

	// Insert all
	for (usz i = 0; i < N; i++)
	{
		swiss.set(key(keys[i]), (int)i);
		std_map.set(key(keys[i]), (int)i);
	}

	// Remove every other entry
	for (usz i = 0; i < N; i += 2)
	{
		TestKey k = key(keys[i]);
		if (swiss.has_key(k))
		{
			swiss.remove(k)!!;
			std_map.remove(k)!!;
		}
	}
	test::eq(swiss.len(), std_map.len());

	// Re-insert removed entries with new values
	for (usz i = 0; i < N; i += 2)
	{
		swiss.set(key(keys[i]), (int)(i + 10000));
		std_map.set(key(keys[i]), (int)(i + 10000));
	}
	test::eq(swiss.len(), std_map.len());

	// Verify every key matches
	for (usz i = 0; i < N; i++)
	{
		TestKey k = key(keys[i]);
		test::eq(swiss.get(k)!!, std_map.get(k)!!);
	}
}

// ============================================================================
// Test: set() correctly clobbers after tombstones (no duplicate keys)
// ============================================================================

fn void test_set_clobber_after_tombstones() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	// Step 1: Insert keys 0..99
	for (int i = 0; i < 100; i++)
	{
		t.set(key(i), i);
	}
	test::eq(t.len(), (usz)100);

	// Step 2: Remove every other key, creating tombstones
	for (int i = 0; i < 100; i += 2)
	{
		t.remove(key(i))!!;
	}
	test::eq(t.len(), (usz)50);

	// Step 3: Re-set the remaining keys with new values.
	// If set() fails to find them past tombstones, it will
	// insert duplicates, inflating len() beyond 50.
	for (int i = 1; i < 100; i += 2)
	{
		bool was_update = t.set(key(i), i + 5000);
		// Must be an update, not a new insert
		test::eq(was_update, true);
	}
	// Length must still be 50 — no duplicates
	test::eq(t.len(), (usz)50);

	// Step 4: Count entries via iteration — must match len()
	usz iter_count = 0;
	swisstable::SwissTableIterator{TestKey, int} it = t.iter();
	while (try entry = it.next())
	{
		iter_count++;
	}
	test::eq(iter_count, t.len());

	// Step 5: Verify values are the updated ones
	for (int i = 1; i < 100; i += 2)
	{
		test::eq(t.get(key(i))!!, i + 5000);
	}
}

// ============================================================================
// Test: heavy set/remove churn — detect duplicates via len vs iteration
// ============================================================================

fn void test_heavy_churn_no_duplicates() @test
{
	Table t;
	t.init(mem);
	defer t.free();

	std::collections::map::HashMap{TestKey, int} ref;
	ref.init(mem);
	defer ref.free();

	// Simulate heavy churn: insert, remove, re-set repeatedly
	for (int round = 0; round < 10; round++)
	{
		int base = round * 20;

		// Insert 20 keys
		for (int i = 0; i < 20; i++)
		{
			t.set(key(base + i), base + i);
			ref.set(key(base + i), base + i);
		}

		// Remove half
		for (int i = 0; i < 20; i += 2)
		{
			t.remove(key(base + i))!!;
			ref.remove(key(base + i))!!;
		}

		// Re-set the surviving keys with new values
		for (int i = 1; i < 20; i += 2)
		{
			t.set(key(base + i), base + i + 9000);
			ref.set(key(base + i), base + i + 9000);
		}

		// Re-insert the removed keys
		for (int i = 0; i < 20; i += 2)
		{
			t.set(key(base + i), base + i + 8000);
			ref.set(key(base + i), base + i + 8000);
		}
	}

	// Length must match
	test::eq(t.len(), ref.len());

	// Count via iteration
	usz iter_count = 0;
	swisstable::SwissTableIterator{TestKey, int} it = t.iter();
	while (try entry = it.next())
	{
		iter_count++;
	}
	test::eq(iter_count, t.len());

	// Every key in ref must be in t with same value
	for (int round = 0; round < 10; round++)
	{
		int base = round * 20;
		for (int i = 0; i < 20; i++)
		{
			TestKey k = key(base + i);
			test::eq(t.get(k)!!, ref.get(k)!!);
		}
	}
}
