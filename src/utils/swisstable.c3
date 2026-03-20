// SwissTable hash map — ported from Rust's hashbrown crate
//
// Original: https://github.com/rust-lang/hashbrown
// Which is itself a Rust port of Google's SwissTable (Abseil C++ library).
//
// hashbrown is licensed under MIT OR Apache-2.0, maintained by the Rust team.
// This C3 implementation was written from scratch following the same algorithm
// (SIMD group probing, H2 metadata bytes, tombstone compaction).

module metaphor::utils::swisstable <Key, Value>;
import std::core::mem;
import std::core::mem::allocator;
import std::math;

// Constants

const usz GROUP_WIDTH = 16;

const char CTRL_EMPTY   = 0xFF;
const char CTRL_DELETED = 0x80;

const usz LOAD_FACTOR_NUMERATOR   = 7;
const usz LOAD_FACTOR_DENOMINATOR = 8;

// Entry — key-value pair stored in the data portion

struct Entry
{
	Key key;
	Value value;
}

// BitMask — iterates over set bits in a group match result

struct BitMask @local
{
	ushort mask;
}

fn bool BitMask.any_bit_set(&self) @inline => self.mask != 0;

fn usz BitMask.lowest_set_bit(&self) @inline
{
	assert(self.mask != 0, "lowest_set_bit called on empty bitmask");
	return (usz)self.mask.ctz();
}

fn void BitMask.remove_lowest_bit(&self) @inline
{
	self.mask &= self.mask - 1;
}

fn usz BitMask.trailing_zeros(&self) @inline
{
	return (usz)self.mask.ctz();
}

fn usz BitMask.leading_zeros(&self) @inline
{
	// For 16-bit mask, clz gives leading zeros from 16-bit perspective
	return (usz)self.mask.clz();
}

// Group — SIMD parallel lookup of 16 control bytes

fn BitMask match_tag(char* ctrl_ptr, char tag) @inline @local
{
	char[<GROUP_WIDTH>] group = *(char[<GROUP_WIDTH>]*)ctrl_ptr;
	char[<GROUP_WIDTH>] splat = (char[<GROUP_WIDTH>])tag;
	bool[<GROUP_WIDTH>] cmp = group.comp_eq(splat);
	return { .mask = cmp.mask_to_int() };
}

fn BitMask match_empty(char* ctrl_ptr) @inline @local
{
	return match_tag(ctrl_ptr, CTRL_EMPTY);
}

fn BitMask match_empty_or_deleted(char* ctrl_ptr) @inline @local
{
	char[<GROUP_WIDTH>] group = *(char[<GROUP_WIDTH>]*)ctrl_ptr;
	char[<GROUP_WIDTH>] high_bit = (char[<GROUP_WIDTH>])0x80;
	bool[<GROUP_WIDTH>] cmp = (group & high_bit).comp_eq(high_bit);
	return { .mask = cmp.mask_to_int() };
}

// Hash splitting

macro usz h1(uint hash) => (usz)hash;
macro char h2(uint hash) => (char)((hash >> 25) & 0x7F);

// ProbeSeq — triangular probe sequence

struct ProbeSeq @local
{
	usz pos;
	usz stride;
}

fn ProbeSeq probe_seq_new(uint hash, usz bucket_mask) @local
{
	return { .pos = h1(hash) & bucket_mask, .stride = 0 };
}

fn void ProbeSeq.move_next(&self, usz bucket_mask) @inline
{
	self.stride += GROUP_WIDTH;
	self.pos = (self.pos + self.stride) & bucket_mask;
}

// SwissTable — the core hash table

struct SwissTable
{
	Allocator allocator;
	char* ctrl;
	Entry* data;
	usz bucket_mask;
	usz items;
	usz growth_left;
}

// Lifecycle

<*
 @param [&inout] allocator : "The allocator to use"
 @param initial_capacity : "The initial capacity to reserve"
*>
fn SwissTable* SwissTable.init(&self, Allocator allocator, usz initial_capacity = 0)
{
	self.allocator = allocator;
	self.ctrl = null;
	self.data = null;
	self.bucket_mask = 0;
	self.items = 0;
	self.growth_left = 0;

	if (initial_capacity > 0)
	{
		usz num_buckets = capacity_to_buckets(initial_capacity);
		self.alloc_table(num_buckets);
	}
	return self;
}

fn SwissTable* SwissTable.tinit(&self, usz initial_capacity = 0)
{
	return self.init(tmem, initial_capacity) @inline;
}

fn void SwissTable.free(&self)
{
	if (self.ctrl == null) return;
	allocator::free(self.allocator, (void*)self.data);
	self.ctrl = null;
	self.data = null;
	self.bucket_mask = 0;
	self.items = 0;
	self.growth_left = 0;
}

// Public API — names match std::collections::map

fn usz SwissTable.len(&self) @inline @operator(len) => self.items;

fn usz SwissTable.num_buckets(&self) @inline => self.bucket_mask + 1;

fn bool SwissTable.is_empty(&self) @inline => self.items == 0;

<*
 Get a reference to a value by key.
 Returns a pointer to the value, or NOT_FOUND fault.
*>
fn Value*? SwissTable.get_ref(&self, Key key)
{
	if (self.is_empty()) return NOT_FOUND~;

	uint hash = key.hash();
	char tag = h2(hash);
	ProbeSeq seq = probe_seq_new(hash, self.bucket_mask);

	while (true)
	{
		BitMask mask = match_tag(&self.ctrl[seq.pos], tag);
		while (mask.any_bit_set())
		{
			usz bit = mask.lowest_set_bit();
			mask.remove_lowest_bit();
			usz index = (seq.pos + bit) & self.bucket_mask;

			Entry* entry = self.entry_at(index);
			if (equals(entry.key, key))
			{
				return &entry.value;
			}
		}

		if (match_empty(&self.ctrl[seq.pos]).any_bit_set())
		{
			return NOT_FOUND~;
		}

		seq.move_next(self.bucket_mask);
	}
}

<*
 Get a value by key. Returns the value or NOT_FOUND fault.
*>
fn Value? SwissTable.get(&self, Key key) @inline @operator([])
{
	return *self.get_ref(key)!;
}

<*
 Check if a key exists in the table.
*>
fn bool SwissTable.has_key(&self, Key key)
{
	return @ok(self.get_ref(key));
}

<*
 Insert or update a key-value pair.
 Returns true if the key already existed (update), false if new insert.
*>
fn bool SwissTable.set(&self, Key key, Value value) @operator([]=)
{
	if (self.ctrl == null)
	{
		usz num_buckets = capacity_to_buckets(1);
		self.alloc_table(num_buckets);
	}

	uint hash = key.hash();
	char tag = h2(hash);

	if (self.items > 0)
	{
		ProbeSeq seq = probe_seq_new(hash, self.bucket_mask);
		while (true)
		{
			BitMask mask = match_tag(&self.ctrl[seq.pos], tag);
			while (mask.any_bit_set())
			{
				usz bit = mask.lowest_set_bit();
				mask.remove_lowest_bit();
				usz index = (seq.pos + bit) & self.bucket_mask;

				Entry* entry = self.entry_at(index);
				if (equals(entry.key, key))
				{
					entry.value = value;
					return true;
				}
			}

			if (match_empty(&self.ctrl[seq.pos]).any_bit_set())
			{
				break;
			}
			seq.move_next(self.bucket_mask);
		}
	}

	if (self.growth_left == 0)
	{
		self.grow();
	}

	self.insert_entry(hash, tag, key, value);
	return false;
}

<*
 Remove a key from the table.
*>
fn void? SwissTable.remove(&self, Key key) @maydiscard
{
	if (self.is_empty()) return NOT_FOUND~;

	uint hash = key.hash();
	char tag = h2(hash);
	ProbeSeq seq = probe_seq_new(hash, self.bucket_mask);

	while (true)
	{
		BitMask mask = match_tag(&self.ctrl[seq.pos], tag);
		while (mask.any_bit_set())
		{
			usz bit = mask.lowest_set_bit();
			mask.remove_lowest_bit();
			usz index = (seq.pos + bit) & self.bucket_mask;

			Entry* entry = self.entry_at(index);
			if (equals(entry.key, key))
			{
				self.erase_at(index);
				return;
			}
		}

		if (match_empty(&self.ctrl[seq.pos]).any_bit_set())
		{
			return NOT_FOUND~;
		}
		seq.move_next(self.bucket_mask);
	}
}

<*
 Remove all entries without freeing the backing allocation.
*>
fn void SwissTable.clear(&self)
{
	if (self.ctrl == null) return;
	usz num_buckets = self.num_buckets();
	self.ctrl[:num_buckets + GROUP_WIDTH] = CTRL_EMPTY;
	self.items = 0;
	self.growth_left = bucket_mask_to_capacity(self.bucket_mask);
}

// Iterator

struct SwissTableIterator
{
	SwissTable* table;
	usz index;
}

fn SwissTableIterator SwissTable.iter(&self)
{
	SwissTableIterator it = { .table = self, .index = 0 };
	if (self.ctrl != null)
	{
		usz num_buckets = self.num_buckets();
		while (it.index < num_buckets && !is_full(self.ctrl[it.index]))
		{
			it.index++;
		}
	}
	return it;
}

fn Entry*? SwissTableIterator.next(&self)
{
	if (self.table.ctrl == null) return NO_MORE_ELEMENT~;

	usz num_buckets = self.table.num_buckets();
	if (self.index >= num_buckets) return NO_MORE_ELEMENT~;

	Entry* entry = self.table.entry_at(self.index);
	self.index++;

	while (self.index < num_buckets && !is_full(self.table.ctrl[self.index]))
	{
		self.index++;
	}

	return entry;
}

typedef SwissTableKeyIterator = SwissTableIterator;
typedef SwissTableValueIterator = SwissTableIterator;

fn SwissTableKeyIterator SwissTable.key_iter(&self)
{
	return (SwissTableKeyIterator)self.iter();
}

fn Key? SwissTableKeyIterator.next(&self)
{
	Entry* e = ((SwissTableIterator*)self).next()!;
	return e.key;
}

fn SwissTableValueIterator SwissTable.value_iter(&self)
{
	return (SwissTableValueIterator)self.iter();
}

fn Value? SwissTableValueIterator.next(&self)
{
	Entry* e = ((SwissTableIterator*)self).next()!;
	return e.value;
}

// Internal helpers

fn bool is_full(char ctrl) @inline @local => (ctrl & 0x80) == 0;

fn Entry* SwissTable.entry_at(&self, usz index) @inline @private
{
	return &self.data[index];
}

fn void SwissTable.set_ctrl(&self, usz index, char ctrl) @local
{
	self.ctrl[index] = ctrl;
	usz mirror = ((index + self.bucket_mask + 1 - GROUP_WIDTH) & self.bucket_mask) + GROUP_WIDTH;
	self.ctrl[mirror] = ctrl;
}

fn void SwissTable.insert_entry(&self, uint hash, char tag, Key key, Value value) @local
{
	ProbeSeq seq = probe_seq_new(hash, self.bucket_mask);

	while (true)
	{
		BitMask mask = match_empty_or_deleted(&self.ctrl[seq.pos]);
		if (mask.any_bit_set())
		{
			usz bit = mask.lowest_set_bit();
			usz index = (seq.pos + bit) & self.bucket_mask;

			if (self.ctrl[index] == CTRL_EMPTY)
			{
				self.growth_left--;
			}

			self.set_ctrl(index, tag);
			Entry* entry = self.entry_at(index);
			entry.key = key;
			entry.value = value;
			self.items++;
			return;
		}
		seq.move_next(self.bucket_mask);
	}
}

fn void SwissTable.erase_at(&self, usz index) @private
{
	assert(is_full(self.ctrl[index]), "erase_at called on non-full slot");
	self.items--;

	// Decide whether to mark the slot EMPTY or DELETED.
	// A slot can be marked EMPTY (reclaiming growth_left) only if it
	// "was never full" — meaning no probe chain could have threaded
	// through this slot. This requires THREE conditions (Abseil algorithm):
	//   1. The group ending at this slot has at least one empty.
	//   2. The group starting at this slot has at least one empty.
	//   3. The combined distance from the nearest empties on each side
	//      is less than GROUP_WIDTH, ensuring no probe group could
	//      span the gap without seeing an empty.
	usz index_before = ((index + self.bucket_mask + 1 - GROUP_WIDTH) & self.bucket_mask);
	BitMask empty_before = match_empty(&self.ctrl[index_before]);
	BitMask empty_after = match_empty(&self.ctrl[index]);
	bool was_never_full = empty_before.any_bit_set()
		&& empty_after.any_bit_set()
		&& (empty_after.trailing_zeros() + empty_before.leading_zeros()) < GROUP_WIDTH;
	if (was_never_full)
	{
		self.set_ctrl(index, CTRL_EMPTY);
		self.growth_left++;
	}
	else
	{
		self.set_ctrl(index, CTRL_DELETED);
	}
}

// Allocation & growth

fn void SwissTable.alloc_table(&self, usz num_buckets) @local
{
	assert(num_buckets > 0 && math::is_power_of_2(num_buckets),
		"num_buckets must be a power of 2");

	usz data_size = Entry.sizeof * num_buckets;
	usz ctrl_size = num_buckets + GROUP_WIDTH;
	usz total_size = data_size + ctrl_size;

	void* raw = allocator::calloc(self.allocator, total_size);

	self.data = (Entry*)raw;
	self.ctrl = (char*)raw + data_size;
	self.bucket_mask = num_buckets - 1;
	self.growth_left = bucket_mask_to_capacity(self.bucket_mask);

	self.ctrl[:ctrl_size] = CTRL_EMPTY;
}

fn void SwissTable.grow(&self) @local
{
	usz old_num_buckets = self.num_buckets();
	usz new_num_buckets;

	if (self.ctrl == null || old_num_buckets == 0)
	{
		new_num_buckets = GROUP_WIDTH;
	}
	else
	{
		new_num_buckets = old_num_buckets * 2;
	}

	char* old_ctrl = self.ctrl;
	Entry* old_data = self.data;
	usz old_items = self.items;

	self.alloc_table(new_num_buckets);
	self.items = 0;

	if (old_ctrl != null && old_items > 0)
	{
		for (usz i = 0; i < old_num_buckets; i++)
		{
			if (is_full(old_ctrl[i]))
			{
				Entry* old_entry = &old_data[i];
				uint hash = old_entry.key.hash();
				char tag = h2(hash);
				self.insert_entry(hash, tag, old_entry.key, old_entry.value);
			}
		}
	}

	if (old_ctrl != null)
	{
		allocator::free(self.allocator, (void*)old_data);
	}
}

// Capacity calculations

fn usz capacity_to_buckets(usz capacity) @local
{
	assert(capacity > 0, "capacity must be positive");

	if (capacity < 8)
	{
		return GROUP_WIDTH;
	}

	usz adjusted = (capacity * LOAD_FACTOR_DENOMINATOR + LOAD_FACTOR_NUMERATOR - 1)
		/ LOAD_FACTOR_NUMERATOR;
	return math::next_power_of_2(adjusted);
}

fn usz bucket_mask_to_capacity(usz bucket_mask) @local
{
	if (bucket_mask < 8)
	{
		return bucket_mask;
	}
	return ((bucket_mask + 1) / LOAD_FACTOR_DENOMINATOR) * LOAD_FACTOR_NUMERATOR;
}
