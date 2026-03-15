module metaphor::utils::hash_map <Key, Value>;
import metaphor::utils::swisstable;
import std::core::mem::allocator;

// HashMap — user-facing generic hash map backed by SwissTable

alias Table = swisstable::SwissTable {Key, Value};

struct HashMap
{
	Table table;
}

// Lifecycle

fn HashMap* HashMap.init(&self, Allocator allocator, usz initial_capacity = 0)
{
	self.table.init(allocator, initial_capacity);
	return self;
}

fn HashMap* HashMap.tinit(&self, usz initial_capacity = 0)
{
	return self.init(tmem, initial_capacity) @inline;
}

fn void HashMap.free(&self)
{
	self.table.free();
}

// Core operations — names match std::collections::map

fn Value? HashMap.get(&self, Key key) @operator([])
{
	return self.table.get(key);
}

fn Value*? HashMap.get_ref(&self, Key key) @operator(&[])
{
	return self.table.get_ref(key);
}

fn bool HashMap.set(&self, Key key, Value value) @operator([]=)
{
	return self.table.set(key, value);
}

fn void? HashMap.remove(&self, Key key) @maydiscard
{
	return self.table.remove(key);
}

fn bool HashMap.has_key(&self, Key key)
{
	return self.table.has_key(key);
}

// Size / capacity

fn usz HashMap.len(&self) @inline @operator(len) => self.table.len();

fn bool HashMap.is_empty(&self) @inline => self.table.is_empty();

fn void HashMap.clear(&self)
{
	self.table.clear();
}

// Iteration

fn swisstable::SwissTableIterator {Key, Value} HashMap.iter(&self)
{
	return self.table.iter();
}

fn swisstable::SwissTableKeyIterator {Key, Value} HashMap.key_iter(&self)
{
	return self.table.key_iter();
}

fn swisstable::SwissTableValueIterator {Key, Value} HashMap.value_iter(&self)
{
	return self.table.value_iter();
}
