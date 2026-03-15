module metaphor::utils::hash_set <Key>;
import metaphor::utils::swisstable;
import std::core::mem::allocator;

// HashSet — set backed by SwissTable

alias Table = swisstable::SwissTable {Key, bool};

struct HashSet
{
	Table table;
}

// Lifecycle

fn HashSet* HashSet.init(&self, Allocator allocator, usz initial_capacity = 0)
{
	self.table.init(allocator, initial_capacity);
	return self;
}

fn HashSet* HashSet.tinit(&self, usz initial_capacity = 0)
{
	return self.init(tmem, initial_capacity) @inline;
}

fn void HashSet.free(&self)
{
	self.table.free();
}

// Core operations

fn bool HashSet.add(&self, Key key)
{
	return self.table.set(key, true);
}

fn bool HashSet.remove(&self, Key key)
{
	if (catch self.table.remove(key)) return false;
	return true;
}

fn bool HashSet.contains(&self, Key key)
{
	return self.table.has_key(key);
}

// Size / capacity

fn usz HashSet.len(&self) @inline @operator(len) => self.table.len();

fn bool HashSet.is_empty(&self) @inline => self.table.is_empty();

fn void HashSet.clear(&self)
{
	self.table.clear();
}

// Iteration

struct KeyIterator @local
{
	swisstable::SwissTableIterator {Key, bool} inner;
}

fn KeyIterator HashSet.key_iter(&self)
{
	return { .inner = self.table.iter() };
}

fn Key? KeyIterator.next(&self)
{
	swisstable::Entry {Key, bool}* e = self.inner.next()!;
	return e.key;
}

// Set operations

fn void HashSet.add_all(&self, HashSet* other)
{
	KeyIterator it = other.key_iter();
	while (try key = it.next())
	{
		self.add(key);
	}
}

fn void HashSet.retain_all(&self, HashSet* other)
{
	usz num_buckets = self.table.num_buckets();
	if (self.table.ctrl == null) return;

	for (usz i = 0; i < num_buckets; i++)
	{
		char ctrl = self.table.ctrl[i];
		if ((ctrl & 0x80) == 0)
		{
			swisstable::Entry {Key, bool}* entry = self.table.entry_at(i);
			if (!other.contains(entry.key))
			{
				self.table.erase_at(i);
			}
		}
	}
}
