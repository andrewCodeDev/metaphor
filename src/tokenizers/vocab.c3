//! Vocabulary
//!
//! Bidirectional mapping between tokens (byte sequences) and IDs.
//! Used by the BPE tokenizer for encoding/decoding text to/from token IDs.
//!

module metaphor::tokenizers::vocab;
import std::core::mem;
import metaphor::utils::hash;
import std::collections::list;
import std::io;

faultdef NOT_FOUND;

const MAX_TOKENS = 256000; // Maximum vocabulary size

// Special tokens

struct SpecialTokens
{
	uint bos;  // <s> or [CLS]
	uint eos;  // </s> or [SEP]
	uint unk;  // <unk>
	uint pad;  // <pad>
	bool has_bos;
	bool has_eos;
	bool has_unk;
	bool has_pad;
}

// MergeRule / MergeTable

struct MergeRule
{
	uint left;
	uint right;
	uint result;
	uint priority;  // Lower = higher priority (order learned)
}

struct MergeTable
{
	list::List{MergeRule} rules;

	// Hash table for O(1) pair lookup: hash(left, right) -> index into rules
	uint* hash_table;    // index values (0xFFFFFFFF = empty)
	uint hash_capacity;
}

fn void MergeTable.init(&self)
{
	self.rules.init(mem);
	self.hash_table = null;
	self.hash_capacity = 0;
}

fn void MergeTable.deinit(&self)
{
	self.rules.free();
	if (self.hash_table != null) mem::free(self.hash_table);
	self.hash_table = null;
}

fn uint pair_hash(uint left, uint right) @inline
{
	ulong combined = ((ulong)left << 32) | (ulong)right;
	// FNV-1a style hash
	ulong h = 14695981039346656037;
	h ^= combined;
	h *= 1099511628211;
	h ^= (combined >> 16);
	h *= 1099511628211;
	return (uint)(h & 0xFFFFFFFF);
}

fn void MergeTable.ensure_hash_capacity(&self)
{
	uint needed = ((uint)self.rules.len() + 1) * 4;  // Load factor ~25%
	if (needed <= self.hash_capacity) return;

	uint new_cap = self.hash_capacity == 0 ? 1024 : self.hash_capacity * 2;
	while (new_cap < needed) new_cap *= 2;

	if (self.hash_table != null) mem::free(self.hash_table);
	self.hash_table = (uint*)mem::malloc(uint.sizeof * new_cap);
	self.hash_capacity = new_cap;

	// Fill with empty sentinel
	for (uint i = 0; i < new_cap; i++) self.hash_table[i] = 0xFFFFFFFF;

	// Rehash all existing rules
	for (uint i = 0; i < (uint)self.rules.len(); i++)
	{
		uint h = pair_hash(self.rules[i].left, self.rules[i].right) % self.hash_capacity;
		while (self.hash_table[h] != 0xFFFFFFFF) h = (h + 1) % self.hash_capacity;
		self.hash_table[h] = i;
	}
}

fn void MergeTable.add_merge(&self, uint left, uint right, uint result)
{
	uint idx = (uint)self.rules.len();
	self.rules.push({ .left = left, .right = right, .result = result, .priority = idx });

	// Insert into hash table
	self.ensure_hash_capacity();
	uint h = pair_hash(left, right) % self.hash_capacity;
	while (self.hash_table[h] != 0xFFFFFFFF) h = (h + 1) % self.hash_capacity;
	self.hash_table[h] = idx;
}

fn MergeRule*? MergeTable.get_merge(&self, uint left, uint right)
{
	if (self.hash_table == null || self.hash_capacity == 0) return NOT_FOUND~;

	uint h = pair_hash(left, right) % self.hash_capacity;
	while (self.hash_table[h] != 0xFFFFFFFF)
	{
		uint idx = self.hash_table[h];
		if (self.rules[idx].left == left && self.rules[idx].right == right)
		{
			return &self.rules[idx];
		}
		h = (h + 1) % self.hash_capacity;
	}
	return NOT_FOUND~;
}

// Vocab

/// Maximum token string length stored inline
const MAX_TOKEN_LEN = 256;

struct TokenEntry
{
	char* str;    // Owned copy of token string
	uint len;     // Token string length
	uint id;      // Token ID
	ulong hash;   // Precomputed hash of token string
}

const MAX_ADDED_TOKENS = 1024;

struct Vocab
{
	// ID -> token mapping (dense array)
	TokenEntry* tokens;
	uint num_tokens;
	uint capacity;

	// Token string -> ID (hash table, open addressing)
	uint* str_hash_table;    // index into tokens (0xFFFFFFFF = empty)
	uint str_hash_capacity;

	// Added tokens — matched literally during encode, bypassing BPE
	uint[MAX_ADDED_TOKENS] added_ids;
	uint num_added;

	SpecialTokens special;
}

fn void Vocab.init(&self)
{
	self.tokens = null;
	self.num_tokens = 0;
	self.capacity = 0;
	self.str_hash_table = null;
	self.str_hash_capacity = 0;
	self.num_added = 0;
	self.special = {};
}

fn void Vocab.deinit(&self)
{
	for (uint i = 0; i < self.num_tokens; i++)
	{
		if (self.tokens[i].str != null) mem::free(self.tokens[i].str);
	}
	if (self.tokens != null) mem::free(self.tokens);
	if (self.str_hash_table != null) mem::free(self.str_hash_table);
	self.tokens = null;
	self.str_hash_table = null;
}



fn bool strings_equal(char* a, uint a_len, char[] b) @inline
{
	if (a_len != (uint)b.len) return false;
	for (uint i = 0; i < a_len; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

fn void Vocab.ensure_str_hash_capacity(&self)
{
	uint needed = (self.num_tokens + 1) * 4;
	if (needed <= self.str_hash_capacity) return;

	uint new_cap = self.str_hash_capacity == 0 ? 1024 : self.str_hash_capacity * 2;
	while (new_cap < needed) new_cap *= 2;

	if (self.str_hash_table != null) mem::free(self.str_hash_table);
	self.str_hash_table = (uint*)mem::malloc(uint.sizeof * new_cap);
	self.str_hash_capacity = new_cap;

	for (uint i = 0; i < new_cap; i++) self.str_hash_table[i] = 0xFFFFFFFF;

	// Rehash
	for (uint i = 0; i < self.num_tokens; i++)
	{
		if (self.tokens[i].str == null) continue;
		uint h = (uint)(self.tokens[i].hash % (ulong)self.str_hash_capacity);
		while (self.str_hash_table[h] != 0xFFFFFFFF) h = (h + 1) % self.str_hash_capacity;
		self.str_hash_table[h] = i;
	}
}

/// Add a token and return its ID. Returns existing ID if duplicate.
fn uint Vocab.add_token(&self, char[] token)
{
	// Check if already exists
	if (try existing = self.get_id(token)) return existing;

	uint id = self.num_tokens;

	// Grow tokens array if needed
	if (id >= self.capacity)
	{
		uint new_cap = self.capacity == 0 ? 512 : self.capacity * 2;
		TokenEntry* new_tokens = (TokenEntry*)mem::malloc(TokenEntry.sizeof * new_cap);
		if (self.tokens != null)
		{
			mem::copy(new_tokens, self.tokens, TokenEntry.sizeof * self.num_tokens);
			mem::free(self.tokens);
		}
		// Zero out new entries
		for (uint i = self.num_tokens; i < new_cap; i++)
		{
			new_tokens[i] = {};
		}
		self.tokens = new_tokens;
		self.capacity = new_cap;
	}

	// Copy token string
	char* owned = (char*)mem::malloc(token.len + 1);
	mem::copy(owned, token.ptr, token.len);
	owned[token.len] = 0;  // null-terminate for convenience

	ulong h = hash::fnv1a(token);
	self.tokens[id] = { .str = owned, .len = (uint)token.len, .id = id, .hash = h };
	self.num_tokens++;

	// Insert into hash table
	self.ensure_str_hash_capacity();
	uint slot = (uint)(h % (ulong)self.str_hash_capacity);
	while (self.str_hash_table[slot] != 0xFFFFFFFF) slot = (slot + 1) % self.str_hash_capacity;
	self.str_hash_table[slot] = id;

	return id;
}

/// Set a token at a specific ID. Used when loading vocabularies.
fn void Vocab.set_token(&self, uint id, char[] token)
{
	// Extend capacity if needed — allocate in bulk, not one-at-a-time
	if (id >= self.capacity)
	{
		uint new_cap = self.capacity == 0 ? 512 : self.capacity;
		while (new_cap <= id) new_cap *= 2;
		TokenEntry* new_tokens = (TokenEntry*)mem::malloc(TokenEntry.sizeof * new_cap);
		if (self.tokens != null)
		{
			mem::copy(new_tokens, self.tokens, TokenEntry.sizeof * self.num_tokens);
			mem::free(self.tokens);
		}
		// Zero out new entries
		for (uint i = self.num_tokens; i < new_cap; i++)
		{
			new_tokens[i] = {};
		}
		self.tokens = new_tokens;
		self.capacity = new_cap;
	}

	// Extend num_tokens to cover this ID (fill placeholders inline)
	while (self.num_tokens <= id)
	{
		// Just mark empty — don't call add_token which triggers hash operations
		if (self.tokens[self.num_tokens].str == null)
		{
			self.tokens[self.num_tokens] = { .str = null, .len = 0, .id = self.num_tokens, .hash = 0 };
		}
		self.num_tokens++;
	}

	// Free old token if present
	if (self.tokens[id].str != null && self.tokens[id].len > 0)
	{
		mem::free(self.tokens[id].str);
	}

	// Copy new token
	char* owned = (char*)mem::malloc(token.len + 1);
	mem::copy(owned, token.ptr, token.len);
	owned[token.len] = 0;

	ulong h = hash::fnv1a(token);
	self.tokens[id] = { .str = owned, .len = (uint)token.len, .id = id, .hash = h };

	// Insert into hash table — just this one entry, NOT a full rebuild
	self.ensure_str_hash_capacity();
	uint slot = (uint)(h % (ulong)self.str_hash_capacity);
	while (self.str_hash_table[slot] != 0xFFFFFFFF) slot = (slot + 1) % self.str_hash_capacity;
	self.str_hash_table[slot] = id;
}

/// Look up ID for a token string.
fn uint? Vocab.get_id(&self, char[] token)
{
	if (self.str_hash_table == null || self.str_hash_capacity == 0) return NOT_FOUND~;

	ulong h = hash::fnv1a(token);
	uint slot = (uint)(h % (ulong)self.str_hash_capacity);

	while (self.str_hash_table[slot] != 0xFFFFFFFF)
	{
		uint idx = self.str_hash_table[slot];
		if (self.tokens[idx].hash == h && strings_equal(self.tokens[idx].str, self.tokens[idx].len, token))
		{
			return self.tokens[idx].id;
		}
		slot = (slot + 1) % self.str_hash_capacity;
	}
	return NOT_FOUND~;
}

/// Look up token string for an ID.
fn char[]? Vocab.get_token(&self, uint id)
{
	if (id >= self.num_tokens) return NOT_FOUND~;
	if (self.tokens[id].str == null || self.tokens[id].len == 0) return NOT_FOUND~;
	return self.tokens[id].str[:self.tokens[id].len];
}

/// Number of tokens in vocabulary.
fn uint Vocab.len(&self)
{
	return self.num_tokens;
}

/// Initialize with byte-level vocabulary (256 single-byte tokens).
fn void Vocab.init_byte_level(&self)
{
	self.init();
	for (uint b = 0; b < 256; b++)
	{
		char[1] buf = { (char)b };
		self.add_token(&buf);
	}
}

/// Register an added token (matched literally during encode, bypassing BPE).
fn void Vocab.add_added_token(&self, uint id)
{
	assert(self.num_added < MAX_ADDED_TOKENS, "Too many added tokens");
	self.added_ids[self.num_added] = id;
	self.num_added++;
}

/// Check and set special token if it matches known patterns.
fn void Vocab.check_special_token(&self, char[] token, uint id)
{
	if (str_eq(token, "<s>") || str_eq(token, "[CLS]") || str_eq(token, "<|begin_of_text|>"))
	{
		self.special.bos = id;
		self.special.has_bos = true;
	}
	else if (str_eq(token, "</s>") || str_eq(token, "[SEP]") || str_eq(token, "<|end_of_text|>") || str_eq(token, "<|endoftext|>"))
	{
		self.special.eos = id;
		self.special.has_eos = true;
	}
	else if (str_eq(token, "<unk>") || str_eq(token, "[UNK]"))
	{
		self.special.unk = id;
		self.special.has_unk = true;
	}
	else if (str_eq(token, "<pad>") || str_eq(token, "[PAD]"))
	{
		self.special.pad = id;
		self.special.has_pad = true;
	}
}

fn bool str_eq(char[] a, char[] b) @inline
{
	if (a.len != b.len) return false;
	for (usz i = 0; i < a.len; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}
