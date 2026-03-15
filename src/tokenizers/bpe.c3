//! Fast BPE Tokenizer
//!
//! Ported from the HuggingFace Tokenizers library (Rust):
//!   https://github.com/huggingface/tokenizers
//!
//! Key source files ported:
//!   - word.rs:        https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/word.rs
//!   - model.rs:       https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/model.rs
//!   - parallelism.rs: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/utils/parallelism.rs
//!
//! Original work Copyright 2021 HuggingFace Inc. (Apache-2.0 license).
//!
//! Key improvements over bpe.c3:
//!
//!   1. Priority-queue (min-heap) merge: O(n log n) vs O(n² × m)
//!   2. Linked-list symbol table: in-place merges without array shifts
//!   3. Per-word result cache: skip re-encoding repeated words
//!
//! This is a parallel implementation. It reuses the existing
//! vocab, pretokenizer, and json_parser modules.

module metaphor::tokenizers::bpe;
import metaphor::tokenizers::vocab;
import metaphor::tokenizers::pretokenizer;
import metaphor::tokenizers::json_parser;
import metaphor::utils::hash;
import std::core::mem;
import std::collections::list;
import std::io;

// Faults

faultdef BPE_ERROR;

// Symbol — linked-list node for in-place merge
//
// Ported from HuggingFace tokenizers/src/models/bpe/word.rs:Symbol
// Each symbol has prev/next indices (-1 = none) so merges can skip
// tombstoned entries without shifting the array.

struct Symbol
{
	uint c;       // Token ID
	int prev;     // Index of previous live symbol (-1 = head)
	int next;     // Index of next live symbol (-1 = tail)
	usz len;      // Byte length this symbol covers (for max_length checks)
}

fn void Symbol.merge_with(&self, Symbol* other, uint new_c)
{
	self.c = new_c;
	self.len += other.len;
	self.next = other.next;
}

// Merge — min-heap entry
//
// Ported from HuggingFace tokenizers/src/models/bpe/word.rs:Merge
// Ordered by rank (lower = higher priority), then by position.

struct Merge
{
	usz pos;      // Position (index into symbols array)
	uint rank;    // Merge priority (lower = apply first)
	uint new_id;  // Result token ID after merge
}

/// Returns true if a should come before b in a min-heap (a has higher priority).
fn bool merge_less_than(Merge* a, Merge* b) @inline
{
	if (a.rank != b.rank) return a.rank < b.rank;
	return a.pos < b.pos;
}

// MergeHeap — simple binary min-heap for Merge entries

const HEAP_INITIAL_CAP = 64;

struct MergeHeap
{
	list::List{Merge} data;
}

fn void MergeHeap.init(&self)
{
	self.data.init(mem);
}

fn void MergeHeap.deinit(&self)
{
	self.data.free();
}

fn void MergeHeap.clear(&self)
{
	self.data.clear();
}

fn void MergeHeap.push(&self, Merge m)
{
	self.data.push(m);
	self.sift_up(self.data.len() - 1);
}

fn Merge MergeHeap.pop(&self)
{
	Merge top = self.data[0];
	if (self.data.len() > 1)
	{
		self.data[0] = self.data[^1];
	}
	self.data.pop()!!;
	if (self.data.len() > 0)
	{
		self.sift_down(0);
	}
	return top;
}

fn bool MergeHeap.is_empty(&self) @inline
{
	return self.data.len() == 0;
}

fn void MergeHeap.sift_up(&self, usz idx)
{
	while (idx > 0)
	{
		usz parent = (idx - 1) / 2;
		if (merge_less_than(&self.data[idx], &self.data[parent]))
		{
			Merge tmp = self.data[idx];
			self.data[idx] = self.data[parent];
			self.data[parent] = tmp;
			idx = parent;
		}
		else { break; }
	}
}

fn void MergeHeap.sift_down(&self, usz idx)
{
	while (true)
	{
		usz smallest = idx;
		usz left = 2 * idx + 1;
		usz right = 2 * idx + 2;
		usz len = self.data.len();

		if (left < len && merge_less_than(&self.data[left], &self.data[smallest]))
		{
			smallest = left;
		}
		if (right < len && merge_less_than(&self.data[right], &self.data[smallest]))
		{
			smallest = right;
		}
		if (smallest != idx)
		{
			Merge tmp = self.data[idx];
			self.data[idx] = self.data[smallest];
			self.data[smallest] = tmp;
			idx = smallest;
		}
		else { break; }
	}
}

// Word — symbol sequence with heap-based merge
//
// Ported from HuggingFace tokenizers/src/models/bpe/word.rs:Word

struct Word
{
	list::List{Symbol} symbols;
}

fn void Word.init(&self)
{
	self.symbols.init(mem);
}

fn void Word.deinit(&self)
{
	self.symbols.free();
}

fn void Word.clear(&self)
{
	self.symbols.clear();
}

fn void Word.add(&self, uint c, usz byte_len)
{
	int prev_idx = (int)(self.symbols.len()) - 1;
	if (prev_idx >= 0)
	{
		self.symbols[prev_idx].next = (int)self.symbols.len();
	}
	self.symbols.push({
		.c = c,
		.prev = prev_idx,
		.next = -1,
		.len = byte_len,
	});
}

/// Priority-queue-based merge: O(n log n) total.
/// Ported from HuggingFace Word::merge_all.
///
/// Staleness detection: when we pop a Merge from the heap, the symbols at
/// that position may have been changed by a prior merge. We validate by
/// re-looking up the current (left, right) pair in the merge table and
/// checking that it still produces the same new_id. This matches the
/// HuggingFace implementation exactly.
fn void Word.merge_all(&self, vocab::MergeTable* merges, MergeHeap* heap)
{
	if (self.symbols.len() < 2) return;

	heap.clear();

	// Seed the heap with all valid adjacent pairs
	for (usz i = 0; i + 1 < self.symbols.len(); i++)
	{
		if (try rule = merges.get_merge(self.symbols[i].c, self.symbols[i + 1].c))
		{
			heap.push({ .pos = i, .rank = rule.priority, .new_id = rule.result });
		}
	}

	while (!heap.is_empty())
	{
		Merge top = heap.pop();
		usz pos = top.pos;

		// Skip if this symbol was already merged (tombstoned: len == 0)
		if (self.symbols[pos].len == 0) continue;

		// Skip if this symbol has no next (it's the tail)
		if (self.symbols[pos].next == -1) continue;

		usz next_pos = (usz)self.symbols[pos].next;
		Symbol right = self.symbols[next_pos];

		// Validate: re-lookup the current pair in merge table.
		// If the pair no longer exists or produces a different new_id,
		// this is a stale entry from before a prior merge changed things.
		if (try rule = merges.get_merge(self.symbols[pos].c, right.c))
		{
			if (rule.result != top.new_id) continue;
		}
		else
		{
			continue;  // Pair no longer valid
		}

		// Perform the merge: update left symbol, tombstone right
		self.symbols[pos].merge_with(&right, top.new_id);

		// Tombstone the right symbol
		self.symbols[next_pos].len = 0;

		// Update the prev pointer of the symbol after right
		if (right.next >= 0 && (usz)right.next < self.symbols.len())
		{
			self.symbols[right.next].prev = (int)pos;
		}

		// Check for new merge with the symbol BEFORE pos
		if (self.symbols[pos].prev >= 0)
		{
			usz prev_pos = (usz)self.symbols[pos].prev;
			if (try new_rule = merges.get_merge(self.symbols[prev_pos].c, self.symbols[pos].c))
			{
				heap.push({ .pos = prev_pos, .rank = new_rule.priority, .new_id = new_rule.result });
			}
		}

		// Check for new merge with the symbol AFTER the merged result
		if (self.symbols[pos].next >= 0 && (usz)self.symbols[pos].next < self.symbols.len())
		{
			usz after_pos = (usz)self.symbols[pos].next;
			if (try new_rule = merges.get_merge(self.symbols[pos].c, self.symbols[after_pos].c))
			{
				heap.push({ .pos = pos, .rank = new_rule.priority, .new_id = new_rule.result });
			}
		}
	}
}

/// Collect final token IDs by walking the linked list.
fn usz Word.collect_tokens(&self, uint* out, usz out_cap)
{
	usz count = 0;
	// Find head: the first symbol with len > 0
	for (usz i = 0; i < self.symbols.len(); i++)
	{
		if (self.symbols[i].len > 0)
		{
			// Walk the linked list from here
			int cur = (int)i;
			while (cur >= 0 && count < out_cap)
			{
				if (self.symbols[cur].len > 0)
				{
					out[count++] = self.symbols[cur].c;
				}
				cur = self.symbols[cur].next;
			}
			break;
		}
	}
	return count;
}

// WordCache — hash map from string → token ID array

const CACHE_CAPACITY = 10000;   // Max cached words
const CACHE_HASH_SIZE = 16384;  // Must be power of 2

struct CacheEntry
{
	char* key;
	usz key_len;
	uint* tokens;
	usz num_tokens;
}

struct WordCache
{
	CacheEntry* entries;
	uint* hash_table;     // indices into entries (0xFFFFFFFF = empty)
	usz num_entries;
	bool initialized;
}

fn void WordCache.init(&self)
{
	self.entries = (CacheEntry*)mem::malloc(CacheEntry.sizeof * CACHE_CAPACITY);
	self.hash_table = (uint*)mem::malloc(uint.sizeof * CACHE_HASH_SIZE);
	for (usz i = 0; i < CACHE_HASH_SIZE; i++) self.hash_table[i] = 0xFFFFFFFF;
	self.num_entries = 0;
	self.initialized = true;
}

fn void WordCache.deinit(&self)
{
	if (!self.initialized) return;
	for (usz i = 0; i < self.num_entries; i++)
	{
		if (self.entries[i].key != null) mem::free(self.entries[i].key);
		if (self.entries[i].tokens != null) mem::free(self.entries[i].tokens);
	}
	mem::free(self.entries);
	mem::free(self.hash_table);
	self.initialized = false;
}



/// Look up cached tokens for a word. Returns null if not found.
fn uint[]? WordCache.get(&self, char[] key)
{
	if (!self.initialized) return BPE_ERROR~;

	ulong h = hash::fnv1a(key);
	uint slot = (uint)(h & (CACHE_HASH_SIZE - 1));
	while (self.hash_table[slot] != 0xFFFFFFFF)
	{
		uint idx = self.hash_table[slot];
		CacheEntry* e = &self.entries[idx];
		if (e.key_len == (usz)key.len)
		{
			bool match = true;
			for (usz i = 0; i < key.len; i++)
			{
				if (e.key[i] != key[i]) { match = false; break; }
			}
			if (match)
			{
				return e.tokens[:e.num_tokens];
			}
		}
		slot = (slot + 1) & (CACHE_HASH_SIZE - 1);
	}
	return BPE_ERROR~;
}

/// Insert tokens for a word into the cache.
fn void WordCache.put(&self, char[] key, uint[] tokens)
{
	if (!self.initialized) return;
	if (self.num_entries >= CACHE_CAPACITY) return;  // Cache full, skip

	// Copy key
	char* owned_key = (char*)mem::malloc(key.len);
	mem::copy(owned_key, key.ptr, key.len);

	// Copy tokens
	uint* owned_tokens = (uint*)mem::malloc(uint.sizeof * tokens.len);
	mem::copy(owned_tokens, tokens.ptr, uint.sizeof * tokens.len);

	usz idx = self.num_entries;
	self.entries[idx] = {
		.key = owned_key,
		.key_len = key.len,
		.tokens = owned_tokens,
		.num_tokens = tokens.len,
	};
	self.num_entries++;

	// Insert into hash table
	ulong h = hash::fnv1a(key);
	uint slot = (uint)(h & (CACHE_HASH_SIZE - 1));
	while (self.hash_table[slot] != 0xFFFFFFFF) slot = (slot + 1) & (CACHE_HASH_SIZE - 1);
	self.hash_table[slot] = (uint)idx;
}

// GPT-2 Byte Mapping (same as bpe.c3, duplicated to keep module standalone)

fn uint byte_to_gpt2_char(char byte)
{
	uint b = (uint)byte;
	if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
	{
		return b;
	}
	if (b <= 32) return 256 + b;
	if (b >= 127 && b <= 160) return 289 + (b - 127);
	if (b == 173) return 323;
	return b;
}

fn int gpt2_char_to_byte(uint codepoint)
{
	if (codepoint <= 255)
	{
		if ((codepoint >= 33 && codepoint <= 126) ||
		    (codepoint >= 161 && codepoint <= 172) ||
		    (codepoint >= 174 && codepoint <= 255))
		{
			return (int)codepoint;
		}
		return -1;
	}
	if (codepoint >= 256 && codepoint <= 288) return (int)(codepoint - 256);
	if (codepoint >= 289 && codepoint <= 322) return (int)(127 + (codepoint - 289));
	if (codepoint == 323) return 173;
	return -1;
}

fn usz utf8_encode(uint cp, char[4]* buf)
{
	if (cp < 0x80)       { (*buf)[0] = (char)cp; return 1; }
	if (cp < 0x800)      { (*buf)[0] = (char)(0xC0 | (cp >> 6)); (*buf)[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
	if (cp < 0x10000)    { (*buf)[0] = (char)(0xE0 | (cp >> 12)); (*buf)[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); (*buf)[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
	(*buf)[0] = (char)(0xF0 | (cp >> 18)); (*buf)[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); (*buf)[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); (*buf)[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

fn usz utf8_codepoint_len(char byte) @inline
{
	if ((byte & 0x80) == 0) return 1;
	if ((byte & 0xE0) == 0xC0) return 2;
	if ((byte & 0xF0) == 0xE0) return 3;
	if ((byte & 0xF8) == 0xF0) return 4;
	return 1;
}

fn uint utf8_decode(char[] bytes)
{
	if (bytes.len == 0) return 0;
	if (bytes.len == 1) return (uint)bytes[0];
	if (bytes.len == 2) return (((uint)bytes[0] & 0x1F) << 6) | ((uint)bytes[1] & 0x3F);
	if (bytes.len == 3) return (((uint)bytes[0] & 0x0F) << 12) | (((uint)bytes[1] & 0x3F) << 6) | ((uint)bytes[2] & 0x3F);
	return (((uint)bytes[0] & 0x07) << 18) | (((uint)bytes[1] & 0x3F) << 12) | (((uint)bytes[2] & 0x3F) << 6) | ((uint)bytes[3] & 0x3F);
}

// Decoder modes (same as bpe.c3)

enum DecoderMode
{
	GPT2,
	SENTENCEPIECE,
}

// Bpe — main tokenizer struct

struct Bpe
{
	vocab::Vocab v;
	vocab::MergeTable merges;
	pretokenizer::Pretokenizer pretok;
	DecoderMode decoder_mode;
	bool byte_fallback;

	// Per-instance scratch (reused across encode calls)
	Word word;
	MergeHeap heap;
	WordCache cache;

	// Output buffer
	uint* out_buf;
	usz out_cap;
}

fn void Bpe.init(&self)
{
	self.v.init();
	self.merges.init();
	self.pretok = { .mode = pretokenizer::Mode.GPT2 };
	self.decoder_mode = DecoderMode.GPT2;
	self.byte_fallback = true;
	self.word.init();
	self.heap.init();
	self.cache.init();
	self.out_buf = null;
	self.out_cap = 0;
}

fn void Bpe.deinit(&self)
{
	self.v.deinit();
	self.merges.deinit();
	self.word.deinit();
	self.heap.deinit();
	self.cache.deinit();
	if (self.out_buf != null) mem::free(self.out_buf);
}

// Tokenizer interface methods

fn uint Bpe.vocab_size(&self)
{
	return self.v.num_tokens;
}

fn uint Bpe.eos_id(&self)
{
	return self.v.special.eos;
}

fn uint Bpe.bos_id(&self)
{
	return self.v.special.bos;
}

fn char[]? Bpe.get_token(&self, uint id)
{
	return self.v.get_token(id);
}

// Loading

/// Load from HuggingFace tokenizer.json (delegates to existing json_parser).
fn void? Bpe.load_huggingface(&self, char[] path)
{
	String p = string::tformat("%s", (ZString)path.ptr);
	char[] data = io::file::load(mem, p)!;

	json_parser::parse_tokenizer_json(data, &self.v, &self.merges)!;

	// Detect decoder mode
	self.decoder_mode = DecoderMode.GPT2;
	for (usz i = 0; i + 12 < data.len; i++)
	{
		if (data[i] == 'B' && data[i+1] == 'y' && data[i+2] == 't' && data[i+3] == 'e' &&
			data[i+4] == 'F' && data[i+5] == 'a' && data[i+6] == 'l' && data[i+7] == 'l')
		{
			self.decoder_mode = DecoderMode.SENTENCEPIECE;
			break;
		}
	}

	mem::free(data.ptr);
}

// Encoding — fast path with heap merge + cache

fn void Bpe.ensure_out(&self, usz needed)
{
	if (needed <= self.out_cap) return;
	usz new_cap = self.out_cap == 0 ? 256 : self.out_cap;
	while (new_cap < needed) new_cap *= 2;
	uint* new_buf = (uint*)mem::malloc(uint.sizeof * new_cap);
	if (self.out_buf != null)
	{
		mem::copy(new_buf, self.out_buf, uint.sizeof * self.out_cap);
		mem::free(self.out_buf);
	}
	self.out_buf = new_buf;
	self.out_cap = new_cap;
}

/// Encode text to token IDs using priority-queue merge.
/// Returns a malloc'd array of token IDs. Caller must free.
fn uint[]? Bpe.encode(&self, char[] text)
{
	if (text.len == 0) return {};

	usz total_tokens = 0;
	usz cap = text.len + 64;
	uint* out = (uint*)mem::malloc(uint.sizeof * cap);
	if (out == null) return BPE_ERROR~;

	// Split on added (special) tokens, encode segments between them
	char[] remaining = text;
	while (remaining.len > 0)
	{
		usz best_pos = remaining.len;
		uint best_id = 0;
		usz best_len = 0;

		for (usz a = 0; a < (usz)self.v.num_added; a++)
		{
			uint aid = self.v.added_ids[a];
			if (try tok_str = self.v.get_token(aid))
			{
				if (tok_str.len > remaining.len) continue;
				for (usz pos = 0; pos + tok_str.len <= remaining.len; pos++)
				{
					bool match = true;
					for (usz k = 0; k < tok_str.len; k++)
					{
						if (remaining[pos + k] != tok_str[k]) { match = false; break; }
					}
					if (match && pos < best_pos)
					{
						best_pos = pos;
						best_id = aid;
						best_len = tok_str.len;
						break;
					}
				}
			}
		}

		if (best_pos == remaining.len)
		{
			self.encode_segment(remaining, &out, &total_tokens, &cap);
			break;
		}

		if (best_pos > 0)
		{
			self.encode_segment(remaining[:best_pos], &out, &total_tokens, &cap);
		}

		// Append added token directly
		if (total_tokens >= cap)
		{
			cap = (total_tokens + 1) * 2;
			uint* new_out = (uint*)mem::malloc(uint.sizeof * cap);
			mem::copy(new_out, out, uint.sizeof * total_tokens);
			mem::free(out);
			out = new_out;
		}
		out[total_tokens++] = best_id;
		remaining = remaining[best_pos + best_len..];
	}

	return out[:total_tokens];
}

/// Encode a text segment (after added-token splitting).
fn void Bpe.encode_segment(&self, char[] text, uint** out_ptr, usz* total_ptr, usz* cap_ptr)
{
	if (text.len == 0) return;

	pretokenizer::SplitResult chunks = self.pretok.split(text);

	for (usz ci = 0; ci < chunks.num_chunks; ci++)
	{
		char[] chunk = chunks.chunks[ci];
		self.encode_chunk(chunk, out_ptr, total_ptr, cap_ptr);
	}

	chunks.deinit();
}

/// Encode a single pretokenized chunk using priority-queue merge + cache.
fn void Bpe.encode_chunk(&self, char[] chunk, uint** out_ptr, usz* total_ptr, usz* cap_ptr)
{
	if (chunk.len == 0) return;

	// Check cache first
	if (try cached = self.cache.get(chunk))
	{
		// Append cached tokens to output
		usz needed = *total_ptr + cached.len;
		if (needed > *cap_ptr)
		{
			*cap_ptr = needed * 2;
			uint* new_out = (uint*)mem::malloc(uint.sizeof * *cap_ptr);
			mem::copy(new_out, *out_ptr, uint.sizeof * *total_ptr);
			mem::free(*out_ptr);
			*out_ptr = new_out;
		}
		mem::copy(&(*out_ptr)[*total_ptr], cached.ptr, uint.sizeof * cached.len);
		*total_ptr += cached.len;
		return;
	}

	// Build word from chunk bytes using GPT-2 byte encoding
	self.word.clear();
	for (usz j = 0; j < chunk.len; j++)
	{
		uint gpt2_cp = byte_to_gpt2_char((char)chunk[j]);
		char[4] utf8_buf;
		usz utf8_len = utf8_encode(gpt2_cp, &utf8_buf);

		if (try id = self.v.get_id(utf8_buf[:utf8_len]))
		{
			self.word.add(id, 1);
		}
		else
		{
			char[1] byte_str = { chunk[j] };
			if (try id2 = self.v.get_id(&byte_str))
			{
				self.word.add(id2, 1);
			}
			else if (self.byte_fallback)
			{
				self.word.add((uint)chunk[j], 1);
			}
		}
	}

	// Apply all merges using the priority-queue algorithm
	self.word.merge_all(&self.merges, &self.heap);

	// Collect results into a temp buffer
	uint[512] temp_buf;
	usz num_result = self.word.collect_tokens(&temp_buf, 512);

	// Cache the result for this chunk
	self.cache.put(chunk, temp_buf[:num_result]);

	// Append to output
	usz needed = *total_ptr + num_result;
	if (needed > *cap_ptr)
	{
		*cap_ptr = needed * 2;
		uint* new_out = (uint*)mem::malloc(uint.sizeof * *cap_ptr);
		mem::copy(new_out, *out_ptr, uint.sizeof * *total_ptr);
		mem::free(*out_ptr);
		*out_ptr = new_out;
	}
	mem::copy(&(*out_ptr)[*total_ptr], &temp_buf, uint.sizeof * num_result);
	*total_ptr += num_result;
}

// Decoding — identical to bpe.c3

fn char[]? Bpe.decode(&self, uint[] tokens)
{
	if (tokens.len == 0) return {};

	if (self.decoder_mode == DecoderMode.SENTENCEPIECE)
	{
		return self.decode_sentencepiece(tokens);
	}
	return self.decode_gpt2(tokens);
}

fn char[]? Bpe.decode_gpt2(&self, uint[] tokens)
{
	usz buf_cap = tokens.len * 4;
	char* buf = (char*)mem::malloc(buf_cap);
	if (buf == null) return BPE_ERROR~;
	usz buf_len = 0;

	for (usz i = 0; i < tokens.len; i++)
	{
		if (try token = self.v.get_token(tokens[i]))
		{
			usz j = 0;
			while (j < token.len)
			{
				usz cp_len = utf8_codepoint_len(token[j]);
				if (j + cp_len > token.len) cp_len = 1;
				uint cp = utf8_decode(token[j:cp_len]);

				if (buf_len + 4 > buf_cap)
				{
					buf_cap *= 2;
					char* new_buf = (char*)mem::malloc(buf_cap);
					mem::copy(new_buf, buf, buf_len);
					mem::free(buf);
					buf = new_buf;
				}

				int raw_byte = gpt2_char_to_byte(cp);
				if (raw_byte >= 0)
				{
					buf[buf_len++] = (char)raw_byte;
				}
				else
				{
					mem::copy(&buf[buf_len], &token[j], cp_len);
					buf_len += cp_len;
				}
				j += cp_len;
			}
		}
	}
	return buf[:buf_len];
}

fn int hex_digit(char c) @private
{
	if (c >= '0' && c <= '9') return (int)(c - '0');
	if (c >= 'a' && c <= 'f') return (int)(c - 'a' + 10);
	if (c >= 'A' && c <= 'F') return (int)(c - 'A' + 10);
	return -1;
}

fn char[]? Bpe.decode_sentencepiece(&self, uint[] tokens)
{
	usz buf_cap = tokens.len * 4;
	char* buf = (char*)mem::malloc(buf_cap);
	if (buf == null) return BPE_ERROR~;
	usz buf_len = 0;

	for (usz i = 0; i < tokens.len; i++)
	{
		if (try token = self.v.get_token(tokens[i]))
		{
			// Check for byte fallback token: <0xNN>
			if (token.len == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>')
			{
				int hi = hex_digit(token[3]);
				int lo = hex_digit(token[4]);
				if (hi >= 0 && lo >= 0)
				{
					if (buf_len + 1 > buf_cap) { buf_cap *= 2; char* nb = (char*)mem::malloc(buf_cap); mem::copy(nb, buf, buf_len); mem::free(buf); buf = nb; }
					buf[buf_len++] = (char)(hi * 16 + lo);
					continue;
				}
			}

			usz j = 0;
			while (j < token.len)
			{
				if (buf_len + 4 > buf_cap)
				{
					buf_cap *= 2;
					char* new_buf = (char*)mem::malloc(buf_cap);
					mem::copy(new_buf, buf, buf_len);
					mem::free(buf);
					buf = new_buf;
				}
				if (j + 2 < token.len && token[j] == (char)0xE2 && token[j+1] == (char)0x96 && token[j+2] == (char)0x81)
				{
					buf[buf_len++] = ' ';
					j += 3;
				}
				else
				{
					buf[buf_len++] = token[j];
					j++;
				}
			}
		}
	}
	return buf[:buf_len];
}
