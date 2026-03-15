///! BPE Tokenizer Tests
///!
///! Validates BPE tokenizer produces correct results.
///! Test cases include HuggingFace reference IDs from tokenizer_compare.c3.

module metaphor::testing::tokenizers::bpe_test;
import metaphor::tokenizers::bpe;
import metaphor::tokenizers::vocab;
import metaphor::tokenizers::pretokenizer;
import std::core::test;
import std::core::mem;

// ============================================================================
// Unit tests: GPT-2 byte mapping (must match bpe.c3)
// ============================================================================

fn void test_fast_gpt2_byte_mapping_roundtrip() @test
{
	for (uint i = 0; i < 256; i++)
	{
		uint cp = bpe::byte_to_gpt2_char((char)i);
		int back = bpe::gpt2_char_to_byte(cp);
		test::eq(back, (int)i);
	}
}

// ============================================================================
// Unit tests: MergeHeap
// ============================================================================

fn void test_merge_heap_ordering() @test
{
	bpe::MergeHeap heap;
	heap.init();
	defer heap.deinit();

	// Push merges in random rank order
	heap.push({ .pos = 5, .rank = 3, .new_id = 100 });
	heap.push({ .pos = 2, .rank = 1, .new_id = 101 });
	heap.push({ .pos = 8, .rank = 5, .new_id = 102 });
	heap.push({ .pos = 1, .rank = 1, .new_id = 103 });  // Same rank as pos=2, lower pos

	// Pop should give lowest rank first, position tiebreak
	bpe::Merge m0 = heap.pop();
	test::eq(m0.rank, (uint)1);
	test::eq(m0.pos, (usz)1);  // pos=1 before pos=2 (both rank 1)

	bpe::Merge m1 = heap.pop();
	test::eq(m1.rank, (uint)1);
	test::eq(m1.pos, (usz)2);

	bpe::Merge m2 = heap.pop();
	test::eq(m2.rank, (uint)3);

	bpe::Merge m3 = heap.pop();
	test::eq(m3.rank, (uint)5);

	assert(heap.is_empty());
}

// ============================================================================
// BPE byte-level encode/decode (matches bpe_test.c3)
// ============================================================================

fn void test_fast_bpe_byte_level_encode_decode() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	// Use NONE pretokenizer for unit testing
	b.pretok = { .mode = pretokenizer::Mode.NONE };

	// Initialize with byte-level vocab
	b.v.deinit();
	b.v.init_byte_level();

	char[] text = "hello";
	uint[]? ids_opt = b.encode(text);
	uint[] ids = ids_opt!!;
	defer mem::free(ids.ptr);

	// Should encode to 5 byte-level tokens
	test::eq(ids.len, (usz)5);

	// Decode should roundtrip
	char[]? decoded_opt = b.decode(ids);
	char[] decoded = decoded_opt!!;
	defer mem::free(decoded.ptr);

	test::eq(decoded.len, text.len);
	for (usz i = 0; i < text.len; i++)
	{
		test::eq(decoded[i], text[i]);
	}
}

// ============================================================================
// BPE with merges (matches bpe_test.c3)
// ============================================================================

fn void test_fast_bpe_with_merges() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	b.pretok = { .mode = pretokenizer::Mode.NONE };

	b.v.deinit();
	b.v.init_byte_level();

	// Add merge: 'h' + 'e' -> he_id
	uint he_id = b.v.add_token("he");
	b.merges.add_merge('h', 'e', he_id);

	// Add merge: he_id + 'l' -> hel_id
	uint hel_id = b.v.add_token("hel");
	b.merges.add_merge(he_id, 'l', hel_id);

	uint[]? ids_opt = b.encode("hello");
	uint[] ids = ids_opt!!;
	defer mem::free(ids.ptr);

	// Should merge: h+e -> he, he+l -> hel, remaining: l, o
	// Result: [hel, l, o]
	test::eq(ids.len, (usz)3);
	test::eq(ids[0], hel_id);
	test::eq(ids[1], (uint)'l');
	test::eq(ids[2], (uint)'o');

	// Decode should roundtrip
	char[]? decoded_opt = b.decode(ids);
	char[] decoded = decoded_opt!!;
	defer mem::free(decoded.ptr);

	test::eq(decoded.len, (usz)5);
	test::eq(decoded[0], 'h');
	test::eq(decoded[1], 'e');
	test::eq(decoded[2], 'l');
	test::eq(decoded[3], 'l');
	test::eq(decoded[4], 'o');
}

// ============================================================================
// Empty input
// ============================================================================

fn void test_fast_bpe_empty_input() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	b.v.deinit();
	b.v.init_byte_level();

	uint[]? ids_opt = b.encode("");
	uint[] ids = ids_opt!!;
	test::eq(ids.len, (usz)0);
}

// ============================================================================
// Cache hit test: encoding the same word twice should use cache
// ============================================================================

fn void test_fast_bpe_cache_hit() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	b.pretok = { .mode = pretokenizer::Mode.NONE };
	b.v.deinit();
	b.v.init_byte_level();

	uint he_id = b.v.add_token("he");
	b.merges.add_merge('h', 'e', he_id);

	// First encode — populates cache
	uint[]? ids1_opt = b.encode("hello");
	uint[] ids1 = ids1_opt!!;
	defer mem::free(ids1.ptr);

	// Second encode — should hit cache
	uint[]? ids2_opt = b.encode("hello");
	uint[] ids2 = ids2_opt!!;
	defer mem::free(ids2.ptr);

	// Results must be identical
	test::eq(ids1.len, ids2.len);
	for (usz i = 0; i < ids1.len; i++)
	{
		test::eq(ids1[i], ids2[i]);
	}
}

// ============================================================================
// Real tokenizer test: Load Mamba tokenizer and compare with reference IDs
// ============================================================================

const char[] TOKENIZER_PATH = "/home/andrew/migration/archived/models/mamba-2.8b-hf/tokenizer.json";

fn void test_fast_bpe_real_tokenizer_hello() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	if (catch b.load_huggingface(TOKENIZER_PATH)) return;

	// "hello" -> [25521] per tokenizer_compare.c3
	uint[]? ids_opt = b.encode("hello");
	uint[] ids = ids_opt!!;
	defer if (ids.len > 0) { mem::free(ids.ptr); };

	test::eq(ids.len, (usz)1);
	test::eq(ids[0], (uint)25521);
}

fn void test_fast_bpe_real_tokenizer_hello_world() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	if (catch b.load_huggingface(TOKENIZER_PATH)) return;

	// "Hello, world!" -> [12092, 13, 1533, 2]
	uint[]? ids_opt = b.encode("Hello, world!");
	uint[] ids = ids_opt!!;
	defer if (ids.len > 0) { mem::free(ids.ptr); };

	uint[4] expected = { 12092, 13, 1533, 2 };
	test::eq(ids.len, (usz)4);
	for (usz i = 0; i < 4; i++)
	{
		test::eq(ids[i], expected[i]);
	}
}

fn void test_fast_bpe_real_tokenizer_quick_fox() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	if (catch b.load_huggingface(TOKENIZER_PATH)) return;

	// "The quick brown fox jumps over the lazy dog."
	uint[10] expected = { 510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370, 15 };
	uint[]? ids_opt = b.encode("The quick brown fox jumps over the lazy dog.");
	uint[] ids = ids_opt!!;
	defer if (ids.len > 0) { mem::free(ids.ptr); };

	test::eq(ids.len, (usz)10);
	for (usz i = 0; i < 10; i++)
	{
		test::eq(ids[i], expected[i]);
	}
}

fn void test_fast_bpe_real_tokenizer_roundtrip() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	if (catch b.load_huggingface(TOKENIZER_PATH)) return;

	char[][] prompts = {
		"Hello, ",
		"The quick brown fox ",
		"Once upon a time, ",
		"Hello, world!",
		"The answer is ",
	};

	for (usz i = 0; i < prompts.len; i++)
	{
		char[] prompt = prompts[i];

		uint[]? ids_opt = b.encode(prompt);
		uint[] ids = ids_opt!!;
		defer if (ids.len > 0) { mem::free(ids.ptr); };

		assert(ids.len > 0);

		char[]? decoded_opt = b.decode(ids);
		char[] decoded = decoded_opt!!;
		defer if (decoded.len > 0) { mem::free(decoded.ptr); };

		test::eq(decoded.len, prompt.len);
		for (usz j = 0; j < prompt.len; j++)
		{
			test::eq(decoded[j], prompt[j]);
		}
	}
}

fn void test_fast_bpe_real_tokenizer_special_chars() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	if (catch b.load_huggingface(TOKENIZER_PATH)) return;

	// "!@#$%^&*()" -> [2, 33, 4, 5, 6, 63, 7, 11, 1082]
	uint[9] expected = { 2, 33, 4, 5, 6, 63, 7, 11, 1082 };
	uint[]? ids_opt = b.encode("!@#$%^&*()");
	uint[] ids = ids_opt!!;
	defer if (ids.len > 0) { mem::free(ids.ptr); };

	test::eq(ids.len, (usz)9);
	for (usz i = 0; i < 9; i++)
	{
		test::eq(ids[i], expected[i]);
	}
}

fn void test_fast_bpe_real_tokenizer_leading_space() @test
{
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	if (catch b.load_huggingface(TOKENIZER_PATH)) return;

	// " leading space" -> [4283, 2317]
	uint[2] expected = { 4283, 2317 };
	uint[]? ids_opt = b.encode(" leading space");
	uint[] ids = ids_opt!!;
	defer if (ids.len > 0) { mem::free(ids.ptr); };

	test::eq(ids.len, (usz)2);
	for (usz i = 0; i < 2; i++)
	{
		test::eq(ids[i], expected[i]);
	}
}
