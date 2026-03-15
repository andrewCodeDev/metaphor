///! Tokenizer Roundtrip Test
///!
///! Verifies that the C3 BPE tokenizer can:
///! 1. Load the real Mamba 2.8B HuggingFace tokenizer
///! 2. Encode prompts to token IDs matching expected values
///! 3. Decode token IDs back to the original text (roundtrip)
///!
///! This validates the tokenizer is suitable for text generation:
///! if Mamba produces token IDs, decode must produce legible text.

module metaphor::testing::tokenizers::tokenizer_roundtrip_test;
import metaphor::tokenizers::bpe;
import std::io;
import std::core::mem;

const char[] TOKENIZER_PATH = "/home/andrew/migration/archived/models/mamba-2.8b-hf/tokenizer.json";

// ============================================================================
// Helper: load tokenizer (shared across tests)
// ============================================================================

fn bpe::Bpe? load_test_tokenizer()
{
	bpe::Bpe b;
	b.init();
	b.load_huggingface(TOKENIZER_PATH)!;
	return b;
}

// ============================================================================
// Test: Encode "Hello, " and verify token IDs
// ============================================================================

fn void test_encode_hello_prompt() @test
{
	bpe::Bpe b = load_test_tokenizer()!!;
	defer b.deinit();

	// Encode "Hello, " - this is our generation prompt
	uint[] ids = b.encode("Hello, ")!!;
	defer if (ids.len > 0) { mem::free(ids.ptr); };

	// "Hello, world!" encodes as [12092, 13, 1533, 2] per tokenizer_compare.c3
	// "Hello, " should encode as the first tokens of that sequence
	// Expected: "Hello" -> 12092, "," -> 13, " " -> 220
	// (GPT-2 tokenizer: space = byte 32 -> codepoint 288 -> Ġ -> token 220)
	assert(ids.len >= 2, "Expected at least 2 tokens for 'Hello, '");

	// First token should be "Hello" = 12092
	assert(ids[0] == 12092, "Expected token 12092 for 'Hello'");
	// Second token should be "," = 13
	assert(ids[1] == 13, "Expected token 13 for ','");
}

// ============================================================================
// Test: Decode known token IDs back to text
// ============================================================================

fn void test_decode_known_ids() @test
{
	bpe::Bpe b = load_test_tokenizer()!!;
	defer b.deinit();

	// Decode the known token IDs for "Hello, world!"
	uint[4] known_ids = { 12092, 13, 1533, 2 };
	char[] decoded = b.decode(&known_ids)!!;
	defer if (decoded.len > 0) { mem::free(decoded.ptr); };

	assert(decoded.len == 13, "Expected 13 chars for 'Hello, world!'");
	assert(str_eq(decoded, "Hello, world!"), "Decoded text should be 'Hello, world!'");
}

// ============================================================================
// Test: Full encode/decode roundtrip for generation prompts
// ============================================================================

fn void test_roundtrip_generation_prompts() @test
{
	bpe::Bpe b = load_test_tokenizer()!!;
	defer b.deinit();

	// Test prompts relevant to text generation
	char[][] prompts = {
		"Hello, ",
		"The quick brown fox ",
		"Once upon a time, ",
		"Hello, world!",
		"def fibonacci(n):\n    ",
		"The answer is ",
		"In the beginning ",
	};

	for (usz i = 0; i < prompts.len; i++)
	{
		char[] prompt = prompts[i];

		// Encode
		uint[] ids = b.encode(prompt)!!;
		defer if (ids.len > 0) { mem::free(ids.ptr); };

		assert(ids.len > 0, "Encoding should produce at least 1 token");

		// Decode
		char[] decoded = b.decode(ids)!!;
		defer if (decoded.len > 0) { mem::free(decoded.ptr); };

		// Verify roundtrip
		if (!str_eq(decoded, prompt))
		{
			io::eprintfn("ROUNDTRIP FAIL for prompt[%d]: '%s'", (int)i, prompt);
			io::eprintfn("  Encoded to %d tokens", (int)ids.len);
			io::eprintf("  Token IDs: [");
			for (usz j = 0; j < ids.len; j++)
			{
				if (j > 0) io::eprintf(", ");
				io::eprintf("%d", ids[j]);
			}
			io::eprintn("]");
			io::eprintfn("  Decoded: '%s' (len=%d)", decoded, (int)decoded.len);
			assert(false, "Roundtrip failed");
		}
	}
}

// ============================================================================
// Test: Decode simulated generation output
// Simulate what Mamba would produce: prompt tokens + generated tokens
// ============================================================================

fn void test_decode_simulated_generation() @test
{
	bpe::Bpe b = load_test_tokenizer()!!;
	defer b.deinit();

	// Encode "Hello, " as prompt
	uint[] prompt_ids = b.encode("Hello, ")!!;
	defer if (prompt_ids.len > 0) { mem::free(prompt_ids.ptr); };

	// Verify prompt roundtrip
	char[] prompt_decoded = b.decode(prompt_ids)!!;
	defer if (prompt_decoded.len > 0) { mem::free(prompt_decoded.ptr); };
	assert(str_eq(prompt_decoded, "Hello, "), "Prompt roundtrip should work");

	// Verify full target roundtrip (simulates generation output)
	uint[] full_ids = b.encode("Hello, world!")!!;
	defer if (full_ids.len > 0) { mem::free(full_ids.ptr); };
	char[] full_decoded = b.decode(full_ids)!!;
	defer if (full_decoded.len > 0) { mem::free(full_decoded.ptr); };
	assert(str_eq(full_decoded, "Hello, world!"), "Full target should decode correctly");
}

// ============================================================================
// Test: Decode individual tokens to verify vocab lookup
// ============================================================================

fn void test_decode_individual_tokens() @test
{
	bpe::Bpe b = load_test_tokenizer()!!;
	defer b.deinit();

	// Verify individual token decoding for common tokens
	uint[6] check_ids = { 12092, 13, 1533, 2, 510, 3158 };
	char[][6] check_expected = { "Hello", ",", " world", "!", "The", " quick" };

	for (usz i = 0; i < 6; i++)
	{
		uint[1] id_arr = { check_ids[i] };
		char[] decoded = b.decode(&id_arr)!!;
		defer if (decoded.len > 0) { mem::free(decoded.ptr); };

		if (!str_eq(decoded, check_expected[i]))
		{
			io::eprintfn("Token %d: expected '%s', got '%s'",
				check_ids[i], check_expected[i], decoded);
			assert(false, "Individual token decode mismatch");
		}
	}
}

// ============================================================================
// Helpers
// ============================================================================

fn bool str_eq(char[] a, char[] b) @inline
{
	if (a.len != b.len) return false;
	for (usz i = 0; i < a.len; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}
