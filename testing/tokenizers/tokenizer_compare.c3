///! Tokenizer Comparison Test
///!
///! Loads the real Mamba tokenizer.json and compares C3 BPE output
///! against HuggingFace reference token IDs.

module metaphor::testing::tokenizers::tokenizer_compare;
import metaphor::tokenizers::bpe;
import std::io;
import std::core::mem;

const char[] TOKENIZER_PATH = "/home/andrew/migration/archived/models/mamba-2.8b-hf/tokenizer.json";

struct TestCase
{
	char[] text;
	uint[] expected_ids;
}

fn void main()
{
	io::printn("=== C3 vs HuggingFace Tokenizer Comparison ===\n");

	// Load tokenizer
	bpe::Bpe b;
	b.init();
	defer b.deinit();

	io::printn("Loading tokenizer...");
	if (catch b.load_huggingface(TOKENIZER_PATH))
	{
		io::printn("FATAL: failed to load tokenizer");
		return;
	}

	io::printfn("Loaded tokenizer: %d vocab tokens, %d merges\n",
		b.v.len(), b.merges.rules.len());

	// Test cases with expected HuggingFace token IDs
	uint[] e0  = { 25521 };
	uint[] e1  = { 12092, 13, 1533, 2 };
	uint[] e2  = { 510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370, 15 };
	uint[] e3  = { 1545, 5713, 251, 42401, 9, 79, 2262, 187, 50274, 338,
	               295, 11049, 337, 27, 187, 50270, 2309, 295, 187, 50274,
	               2309, 5713, 251, 42401, 9, 79, 14, 18, 10, 559, 5713,
	               251, 42401, 9, 79, 14, 19, 10 };
	uint[] e4  = { 23433, 27, 1737, 310, 374, 12, 19, 32, 187, 32869, 27, 577, 187, 29, 8112, 31 };
	uint[] e5  = { 27363, 27, 187, 3845, 2073, 25521, 1533, 3401, 187, 29, 8112, 31 };
	uint[] e6  = { 4283, 2317 };
	uint[] e7  = { 33754, 186, 1568 };
	uint[] e8  = { 1826, 8737, 535, 187, 34263 };
	uint[] e10 = { 66 };
	uint[] e11 = { 50275 };
	uint[] e12 = { 2, 33, 4, 5, 6, 63, 7, 11, 1082 };
	uint[] e13 = { 68, 2320, 860, 9568, 2204, 860, 36060 };
	uint[] e14 = { 34576, 34576, 34576, 34576, 34576, 34576, 34576, 34576,
	               34576, 34576, 34576, 34576, 16321 };

	TestCase[15] tests = {
		{ .text = "hello",                                             .expected_ids = e0  },
		{ .text = "Hello, world!",                                     .expected_ids = e1  },
		{ .text = "The quick brown fox jumps over the lazy dog.",       .expected_ids = e2  },
		{ .text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", .expected_ids = e3 },
		{ .text = "Question: What is 2+2?\nAnswer: 4\n<END>",          .expected_ids = e4  },
		{ .text = "Python:\nprint('hello world')\n<END>",              .expected_ids = e5  },
		{ .text = " leading space",                                    .expected_ids = e6  },
		{ .text = "tabs\there",                                        .expected_ids = e7  },
		{ .text = "newlines\n\n\nmultiple",                            .expected_ids = e8  },
		{ .text = "",                                                  .expected_ids = {} },
		{ .text = "a",                                                 .expected_ids = e10 },
		{ .text = "   ",                                               .expected_ids = e11 },
		{ .text = "!@#$%^&*()",                                        .expected_ids = e12 },
		{ .text = "café résumé naïve",                                 .expected_ids = e13 },
		{ .text = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", .expected_ids = e14 },
	};

	uint passed = 0;
	uint failed = 0;

	for (usz i = 0; i < 15; i++)
	{
		TestCase* tc = &tests[i];

		io::printfn("[%2d] Testing: \"%s\"", (int)i, tc.text);

		uint[]? ids_opt = b.encode(tc.text);
		if (catch ids_opt)
		{
			io::printfn("     FAIL: encode returned error");
			failed++;
			continue;
		}
		uint[] ids = ids_opt;

		// Compare length
		if (ids.len != tc.expected_ids.len)
		{
			io::printfn("     FAIL: length mismatch: got %d, expected %d",
				(int)ids.len, (int)tc.expected_ids.len);
			io::printf("     Got:      [");
			for (usz j = 0; j < ids.len; j++)
			{
				if (j > 0) io::printf(", ");
				io::printf("%d", ids[j]);
			}
			io::printn("]");
			io::printf("     Expected: [");
			for (usz j = 0; j < tc.expected_ids.len; j++)
			{
				if (j > 0) io::printf(", ");
				io::printf("%d", tc.expected_ids[j]);
			}
			io::printn("]");
			failed++;
			if (ids.len > 0) mem::free(ids.ptr);
			continue;
		}

		// Compare token IDs
		bool match = true;
		for (usz j = 0; j < ids.len; j++)
		{
			if (ids[j] != tc.expected_ids[j])
			{
				io::printfn("     FAIL: token[%d] mismatch: got %d, expected %d",
					(int)j, ids[j], tc.expected_ids[j]);
				match = false;
			}
		}

		if (match)
		{
			io::printfn("     PASS (%d tokens)", (int)ids.len);
			passed++;
		}
		else
		{
			failed++;
		}

		if (ids.len > 0) mem::free(ids.ptr);
	}

	io::printfn("\n=== Results: %d passed, %d failed ===", passed, failed);
	if (failed == 0)
	{
		io::printn("SUCCESS: C3 tokenizer matches HuggingFace byte-for-byte!");
	}
}
