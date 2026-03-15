///! BPE Tokenizer Benchmark
///!
///! Benchmarks the BPE tokenizer using real Wikipedia text data.
///!
///! Build/run:
///!   cc -c -o /tmp/cJSON.o src/utils/cjson/cJSON.c -I src/utils/cjson
///!   c3c compile-benchmark src/tokenizers/*.c3 src/utils/cjson_bindings.c3 \
///!     testing/tokenizers/bpe_benchmark.c3 -z /tmp/cJSON.o -O3
///!
///! NOTE: C3's benchmark runtime has a bug where max_iterations < 100
///! causes a division-by-zero (runtime_benchmark.c3:127 computes
///! print_step = iterations / 100, then does modulo on line 131).
///! We must use >= 100 iterations.

module metaphor::testing::tokenizers::bpe_benchmark;

import metaphor::tokenizers::bpe;
import std::io;
import std::core::mem;

const char[] TOKENIZER_PATH = "/home/andrew/migration/archived/models/mamba-2.8b-hf/tokenizer.json";
const char[] WIKI_DATA_PATH = "/home/andrew/migration/metaphor/testing/tokenizers/wiki_test_data.txt";

// Global state
bpe::Bpe tokenizer;
char[] wiki_text;
bool loaded = false;

fn void ensure_loaded() @inline
{
	if (loaded) return;
	loaded = true;

	tokenizer.init();
	tokenizer.load_huggingface(TOKENIZER_PATH)!!;

	String p = string::tformat("%s", (ZString)WIKI_DATA_PATH.ptr);
	wiki_text = io::file::load(mem, p)!!;
	io::printfn("Loaded tokenizer + %d bytes of Wikipedia text", wiki_text.len);
}

fn void bench_setup() @init
{
	// Must be >= 100 due to C3 runtime bug (division by zero in progress printing).
	set_benchmark_warmup_iterations(3);
	set_benchmark_max_iterations(100);
}

// ============================================================================

module metaphor::testing::tokenizers::bpe_benchmark @benchmark;
import metaphor::tokenizers::bpe;
import std::core::mem;

fn void bpe_encode()
{
	ensure_loaded();
	uint[]? ids_opt = tokenizer.encode(wiki_text);
	if (try ids = ids_opt)
	{
		if (ids.len > 0) mem::free(ids.ptr);
	}
}

fn void bpe_roundtrip()
{
	ensure_loaded();
	uint[]? ids_opt = tokenizer.encode(wiki_text);
	if (try ids = ids_opt)
	{
		char[]? dec_opt = tokenizer.decode(ids);
		if (try dec = dec_opt)
		{
			if (dec.len > 0) mem::free(dec.ptr);
		}
		if (ids.len > 0) mem::free(ids.ptr);
	}
}
