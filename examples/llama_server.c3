//! LLaMA 3.1 8B Instruct — Pipe-based Inference Server
//!
//! Pure token-in/token-out backend. Reads commands from stdin,
//! writes responses to stdout. No tokenizer, no chat template,
//! no tool execution — all orchestration lives in the Python frontend.
//!
//! Protocol:
//!   → PREFILL <id1> <id2> ...    Prefill KV cache with token IDs
//!   ← OK
//!   → GENERATE <max_tokens>      Generate tokens, streaming results
//!   ← TOKEN <id>                 (one per generated token)
//!   ← DONE
//!   → RESET                      Clear KV cache + state
//!   ← OK
//!   → QUIT                       Shut down
//!
//! Run: c3c build llama_server && python3 ../data/spook_chat/chat.py

module metaphor::examples::llama_server;
import metaphor::examples::llama_generate;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::utils::sampling;
import metaphor::rtti;
import std::io;
import std::core::mem;
import libc;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HipDevice = hip_device::HipDevice;
alias LlamaModel = llama_generate::LlamaModel;
alias LlamaConfig = llama_generate::LlamaConfig;
alias GenerationState = llama_generate::GenerationState;
alias Datatype = rtti::Datatype;

// LLaMA 3.1 special tokens
const uint SPECIAL_TOKEN_BASE = 128000;
const uint PYTHON_TAG = 128010;       // <|python_tag|> — signals tool call, must pass through
const uint EOT_ID = 128009;           // <|eot_id|> — end of turn

// Prefill state saved between PREFILL and GENERATE
float[] g_prefill_logits;       // [vocab] float — host-side, ready for sampling
bool g_has_prefill_logits = false;

// ============================================================================
// Protocol I/O
// ============================================================================

fn void protocol_send(char[] msg)
{
	io::stdout().write(msg)!!;
	io::stdout().write("\n")!!;
	io::stdout().flush()!!;
}

fn void protocol_send_token(uint token_id)
{
	io::printfn("TOKEN %d", token_id);
	io::stdout().flush()!!;
}

fn usz read_line(char* buf, usz cap)
{
	usz len = 0;
	while (len < cap - 1)
	{
		char[1] ch;
		usz n = io::stdin().read(&ch) ?? 0;
		if (n == 0)
		{
			if (len == 0) return 0;  // EOF
			break;
		}
		if (ch[0] == '\n') break;
		buf[len++] = ch[0];
	}
	return len;
}

fn bool starts_with(char[] s, char[] prefix)
{
	if (s.len < prefix.len) return false;
	for (usz i = 0; i < prefix.len; i++)
	{
		if (s[i] != prefix[i]) return false;
	}
	return true;
}

/// Parse space-separated unsigned integers from a string.
/// Returns the number of tokens parsed.
fn usz parse_token_ids(char[] s, uint* out, usz cap)
{
	usz count = 0;
	usz i = 0;

	while (i < s.len && count < cap)
	{
		// Skip spaces
		while (i < s.len && s[i] == ' ') i++;
		if (i >= s.len) break;

		// Parse integer
		uint val = 0;
		bool got_digit = false;
		while (i < s.len && s[i] >= '0' && s[i] <= '9')
		{
			val = val * 10 + (uint)(s[i] - '0');
			got_digit = true;
			i++;
		}
		if (got_digit)
		{
			out[count++] = val;
		}
	}
	return count;
}

/// Parse an unsigned integer from a string.
fn usz parse_uint(char[] s)
{
	usz val = 0;
	usz i = 0;
	while (i < s.len && s[i] == ' ') i++;
	while (i < s.len && s[i] >= '0' && s[i] <= '9')
	{
		val = val * 10 + (usz)(s[i] - '0');
		i++;
	}
	return val;
}

// ============================================================================
// Command Handlers
// ============================================================================

/// Process a single token through model_step (no prefill).
/// For debugging: step-by-step comparison with reference_llama.py.
/// Protocol: STEP <token_id> → STEP_OK <logit_0> <logit_1> ... <logit_7>
fn void handle_step(
	LlamaModel* model,
	GenerationState* state,
	char[] args,
	usz vocab_size
)
{
	uint[1] tokens;
	usz n = parse_token_ids(args, &tokens, 1);
	if (n == 0)
	{
		protocol_send("ERROR empty step");
		return;
	}

	state.append(tokens[0]);

	// Run the decode graph for this single token
	llama_generate::model_step(model, (ulong)tokens[0])!!;

	// Load logits into sampler's probs buffer (DtoH + dtype conversion)
	model.logits.collect()!!;
	sampling::Sampler* sampler = &llama_generate::g_sampler;
	usz sample_vocab = sampler.load_logits(model.logits, model.dev);
	float[] flogits = sampler.probs[:sample_vocab];

	// Find argmax
	uint argmax_idx = 0;
	float argmax_val = flogits[0];
	for (usz i = 1; i < sample_vocab; i++)
	{
		if (flogits[i] > argmax_val)
		{
			argmax_val = flogits[i];
			argmax_idx = (uint)i;
		}
	}

	io::eprintf("[STEP_LOGITS] first 8 = [");
	usz dump_n = 8;
	if (dump_n > sample_vocab) dump_n = sample_vocab;
	for (usz i = 0; i < dump_n; i++)
	{
		if (i > 0) io::eprintf(", ");
		io::eprintf("%.4f", flogits[i]);
	}
	io::eprintfn("]");
	io::eprintfn("[STEP_LOGITS] argmax = %d (val=%.4f)", argmax_idx, argmax_val);

	io::printfn("STEP_OK %d", argmax_idx);
	io::stdout().flush()!!;
}

fn void handle_prefill(
	LlamaModel* model,
	GenerationState* state,
	char[] args
)
{
	uint[8192] tokens;
	usz n_tokens = parse_token_ids(args, &tokens, tokens.len);

	if (n_tokens == 0)
	{
		protocol_send("ERROR empty prefill");
		return;
	}

	// Append tokens to state (for sampling context)
	for (usz i = 0; i < n_tokens; i++)
	{
		state.append(tokens[i]);
	}

	// Run prefill — returns host-side float[vocab] logits
	g_prefill_logits = llama_generate::model_prefill(model, tokens[:n_tokens])!!;
	g_has_prefill_logits = true;

	protocol_send("OK");
}

fn void handle_generate(
	LlamaModel* model,
	GenerationState* state,
	char[] args,
	usz vocab_size
)
{
	usz max_tokens = parse_uint(args);
	if (max_tokens == 0) max_tokens = 512;

	for (usz i = 0; i < max_tokens; i++)
	{
		uint[] ctx = state.all_tokens[:state.num_tokens];
		uint next_token;

		if (i == 0 && g_has_prefill_logits)
		{
			// Prefill logits are host-side float[] — sample directly
			next_token = llama_generate::g_sampler.sample_with_context(g_prefill_logits, ctx);
			g_has_prefill_logits = false;
		}
		else
		{
			// Decode logits from graph
			model.logits.collect()!!;
			next_token = llama_generate::g_sampler.sample_tensor(model.logits, model.dev, ctx);
		}

		// Stop on end-of-turn special tokens, but let <|python_tag|> through
		// so the Python frontend can detect and handle tool calls
		if (next_token >= SPECIAL_TOKEN_BASE && next_token != PYTHON_TAG)
		{
			break;
		}

		state.append(next_token);
		protocol_send_token(next_token);

		// Advance model
		llama_generate::model_step(model, (ulong)next_token)!!;
	}

	protocol_send("DONE");
}

// ============================================================================
// Main
// ============================================================================

fn void main(String[] args) @if($feature(LLAMA_SERVER_MODE))
{
	ZString weights_path = null;
	usz max_seq_len = 2048;
	float temperature = 0.7;
	uint top_k = 40;
	float top_p = 0.9;
	ulong seed = 0;
	usz vocab_size = 128256;

	usz i = 1;
	while (i < args.len)
	{
		String arg = args[i];
		if (arg == "--weights" && i + 1 < args.len)
		{
			weights_path = (ZString)args[i + 1].ptr; i += 2;
		}
		else if (arg == "--max-seq-len" && i + 1 < args.len)
		{
			max_seq_len = (usz)(args[i + 1].to_int() ?? 2048); i += 2;
		}
		else if (arg == "--temperature" && i + 1 < args.len)
		{
			temperature = (float)(args[i + 1].to_double() ?? 0.7); i += 2;
		}
		else if (arg == "--top-k" && i + 1 < args.len)
		{
			top_k = (uint)(args[i + 1].to_int() ?? 40); i += 2;
		}
		else if (arg == "--top-p" && i + 1 < args.len)
		{
			top_p = (float)(args[i + 1].to_double() ?? 0.9); i += 2;
		}
		else if (arg == "--seed" && i + 1 < args.len)
		{
			seed = (ulong)(args[i + 1].to_long() ?? 0); i += 2;
		}
		else if (arg == "--vocab-size" && i + 1 < args.len)
		{
			vocab_size = (usz)(args[i + 1].to_int() ?? 128256); i += 2;
		}
		else
		{
			io::printfn("Unknown argument: %s", arg); return;
		}
	}

	if (weights_path == null)
	{
		io::printfn("Usage: llama_server --weights <path> [options]");
		io::printfn("  --max-seq-len <int>    KV cache size (default: 2048)");
		io::printfn("  --temperature <float>  Sampling temp (default: 0.7)");
		io::printfn("  --top-k <int>          Top-k (default: 40)");
		io::printfn("  --top-p <float>        Nucleus (default: 0.9)");
		io::printfn("  --seed <int>           Random seed (default: 0=random)");
		io::printfn("  --vocab-size <int>     Vocab size for sampling (default: 128256)");
		return;
	}

	// Init GPU + model
	HipDevice gpu = hip_device::hip_device_create({ .device_id = 0 });
	defer gpu.deinit();
	DeviceReference dev = gpu.reference();

	graph::init();


	LlamaConfig cfg = llama_generate::LLAMA_31_8B;
	cfg.max_seq_len = (ulong)max_seq_len;

	io::printfn("Building model...");
	LlamaModel model = llama_generate::model_init(&cfg, dev)!!;

	usz wlen = libc::strlen(weights_path);
	llama_generate::load_weights(&model, weights_path[:wlen])!!;

	// Sampling
	sampling::SamplingConfig sample_cfg = {
		.temperature = temperature,
		.top_k = top_k,
		.top_p = top_p,
		.repetition_penalty = 1.1,
		.ngram_size = 4,
		.frequency_penalty = 0.0,
		.penalty_exempt_below = 0,
		.penalty_length_threshold = 4,
		.seed = seed,
	};

	llama_generate::g_sampler.init((uint)vocab_size, sample_cfg);
	defer llama_generate::g_sampler.deinit();

	GenerationState state;
	state.init();
	defer state.deinit();

	// Signal ready
	protocol_send("READY");

	// Command loop
	char[65536] line_buf;
	while (true)
	{
		usz line_len = read_line(&line_buf, line_buf.len);
		if (line_len == 0) break;  // EOF → exit

		char[] line = ((char*)&line_buf)[:line_len];

		if (starts_with(line, "PREFILL "))
		{
			handle_prefill(&model, &state, line[8..]);
		}
		else if (starts_with(line, "GENERATE"))
		{
			char[] gen_args = line.len > 9 ? line[9..] : "";
			handle_generate(&model, &state, gen_args, vocab_size);
		}
		else if (starts_with(line, "STEP "))
		{
			handle_step(&model, &state, line[5..], vocab_size);
		}
		else if (starts_with(line, "RESET"))
		{
			llama_generate::model_reset(&model);
			state.reset();
			g_has_prefill_logits = false;
			protocol_send("OK");
		}
		else if (starts_with(line, "QUIT"))
		{
			break;
		}
		else
		{
			io::printfn("ERROR unknown command");
			io::stdout().flush()!!;
		}
	}
}
