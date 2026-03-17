//! LLaMA LoRA Training (HIP)
//!
//! Trains LoRA adapters on LLaMA 3.1 8B with causal masked loss.
//! Only LoRA parameters are trained; base model weights remain frozen.
//!
//! Required:
//!   --data-dir <path>     Directory with train_tokens.bin + train_meta.json
//!   --weights <path>      Pretrained base weights (.safetensors)
//!
//! Optional:
//!   --lora-out <path>     Output checkpoint (default: llama_lora_output.safetensors)
//!   --resume-lora <path>  Resume from a LoRA checkpoint
//!   --lr <float>          Learning rate (default: 2e-4)
//!   --epochs <int>        Number of epochs (default: 1)
//!   --warmup <int>        Warmup steps (default: 50)
//!   --log-interval <int>  Log every N steps (default: 10)
//!   --save-interval <int> Save checkpoint every N steps (default: 500)
//!   --max-steps <int>     Stop after N steps (default: 0=unlimited)
//!   --start-step <int>    Skip N sequences to resume (default: 0)
//!   --seed <int>          PRNG seed for shuffling (default: 0=sequential)
//!   --layers <int>        Number of layers (default: all)
//!   --train-layers <int>  Only train last N layers with LoRA (default: all)
//!   --lora-rank <int>     LoRA rank (default: 16)
//!   --target-loss <float> Stop when avg loss drops below this (default: 0=disabled)
//!
//!   Sequence length is derived automatically from the dataset's pred_len.

module metaphor::examples::llama_lora_train;
import metaphor::examples::llama;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::rtti;
import metaphor::utils::token_dataset;
import metaphor::events;
import metaphor::logging;
import metaphor::graph::execution_sequence;
import std::io;
import std::time::clock;
import std::core::mem;
import libc;

faultdef INVALID_ARGS;

// ============================================================================
// NaN Detection Hook
// ============================================================================

CFile nan_log;
bool nan_found = false;
int nan_exec_count = 0;

fn void nan_detect_hook(tensor_core::TensorCore* tc, events::EventKind kind)
{
	if (kind != events::EventKind.EXECUTE) return;
	if (!tc.has_data()) return;

	nan_exec_count++;

	rtti::Datatype dt = tc.dtype();
	usz n_elems = tc.shape().size();
	// Check up to 32768 elements to catch NaN deeper in large tensors
	// Full check for labeled tensors or late-stage ops (last 10), quick check for others
	bool is_labeled = tc.label_or_null() != null;
	bool is_late = nan_exec_count > 1510;  // last ~12 ops of 1522 forward ops
	usz check_n = (is_labeled || is_late) ? n_elems : (n_elems < 32768 ? n_elems : 32768);
	usz print_n = n_elems < 16 ? n_elems : 16;

	ZString label = tc.label_or_null();
	if (label == null) label = "(unlabeled)";

	tensor::Tensor t = { ._core = tc };
	bool has_nan = false;
	int nan_count = 0;

	// Print shape for identification
	if (nan_log != null)
	{
		libc::fprintf(nan_log, "[%d] %s (%s, %d elems, shape=[",
			nan_exec_count, label,
			dt == rtti::Datatype.BF16 ? "bf16" : (dt == rtti::Datatype.F32 ? "f32" : "other"),
			n_elems);
		for (usz di = 0; di < tc.shape().len; di++)
		{
			if (di > 0) libc::fprintf(nan_log, ",");
			libc::fprintf(nan_log, "%d", tc.shape().get(di));
		}
		libc::fprintf(nan_log, "]):");
	}

	if (dt == rtti::Datatype.BF16)
	{
		// Heap-allocate for large checks
		char* raw = (char*)libc::malloc(check_n * bfloat.sizeof);
		assert(raw != null, "nan_detect_hook: malloc failed");
		bfloat[] buf = ((bfloat*)raw)[:check_n];
		t.get(buf);

		if (nan_log != null)
		{
			for (usz i = 0; i < check_n; i++)
			{
				float v = (float)buf[i];
				uint vi = *((uint*)&v);
				bool is_nan_val = ((vi & 0x7F800000) == 0x7F800000) && ((vi & 0x007FFFFF) != 0);
				if (is_nan_val) { has_nan = true; nan_count++; }
				if (i < print_n) libc::fprintf(nan_log, " %.4f", (double)v);
			}
			if (has_nan) {
				libc::fprintf(nan_log, " *** NAN (%d/%d checked) ***\n", nan_count, (int)check_n);
			} else {
				libc::fprintf(nan_log, "\n");
			}
		}
		libc::free(raw);
	}
	else if (dt == rtti::Datatype.F32)
	{
		char* raw = (char*)libc::malloc(check_n * float.sizeof);
		assert(raw != null, "nan_detect_hook: malloc failed");
		float[] buf = ((float*)raw)[:check_n];
		t.get(buf);

		if (nan_log != null)
		{
			for (usz i = 0; i < check_n; i++)
			{
				uint vi = *((uint*)&buf[i]);
				bool is_nan_val = ((vi & 0x7F800000) == 0x7F800000) && ((vi & 0x007FFFFF) != 0);
				if (is_nan_val) { has_nan = true; nan_count++; }
				if (i < print_n) libc::fprintf(nan_log, " %.4f", (double)buf[i]);
			}
			if (has_nan) {
				libc::fprintf(nan_log, " *** NAN (%d/%d checked) ***\n", nan_count, (int)check_n);
			} else {
				libc::fprintf(nan_log, "\n");
			}
		}
		libc::free(raw);
	}

	if (has_nan && !nan_found)
	{
		nan_found = true;
		io::printfn("!!! NaN detected at exec #%d: %s (dtype=%d, %d elems)",
			nan_exec_count, label, (int)dt, n_elems);
		if (nan_log != null) { libc::fflush(nan_log); libc::fclose(nan_log); }
		libc::exit(1);
	}
}

// ============================================================================
// Configuration
// ============================================================================

struct TrainConfig
{
	ZString data_dir;
	ZString weights;
	ZString lora_output;
	ZString resume_lora;
	float lr;
	float weight_decay;
	int epochs;
	int warmup_steps;
	int log_interval;
	int save_interval;
	int max_steps;
	int start_step;
	ulong seed;
	int layers;
	int train_layers;
	int lora_rank;
	int seq_len;
	float target_loss;
	int grad_accum;           // 0 = disabled, N = accumulate over N micro-batches
}

fn void print_usage(ZString prog)
{
	io::printfn("Usage: %s --data-dir <path> --weights <path> [options]", prog);
	io::printfn("");
	io::printfn("Required:");
	io::printfn("  --data-dir <path>     Directory with train_tokens.bin + train_meta.json");
	io::printfn("  --weights <path>      Pretrained base weights (.safetensors)");
	io::printfn("");
	io::printfn("Optional:");
	io::printfn("  --lora-out <path>     Output LoRA checkpoint (default: llama_lora_output.safetensors)");
	io::printfn("  --resume-lora <path>  Resume from a LoRA checkpoint");
	io::printfn("  --lr <float>          Learning rate (default: 2e-4)");
	io::printfn("  --weight-decay <float> Weight decay (default: 0.01)");
	io::printfn("  --epochs <int>        Number of epochs (default: 1)");
	io::printfn("  --warmup <int>        Warmup steps (default: 100)");
	io::printfn("  --log-interval <int>  Log every N steps (default: 10)");
	io::printfn("  --save-interval <int> Save checkpoint every N steps (default: 500)");
	io::printfn("  --max-steps <int>     Stop after N steps, 0=unlimited (default: 0)");
	io::printfn("  --start-step <int>    Skip N sequences to resume (default: 0)");
	io::printfn("  --seed <int>          PRNG seed, 0=sequential (default: 0)");
	io::printfn("  --layers <int>        Number of layers (default: all)");
	io::printfn("  --train-layers <int>  Only train last N layers with LoRA (default: all)");
	io::printfn("  --lora-rank <int>     LoRA rank (default: 16)");
	io::printfn("  --seq-len <int>       Training sequence length (default: 512)");
	io::printfn("  --target-loss <float> Stop when avg loss drops below this (default: 0=disabled)");
	io::printfn("  --grad-accum <int>    Gradient accumulation steps, 0=disabled (default: 0)");
}

fn TrainConfig? parse_args(String[] args)
{
	TrainConfig cfg = {
		.data_dir = null,
		.weights = null,
		.lora_output = null,
		.resume_lora = null,
		.lr = 2e-4,
		.weight_decay = 0.01,
		.epochs = 1,
		.warmup_steps = 100,
		.log_interval = 10,
		.save_interval = 500,
		.max_steps = 0,
		.start_step = 0,
		.seed = 0,
		.layers = 0,
		.train_layers = 0,
		.lora_rank = 0,
		.seq_len = 512,
	};

	usz i = 1;
	while (i < args.len)
	{
		String arg = args[i];
		if (arg == "--data-dir" && i + 1 < args.len) { cfg.data_dir = (ZString)args[i + 1].ptr; i += 2; }
		else if (arg == "--weights" && i + 1 < args.len) { cfg.weights = (ZString)args[i + 1].ptr; i += 2; }
		else if (arg == "--lora-out" && i + 1 < args.len) { cfg.lora_output = (ZString)args[i + 1].ptr; i += 2; }
		else if (arg == "--resume-lora" && i + 1 < args.len) { cfg.resume_lora = (ZString)args[i + 1].ptr; i += 2; }
		else if (arg == "--lr" && i + 1 < args.len) { cfg.lr = args[i + 1].to_float() ?? 2e-4; i += 2; }
		else if (arg == "--weight-decay" && i + 1 < args.len) { cfg.weight_decay = args[i + 1].to_float() ?? 0.01; i += 2; }
		else if (arg == "--epochs" && i + 1 < args.len) { cfg.epochs = args[i + 1].to_int() ?? 1; i += 2; }
		else if (arg == "--warmup" && i + 1 < args.len) { cfg.warmup_steps = args[i + 1].to_int() ?? 50; i += 2; }
		else if (arg == "--log-interval" && i + 1 < args.len) { cfg.log_interval = args[i + 1].to_int() ?? 10; i += 2; }
		else if (arg == "--save-interval" && i + 1 < args.len) { cfg.save_interval = args[i + 1].to_int() ?? 500; i += 2; }
		else if (arg == "--max-steps" && i + 1 < args.len) { cfg.max_steps = args[i + 1].to_int() ?? 0; i += 2; }
		else if (arg == "--start-step" && i + 1 < args.len) { cfg.start_step = args[i + 1].to_int() ?? 0; i += 2; }
		else if (arg == "--seed" && i + 1 < args.len) { cfg.seed = (ulong)(args[i + 1].to_long() ?? 0); i += 2; }
		else if (arg == "--layers" && i + 1 < args.len) { cfg.layers = args[i + 1].to_int() ?? 0; i += 2; }
		else if (arg == "--train-layers" && i + 1 < args.len) { cfg.train_layers = args[i + 1].to_int() ?? 0; i += 2; }
		else if (arg == "--lora-rank" && i + 1 < args.len) { cfg.lora_rank = args[i + 1].to_int() ?? 0; i += 2; }
		else if (arg == "--seq-len" && i + 1 < args.len) { cfg.seq_len = args[i + 1].to_int() ?? 512; i += 2; }
		else if (arg == "--target-loss" && i + 1 < args.len) { cfg.target_loss = args[i + 1].to_float() ?? 0.0; i += 2; }
		else if (arg == "--grad-accum" && i + 1 < args.len) { cfg.grad_accum = args[i + 1].to_int() ?? 0; i += 2; }
		else if (arg == "--help" || arg == "-h") { print_usage((ZString)args[0].ptr); return INVALID_ARGS~; }
		else { io::printfn("Unknown argument: %s", arg); print_usage((ZString)args[0].ptr); return INVALID_ARGS~; }
	}

	if (cfg.data_dir == null) { io::printfn("Error: --data-dir is required"); print_usage((ZString)args[0].ptr); return INVALID_ARGS~; }
	if (cfg.weights == null) { io::printfn("Error: --weights is required"); print_usage((ZString)args[0].ptr); return INVALID_ARGS~; }

	if (cfg.lora_output == null)
	{
		cfg.lora_output = "llama_lora_output.safetensors";
	}

	return cfg;
}

// ============================================================================
// Main
// ============================================================================

fn void main(String[] args)
{
	// --- Parse CLI args ---
	TrainConfig tcfg;
	if (catch parse_args(args)) { return; }
	tcfg = parse_args(args)!!;

	// --- Device ---
	hip_device::HipDevice gpu = hip_device::create();
	defer gpu.deinit();
	device_reference::DeviceReference dev = gpu.reference();

	graph::init();
	defer graph::deinit();

	logging::enable(logging::Scope.CODEGEN, logging::Level.ERR);

	// --- Load dataset (early, to derive seq_len) ---
	io::printfn("\nLoading dataset...");
	token_dataset::TokenDataset ds = token_dataset::init(tcfg.data_dir)!!;
	defer ds.free();

	// Shuffle sequences if seed is provided
	if (tcfg.seed != 0)
	{
		ds.shuffle(tcfg.seed);
	}

	usz ds_pred_len = ds.pred_len();
	if (ds.has_weights())
	{
		io::printfn("  Loss weights: loaded from train_weights.bin");
	}
	else
	{
		io::printfn("  Loss weights: not found, using uniform weights (all 1.0)");
	}

	// --- Model config ---
	llama::LlamaConfig cfg = llama::llama_31_8b();
	if (tcfg.layers > 0)
	{
		cfg.n_layers = (ulong)tcfg.layers;
	}
	cfg.seq_len = (ulong)ds_pred_len;

	ulong lora_rank = tcfg.lora_rank > 0 ? (ulong)tcfg.lora_rank : 16;
	llama::LoRAConfig lcfg = {
		.rank = lora_rank,
		.alpha = (float)(lora_rank * 2),
		.adapt_q_proj = true,
		.adapt_k_proj = false,
		.adapt_v_proj = true,
		.adapt_o_proj = true,
		.adapt_gate_proj = false,
		.adapt_up_proj = false,
		.adapt_down_proj = false,
	};

	// --- Print configuration ---
	io::printfn("\n================================================================");
	io::printfn("LoRA LLaMA Training");
	io::printfn("================================================================\n");
	io::printfn("Device: HIP");
	io::printfn("Model:  d_model=%d, n_heads=%d, n_kv_heads=%d, n_layers=%d, vocab=%d",
		cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.n_layers, cfg.vocab_size);
	io::printfn("LoRA:   rank=%d, alpha=%.0f, q=%s, k=%s, v=%s, o=%s, gate=%s, up=%s, down=%s",
		lcfg.rank, lcfg.alpha,
		lcfg.adapt_q_proj ? "yes" : "no",
		lcfg.adapt_k_proj ? "yes" : "no",
		lcfg.adapt_v_proj ? "yes" : "no",
		lcfg.adapt_o_proj ? "yes" : "no",
		lcfg.adapt_gate_proj ? "yes" : "no",
		lcfg.adapt_up_proj ? "yes" : "no",
		lcfg.adapt_down_proj ? "yes" : "no");
	if (tcfg.train_layers > 0)
	{
		io::printfn("        training last %d of %d layers", tcfg.train_layers, cfg.n_layers);
	}
	io::printfn("Data:   %s", tcfg.data_dir);
	io::printfn("LR:     %.1e, wd=%.3f, epochs=%d, warmup=%d, save_every=%d", tcfg.lr, tcfg.weight_decay, tcfg.epochs, tcfg.warmup_steps, tcfg.save_interval);
	if (tcfg.grad_accum > 0)
	{
		io::printfn("Accum:  %d micro-batches per update", tcfg.grad_accum);
	}
	io::printfn("Seq:    %d tokens (from dataset pred_len)", cfg.seq_len);
	if (tcfg.target_loss > 0.0f)
	{
		io::printfn("Target: stop when avg loss < %.2f", tcfg.target_loss);
	}
	io::printfn("Output: %s", tcfg.lora_output);

	// --- Build model ---
	io::printfn("\nBuilding model...");
	llama::LoRALlamaModel model = llama::model_init(cfg, lcfg, dev, tcfg.train_layers)!!;
	cfg = model.config;

	ulong base_params = llama::total_base_params(cfg);
	ulong lora_params = llama::total_lora_params(&model);
	io::printfn("  Base params (frozen):    %d", base_params);
	if (lora_params > 0)
	{
		io::printfn("  LoRA params (trainable): %d", lora_params);
		io::printfn("  LoRA / Base ratio:       %.2f%%",
			(double)lora_params / (double)base_params * 100.0);
	}

	// --- Load pretrained base weights ---
	usz weights_len = libc::strlen(tcfg.weights);
	llama::load_pretrained_weights(&model, tcfg.weights[:weights_len], dev)!!;

	// --- Resume LoRA weights if provided ---
	if (tcfg.resume_lora != null)
	{
		usz lora_path_len = libc::strlen(tcfg.resume_lora);
		llama::load_lora_checkpoint(&model, tcfg.resume_lora[:lora_path_len], dev)!!;
		io::printfn("  Resumed LoRA weights from: %s", tcfg.resume_lora);
	}

	// --- Prepare host buffers ---
	usz model_seq_len = (usz)cfg.seq_len;
	usz vocab = (usz)cfg.vocab_size;
	usz onehot_floats = model_seq_len * vocab;

	ulong* host_tokens = (ulong*)libc::malloc(model_seq_len * ulong.sizeof);
	assert(host_tokens != null, "Failed to allocate host token buffer");
	defer libc::free(host_tokens);

	float* host_onehot = (float*)libc::calloc(onehot_floats, float.sizeof);
	assert(host_onehot != null, "Failed to allocate host onehot buffer");
	defer libc::free(host_onehot);

	uint* input_buf = (uint*)libc::malloc(ds_pred_len * uint.sizeof);
	defer libc::free(input_buf);

	uint* target_buf = (uint*)libc::malloc(ds_pred_len * uint.sizeof);
	assert(input_buf != null && target_buf != null, "Failed to allocate token buffers");
	defer libc::free(target_buf);

	float[] host_weights = mem::alloc_array(float, model_seq_len);
	defer mem::free(host_weights.ptr);

	// --- Build loss graph ---
	// Target one-hot: [seq_len, vocab_size]
	tensor::Tensor target_onehot = tensor::zeros(F32, dev, { cfg.seq_len, cfg.vocab_size })!!;

	// Per-position loss weight: [seq_len]
	tensor::Tensor weight_tensor = tensor::ones(F32, dev, { cfg.seq_len })!!;

	// Squeeze logits: [batch=1, seq_len, vocab] -> [seq_len, vocab], cast to F32 for loss
	// Use squeeze(0) instead of reduce("bsv->sv") — squeeze is zero-copy (share_data_with)
	// while reduce goes through hipTENSOR which introduces NaN on re-execution.
	tensor::Tensor logits_2d = model.logits.squeeze(0)!!.cast(F32)!!;
	logits_2d.set_label("logits_2d");

	tensor::Tensor loss_t = loss::softmax_cross_entropy(logits_2d, target_onehot, weight_tensor)!!;
	loss_t = loss_t.stable();
	loss_t.set_label("loss");

	// --- Backward ---
	loss_t.backward();

	// --- Optimizer ---
	int total_micro_batches = tcfg.epochs * (int)ds.num_sequences();
	// When using gradient accumulation, the scheduler counts optimizer updates (not micro-batches)
	int total_steps = total_micro_batches;
	int warmup = tcfg.warmup_steps;
	if (tcfg.grad_accum > 1)
	{
		total_steps = total_micro_batches / tcfg.grad_accum;
		warmup = tcfg.warmup_steps / tcfg.grad_accum;
		if (warmup < 1) { warmup = 1; }
	}
	optimizer::Optimizer opt = optimizer::init({
		.lr = tcfg.lr,
		.scheduler = optimizer::scheduler_warmup_cosine(
			(uint)warmup, (uint)total_steps, 1e-6),
		.global_clip_norm = 1.0,
		.grad_accum_interval = (uint)tcfg.grad_accum,
	});
	usz total_opt_params = llama::register_lora_params(&model, &opt, {
		...optimizer::DEFAULT_ADAM_OPTIONS,
		.weight_decay = tcfg.weight_decay,
		.clip_grad = 0.0,  // global clip handles it; per-param fused clip disabled
	});
	io::printfn("  Optimizer: %d param groups, %d total steps (%d micro-batches)", total_opt_params, total_steps, total_micro_batches);
	if (tcfg.seed != 0) { io::printfn("  Seed:      %d", tcfg.seed); }
	if (tcfg.start_step > 0) { io::printfn("  Resuming from step %d", tcfg.start_step); }

	// --- Training loop ---

	io::printfn("\n================================================================");
	io::printfn("Training");
	io::printfn("================================================================\n");
	io::printfn("Step    Loss       LR         Used(MB)  Peak(MB)");
	io::printfn("------  ---------  ---------  --------  --------");

	int step = 0;
	float running_loss = 0.0;
	int running_count = 0;
	float last_epoch_loss = 0.0;
	int last_epoch_count = 0;
	Clock elapsed_clock;

	for (int epoch = 0; epoch < tcfg.epochs; epoch++)
	{
		ds.reset();
		// Re-shuffle each epoch (different permutation via epoch-dependent seed)
		if (tcfg.seed != 0)
		{
			ds.shuffle(tcfg.seed + (ulong)epoch);
		}
		float epoch_loss = 0.0;
		int epoch_count = 0;

		// Skip sequences if resuming
		if (tcfg.start_step > 0 && epoch == 0)
		{
			int to_skip = tcfg.start_step;
			io::printfn("  Skipping %d sequences to resume...", to_skip);
			while (to_skip > 0 && ds.next(input_buf[:ds_pred_len], target_buf[:ds_pred_len]))
			{
				to_skip--;
				step++;
			}
			io::printfn("  Resumed at step %d", step);
		}

		while (ds.next(input_buf[:ds_pred_len], target_buf[:ds_pred_len]))
		{
			// Convert uint tokens to ulong (truncate to model_seq_len)
			for (usz i = 0; i < model_seq_len; i++)
			{
				host_tokens[i] = (ulong)input_buf[i];
			}

			// Build one-hot target on host
			libc::memset(host_onehot, 0, onehot_floats * float.sizeof);
			for (usz i = 0; i < model_seq_len; i++)
			{
				usz tok = (usz)target_buf[i];
				if (tok < vocab)
				{
					host_onehot[i * vocab + tok] = 1.0;
				}
			}

			// Transfer to GPU
			model.tokens.set(host_tokens[:model_seq_len]);
			target_onehot.set(host_onehot[:onehot_floats]);

			// Transfer loss weights if available
			if (ds.get_weights(host_weights))
			{
				weight_tensor.set(host_weights);
			}

			// Forward
			loss_t.collect()!!;
			gpu.sync();

			// Read loss
			float loss_val = llama::read_loss_gpu(loss_t);

			running_loss += loss_val;
			running_count++;
			epoch_loss += loss_val;
			epoch_count++;

			// Backward + optimizer
			opt.step();
			gpu.sync();

			// Log (after step so we can show grad norm)
			if (step < 15 || step % tcfg.log_interval == 0 || step == total_micro_batches - 1)
			{
				double elapsed_secs = (double)elapsed_clock.mark() / 1_000_000_000.0;
				usz used = dev.mem_used() / (1024 * 1024);
				usz peak = dev.mem_peak() / (1024 * 1024);
				float ep_avg = epoch_loss / (float)epoch_count;
				io::printfn("%6d  %.6f  %.2e  gnorm=%.2f  %8d  %8d  ep_avg=%.4f  epoch=%d  %.1fs",
					step, loss_val, opt.lr, opt.last_grad_norm, used, peak, ep_avg, epoch + 1, elapsed_secs);
			}

			// Periodic checkpoint
			if (tcfg.save_interval > 0 && step > 0 && step % tcfg.save_interval == 0)
			{
				char[512] ckpt_path;
				usz out_len = libc::strlen(tcfg.lora_output);
				char[] out_str = tcfg.lora_output[:out_len];
				char[] ext = ".safetensors";
				usz base_len = out_len;
				if (out_len > ext.len)
				{
					bool has_ext = true;
					for (usz ei = 0; ei < ext.len; ei++)
					{
						if (out_str[out_len - ext.len + ei] != ext[ei]) { has_ext = false; break; }
					}
					if (has_ext) { base_len = out_len - ext.len; }
				}
				char[] base_str = out_str[:base_len];
				char[] ckpt_name;
				if (tcfg.seed != 0)
				{
					ckpt_name = io::bprintf(&ckpt_path, "%s_s%d_step_%d.safetensors",
						(String)base_str, tcfg.seed, step)!!;
				}
				else
				{
					ckpt_name = io::bprintf(&ckpt_path, "%s_step_%d.safetensors",
						(String)base_str, step)!!;
				}
				io::printfn("  [checkpoint] Saving at step %d -> %s", step, (String)ckpt_name);
				llama::save_lora_checkpoint(&model, ckpt_name);
			}

			step++;

			if (tcfg.max_steps > 0 && step >= tcfg.max_steps)
			{
				io::printfn("--- Reached max_steps=%d, stopping ---", tcfg.max_steps);
				break;
			}

			if (step >= total_micro_batches)
			{
				io::printfn("--- Completed %d micro-batches (%d optimizer steps), stopping ---", total_micro_batches, total_steps);
				break;
			}

			float ep_avg = epoch_loss / (float)epoch_count;
			if (tcfg.target_loss > 0.0f && ep_avg < tcfg.target_loss)
			{
				io::printfn("--- Epoch avg loss %.4f < target %.4f, stopping ---", ep_avg, tcfg.target_loss);
				break;
			}
		}

		if (tcfg.max_steps > 0 && step >= tcfg.max_steps) { break; }
		if (step >= total_micro_batches) { break; }
		last_epoch_loss = epoch_loss;
		last_epoch_count = epoch_count;
		if (tcfg.target_loss > 0.0f && epoch_loss / (float)epoch_count < tcfg.target_loss) { break; }
		io::printfn("--- Epoch %d/%d complete (step %d, ep_avg=%.4f) ---", epoch + 1, tcfg.epochs, step, epoch_loss / (float)epoch_count);
	}

	// --- Save final checkpoint ---
	usz output_len = libc::strlen(tcfg.lora_output);
	llama::save_lora_checkpoint(&model, tcfg.lora_output[:output_len]);

	// --- Summary ---
	io::printfn("\n================================================================");
	io::printfn("Training Complete");
	io::printfn("================================================================");
	io::printfn("  Total steps:  %d", step);
	io::printfn("  Final avg:    %.4f (global), %.4f (last epoch)",
		running_loss / (float)running_count,
		last_epoch_count > 0 ? (double)(last_epoch_loss / (float)last_epoch_count) : 0.0);
	io::printfn("  Checkpoint:   %s", tcfg.lora_output);
	usz peak = dev.mem_peak();
	usz reserved = dev.mem_reserved();
	io::printfn("  Peak GPU:     %.1f MB", (float)peak / (1024.0 * 1024.0));
	io::printfn("  Reserved GPU: %.1f MB", (float)reserved / (1024.0 * 1024.0));
}
