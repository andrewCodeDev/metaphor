//! Mamba Training (CUDA)
//!
//! Trains LoRA adapters + embed_ext with causal masked loss.
//! Only response tokens (from <|thinking|> onward) contribute to the loss.
//!
//! Required:
//!   --data-dir <path>     Directory with train_tokens.bin + train_meta.json
//!   --weights <path>      Pretrained base weights (.safetensors)
//!
//! Optional:
//!   --lora-out <path>     Output checkpoint (default: lora_output.safetensors)
//!   --lr <float>          Learning rate (default: 3e-3)
//!   --epochs <int>        Number of epochs (default: 2)
//!   --warmup <int>        Warmup steps (default: 20)
//!   --log-interval <int>  Log every N steps (default: 10)
//!   --layers <int>        Number of layers (default: all from model config)
//!   --seq-len <int>       Sequence length (default: 64)
//!
//! Pipeline:
//!   1. ./mamba_train.sh train   — LoRA rank 64 with causal masked loss
//!   2. ./mamba_train.sh fuse    — merge LoRA + embed_ext into base weights

module metaphor::examples::mamba_lora_train;
import metaphor::examples::mamba;
import metaphor::device::host_device;
import metaphor::device::cuda_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::rtti;
import metaphor::utils::token_dataset;
import metaphor::logging;
import std::io;
import std::time::clock;
import std::core::mem;
import libc;

const bool USE_CUDA = true;

faultdef INVALID_ARGS;

// ============================================================================
// Configuration — all from CLI args, no hardcoded paths
// ============================================================================

struct TrainConfig
{
	ZString data_dir;
	ZString weights;
	ZString lora_output;
	ZString resume_lora;      // path to LoRA checkpoint to resume from
	ZString model_size;       // "130m", "370m", "790m", "2.8b"
	float lr;
	int epochs;
	int warmup_steps;
	int log_interval;
	int save_interval;        // 0 = save only at end
	int max_steps;            // 0 = unlimited
	int start_step;           // skip this many sequences on resume
	ulong seed;               // PRNG seed for shuffling, 0 = sequential
	int layers;               // 0 = use model default
	int train_layers;         // 0 = all layers get LoRA, N = only last N layers
	int lora_rank;            // 0 = default (16)
	int seq_len;
	float target_loss;        // 0 = disabled, stop when avg loss drops below this
	int vocab_size;           // 0 = use model default, else override
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
	io::printfn("  --lora-out <path>     Output LoRA checkpoint (default: lora_output.safetensors)");
	io::printfn("  --resume-lora <path>  Resume from a LoRA checkpoint (load weights before training)");
	io::printfn("  --lr <float>          Learning rate (default: 3e-3)");
	io::printfn("  --epochs <int>        Number of epochs (default: 2)");
	io::printfn("  --warmup <int>        Warmup steps (default: 20)");
	io::printfn("  --log-interval <int>  Log every N steps (default: 10)");
	io::printfn("  --save-interval <int> Save checkpoint every N steps (default: 500)");
	io::printfn("  --max-steps <int>     Stop after N steps, 0=unlimited (default: 0)");
	io::printfn("  --start-step <int>   Skip N sequences to resume from checkpoint (default: 0)");
	io::printfn("  --seed <int>         PRNG seed, 0=sequential (default: 0)");
	io::printfn("  --model <size>        Model config: 130m, 370m, 790m, 2.8b (default: 130m)");
	io::printfn("  --layers <int>        Number of layers (default: all from model config)");
	io::printfn("  --train-layers <int>  Only train last N layers with LoRA (default: all)");
	io::printfn("  --lora-rank <int>     LoRA rank (default: 64)");
	io::printfn("  --seq-len <int>       Sequence length (default: 64)");
	io::printfn("  --target-loss <float> Stop when avg loss drops below this (default: 0=disabled)");
	io::printfn("  --vocab-size <int>    Override vocab size (default: from model config)");
	io::printfn("  --grad-accum <int>    Gradient accumulation steps, 0=disabled (default: 0)");
}

fn TrainConfig? parse_args(String[] args)
{
	TrainConfig cfg = {
		.data_dir = null,
		.weights = null,
		.lora_output = null,
		.model_size = null,
		.lr = 3e-3,
		.epochs = 2,
		.warmup_steps = 20,
		.log_interval = 10,
		.save_interval = 500,
		.max_steps = 0,
		.start_step = 0,
		.seed = 0,
		.layers = 0,
		.train_layers = 0,
		.lora_rank = 0,
		.seq_len = 64,
	};

	usz i = 1;
	while (i < args.len)
	{
		String arg = args[i];
		if (arg == "--data-dir" && i + 1 < args.len)
		{
			cfg.data_dir = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--weights" && i + 1 < args.len)
		{
			cfg.weights = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--lora-out" && i + 1 < args.len)
		{
			cfg.lora_output = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--resume-lora" && i + 1 < args.len)
		{
			cfg.resume_lora = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--model" && i + 1 < args.len)
		{
			cfg.model_size = (ZString)args[i + 1].ptr;
			i += 2;
		}
		else if (arg == "--lr" && i + 1 < args.len)
		{
			cfg.lr = args[i + 1].to_float() ?? 3e-3;
			i += 2;
		}
		else if (arg == "--epochs" && i + 1 < args.len)
		{
			cfg.epochs = args[i + 1].to_int() ?? 2;
			i += 2;
		}
		else if (arg == "--warmup" && i + 1 < args.len)
		{
			cfg.warmup_steps = args[i + 1].to_int() ?? 20;
			i += 2;
		}
		else if (arg == "--log-interval" && i + 1 < args.len)
		{
			cfg.log_interval = args[i + 1].to_int() ?? 10;
			i += 2;
		}
		else if (arg == "--save-interval" && i + 1 < args.len)
		{
			cfg.save_interval = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--max-steps" && i + 1 < args.len)
		{
			cfg.max_steps = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--start-step" && i + 1 < args.len)
		{
			cfg.start_step = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--seed" && i + 1 < args.len)
		{
			cfg.seed = (ulong)(args[i + 1].to_long() ?? 0);
			i += 2;
		}
		else if (arg == "--layers" && i + 1 < args.len)
		{
			cfg.layers = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--train-layers" && i + 1 < args.len)
		{
			cfg.train_layers = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--lora-rank" && i + 1 < args.len)
		{
			cfg.lora_rank = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--seq-len" && i + 1 < args.len)
		{
			cfg.seq_len = args[i + 1].to_int() ?? 64;
			i += 2;
		}
		else if (arg == "--target-loss" && i + 1 < args.len)
		{
			cfg.target_loss = args[i + 1].to_float() ?? 0.0;
			i += 2;
		}
		else if (arg == "--vocab-size" && i + 1 < args.len)
		{
			cfg.vocab_size = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--grad-accum" && i + 1 < args.len)
		{
			cfg.grad_accum = args[i + 1].to_int() ?? 0;
			i += 2;
		}
		else if (arg == "--help" || arg == "-h")
		{
			print_usage((ZString)args[0].ptr);
			return INVALID_ARGS~;
		}
		else
		{
			io::printfn("Unknown argument: %s", arg);
			print_usage((ZString)args[0].ptr);
			return INVALID_ARGS~;
		}
	}

	if (cfg.data_dir == null)
	{
		io::printfn("Error: --data-dir is required");
		print_usage((ZString)args[0].ptr);
		return INVALID_ARGS~;
	}
	if (cfg.weights == null)
	{
		io::printfn("Error: --weights is required");
		print_usage((ZString)args[0].ptr);
		return INVALID_ARGS~;
	}

	// Default lora output name includes model size
	if (cfg.lora_output == null)
	{
		static char[128] default_name;
		ZString model = cfg.model_size != null ? cfg.model_size : "unknown";
		io::bprintf(&default_name, "lora_%s_output.safetensors", model)!!;
		cfg.lora_output = (ZString)&default_name;
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
	if (catch parse_args(args))
	{
		return;
	}
	tcfg = parse_args(args)!!;

	// --- Device ---
	$if USE_CUDA:
		cuda_device::CudaDevice gpu = cuda_device::cuda_device_create({ .device_id = 0 });
		defer gpu.deinit();
		device_reference::DeviceReference dev = gpu.reference();
	$else
		host_device::HostDevice cpu;
		cpu.init();
		defer cpu.deinit();
		device_reference::DeviceReference dev = cpu.reference();
	$endif

	graph::init();
	defer graph::deinit();

	// Enable CODEGEN error-only logging
	logging::enable(logging::Scope.CODEGEN, logging::Level.ERR);

	// --- Model config ---
	mamba::MambaConfig cfg = mamba::mamba_130m();
	if (tcfg.model_size != null)
	{
		usz mlen = libc::strlen(tcfg.model_size);
		char[] ms = tcfg.model_size[:mlen];
		if (ms == "370m" || ms == "370M")
		{
			cfg = mamba::mamba_370m();
		}
		else if (ms == "790m" || ms == "790M")
		{
			cfg = mamba::mamba_790m();
		}
		else if (ms == "2.8b" || ms == "2.8B" || ms == "2800m")
		{
			cfg = mamba::mamba_2_8b();
		}
		else if (ms != "130m" && ms != "130M")
		{
			io::printfn("Unknown model size: %s (use 130m, 370m, 790m, or 2.8b)", tcfg.model_size);
			return;
		}
	}
	if (tcfg.layers > 0)
	{
		cfg.n_layers = (ulong)tcfg.layers;
	}
	cfg.seq_len = (ulong)tcfg.seq_len;
	if (tcfg.vocab_size > 0)
	{
		cfg.vocab_size = (ulong)tcfg.vocab_size;
	}

	ulong lora_rank = tcfg.lora_rank > 0 ? (ulong)tcfg.lora_rank : 64;
	mamba::LoRAConfig lcfg = {
		.rank = lora_rank,
		.alpha = (float)lora_rank,
		.adapt_in_proj = true,
		.adapt_out_proj = true,
		.adapt_x_proj = false,
		.adapt_dt_proj = false,
	};

	// --- Print configuration ---
	io::printfn("\n================================================================");
	io::printfn("LoRA Mamba Training");
	io::printfn("================================================================\n");
	$if USE_CUDA:
		io::printfn("Device: CUDA");
	$else
		io::printfn("Device: CPU");
	$endif
	io::printfn("Model:  d_model=%d, n_layers=%d, d_state=%d, dt_rank=%d, vocab=%d",
		cfg.d_model, cfg.n_layers, cfg.d_state, cfg.get_dt_rank(), cfg.vocab_size);
	io::printfn("LoRA:   rank=%d, alpha=%.0f, in_proj=%s, out_proj=%s, train_layers=%s",
		lcfg.rank, lcfg.alpha,
		lcfg.adapt_in_proj ? "yes" : "no",
		lcfg.adapt_out_proj ? "yes" : "no",
		tcfg.train_layers > 0 ? "last" : "all");
	if (tcfg.train_layers > 0)
	{
		io::printfn("        training last %d of %d layers", tcfg.train_layers, cfg.n_layers);
	}
	io::printfn("Data:   %s", tcfg.data_dir);
	io::printfn("LR:     %.1e, epochs=%d, warmup=%d, save_every=%d", tcfg.lr, tcfg.epochs, tcfg.warmup_steps, tcfg.save_interval);
	if (tcfg.grad_accum > 0)
	{
		io::printfn("Accum:  %d micro-batches per update (effective batch=%d)", tcfg.grad_accum, tcfg.grad_accum);
	}
	io::printfn("Loss:   causal masked (response tokens only)");
	if (tcfg.target_loss > 0.0f)
	{
		io::printfn("Target: stop when avg loss < %.2f", tcfg.target_loss);
	}
	io::printfn("Output: %s", tcfg.lora_output);

	// --- Build model ---
	io::printfn("\nBuilding model...");
	mamba::LoRAMambaModel model = mamba::model_init(cfg, lcfg, dev, tcfg.train_layers)!!;
	cfg = model.config;

	ulong base_params = mamba::total_base_params(cfg);
	ulong lora_params = mamba::total_lora_params(&model);
	io::printfn("  Base params (frozen):    %d", base_params);
	if (lora_params > 0)
	{
		io::printfn("  LoRA params (trainable): %d", lora_params);
		io::printfn("  LoRA / Base ratio:       %.2f%%",
			(double)lora_params / (double)base_params * 100.0);
	}

	// --- Load pretrained base weights ---
	usz weights_len = libc::strlen(tcfg.weights);
	mamba::load_pretrained_weights(&model, tcfg.weights[:weights_len], dev)!!;

	// --- Resume LoRA weights if provided ---
	if (tcfg.resume_lora != null)
	{
		usz lora_path_len = libc::strlen(tcfg.resume_lora);
		mamba::load_lora_checkpoint(&model, tcfg.resume_lora[:lora_path_len], dev)!!;
		io::printfn("  Resumed LoRA weights from: %s", tcfg.resume_lora);
	}

	// --- Load dataset ---
	io::printfn("\nLoading dataset...");
	token_dataset::TokenDataset ds = token_dataset::init(tcfg.data_dir)!!;
	defer ds.free();

	usz ds_pred_len = ds.pred_len();
	usz model_seq_len = (usz)cfg.seq_len;
	assert(ds_pred_len >= model_seq_len, "Dataset pred_len must be >= model seq_len");
	if (ds_pred_len > model_seq_len)
	{
		io::printfn("  Truncating sequences: %d -> %d tokens", ds_pred_len, model_seq_len);
	}
	if (ds.has_weights())
	{
		io::printfn("  Loss weights: loaded from train_weights.bin");
	}
	else
	{
		io::printfn("  Loss weights: not found, using uniform weights (all 1.0)");
	}

	// --- Prepare host buffers for batch transfer ---
	usz vocab = (usz)cfg.vocab_size;
	usz onehot_floats = model_seq_len * vocab;

	// Input token buffer (on host, will transfer to GPU)
	ulong* host_tokens = (ulong*)libc::malloc(model_seq_len * ulong.sizeof);
	assert(host_tokens != null, "Failed to allocate host token buffer");
	defer libc::free(host_tokens);

	// One-hot target buffer (on host, will transfer to GPU)
	float* host_onehot = (float*)libc::calloc(onehot_floats, float.sizeof);
	assert(host_onehot != null, "Failed to allocate host onehot buffer");
	defer libc::free(host_onehot);

	// Token dataset returns uint — need full-size buffers for reading
	uint* input_buf = (uint*)libc::malloc(ds_pred_len * uint.sizeof);
	defer libc::free(input_buf);

	uint* target_buf = (uint*)libc::malloc(ds_pred_len * uint.sizeof);
	assert(input_buf != null && target_buf != null, "Failed to allocate token buffers");

	defer libc::free(target_buf);

	// Host buffer for causal masked loss weights
	float[] host_weights = mem::alloc_array(float, model_seq_len);
	defer mem::free(host_weights.ptr);

	// --- Build loss graph ---
	// Target one-hot tensor on device: [seq_len, vocab_size]
	tensor::Tensor target_onehot = tensor::zeros(F32, dev, { cfg.seq_len, cfg.vocab_size })!!;

	// Per-position loss weight tensor for causal masked loss: [seq_len]
	tensor::Tensor weight_tensor = tensor::ones(F32, dev, { cfg.seq_len })!!;

	// Squeeze logits: [batch, seq_len, vocab] -> [seq_len, vocab]
	tensor::Tensor logits_2d = model.logits.reduce("bsv->sv")!!;
	logits_2d.set_label("logits_2d");

	tensor::Tensor loss_t = loss::softmax_cross_entropy(logits_2d, target_onehot, weight_tensor)!!;
	loss_t = loss_t.stable();
	loss_t.set_label("loss");

	// --- Backward ---
	loss_t.backward();

	// --- Optimizer ---
	int total_steps = tcfg.epochs * (int)ds.num_sequences();
	optimizer::Optimizer opt = optimizer::init({
		.lr = tcfg.lr,
		.scheduler = optimizer::scheduler_warmup_cosine(
			(uint)tcfg.warmup_steps, (uint)total_steps, 1e-6),
		.global_clip_norm = 1.0,
		.grad_accum_interval = (uint)tcfg.grad_accum,
	});
	usz total_opt_params = mamba::register_lora_params(&model, &opt);
	io::printfn("  Optimizer: %d param groups, %d total steps", total_opt_params, total_steps);
	if (tcfg.seed != 0)
	{
		io::printfn("  Seed:      %d", tcfg.seed);
	}
	if (tcfg.start_step > 0)
	{
		io::printfn("  Resuming from step %d", tcfg.start_step);
	}

	// --- Training loop ---
	io::printfn("\n================================================================");
	io::printfn("Training");
	io::printfn("================================================================\n");
	io::printfn("Step    Loss       LR         Used(MB)  Peak(MB)");
	io::printfn("------  ---------  ---------  --------  --------");

	graph::Graph* g = graph::get();
	int step = 0;
	float running_loss = 0.0;
	int running_count = 0;
	float last_epoch_loss = 0.0;
	int last_epoch_count = 0;
	Clock elapsed_clock;

	for (int epoch = 0; epoch < tcfg.epochs; epoch++)
	{
		ds.reset();
		float epoch_loss = 0.0;
		int epoch_count = 0;

		// Skip sequences if resuming from a checkpoint
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
			// Convert uint tokens to ulong for model input (truncate to model_seq_len)
			for (usz i = 0; i < model_seq_len; i++)
			{
				host_tokens[i] = (ulong)input_buf[i];
			}

			// Build one-hot target on host (truncate to model_seq_len)
			libc::memset(host_onehot, 0, onehot_floats * float.sizeof);
			for (usz i = 0; i < model_seq_len; i++)
			{
				usz tok = (usz)target_buf[i];
				if (tok < vocab)
				{
					host_onehot[i * vocab + tok] = 1.0;
				}
			}

			// Transfer input tokens to GPU
			char[] host_tok_bytes = ((char*)host_tokens)[:model_seq_len * ulong.sizeof];
			char[] dev_tok_bytes = model.tokens.value_bytes();
			dev.mem_transfer(host_tok_bytes, dev_tok_bytes, device_reference::TransferDirection.HTOD);

			// Transfer one-hot target to GPU
			char[] host_oh_bytes = ((char*)host_onehot)[:onehot_floats * float.sizeof];
			char[] dev_oh_bytes = target_onehot.value_bytes();
			dev.mem_transfer(host_oh_bytes, dev_oh_bytes, device_reference::TransferDirection.HTOD);

			// Transfer pre-computed loss weights (from train_weights.bin)
			if (ds.get_weights(host_weights))
			{
				dev.@transfer_to_device(host_weights, weight_tensor.value_bytes());
			}

			// Forward + backward
			g.advance_generation();
			loss_t.collect()!!;

			$if USE_CUDA:
				gpu.sync();
			$endif

			// Read loss
			float loss_val;
			$if USE_CUDA:
				loss_val = mamba::read_loss_cuda(loss_t, dev);
			$else
				char[] loss_bytes = loss_t.value_bytes();
				loss_val = *((float*)loss_bytes.ptr);
			$endif

			running_loss += loss_val;
			running_count++;
			epoch_loss += loss_val;
			epoch_count++;

			// Log
			if (step % tcfg.log_interval == 0 || step == total_steps - 1)
			{
				double elapsed_secs = (double)elapsed_clock.mark() / 1_000_000_000.0;
				usz used = dev.mem_used() / (1024 * 1024);
				usz peak = dev.mem_peak() / (1024 * 1024);
				float ep_avg = epoch_loss / (float)epoch_count;
				io::printfn("%6d  %.6f  %.2e  %8d  %8d  ep_avg=%.4f  epoch=%d  %.1fs",
					step, loss_val, opt.lr, used, peak, ep_avg, epoch + 1, elapsed_secs);
			}

			// Periodic checkpoint save (numbered: lora_output_step_N.safetensors)
			if (tcfg.save_interval > 0 && step > 0 && step % tcfg.save_interval == 0)
			{
				char[512] ckpt_path;
				// Strip .safetensors extension if present, append _step_N.safetensors
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
				mamba::save_lora_checkpoint(&model, ckpt_name);
			}

			// Optimizer step
			opt.step();

			$if USE_CUDA:
				gpu.sync();
			$endif

			step++;

			if (tcfg.max_steps > 0 && step >= tcfg.max_steps)
			{
				io::printfn("--- Reached max_steps=%d, stopping ---", tcfg.max_steps);
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
		last_epoch_loss = epoch_loss;
		last_epoch_count = epoch_count;
		if (tcfg.target_loss > 0.0f && epoch_loss / (float)epoch_count < tcfg.target_loss) { break; }
		io::printfn("--- Epoch %d/%d complete (step %d, ep_avg=%.4f) ---", epoch + 1, tcfg.epochs, step, epoch_loss / (float)epoch_count);
	}

	// --- Save checkpoint ---
	usz output_len = libc::strlen(tcfg.lora_output);
	mamba::save_lora_checkpoint(&model, tcfg.lora_output[:output_len]);

	// --- Summary ---
	io::printfn("\n================================================================");
	io::printfn("Training Complete");
	io::printfn("================================================================");
	io::printfn("  Total steps:  %d", step);
	io::printfn("  Final avg:    %.4f (global), %.4f (last epoch)", running_loss / (float)running_count, last_epoch_loss / (float)last_epoch_count);
	io::printfn("  Checkpoint:   %s", tcfg.lora_output);
	usz peak = dev.mem_peak();
	usz reserved = dev.mem_reserved();
	io::printfn("  Peak GPU:     %.1f MB", (float)peak / (1024.0 * 1024.0));
	io::printfn("  Reserved GPU: %.1f MB", (float)reserved / (1024.0 * 1024.0));
}
