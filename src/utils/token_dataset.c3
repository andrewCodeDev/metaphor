//! Token Dataset Reader
//!
//! Loads pre-tokenized binary data (u32 LE) from a directory containing
//! train_tokens.bin and train_meta.json. Optionally loads train_weights.bin
//! for per-token loss weights. Serves batches for next-token prediction
//! (input = tokens[0..seq_len-1], target = tokens[1..seq_len]).
//!
//! Usage:
//!   TokenDataset ds = token_dataset::init("data/processed/grounded")!!;
//!   defer ds.free();
//!   uint[512] input_buf;
//!   uint[512] target_buf;
//!   float[511] weight_buf;
//!   while (ds.next(&input_buf, &target_buf)) {
//!       ds.get_weights(&weight_buf);  // call after next()
//!       // fill tensors with input_buf / target_buf / weight_buf
//!   }
//!   ds.reset();

module metaphor::utils::token_dataset;
import metaphor::utils::cjson;
import std::io;
import std::core::mem;

alias CJson = cjson::CJson;

// Faults

faultdef TOKEN_DATASET_LOAD_FAILED;
faultdef TOKEN_DATASET_META_PARSE_FAILED;

// TokenDataset

struct TokenDataset
{
	uint* data;           // loaded token data (all sequences flat)
	float* weights;       // per-token loss weights (all sequences flat), null if not present
	usz n_sequences;      // number of sequences
	usz seq_len;          // tokens per sequence (from meta, includes target shift)
	usz vocab_size;       // vocabulary size (from meta)
	usz total_tokens;     // n_sequences * seq_len
	usz current;          // current sequence index (batch cursor)
}

/// Load a token dataset from a directory containing train_tokens.bin and train_meta.json.
/// Optionally loads train_weights.bin for per-token loss weights.
fn TokenDataset? init(ZString dir_path)
{
	// --- Parse train_meta.json ---
	String meta_path = string::tformat("%s/train_meta.json", dir_path);
	char[] meta_data = io::file::load(mem, meta_path)!;
	defer mem::free(meta_data.ptr);

	// Null-terminate for cJSON
	char* meta_cstr = (char*)mem::malloc(meta_data.len + 1);
	assert(meta_cstr != null, "token_dataset: meta allocation failed");
	mem::copy(meta_cstr, meta_data.ptr, meta_data.len);
	meta_cstr[meta_data.len] = 0;

	CJson* root = cjson::cJSON_Parse(meta_cstr);
	mem::free(meta_cstr);

	if (root == null)
	{
		io::printfn("token_dataset: failed to parse train_meta.json");
		return TOKEN_DATASET_META_PARSE_FAILED~;
	}

	CJson* n_seq_item = cjson::cJSON_GetObjectItemCaseSensitive(root, "n_sequences");
	CJson* seq_len_item = cjson::cJSON_GetObjectItemCaseSensitive(root, "seq_len");
	CJson* vocab_item = cjson::cJSON_GetObjectItemCaseSensitive(root, "vocab_size");

	if (n_seq_item == null || seq_len_item == null || vocab_item == null)
	{
		io::printfn("token_dataset: missing fields in train_meta.json (need n_sequences, seq_len, vocab_size)");
		cjson::cJSON_Delete(root);
		return TOKEN_DATASET_META_PARSE_FAILED~;
	}

	usz n_sequences = (usz)cjson::cJSON_GetNumberValue(n_seq_item);
	usz seq_len = (usz)cjson::cJSON_GetNumberValue(seq_len_item);
	usz vocab_size = (usz)cjson::cJSON_GetNumberValue(vocab_item);
	cjson::cJSON_Delete(root);

	io::printfn("token_dataset: n_sequences=%d, seq_len=%d, vocab_size=%d", n_sequences, seq_len, vocab_size);

	// --- Load train_tokens.bin ---
	String bin_path = string::tformat("%s/train_tokens.bin", dir_path);
	char[] bin_data = io::file::load(mem, bin_path)!;

	usz total_tokens = n_sequences * seq_len;
	usz data_bytes = total_tokens * uint.sizeof;

	if (bin_data.len != data_bytes)
	{
		io::printfn("token_dataset: expected %d bytes, got %d", data_bytes, bin_data.len);
		mem::free(bin_data.ptr);
		return TOKEN_DATASET_LOAD_FAILED~;
	}

	io::printfn("token_dataset: loaded %d tokens (%d bytes)", total_tokens, data_bytes);

	// --- Load train_weights.bin (optional) ---
	float* weights = null;
	String weights_path = string::tformat("%s/train_weights.bin", dir_path);
	if (try w_data = io::file::load(mem, weights_path))
	{
		usz weights_bytes = total_tokens * float.sizeof;
		if (w_data.len == weights_bytes)
		{
			weights = (float*)w_data.ptr;
			io::printfn("token_dataset: loaded loss weights (%d bytes)", weights_bytes);
		}
		else
		{
			io::printfn("token_dataset: weights size mismatch, expected %d bytes, got %d — ignoring", weights_bytes, w_data.len);
			mem::free(w_data.ptr);
		}
	}

	return {
		.data = (uint*)bin_data.ptr,
		.weights = weights,
		.n_sequences = n_sequences,
		.seq_len = seq_len,
		.vocab_size = vocab_size,
		.total_tokens = total_tokens,
		.current = 0,
	};
}

/// Free all allocated memory.
fn void TokenDataset.free(&self)
{
	if (self.data != null)
	{
		mem::free(self.data);
		self.data = null;
	}
	if (self.weights != null)
	{
		mem::free(self.weights);
		self.weights = null;
	}
}

/// Get the next sequence pair for next-token prediction.
/// Fills input_buf with tokens[0..seq_len-1] and target_buf with tokens[1..seq_len].
/// Returns false when all sequences have been consumed (call reset() to restart).
/// input_buf and target_buf must each have at least (seq_len - 1) elements.
fn bool TokenDataset.next(&self, uint[] input_buf, uint[] target_buf)
{
	if (self.current >= self.n_sequences)
	{
		return false;
	}

	usz offset = self.current * self.seq_len;
	usz pred_len = self.seq_len - 1;  // input/target length for next-token prediction

	for (usz i = 0; i < pred_len; i++)
	{
		input_buf[i] = self.data[offset + i];
		target_buf[i] = self.data[offset + i + 1];
	}

	self.current++;
	return true;
}

/// Get loss weights for the last sequence returned by next().
/// Fills weight_buf with weights aligned to target tokens (shifted by 1).
/// weight_buf must have at least (seq_len - 1) elements.
/// Returns false if no weights are loaded or next() hasn't been called.
fn bool TokenDataset.get_weights(&self, float[] weight_buf)
{
	if (self.weights == null) return false;
	if (self.current == 0) return false;

	usz offset = (self.current - 1) * self.seq_len;
	usz pred_len = self.seq_len - 1;

	for (usz i = 0; i < pred_len; i++)
	{
		weight_buf[i] = self.weights[offset + i + 1];  // aligned to target
	}
	return true;
}

/// Whether pre-computed loss weights are available.
fn bool TokenDataset.has_weights(&self)
{
	return self.weights != null;
}

/// Reset the cursor for a new epoch.
fn void TokenDataset.reset(&self)
{
	self.current = 0;
}

/// Shuffle sequences in-place using Fisher-Yates with a simple LCG PRNG.
/// Shuffles both token data and weights (if present) together.
/// Pass seed=0 to skip shuffling.
fn void TokenDataset.shuffle(&self, ulong seed)
{
	if (seed == 0 || self.n_sequences <= 1) return;

	// LCG parameters (Numerical Recipes)
	ulong state = seed;

	for (usz i = self.n_sequences - 1; i > 0; i--)
	{
		state = state * 6364136223846793005 + 1442695040888963407;
		usz j = (usz)(state >> 33) % (i + 1);  // uniform in [0, i]

		if (i == j) continue;

		// Swap sequences i and j in token data
		usz off_i = i * self.seq_len;
		usz off_j = j * self.seq_len;
		for (usz k = 0; k < self.seq_len; k++)
		{
			uint tmp = self.data[off_i + k];
			self.data[off_i + k] = self.data[off_j + k];
			self.data[off_j + k] = tmp;
		}

		// Swap in weights if present
		if (self.weights != null)
		{
			for (usz k = 0; k < self.seq_len; k++)
			{
				float tmp = self.weights[off_i + k];
				self.weights[off_i + k] = self.weights[off_j + k];
				self.weights[off_j + k] = tmp;
			}
		}
	}
}

/// Number of sequences in the dataset.
fn usz TokenDataset.num_sequences(&self)
{
	return self.n_sequences;
}

/// Prediction length (seq_len - 1, since we shift by 1 for next-token prediction).
fn usz TokenDataset.pred_len(&self)
{
	return self.seq_len - 1;
}
