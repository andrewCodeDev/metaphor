//! Sampling
//!
//! Token sampling from logit distributions for text generation.
//! Supports temperature, top-k, top-p (nucleus), repetition penalty,
//! and greedy (argmax) decoding.
//!

module metaphor::utils::sampling;
import metaphor::device::device_reference;
import metaphor::tensor;
import metaphor::rtti;
import std::core::mem;
import std::math;

// Config

struct SamplingConfig
{
	/// Temperature for logit scaling. 1.0 = no change, <1 = sharper, >1 = softer.
	float temperature;
	/// Top-k: only sample from the k most likely tokens. 0 = disabled.
	uint top_k;
	/// Top-p (nucleus): sample from smallest set with cumulative prob >= p. 0 = disabled.
	float top_p;
	/// Repetition penalty for tokens in context. 1.0 = disabled.
	float repetition_penalty;
	/// Block repeated n-grams of this size. 0 = disabled.
	uint ngram_size;
	/// Per-occurrence frequency penalty. 0.0 = disabled.
	float frequency_penalty;
	/// Tokens below this ID are exempt from repetition/frequency/n-gram penalties.
	/// GPT-2 BPE: 0-255 are byte-level tokens (punctuation, spaces, digits).
	/// Set to 256 to protect all single-byte tokens. 0 = no exemptions.
	uint penalty_exempt_below;
	/// Scale penalties by decoded token length.
	/// Tokens of 1 byte get no penalty, scaling linearly to full penalty
	/// at penalty_length_threshold bytes. 0 = disabled (use hard exempt cutoff only).
	uint penalty_length_threshold;
	/// Random seed.
	ulong seed;
}

const SamplingConfig DEFAULT_CONFIG = {
	.temperature = 1.0,
	.top_k = 0,
	.top_p = 0.0,
	.repetition_penalty = 1.0,
	.ngram_size = 0,
	.frequency_penalty = 0.0,
	.penalty_exempt_below = 0,
	.penalty_length_threshold = 0,
	.seed = 0,
};

// Simple xoshiro256** PRNG

struct Prng
{
	ulong[4] state;
}

fn Prng prng_init(ulong seed)
{
	// SplitMix64 to seed the state
	Prng p;
	ulong s = seed;
	for (int i = 0; i < 4; i++)
	{
		s += 0x9e3779b97f4a7c15;
		ulong z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		z = z ^ (z >> 31);
		p.state[i] = z;
	}
	return p;
}

fn ulong rotl(ulong x, int k) @inline
{
	return (x << k) | (x >> (64 - k));
}

fn ulong Prng.next(&self)
{
	ulong result = rotl(self.state[1] * 5, 7) * 9;
	ulong t = self.state[1] << 17;

	self.state[2] ^= self.state[0];
	self.state[3] ^= self.state[1];
	self.state[1] ^= self.state[2];
	self.state[0] ^= self.state[3];

	self.state[2] ^= t;
	self.state[3] = rotl(self.state[3], 45);

	return result;
}

/// Generate a random float in [0, 1).
fn float Prng.float01(&self)
{
	ulong bits = self.next();
	// Use top 24 bits for float precision
	return (float)(bits >> 40) / (float)(1 << 24);
}

// Global RNG — seeded once, used by all device mem_random calls

Prng global_rng;
bool global_rng_initialized = false;

/// Seed the global RNG. Call once before any random tensor creation.
fn void seed(ulong s)
{
	global_rng = prng_init(s);
	global_rng_initialized = true;
}

/// Get the global RNG, auto-initializing with a default seed if needed.
fn Prng* get_rng()
{
	if (!global_rng_initialized)
	{
		global_rng = prng_init(81761);
		global_rng_initialized = true;
	}
	return &global_rng;
}

// Random fill

/// Fill a float buffer with random values from the given distribution.
fn void fill_random(float[] data, device_reference::RandType rand_type, Prng* rng, device_reference::Scale scale = device_reference::DEFAULT_SCALE)
{
	float s = (float)scale.resolve();
	switch (rand_type)
	{
		case UNIFORM:
			for (usz i = 0; i < data.len; i++)
			{
				data[i] = rng.float01() * s;
			}
		case NORMAL:
			// Box-Muller transform for pairs
			usz i = 0;
			while (i < data.len)
			{
				float u1 = rng.float01();
				float u2 = rng.float01();
				if (u1 < 1e-10) u1 = 1e-10;
				float r = math::sqrt(-2.0f * math::ln(u1));
				float theta = 2.0f * (float)math::PI * u2;
				data[i] = r * math::cos(theta) * s;
				if (i + 1 < data.len)
				{
					data[i + 1] = r * math::sin(theta) * s;
				}
				i += 2;
			}
	}
}

// Sampler

struct Sampler
{
	Prng prng;
	SamplingConfig config;

	// Scratch buffers (reused across calls)
	uint* indices;
	float* probs;
	char* raw_buf;   // Host buffer for DtoH transfer (sized for largest dtype)
	usz buf_size;

	// Token length table for length-scaled penalties (optional, set via set_token_lengths)
	char* token_lengths;  // one byte per vocab entry, 0 = unknown
}

fn void Sampler.init(&self, uint vocab_size, SamplingConfig config = DEFAULT_CONFIG)
{
	ulong seed = config.seed;
	if (seed == 0)
	{
		// Use address of self as crude entropy source
		seed = (ulong)(uptr)self ^ 0xdeadbeef12345678;
	}

	self.prng = prng_init(seed);
	self.config = config;
	self.buf_size = (usz)vocab_size;
	self.indices = (uint*)mem::malloc(uint.sizeof * self.buf_size);
	self.probs = (float*)mem::malloc(float.sizeof * self.buf_size);
	self.raw_buf = (char*)mem::malloc(float.sizeof * self.buf_size); // max 4 bytes per element
	self.token_lengths = null;
}

/// Set token length table for length-scaled penalties.
/// lengths[i] = decoded byte length of token i (capped to 255).
fn void Sampler.set_token_lengths(&self, char* lengths)
{
	self.token_lengths = lengths;
}

fn void Sampler.deinit(&self)
{
	mem::free(self.indices);
	mem::free(self.probs);
	mem::free(self.raw_buf);
}

/// Sample a token from logits using configured sampling strategy.
fn uint Sampler.sample(&self, float[] logits)
{
	uint[] empty;
	return self.sample_with_context(logits, empty);
}

/// Transfer logits from a collected tensor to the probs buffer as float.
/// Does DtoH transfer + dtype conversion without sampling.
/// The tensor must already be collected (have data on device).
/// After this call, self.probs[:n] contains the raw float logits.
/// Returns the number of elements loaded.
fn usz Sampler.load_logits(&self, tensor::Tensor logits, device_reference::DeviceReference dev)
{
	rtti::Datatype dtype = logits.dtype();
	usz n_elements = logits.shape().size();
	if (n_elements > self.buf_size) { n_elements = self.buf_size; }

	usz byte_size = n_elements * dtype.size_of();
	char[] device_bytes = logits.value_bytes()[:byte_size];
	dev.@transfer_to_host(device_bytes, self.raw_buf[:byte_size]);

	// Convert to float probs buffer based on dtype
	switch (dtype)
	{
		case BF16:
			bfloat* src = (bfloat*)self.raw_buf;
			for (usz i = 0; i < n_elements; i++) { self.probs[i] = (float)src[i]; }
		case F16:
			float16* src = (float16*)self.raw_buf;
			for (usz i = 0; i < n_elements; i++) { self.probs[i] = (float)src[i]; }
		case F32:
			float* src = (float*)self.raw_buf;
			for (usz i = 0; i < n_elements; i++) { self.probs[i] = src[i]; }
		default:
			unreachable("load_logits: unsupported dtype");
	}

	return n_elements;
}

/// Sample from a collected Tensor (any dtype). Handles DtoH transfer internally.
/// The tensor must already be collected (have data on device).
fn uint Sampler.sample_tensor(&self, tensor::Tensor logits, device_reference::DeviceReference dev, uint[] context)
{
	usz n_elements = self.load_logits(logits, dev);
	return self.sample_from_probs(n_elements, context);
}

/// Sample with repetition penalty applied to context tokens.
fn uint Sampler.sample_with_context(&self, float[] logits, uint[] context)
{
	usz n = logits.len;
	if (n > self.buf_size) { n = self.buf_size; }

	// Copy logits to working buffer
	for (usz i = 0; i < n; i++) { self.probs[i] = logits[i]; }

	return self.sample_from_probs(n, context);
}

/// Core sampling logic — operates on self.probs[:n] already populated with logits.
fn uint Sampler.sample_from_probs(&self, usz n, uint[] context)
{

	// Apply n-gram blocking
	uint exempt = self.config.penalty_exempt_below;
	uint len_thresh = self.config.penalty_length_threshold;
	if (self.config.ngram_size > 0 && context.len >= self.config.ngram_size)
	{
		apply_ngram_blocking(self.probs[:n], context, self.config.ngram_size, exempt, self.token_lengths, len_thresh);
	}

	// Apply frequency penalty (no length exemptions — every token accumulates)
	if (self.config.frequency_penalty > 0.0)
	{
		apply_frequency_penalty(self.probs[:n], context, self.config.frequency_penalty);
	}

	// Apply repetition penalty
	if (self.config.repetition_penalty != 1.0)
	{
		apply_repetition_penalty(self.probs[:n], context, self.config.repetition_penalty, exempt, self.token_lengths, len_thresh);
	}

	// Apply temperature
	if (self.config.temperature != 1.0 && self.config.temperature > 0)
	{
		apply_temperature(self.probs[:n], self.config.temperature);
	}

	// Initialize indices
	for (usz i = 0; i < n; i++) { self.indices[i] = (uint)i; }

	// Apply top-k (sorts and truncates)
	usz active_n = n;
	if (self.config.top_k > 0)
	{
		active_n = apply_top_k(self.probs[:n], self.indices[:n], self.config.top_k);
	}

	// Convert to probabilities (softmax)
	softmax(self.probs[:active_n]);

	// Apply top-p (nucleus sampling)
	if (self.config.top_p > 0.0)
	{
		active_n = apply_top_p(self.probs[:active_n], self.indices[:active_n], self.config.top_p);
	}

	// Sample from distribution
	return sample_from_distribution(self.probs[:active_n], self.indices[:active_n], &self.prng);
}

// Sampling operations

/// Apply temperature scaling to logits in-place.
fn void apply_temperature(float[] logits, float temperature)
{
	float inv_temp = 1.0 / temperature;
	for (usz i = 0; i < logits.len; i++)
	{
		logits[i] *= inv_temp;
	}
}

/// Compute penalty scale factor [0, 1] for a token based on its decoded length.
/// Returns 0.0 (no penalty) for 1-byte tokens, scaling linearly to 1.0 at threshold.
/// If token_lengths is null or threshold is 0, falls back to hard exempt_below cutoff.
fn float penalty_scale(uint token, uint exempt_below, char* token_lengths, uint threshold) @inline
{
	if (token < exempt_below) { return 0.0; }
	if (threshold > 0 && token_lengths != null)
	{
		uint len = (uint)token_lengths[token];
		if (len <= 1) { return 0.0; }
		if (len >= threshold) { return 1.0; }
		return (float)(len - 1) / (float)(threshold - 1);
	}
	return 1.0;
}

/// Apply repetition penalty to tokens that appear in context.
fn void apply_repetition_penalty(float[] logits, uint[] context, float penalty, uint exempt_below, char* token_lengths, uint len_thresh)
{
	for (usz i = 0; i < context.len; i++)
	{
		uint token = context[i];
		if (token >= (uint)logits.len) { continue; }
		float scale = penalty_scale(token, exempt_below, token_lengths, len_thresh);
		if (scale <= 0.0) { continue; }

		// Interpolate penalty: effective = 1.0 + scale * (penalty - 1.0)
		float eff = 1.0 + scale * (penalty - 1.0);
		if (logits[token] > 0)
		{
			logits[token] /= eff;
		}
		else
		{
			logits[token] *= eff;
		}
	}
}

/// Apply n-gram blocking: set logits to -inf for any token that would
/// complete a repeated n-gram in context.
fn void apply_ngram_blocking(float[] logits, uint[] context, uint ngram_size, uint exempt_below, char* token_lengths, uint len_thresh)
{
	usz n = (usz)ngram_size;
	if (context.len < n) { return; }

	// The last (n-1) tokens form the prefix we're trying to extend
	usz prefix_start = context.len - (n - 1);
	uint[] prefix = context[prefix_start:n - 1];

	// Scan context for all windows of size (n-1) that match this prefix
	// If match found at position p, then context[p + n - 1] is the token
	// that completed that n-gram — block it from being generated again.
	usz scan_end = context.len - (n - 1);
	for (usz p = 0; p < scan_end; p++)
	{
		// Check if context[p..p+n-1] matches prefix
		bool match = true;
		for (usz k = 0; k < n - 1; k++)
		{
			if (context[p + k] != prefix[k])
			{
				match = false;
				break;
			}
		}
		if (match)
		{
			uint blocked_token = context[p + n - 1];
			if (blocked_token >= (uint)logits.len) { continue; }
			float scale = penalty_scale(blocked_token, exempt_below, token_lengths, len_thresh);
			if (scale <= 0.0) { continue; }
			// For n-gram blocking, scale determines how strongly we suppress:
			// full scale = -inf, partial scale = proportional reduction
			if (scale >= 1.0)
			{
				logits[blocked_token] = -1e9;
			}
			else
			{
				logits[blocked_token] -= scale * 1e9;
			}
		}
	}
}

/// Apply frequency penalty: penalize tokens proportional to their occurrence
/// count in context. First occurrence is free, each additional costs frequency_penalty.
fn void apply_frequency_penalty(float[] logits, uint[] context, float penalty)
{
	for (usz i = 0; i < context.len; i++)
	{
		uint token = context[i];
		if (token >= (uint)logits.len) { continue; }
		logits[token] -= penalty;
	}
}

/// Apply top-k filtering: keep only the k highest logits.
fn usz apply_top_k(float[] logits, uint[] indices, uint k)
{
	usz n = logits.len;
	if ((usz)k >= n) { return n; }

	partial_sort(logits, indices, k);
	return (usz)k;
}

/// Apply top-p (nucleus) filtering.
fn usz apply_top_p(float[] probs, uint[] indices, float p)
{
	usz n = probs.len;
	if (n == 0) { return 0; }

	sort_by_value_descending(probs, indices);

	float cumsum = 0;
	usz cutoff = n;
	for (usz i = 0; i < n; i++)
	{
		cumsum += probs[i];
		if (cumsum >= p)
		{
			cutoff = i + 1;
			break;
		}
	}

	// Renormalize
	if (cutoff < n)
	{
		float sum = 0;
		for (usz i = 0; i < cutoff; i++) { sum += probs[i]; }
		if (sum > 0)
		{
			float inv_sum = 1.0 / sum;
			for (usz i = 0; i < cutoff; i++) { probs[i] *= inv_sum; }
		}
	}

	return cutoff;
}

/// Convert logits to probabilities using softmax.
fn void softmax(float[] logits)
{
	if (logits.len == 0) { return; }

	// Find max for numerical stability
	float max_val = logits[0];
	for (usz i = 1; i < logits.len; i++)
	{
		if (logits[i] > max_val) { max_val = logits[i]; }
	}

	// exp(x - max) and sum
	float sum = 0;
	for (usz i = 0; i < logits.len; i++)
	{
		logits[i] = math::exp(logits[i] - max_val);
		sum += logits[i];
	}

	// Normalize
	if (sum > 0)
	{
		float inv_sum = 1.0 / sum;
		for (usz i = 0; i < logits.len; i++)
		{
			logits[i] *= inv_sum;
		}
	}
}

/// Sample a token index from a probability distribution.
fn uint sample_from_distribution(float[] probs, uint[] indices, Prng* rng)
{
	if (probs.len == 0) { return 0; }
	if (probs.len == 1) { return indices[0]; }

	float r = rng.float01();
	float cumsum = 0;

	for (usz i = 0; i < probs.len; i++)
	{
		cumsum += probs[i];
		if (r < cumsum)
		{
			return indices[i];
		}
	}

	// Fallback to last token
	return indices[probs.len - 1];
}

/// Convenience function: argmax (greedy decoding).
fn uint argmax(float[] logits)
{
	if (logits.len == 0) { return 0; }

	uint max_idx = 0;
	float max_val = logits[0];
	for (usz i = 1; i < logits.len; i++)
	{
		if (logits[i] > max_val)
		{
			max_val = logits[i];
			max_idx = (uint)i;
		}
	}
	return max_idx;
}

// Sorting helpers

/// Partial sort: move the k largest elements to the front.
fn void partial_sort(float[] values, uint[] indices, uint k)
{
	usz n = values.len;
	usz k_clamped = (usz)k;
	if (k_clamped > n) { k_clamped = n; }

	for (usz i = 0; i < k_clamped; i++)
	{
		usz max_idx = i;
		float max_val = values[i];
		for (usz j = i + 1; j < n; j++)
		{
			if (values[j] > max_val)
			{
				max_val = values[j];
				max_idx = j;
			}
		}

		if (max_idx != i)
		{
			// Swap values
			float tmp_v = values[i];
			values[i] = values[max_idx];
			values[max_idx] = tmp_v;
			// Swap indices
			uint tmp_i = indices[i];
			indices[i] = indices[max_idx];
			indices[max_idx] = tmp_i;
		}
	}
}

/// Sort values and indices by value descending (insertion sort).
fn void sort_by_value_descending(float[] values, uint[] indices)
{
	usz n = values.len;
	for (usz i = 1; i < n; i++)
	{
		float val = values[i];
		uint idx = indices[i];
		usz j = i;
		while (j > 0 && values[j - 1] < val)
		{
			values[j] = values[j - 1];
			indices[j] = indices[j - 1];
			j--;
		}
		values[j] = val;
		indices[j] = idx;
	}
}
