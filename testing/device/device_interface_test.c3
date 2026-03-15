//! Device Interface Tests
//!
//! Tests the new Device interface methods: mem_fill, mem_dupe, mem_sequence,
//! mem_cast, mem_equal, sample, argmax — exercised through DeviceReference
//! on a real HostDevice.
//!
//! Mirrors tests from:

module metaphor::testing::device::device_interface_test;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::rtti;
import metaphor::utils::sampling;
import std::math;

// ============================================================================
// mem_fill — fill device memory with a constant value
// ============================================================================

fn void test_mem_fill_f32() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer dev.mem_free(buf);

	dev.mem_fill(rtti::Datatype.F32, buf, 3.14);

	float* fp = (float*)buf.ptr;
	for (usz i = 0; i < 4; i++)
	{
		assert(math::abs(fp[i] - 3.14f) < 1e-5, "mem_fill f32 value mismatch");
	}
}

fn void test_mem_fill_f64() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F64, 3);
	defer dev.mem_free(buf);

	dev.mem_fill(rtti::Datatype.F64, buf, -2.5);

	double* dp = (double*)buf.ptr;
	for (usz i = 0; i < 3; i++)
	{
		assert(math::abs(dp[i] - (-2.5)) < 1e-10, "mem_fill f64 value mismatch");
	}
}

fn void test_mem_fill_zero() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer dev.mem_free(buf);

	// Fill with non-zero first
	dev.mem_fill(rtti::Datatype.F32, buf, 99.0);
	// Then fill with zero
	dev.mem_fill(rtti::Datatype.F32, buf, 0.0);

	float* fp = (float*)buf.ptr;
	for (usz i = 0; i < 4; i++)
	{
		assert(fp[i] == 0.0, "mem_fill zero should produce 0.0");
	}
}

// ============================================================================
// mem_dupe — allocate + copy
// ============================================================================

fn void test_mem_dupe_f32() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] src = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer dev.mem_free(src);

	float* sp = (float*)src.ptr;
	sp[0] = 1.0; sp[1] = 2.0; sp[2] = 3.0; sp[3] = 4.0;

	char[] dst = dev.mem_dupe(rtti::Datatype.F32, src);
	defer dev.mem_free(dst);

	assert(dst.len == src.len, "duped buffer should have same length");
	assert(dst.ptr != src.ptr, "duped buffer should be separate allocation");

	float* dp = (float*)dst.ptr;
	for (usz i = 0; i < 4; i++)
	{
		assert(dp[i] == sp[i], "duped values should match");
	}
}

fn void test_mem_dupe_empty() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] empty = {};
	char[] result = dev.mem_dupe(rtti::Datatype.F32, empty);
	assert(result.len == 0, "duping empty should return empty");
}

// ============================================================================
// mem_sequence — arithmetic sequence fill
// ============================================================================

fn void test_mem_sequence_f32() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 5);
	defer dev.mem_free(buf);

	dev.mem_sequence(rtti::Datatype.F32, buf, 1.0, 0.5);

	float* fp = (float*)buf.ptr;
	// Expected: 1.0, 1.5, 2.0, 2.5, 3.0
	assert(math::abs(fp[0] - 1.0) < 1e-5, "seq[0]");
	assert(math::abs(fp[1] - 1.5) < 1e-5, "seq[1]");
	assert(math::abs(fp[2] - 2.0) < 1e-5, "seq[2]");
	assert(math::abs(fp[3] - 2.5) < 1e-5, "seq[3]");
	assert(math::abs(fp[4] - 3.0) < 1e-5, "seq[4]");
}

fn void test_mem_sequence_f64() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F64, 4);
	defer dev.mem_free(buf);

	dev.mem_sequence(rtti::Datatype.F64, buf, 0.0, 1.0);

	double* dp = (double*)buf.ptr;
	for (usz i = 0; i < 4; i++)
	{
		assert(math::abs(dp[i] - (double)i) < 1e-10, "seq f64 mismatch");
	}
}

fn void test_mem_sequence_negative_step() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 3);
	defer dev.mem_free(buf);

	dev.mem_sequence(rtti::Datatype.F32, buf, 10.0, -3.0);

	float* fp = (float*)buf.ptr;
	// Expected: 10.0, 7.0, 4.0
	assert(math::abs(fp[0] - 10.0) < 1e-5, "neg step [0]");
	assert(math::abs(fp[1] - 7.0) < 1e-5, "neg step [1]");
	assert(math::abs(fp[2] - 4.0) < 1e-5, "neg step [2]");
}

fn void test_mem_fill_via_sequence() @test
{
	// mem_fill delegates to mem_sequence(value, 0) — verify equivalence
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf_fill = dev.mem_alloc(rtti::Datatype.F32, 4);
	char[] buf_seq = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer { dev.mem_free(buf_fill); dev.mem_free(buf_seq); }

	dev.mem_fill(rtti::Datatype.F32, buf_fill, 7.5);
	dev.mem_sequence(rtti::Datatype.F32, buf_seq, 7.5, 0.0);

	float* fp_fill = (float*)buf_fill.ptr;
	float* fp_seq = (float*)buf_seq.ptr;
	for (usz i = 0; i < 4; i++)
	{
		assert(fp_fill[i] == fp_seq[i], "fill and sequence(v,0) should match");
	}
}

// ============================================================================
// mem_cast — dtype conversion through device interface
// ============================================================================

fn void test_mem_cast_f32_to_f64() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] src = dev.mem_alloc(rtti::Datatype.F32, 4);
	char[] dst = dev.mem_alloc(rtti::Datatype.F64, 4);
	defer { dev.mem_free(src); dev.mem_free(dst); }

	float* sp = (float*)src.ptr;
	sp[0] = 1.0; sp[1] = 2.5; sp[2] = -3.0; sp[3] = 4.5;

	dev.mem_cast(rtti::Datatype.F32, rtti::Datatype.F64, src, dst);

	double* dp = (double*)dst.ptr;
	assert(math::abs(dp[0] - 1.0) < 1e-6, "cast f32->f64 [0]");
	assert(math::abs(dp[1] - 2.5) < 1e-6, "cast f32->f64 [1]");
	assert(math::abs(dp[2] - (-3.0)) < 1e-6, "cast f32->f64 [2]");
	assert(math::abs(dp[3] - 4.5) < 1e-6, "cast f32->f64 [3]");
}

fn void test_mem_cast_f64_to_f32() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] src = dev.mem_alloc(rtti::Datatype.F64, 4);
	char[] dst = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer { dev.mem_free(src); dev.mem_free(dst); }

	double* sp = (double*)src.ptr;
	sp[0] = 1.0; sp[1] = 2.5; sp[2] = -3.0; sp[3] = 4.5;

	dev.mem_cast(rtti::Datatype.F64, rtti::Datatype.F32, src, dst);

	float* dp = (float*)dst.ptr;
	assert(math::abs(dp[0] - 1.0) < 1e-6, "cast f64->f32 [0]");
	assert(math::abs(dp[1] - 2.5) < 1e-6, "cast f64->f32 [1]");
	assert(math::abs(dp[2] - (-3.0)) < 1e-6, "cast f64->f32 [2]");
	assert(math::abs(dp[3] - 4.5) < 1e-6, "cast f64->f32 [3]");
}

// ============================================================================
// mem_equal — exact and approximate comparison
// ============================================================================

fn void test_mem_equal_exact_match() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] a = dev.mem_alloc(rtti::Datatype.F32, 4);
	char[] b = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer { dev.mem_free(a); dev.mem_free(b); }

	float* ap = (float*)a.ptr;
	float* bp = (float*)b.ptr;
	for (usz i = 0; i < 4; i++)
	{
		ap[i] = (float)i * 1.5;
		bp[i] = (float)i * 1.5;
	}

	assert(dev.mem_equal(rtti::Datatype.F32, a, b, EqualMode.EXACT), "identical buffers should be exact equal");
}

fn void test_mem_equal_exact_mismatch() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] a = dev.mem_alloc(rtti::Datatype.F32, 4);
	char[] b = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer { dev.mem_free(a); dev.mem_free(b); }

	float* ap = (float*)a.ptr;
	float* bp = (float*)b.ptr;
	for (usz i = 0; i < 4; i++)
	{
		ap[i] = (float)i;
		bp[i] = (float)i;
	}
	bp[2] = 999.0; // differ at one element

	assert(!dev.mem_equal(rtti::Datatype.F32, a, b, EqualMode.EXACT), "different buffers should not be exact equal");
}

fn void test_mem_equal_approx_within_tolerance() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] a = dev.mem_alloc(rtti::Datatype.F32, 3);
	char[] b = dev.mem_alloc(rtti::Datatype.F32, 3);
	defer { dev.mem_free(a); dev.mem_free(b); }

	float* ap = (float*)a.ptr;
	float* bp = (float*)b.ptr;
	ap[0] = 1.0;       bp[0] = 1.0 + 1e-7;
	ap[1] = 100.0;     bp[1] = 100.0 + 5e-4;
	ap[2] = -0.5;      bp[2] = -0.5 + 1e-7;

	assert(dev.mem_equal(rtti::Datatype.F32, a, b, EqualMode.APPROX), "nearly identical should pass approx");
}

fn void test_mem_equal_approx_outside_tolerance() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] a = dev.mem_alloc(rtti::Datatype.F32, 2);
	char[] b = dev.mem_alloc(rtti::Datatype.F32, 2);
	defer { dev.mem_free(a); dev.mem_free(b); }

	float* ap = (float*)a.ptr;
	float* bp = (float*)b.ptr;
	ap[0] = 1.0;  bp[0] = 1.0;
	ap[1] = 1.0;  bp[1] = 2.0; // large difference

	assert(!dev.mem_equal(rtti::Datatype.F32, a, b, EqualMode.APPROX), "large diff should fail approx");
}

fn void test_mem_equal_different_lengths() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] a = dev.mem_alloc(rtti::Datatype.F32, 4);
	char[] b = dev.mem_alloc(rtti::Datatype.F32, 8);
	defer { dev.mem_free(a); dev.mem_free(b); }

	assert(!dev.mem_equal(rtti::Datatype.F32, a, b, EqualMode.EXACT), "different length buffers should not be equal");
}

fn void test_mem_equal_f64_approx() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] a = dev.mem_alloc(rtti::Datatype.F64, 2);
	char[] b = dev.mem_alloc(rtti::Datatype.F64, 2);
	defer { dev.mem_free(a); dev.mem_free(b); }

	double* ap = (double*)a.ptr;
	double* bp = (double*)b.ptr;
	ap[0] = 3.141592653589793;  bp[0] = 3.141592653589793 + 1e-12;
	ap[1] = -1e10;              bp[1] = -1e10 + 1.0;

	assert(dev.mem_equal(rtti::Datatype.F64, a, b, EqualMode.APPROX), "f64 near-identical should pass approx");
}

// ============================================================================
// argmax — greedy decoding via device interface
// ============================================================================

fn void test_argmax_basic() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 4);
	defer dev.mem_free(buf);

	float* fp = (float*)buf.ptr;
	fp[0] = 1.0; fp[1] = 3.0; fp[2] = 2.0; fp[3] = 0.5;

	uint result = dev.argmax(rtti::Datatype.F32, buf, 4);
	assert(result == 1, "argmax should return index 1 (value 3.0)");
}

fn void test_argmax_first_element() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 3);
	defer dev.mem_free(buf);

	float* fp = (float*)buf.ptr;
	fp[0] = 10.0; fp[1] = 5.0; fp[2] = 1.0;

	uint result = dev.argmax(rtti::Datatype.F32, buf, 3);
	assert(result == 0, "argmax should return index 0");
}

fn void test_argmax_last_element() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F32, 3);
	defer dev.mem_free(buf);

	float* fp = (float*)buf.ptr;
	fp[0] = 1.0; fp[1] = 2.0; fp[2] = 100.0;

	uint result = dev.argmax(rtti::Datatype.F32, buf, 3);
	assert(result == 2, "argmax should return index 2");
}

fn void test_argmax_f64() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	char[] buf = dev.mem_alloc(rtti::Datatype.F64, 4);
	defer dev.mem_free(buf);

	double* dp = (double*)buf.ptr;
	dp[0] = -1.0; dp[1] = -0.5; dp[2] = -2.0; dp[3] = -0.1;

	uint result = dev.argmax(rtti::Datatype.F64, buf, 4);
	assert(result == 3, "argmax should return index 3 (value -0.1)");
}

// ============================================================================
// sample — token sampling via device interface
// ============================================================================

fn void test_sample_returns_valid_token() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	usz vocab = 100;
	char[] buf = dev.mem_alloc(rtti::Datatype.F32, vocab);
	defer dev.mem_free(buf);

	// Create monotonically increasing logits
	float* fp = (float*)buf.ptr;
	for (usz i = 0; i < vocab; i++)
	{
		fp[i] = (float)i * 0.1;
	}

	sampling::SamplingConfig config = {
		.temperature = 0.8,
		.top_k = 10,
		.top_p = 0.9,
		.repetition_penalty = 1.0,
		.seed = 42,
	};

	char[] empty_ctx = {};
	uint token = dev.sample(rtti::Datatype.F32, buf, vocab, empty_ctx, 0, &config);
	assert(token < (uint)vocab, "sampled token should be within vocab range");
}

fn void test_sample_low_temperature_favors_max() @test
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	usz vocab = 10;
	char[] buf = dev.mem_alloc(rtti::Datatype.F32, vocab);
	defer dev.mem_free(buf);

	float* fp = (float*)buf.ptr;
	// One very dominant logit at index 7
	for (usz i = 0; i < vocab; i++) { fp[i] = 0.0; }
	fp[7] = 100.0;

	sampling::SamplingConfig config = {
		.temperature = 0.01,
		.top_k = 0,
		.top_p = 0.0,
		.repetition_penalty = 1.0,
		.seed = 123,
	};

	char[] empty_ctx = {};
	// With very low temperature and dominant logit, should always pick index 7
	uint token = dev.sample(rtti::Datatype.F32, buf, vocab, empty_ctx, 0, &config);
	assert(token == 7, "very low temperature should pick dominant logit");
}

// ============================================================================
// ============================================================================

fn void test_sampling_softmax() @test
{
	float[3] logits = { 1.0, 2.0, 3.0 };
	sampling::softmax(&logits);

	// Check probabilities sum to 1
	float sum = 0;
	for (usz i = 0; i < 3; i++) { sum += logits[i]; }
	assert(math::abs(sum - 1.0) < 0.0001, "softmax should sum to 1");

	// Ordering preserved
	assert(logits[2] > logits[1], "highest logit = highest prob");
	assert(logits[1] > logits[0], "middle logit = middle prob");
}

fn void test_sampling_temperature() @test
{
	// Low temperature should make distribution sharper
	float[3] logits_low = { 1.0, 2.0, 3.0 };
	sampling::apply_temperature(&logits_low, 0.5);
	sampling::softmax(&logits_low);

	float[3] logits_high = { 1.0, 2.0, 3.0 };
	sampling::apply_temperature(&logits_high, 2.0);
	sampling::softmax(&logits_high);

	// Low temp: highest prob should be higher
	assert(logits_low[2] > logits_high[2], "low temp should sharpen peak");
	// High temp: lowest prob should be higher (more uniform)
	assert(logits_high[0] > logits_low[0], "high temp should flatten distribution");
}

fn void test_sampling_top_k() @test
{
	float[4] logits = { 1.0, 4.0, 2.0, 3.0 };
	uint[4] indices = { 0, 1, 2, 3 };

	usz n = sampling::apply_top_k(&logits, &indices, 2);
	assert(n == 2, "top_k should return k");

	// Top 2 should be sorted descending in top-k portion
	assert(logits[0] >= logits[1], "top-k values sorted descending");
}

fn void test_sampling_argmax_utility() @test
{
	float[4] logits = { 1.0, 3.0, 2.0, 0.5 };
	uint result = sampling::argmax(&logits);
	assert(result == 1, "argmax should return 1");
}

fn void test_sampler_integration() @test
{
	sampling::SamplingConfig config = {
		.temperature = 0.8,
		.top_k = 10,
		.top_p = 0.9,
		.repetition_penalty = 1.0,
		.seed = 42,
	};

	sampling::Sampler sampler;
	sampler.init(100, config);
	defer sampler.deinit();

	// Create test logits
	float[100] logits;
	for (usz i = 0; i < 100; i++)
	{
		logits[i] = (float)i * 0.1;
	}

	// Sample should return valid token
	uint token = sampler.sample(&logits);
	assert(token < 100, "sampled token should be in range");
}
