//! Minimal prefill test — isolates batched self-attention from full LLaMA.
//!
//! Tests the core prefill operations (einsum projections, batched RoPE,
//! causal masked self-attention) with small dimensions on the HIP device.
//!
//! Run: c3c build test_prefill && ./build/test_prefill

module metaphor::examples::test_prefill;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::device::common::operations;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::rtti;
import metaphor::tensor::tensor_core;
import std::math;
import std::io;
import std::core::mem;
import libc;

alias DeviceReference = device_reference::DeviceReference;
alias TransferDirection = device_reference::TransferDirection;
alias Tensor = tensor::Tensor;
alias Datatype = rtti::Datatype;
alias BinaryOp = operations::BinaryOp;

// Small model config for testing
const usz D_MODEL = 64;
const usz N_HEADS = 4;
const usz HEAD_DIM = 16; // D_MODEL / N_HEADS
const usz HALF_DIM = 8;  // HEAD_DIM / 2

fn void read_bf16_as_f32(DeviceReference dev, Tensor t, float* dst, usz count)
{
	char[] gpu_bytes = t.value_bytes();
	usz bf_bytes = count * ushort.sizeof;
	ushort* hbf = (ushort*)mem::malloc(bf_bytes);
	dev.mem_transfer(gpu_bytes[:bf_bytes], ((char*)hbf)[:bf_bytes], TransferDirection.DTOH);
	for (usz i = 0; i < count; i++)
	{
		uint bits = (uint)hbf[i] << 16;
		dst[i] = *(float*)&bits;
	}
	mem::free(hbf);
}

fn void dump_tensor(DeviceReference dev, Tensor t, char[] label, usz max_elems)
{
	usz total = t.core().shape().size();
	usz count = total < max_elems ? total : max_elems;
	float* buf = (float*)mem::malloc(count * float.sizeof);
	read_bf16_as_f32(dev, t, buf, count);
	io::printf("%s [", label);
	for (usz i = 0; i < (usz)t.core().shape().len; i++)
	{
		if (i > 0) io::printf(",");
		io::printf("%d", t.core().shape().get(i));
	}
	io::printf("] first %d: ", count);
	for (usz i = 0; i < count; i++)
	{
		io::printf("%.4f ", buf[i]);
	}
	io::printn("");
	mem::free(buf);
}

/// Apply RoPE for decode: x [B, H, head_dim], cos/sin [1, 1, half]
fn Tensor? apply_rope(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(2) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 2)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 2)!;
	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	Tensor[2] parts = { rot1, rot2 };
	return tensor::cat(&parts, 2);
}

/// Apply RoPE batched: x [B, S, H, head_dim], cos/sin [1, S, 1, half]
fn Tensor? apply_rope_batched(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(3) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 3)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 3)!;
	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	Tensor[2] parts = { rot1, rot2 };
	return tensor::cat(&parts, 3);
}

/// Single-token layer forward (decode path)
fn Tensor? decode_layer(Tensor x, Tensor q_proj, Tensor k_proj, Tensor v_proj, Tensor o_proj,
	Tensor ln_w, Tensor cos_pos, Tensor sin_pos, Tensor scale, double eps,
	DeviceReference dev, bool verbose)
{
	Tensor normed = functional::rms_norm(x, ln_w, eps)!;

	// Project: [B, d] × [H, K, d] → [B, H, K]
	Tensor q = normed.einsum(q_proj, "bd,hkd->bhk")!;
	Tensor k = normed.einsum(k_proj, "bd,hkd->bhk")!;
	Tensor v = normed.einsum(v_proj, "bd,hkd->bhk")!;

	q = apply_rope(q, cos_pos, sin_pos)!;
	k = apply_rope(k, cos_pos, sin_pos)!;

	if (verbose) { q.stable().collect()!; dev.sync(); dump_tensor(dev, q, "  decode q after rope", 8); }

	// Self-attention (single token: Q=K=V, no cache)
	Tensor q_scaled = q * scale;
	if (verbose) { q_scaled.stable().collect()!; dev.sync(); dump_tensor(dev, q_scaled, "  decode q_scaled", 16); }
	if (verbose) { k.stable().collect()!; dev.sync(); dump_tensor(dev, k, "  decode k", 16); }

	// [B,H,K] → [B,H,1,K]
	Tensor q_4d = q_scaled.unsqueeze(2)!;
	Tensor k_4d = k.unsqueeze(2)!;
	Tensor v_4d = v.unsqueeze(2)!;

	if (verbose) { q_4d.stable().collect()!; dev.sync(); dump_tensor(dev, q_4d, "  decode q_4d", 16); }
	if (verbose) { k_4d.stable().collect()!; dev.sync(); dump_tensor(dev, k_4d, "  decode k_4d", 16); }

	// Scores: [B,H,1,K] × [B,H,1,K] → [B,H,1,1]
	Tensor scores = q_4d.einsum(k_4d, "bhik,bhjk->bhij")!;
	if (verbose) { scores.stable().collect()!; dev.sync(); dump_tensor(dev, scores, "  decode scores", 8); }

	// Softmax over j (dim 3) — single element, always 1.0
	Tensor weights = functional::softmax(scores, 3)!;
	if (verbose) { weights.stable().collect()!; dev.sync(); dump_tensor(dev, weights, "  decode weights", 8); }

	// Weighted sum → [B,H,1,K] → squeeze → [B,H,K]
	Tensor attn_out = weights.einsum(v_4d, "bhij,bhjk->bhik")!.squeeze(2)!;
	if (verbose) { attn_out.stable().collect()!; dev.sync(); dump_tensor(dev, attn_out, "  decode attn_out", 8); }

	// Output projection: [B,H,K] → [B,d]
	Tensor projected = attn_out.einsum(o_proj, "bhk,dhk->bd")!;
	if (verbose) { projected.stable().collect()!; dev.sync(); dump_tensor(dev, projected, "  decode projected", 8); }

	return x + projected;
}

/// Batched prefill layer forward
fn Tensor? prefill_layer(Tensor x, Tensor q_proj, Tensor k_proj, Tensor v_proj, Tensor o_proj,
	Tensor ln_w, Tensor cos_pos, Tensor sin_pos, Tensor causal_mask, Tensor scale, double eps,
	DeviceReference dev, bool verbose)
{
	Tensor normed = functional::rms_norm(x, ln_w, eps)!;

	// Project: [B, S, d] × [H, K, d] → [B, S, H, K]
	Tensor q = normed.einsum(q_proj, "bsd,hkd->bshk")!;
	Tensor k = normed.einsum(k_proj, "bsd,hkd->bshk")!;
	Tensor v = normed.einsum(v_proj, "bsd,hkd->bshk")!;

	q = apply_rope_batched(q, cos_pos, sin_pos)!;
	k = apply_rope_batched(k, cos_pos, sin_pos)!;

	if (verbose) { q.stable().collect()!; dev.sync(); dump_tensor(dev, q, "  prefill q after rope", 8); }

	// Self-attention
	Tensor q_scaled = q * scale;
	if (verbose) { q_scaled.stable().collect()!; dev.sync(); dump_tensor(dev, q_scaled, "  prefill q_scaled", 16); }
	if (verbose) { k.stable().collect()!; dev.sync(); dump_tensor(dev, k, "  prefill k", 16); }

	// Scores: Q[B,S,H,K] × K[B,T,H,K] → [B,H,S,T]
	Tensor scores = q_scaled.einsum(k, "bshk,bthk->bhst")!;
	if (verbose) { scores.stable().collect()!; dev.sync(); dump_tensor(dev, scores, "  prefill scores", 8); }

	// Causal mask: [S,T] add
	Tensor masked = scores.einsum(causal_mask, "bhst,st->bhst", inner: BinaryOp.ADD)!;
	Tensor weights = functional::softmax(masked, 3)!;
	if (verbose) { weights.stable().collect()!; dev.sync(); dump_tensor(dev, weights, "  prefill weights", 8); }

	// Weighted sum: [B,H,S,T] × V[B,T,H,K] → [B,S,H,K]
	Tensor attn_out = weights.einsum(v, "bhst,bthk->bshk")!;
	if (verbose) { attn_out.stable().collect()!; dev.sync(); dump_tensor(dev, attn_out, "  prefill attn_out", 8); }

	// Output projection: [B,S,H,K] → [B,S,d]
	Tensor projected = attn_out.einsum(o_proj, "bshk,dhk->bsd")!;
	if (verbose) { projected.stable().collect()!; dev.sync(); dump_tensor(dev, projected, "  prefill projected", 8); }

	return x + projected;
}

fn int main()
{
	hip_device::HipDevice gpu = hip_device::create();
	DeviceReference dev = gpu.reference();
	Datatype dt = Datatype.BF16;
	double eps = 1e-5;

	io::printn("=== Prefill Test ===");

	usz n_tokens = 1;
	ulong n = (ulong)n_tokens;

	graph::@subgraph()
	{
		// --- Create random weights ---
		ulong[3] proj_shape = { N_HEADS, HEAD_DIM, D_MODEL };
		Tensor q_proj = tensor::random(dt, dev, &proj_shape, device_reference::RandType.NORMAL)!!;
		Tensor k_proj = tensor::random(dt, dev, &proj_shape, device_reference::RandType.NORMAL)!!;
		Tensor v_proj = tensor::random(dt, dev, &proj_shape, device_reference::RandType.NORMAL)!!;

		Tensor o_proj = tensor::random(dt, dev, { D_MODEL, N_HEADS, HEAD_DIM }, device_reference::RandType.NORMAL)!!;

		Tensor ln_w = tensor::constant(dt, dev, { D_MODEL }, 1.0)!!;

		// Scale: [HEAD_DIM] filled with 1/sqrt(head_dim)
		float scale_val = 1.0 / math::sqrt((float)HEAD_DIM);
		Tensor scale = tensor::constant(dt, dev, { HEAD_DIM }, scale_val)!!;

		// --- Create input: [1, N, d_model] random hidden states ---
		Tensor input_2d = tensor::random(dt, dev, { n, D_MODEL }, device_reference::RandType.NORMAL)!!;
		// Collect so we can reuse the same data for both paths
		input_2d.stable().collect()!!;

		// --- RoPE: just use ones for simplicity (position 0 for all) ---
		ulong[3] cos_sin_decode_shape = { 1, 1, HALF_DIM };
		Tensor cos_decode = tensor::constant(dt, dev, &cos_sin_decode_shape, 1.0)!!;
		Tensor sin_decode = tensor::constant(dt, dev, &cos_sin_decode_shape, 0.0)!!;

		ulong[4] cos_sin_batch_shape = { 1, n, 1, HALF_DIM };
		Tensor cos_batch = tensor::constant(dt, dev, &cos_sin_batch_shape, 1.0)!!;
		Tensor sin_batch = tensor::constant(dt, dev, &cos_sin_batch_shape, 0.0)!!;

		// --- Causal mask: [N, N] ---
		Tensor causal_mask = tensor::empty(dt, dev, { n, n })!!;
		causal_mask.core().alloc_data();
		{
			usz mask_elems = n_tokens * n_tokens;
			float* host_f = (float*)mem::malloc(mask_elems * float.sizeof);
			ushort* host_bf = (ushort*)mem::malloc(mask_elems * ushort.sizeof);
			for (usz i = 0; i < n_tokens; i++)
			{
				for (usz j = 0; j < n_tokens; j++)
				{
					host_f[i * n_tokens + j] = j <= i ? 0.0 : -1e9;
				}
			}
			for (usz k = 0; k < mask_elems; k++)
			{
				uint bits = *(uint*)&host_f[k];
				host_bf[k] = (ushort)(bits >> 16);
			}
			usz bf_bytes = mask_elems * ushort.sizeof;
			dev.mem_transfer(((char*)host_bf)[:bf_bytes],
				causal_mask.value_bytes()[:bf_bytes], TransferDirection.HTOD);
			mem::free(host_f);
			mem::free(host_bf);
		}

		// ============================================================
		// TEST 1: RMS norm 3D vs 2D
		// ============================================================
		io::printn("\n--- RMS norm test ---");
		Tensor x_3d = input_2d.unsqueeze(0)!!;  // [1,1,64]
		Tensor x_2d = input_2d.slice({ { 0, 1, 1 } }, offset: 0)!!; // [1, 64]

		Tensor norm_3d = functional::rms_norm(x_3d, ln_w, eps)!!;
		norm_3d.stable().collect()!!;
		dev.sync();
		dump_tensor(dev, norm_3d, "rms_norm 3D [1,1,64]", 16);

		Tensor norm_2d = functional::rms_norm(x_2d, ln_w, eps)!!;
		norm_2d.stable().collect()!!;
		dev.sync();
		dump_tensor(dev, norm_2d, "rms_norm 2D [1,64]  ", 16);

		// ============================================================
		// TEST 2: RMS norm + projection
		// ============================================================
		io::printn("\n--- RMS norm + projection test ---");
		Tensor qa_3d = norm_3d.einsum(q_proj, "bsd,hkd->bshk")!!;
		qa_3d.stable().collect()!!;
		dev.sync();
		dump_tensor(dev, qa_3d, "norm+proj 3D", 16);

		Tensor qa_2d = norm_2d.einsum(q_proj, "bd,hkd->bhk")!!;
		qa_2d.stable().collect()!!;
		dev.sync();
		dump_tensor(dev, qa_2d, "norm+proj 2D", 16);

		// ============================================================
		// TEST 3: Full layer, single token
		// ============================================================
		io::printn("\n--- Full layer test (1 token) ---");
		Tensor prefill_out = prefill_layer(x_3d, q_proj, k_proj, v_proj, o_proj,
			ln_w, cos_batch, sin_batch, causal_mask, scale, eps, dev, true)!!;
		prefill_out.stable().collect()!!;
		dev.sync();
		dump_tensor(dev, prefill_out, "prefill layer out", 16);

		Tensor decode_out = decode_layer(x_2d, q_proj, k_proj, v_proj, o_proj,
			ln_w, cos_decode, sin_decode, scale, eps, dev, true)!!;
		decode_out.stable().collect()!!;
		dev.sync();
		dump_tensor(dev, decode_out, "decode  layer out", 16);
	};

	io::printn("\n=== Done ===");
	return 0;
}
