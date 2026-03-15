//! Transformer Attention Block Example
//!
//! Multi-head self-attention with feed-forward network and residual
//! connections, using Metaphor's computation graph model.
//!
//! Architecture:
//!   Input → MultiHeadAttention → Residual → FFN → Residual → Output
//!
//! Key einsum patterns:
//!   Q/K/V projection: bsd,hdk→bhsk (batched head projection)
//!   Attention scores: bhik,bhjk→bhij (batched dot product)
//!   Attention output: bhij,bhjk→bhik (weighted sum)
//!   Output project:   bhsk,hkd→bsd   (merge heads)
//!

module metaphor::examples::attention;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::rtti;
import metaphor::utils::sampling;
import std::math;
import std::io;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HostDevice = host_device::HostDevice;
alias Datatype = rtti::Datatype;

// ============================================================================
// MultiHeadAttention
// ============================================================================

struct MultiHeadAttention
{
	Tensor w_q;
	Tensor w_k;
	Tensor w_v;
	Tensor w_o;
	Tensor input;
	Tensor output;
	ulong num_heads;
	ulong d_model;
	ulong d_k;
	ulong batch_size;
	ulong seq_len;
}

fn MultiHeadAttention mha_init(
	ulong d_model,
	ulong num_heads,
	ulong batch_size,
	ulong seq_len,
	DeviceReference dev)
{
	ulong d_k = d_model / num_heads;

	ulong[3] qkv_shape = { num_heads, d_model, d_k };
	Tensor w_q = tensor::empty(Datatype.F32, dev, &qkv_shape)!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)d_model));
	Tensor w_k = tensor::empty(Datatype.F32, dev, &qkv_shape)!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)d_model));
	Tensor w_v = tensor::empty(Datatype.F32, dev, &qkv_shape)!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)d_model));

	Tensor w_o = tensor::empty(Datatype.F32, dev, { num_heads, d_k, d_model })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)(d_k * num_heads)));

	Tensor input = tensor::empty(Datatype.F32, dev, { batch_size, seq_len, d_model })!!;

	// Q, K, V projections
	Tensor q = input.einsum(w_q, "bsd,hdk->bhsk")!!;
	Tensor k = input.einsum(w_k, "bsd,hdk->bhsk")!!;
	Tensor v = input.einsum(w_v, "bsd,hdk->bhsk")!!;

	// Scaled dot-product attention
	Tensor scores = q.einsum(k, "bhik,bhjk->bhij")!!;
	float scale_val = 1.0f / math::sqrt((float)d_k);
	Tensor scale = tensor::constant(Datatype.F32, dev, { 1 }, scale_val)!!;
	Tensor scaled_scores = scores.mul(scale)!!;

	// Numerically stable softmax over last dimension (key positions)
	Tensor attn_weights = functional::softmax(scaled_scores, 3)!!;

	// Weighted sum
	Tensor attn_output = attn_weights.einsum(v, "bhij,bhjk->bhik")!!;

	// Project back and merge heads
	Tensor output = attn_output.einsum(w_o, "bhsk,hkd->bsd")!!.stable();

	return {
		.w_q = w_q, .w_k = w_k, .w_v = w_v, .w_o = w_o,
		.input = input, .output = output,
		.num_heads = num_heads, .d_model = d_model, .d_k = d_k,
		.batch_size = batch_size, .seq_len = seq_len,
	};
}

// ============================================================================
// FeedForward Network
// ============================================================================

struct FeedForward
{
	Tensor w1;
	Tensor b1;
	Tensor w2;
	Tensor b2;
	Tensor input;
	Tensor output;
}

fn FeedForward ffn_init(
	ulong d_model,
	ulong d_ff,
	ulong batch_size,
	ulong seq_len,
	Tensor input,
	DeviceReference dev)
{
	Tensor w1 = tensor::empty(Datatype.F32, dev, { d_model, d_ff })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)d_model));

	Tensor b1 = tensor::empty(Datatype.F32, dev, { d_ff })!!.enable_grad();
	b1.fill(0.0);

	Tensor w2 = tensor::empty(Datatype.F32, dev, { d_ff, d_model })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)d_ff));

	Tensor b2 = tensor::empty(Datatype.F32, dev, { d_model })!!.enable_grad();
	b2.fill(0.0);

	// Forward: ReLU(x @ W1 + b1) @ W2 + b2
	Tensor hidden = input.einsum(w1, "bsi,ij->bsj")!!;
	Tensor hidden_bias = hidden.einsum(b1, "bsj,j->bsj")!!;
	Tensor hidden_act = hidden_bias.relu()!!;
	Tensor out = hidden_act.einsum(w2, "bsj,ji->bsi")!!;
	Tensor output = out.einsum(b2, "bsi,i->bsi")!!.stable();

	return {
		.w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2,
		.input = input, .output = output,
	};
}

// ============================================================================
// Transformer Block
// ============================================================================

struct TransformerBlock
{
	MultiHeadAttention attention;
	FeedForward ffn;
	Tensor input;
	Tensor output;
	ulong batch_size;
	ulong seq_len;
	ulong d_model;
}

fn TransformerBlock transformer_init(
	ulong d_model,
	ulong num_heads,
	ulong d_ff,
	ulong batch_size,
	ulong seq_len,
	DeviceReference dev)
{
	MultiHeadAttention attn = mha_init(
		d_model, num_heads, batch_size, seq_len, dev);

	// Residual connection after attention
	Tensor after_attn = attn.input.add(attn.output)!!;

	// FFN with residual
	FeedForward ffn = ffn_init(
		d_model, d_ff, batch_size, seq_len, after_attn, dev);
	Tensor output = after_attn.add(ffn.output)!!.stable();

	return {
		.attention = attn,
		.ffn = ffn,
		.input = attn.input,
		.output = output,
		.batch_size = batch_size,
		.seq_len = seq_len,
		.d_model = d_model,
	};
}

// ============================================================================
// Main
// ============================================================================

fn void main()
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::init();
	defer graph::deinit();

	sampling::seed(42);

	ulong batch_size = 2;
	ulong seq_len = 8;
	ulong d_model = 64;
	ulong num_heads = 4;
	ulong d_ff = 256;

	io::printfn("\n=== Transformer Attention Block Example ===\n");
	io::printfn("Configuration:");
	io::printfn("  batch_size: %d", batch_size);
	io::printfn("  seq_len:    %d", seq_len);
	io::printfn("  d_model:    %d", d_model);
	io::printfn("  num_heads:  %d", num_heads);
	io::printfn("  d_ff:       %d", d_ff);
	io::printfn("  d_k:        %d (d_model / num_heads)\n", d_model / num_heads);

	io::printfn("Building transformer block...");

	TransformerBlock block = transformer_init(
		d_model, num_heads, d_ff, batch_size, seq_len, dev);

	io::printfn("  Input shape:  [%d, %d, %d]",
		block.input.shape().get(0),
		block.input.shape().get(1),
		block.input.shape().get(2));
	io::printfn("  Output shape: [%d, %d, %d]\n",
		block.output.shape().get(0),
		block.output.shape().get(1),
		block.output.shape().get(2));

	// Count parameters
	ulong d_k = d_model / num_heads;
	ulong attn_params = 4 * num_heads * d_model * d_k;
	ulong ffn_params = d_model * d_ff + d_ff + d_ff * d_model + d_model;
	io::printfn("Parameters:");
	io::printfn("  Attention: %d (4 x %d x %d x %d)",
		attn_params, num_heads, d_model, d_k);
	io::printfn("  FFN:       %d", ffn_params);
	io::printfn("  Total:     %d\n", attn_params + ffn_params);

	// Evaluate
	io::printfn("Evaluating...");
	block.output.collect()!!;

	char[] out_bytes = block.output.value_bytes();
	float* out_data = (float*)out_bytes.ptr;
	usz out_count = out_bytes.len / float.sizeof;

	float sum = 0;
	for (usz i = 0; i < out_count; i++)
	{
		sum += out_data[i];
	}
	float mean = sum / (float)out_count;

	io::printfn("  Output mean: %.6f", mean);
	io::printfn("  Output elements: %d\n", out_count);

	io::printfn("=== Complete ===\n");
	io::printfn("Key einsum patterns:");
	io::printfn("  Q/K/V projection: bsd,hdk->bhsk");
	io::printfn("  Attention scores: bhik,bhjk->bhij");
	io::printfn("  Attention output: bhij,bhjk->bhik");
	io::printfn("  Output project:   bhsk,hkd->bsd");
}
