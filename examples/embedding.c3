//! Embedding Table Training Example
//!
//! Trains a character-level embedding table to predict the next character's
//! embedding using gather-based lookup.
//!
//! Architecture:
//!   input_ids [batch, seq_len] (u64)
//!       | gather(embedding_table, dim=0)
//!   embeddings [batch, seq_len, embed_dim]
//!       | reduce("ijk->ik") / seq_len
//!   pooled [batch, embed_dim]
//!       | matmul(projection)
//!   predicted [batch, embed_dim]
//!
//! Loss: MSE between predicted embedding and target character embedding
//!

module metaphor::examples::embedding;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::tensor;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::rtti;
import metaphor::utils::sampling;
import std::io;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HostDevice = host_device::HostDevice;
alias Datatype = rtti::Datatype;
alias Optimizer = optimizer::Optimizer;

// ============================================================================
// Character Tokenizer
// ============================================================================

const ulong VOCAB_SIZE = 27; // a-z + space

fn ulong char_to_id(char c)
{
	if (c >= 'a' && c <= 'z') return (ulong)(c - 'a');
	return 26;
}

fn char id_to_char(ulong id)
{
	if (id < 26) return (char)('a' + id);
	return ' ';
}

fn usz encode(char[] text, ulong[] output)
{
	usz count = 0;
	for (usz i = 0; i < text.len && count < output.len; i++)
	{
		output[count] = char_to_id(text[i]);
		count++;
	}
	return count;
}

// ============================================================================
// Constants
// ============================================================================

const ulong EMBED_DIM = 32;
const ulong BATCH_SIZE = 4;
const ulong SEQ_LEN = 8;
const float LEARNING_RATE = 0.01;
const usz NUM_EPOCHS = 20;

// ============================================================================
// Main
// ============================================================================

fn void main()
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	sampling::seed(42);

	io::printfn("\n=== Embedding Training Example ===");
	io::printfn("Vocab size: %d", VOCAB_SIZE);
	io::printfn("Embedding dim: %d", EMBED_DIM);
	io::printfn("Batch size: %d", BATCH_SIZE);
	io::printfn("Sequence length: %d", SEQ_LEN);
	io::printfn("Learning rate: %f", LEARNING_RATE);
	io::printfn("Epochs: %d\n", NUM_EPOCHS);

	// --- Trainable parameters ---

	Tensor embedding_table = tensor::empty(
		Datatype.F32, dev, { VOCAB_SIZE, EMBED_DIM })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)EMBED_DIM));

	Tensor projection = tensor::empty(
		Datatype.F32, dev, { EMBED_DIM, EMBED_DIM })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)EMBED_DIM));

	io::printfn("Model initialized:");
	io::printfn("  - Embedding table: [%d, %d]",
		embedding_table.shape().get(0), embedding_table.shape().get(1));
	io::printfn("  - Projection: [%d, %d]\n",
		projection.shape().get(0), projection.shape().get(1));

	// --- Input placeholders ---

	Tensor input_ids = tensor::empty(Datatype.U64, dev, { BATCH_SIZE, SEQ_LEN })!!;

	Tensor target_ids = tensor::empty(Datatype.U64, dev, { BATCH_SIZE })!!;

	// --- Forward pass graph ---

	Tensor input_embeddings = embedding_table.gather(0, input_ids)!!;

	Tensor pooled = input_embeddings.reduce("ijk->ik")!!;
	Tensor mean_pooled = pooled / (float)SEQ_LEN;

	Tensor predicted = mean_pooled.matmul(projection)!!.stable();

	// --- Target embedding + Loss ---

	Tensor target_emb = embedding_table.gather(0, target_ids)!!.stable();

	Tensor mse = loss::mse_loss(predicted, target_emb)!!.stable();
	mse.backward();

	// Optimizer
	Tensor[2] params = { embedding_table, projection };
	Optimizer optim = optimizer::init({ .lr = LEARNING_RATE });
	optim.sgd(&params, {});
	defer optim.deinit();

	// --- Training data ---
	char[][4] texts = {
		"hello wo",
		"the quic",
		"metaphor",
		"embeddin"
	};
	char[4] next_chars = { 'r', 'k', 's', 'g' };

	ulong[BATCH_SIZE * SEQ_LEN] input_tokens;
	for (usz i = 0; i < (usz)BATCH_SIZE; i++)
	{
		ulong[SEQ_LEN] tokens;
		encode(texts[i][:(usz)SEQ_LEN], &tokens);
		for (usz s = 0; s < (usz)SEQ_LEN; s++)
		{
			input_tokens[i * (usz)SEQ_LEN + s] = tokens[s];
		}
	}

	ulong[BATCH_SIZE] target_token_ids;
	for (usz i = 0; i < (usz)BATCH_SIZE; i++)
	{
		target_token_ids[i] = char_to_id(next_chars[i]);
	}

	io::printfn("Training data:");
	for (usz i = 0; i < (usz)BATCH_SIZE; i++)
	{
		io::printfn("  \"%s\" -> '%c'", texts[i], next_chars[i]);
	}
	io::printfn("");

	// --- Training loop ---
	io::printfn("Training...");
	io::printfn("----------------------------------------");

	for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
	{
		input_ids.set(input_tokens);
		target_ids.set(target_token_ids);

		mse.collect()!!;
		optim.step();

		float loss_val;
		mse.get(&loss_val);
		io::printfn("Epoch %3d: loss = %.6f", epoch + 1, loss_val);
	}

	io::printfn("----------------------------------------");
	io::printfn("Training complete!\n");

	// --- Inference demo ---
	io::printfn("Inference demo:");
	io::printfn("  Input: \"hello wo\"");

	predicted.collect()!!;
	float[4] pred_first;
	predicted.get(&pred_first);
	io::printfn("  Predicted embedding (first 4 dims): [%.4f, %.4f, %.4f, %.4f]",
		pred_first[0], pred_first[1], pred_first[2], pred_first[3]);

	io::printfn("\nDone!");
}
