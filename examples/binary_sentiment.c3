//! Binary Sentiment Classification Example
//!
//! Trains a simple sentiment classifier with embedding + MLP + sigmoid.
//!
//! Architecture:
//!   input_ids [batch, seq_len] (u64)
//!       | gather(embedding_table, dim=0)
//!   embeddings [batch, seq_len, embed_dim]
//!       | reduce("ijk->ik") / seq_len  (mean pool)
//!   pooled [batch, embed_dim]
//!       | matmul(w0) + b0 → ReLU
//!   hidden [batch, hidden_dim]
//!       | matmul(w1) + b1
//!   logits [batch, 1]
//!       | sigmoid
//!   output [batch, 1] (probability)
//!

module metaphor::examples::binary_sentiment;
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
// Configuration
// ============================================================================

const ulong VOCAB_SIZE = 100;
const ulong EMBED_DIM = 16;
const ulong HIDDEN_DIM = 8;
const ulong SEQ_LEN = 6;
const ulong BATCH_SIZE = 4;
const float LEARNING_RATE = 0.1;
const usz NUM_EPOCHS = 50;

// ============================================================================
// Main
// ============================================================================

fn void main()
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	sampling::seed(42);

	io::printfn("\n=== Binary Sentiment Classification ===\n");

	// --- Trainable parameters ---

	Tensor embedding_table = tensor::empty(
		Datatype.F32, dev, { VOCAB_SIZE, EMBED_DIM })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)VOCAB_SIZE));

	Tensor w0 = tensor::empty(
		Datatype.F32, dev, { EMBED_DIM, HIDDEN_DIM })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)EMBED_DIM));

	Tensor b0 = tensor::empty(Datatype.F32, dev, { HIDDEN_DIM })!!.enable_grad();
	b0.fill(0.0);

	Tensor w1 = tensor::empty(
		Datatype.F32, dev, { HIDDEN_DIM, 1 })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)HIDDEN_DIM));

	Tensor b1 = tensor::empty(Datatype.F32, dev, { 1 })!!.enable_grad();
	b1.fill(0.0);

	// --- Input placeholders ---

	// Input token IDs [batch, seq_len] — u64 for gather
	Tensor input_ids = tensor::empty(Datatype.U64, dev, { BATCH_SIZE, SEQ_LEN })!!;

	// --- Forward graph ---

	// 1. Embed: [batch, seq_len] -> [batch, seq_len, embed_dim]
	Tensor embeddings = embedding_table.gather(0, input_ids)!!;

	// 2. Mean pool: reduce sum over seq_len, then divide
	Tensor summed = embeddings.reduce("ijk->ik")!!;
	Tensor pooled = summed / (float)SEQ_LEN;

	// 3. MLP: Linear(embed_dim -> hidden) -> ReLU -> Linear(hidden -> 1)
	Tensor z0 = pooled.matmul(w0)!! + b0;
	Tensor a0 = z0.relu()!!;
	Tensor logits = (a0.matmul(w1)!! + b1).stable();

	// 4. Sigmoid
	Tensor output = logits.sigmoid()!!.stable();

	// --- Loss ---
	Tensor target = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, 1 })!!;

	Tensor bce = loss::binary_cross_entropy(output, target)!!.stable();
	bce.backward();

	// --- Optimizer ---
	Tensor[5] params = {
		embedding_table, w0, b0, w1, b1,
	};
	Optimizer optim = optimizer::init({ .lr = LEARNING_RATE });
	optim.sgd(&params, {});
	defer optim.deinit();

	// --- Synthetic training data ---
	// "Positive" tokens: IDs 1-10 (low range)
	// "Negative" tokens: IDs 50-60 (high range)
	ulong[BATCH_SIZE * SEQ_LEN] token_data = {
		// Sample 0: positive (label 1)
		1, 3, 5, 2, 7, 4,
		// Sample 1: negative (label 0)
		50, 53, 55, 52, 57, 54,
		// Sample 2: positive (label 1)
		2, 8, 1, 6, 3, 9,
		// Sample 3: negative (label 0)
		51, 58, 56, 53, 59, 52,
	};
	float[BATCH_SIZE] label_data = { 1.0, 0.0, 1.0, 0.0 };

	usz num_params = (usz)(VOCAB_SIZE * EMBED_DIM +
		EMBED_DIM * HIDDEN_DIM + HIDDEN_DIM +
		HIDDEN_DIM * 1 + 1);
	io::printfn("Model parameters: %d", num_params);
	io::printfn("  Embedding: [%d, %d]", VOCAB_SIZE, EMBED_DIM);
	io::printfn("  Hidden: [%d, %d]", EMBED_DIM, HIDDEN_DIM);
	io::printfn("  Output: [%d, 1]\n", HIDDEN_DIM);

	io::printfn("Training...");
	io::printfn("Epochs: %d, Batch size: %d, LR: %.4f\n",
		NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
	io::printfn("----------------------------------------");

	for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
	{
		// Fill input tokens (u64)
		input_ids.set(token_data);
		// Fill target labels (f32, reshaped to [batch, 1])
		target.set(label_data);

		bce.collect()!!;
		optim.step();

		if ((epoch + 1) % 5 == 0 || epoch == 0)
		{
			float loss_val;
			bce.get(&loss_val);
			io::printfn("Epoch %3d: loss = %.6f", epoch + 1, loss_val);
		}

	}

	io::printfn("----------------------------------------");

	// Print final predictions
	output.collect()!!;
	float[BATCH_SIZE] probs;
	output.get(&probs);

	io::printfn("\nPredictions (prob of positive):");
	for (usz i = 0; i < (usz)BATCH_SIZE; i++)
	{
		float expected = label_data[i];
		io::printfn("  Sample %d: %.4f (expected %.0f)", i, probs[i], expected);
	}

	io::printfn("\nTraining complete!");
}
