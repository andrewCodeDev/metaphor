//! MNIST Training Example
//!
//! Trains a simple MLP classifier on the MNIST dataset using the static
//! pipeline pattern.
//!
//! Architecture:
//!   Input [batch, 784] -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> Softmax
//!
//!
//! Usage:
//!   1. Run download_data.sh to download MNIST data
//!   2. c3c run mnist

module metaphor::examples::mnist;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::nn::functional;
import metaphor::rtti;
import metaphor::tensor::tensor_core;
import metaphor::utils::dataset;
import metaphor::utils::sampling;
import std::io;
import std::time;

// ============================================================================
// Aliases
// ============================================================================

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HostDevice = host_device::HostDevice;
alias Datatype = rtti::Datatype;
alias DatasetReader = dataset::DatasetReader;
alias Optimizer = optimizer::Optimizer;
alias Clock = time::Clock;
alias NanoDuration = time::NanoDuration;

// ============================================================================
// Configuration
// ============================================================================

const ulong BATCH_SIZE = 64;
const ulong INPUT_DIM = 784;   // 28x28
const ulong HIDDEN1 = 128;
const ulong HIDDEN2 = 64;
const ulong OUTPUT_DIM = 10;
const float LEARNING_RATE = 0.01;
const usz NUM_EPOCHS = 4;
const usz WARMUP_EPOCHS = 1;

// ============================================================================
// Model
// ============================================================================

struct MnistModel
{
	Tensor weights0;
	Tensor biases0;
	Tensor weights1;
	Tensor biases1;
	Tensor weights2;
	Tensor biases2;
	Tensor input;
	Tensor target;
	Tensor logits;
	Tensor loss_tensor;
}

fn MnistModel build_model(DeviceReference dev)
{
	// === Trainable Parameters ===

	Tensor w0 = tensor::empty(Datatype.F32, dev, { INPUT_DIM, HIDDEN1 })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)INPUT_DIM));

	Tensor b0 = tensor::empty(Datatype.F32, dev, { HIDDEN1 })!!.enable_grad();
	b0.fill(0.0);

	Tensor w1 = tensor::empty(Datatype.F32, dev, { HIDDEN1, HIDDEN2 })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)HIDDEN1));

	Tensor b1 = tensor::empty(Datatype.F32, dev, { HIDDEN2 })!!.enable_grad();
	b1.fill(0.0);

	Tensor w2 = tensor::empty(Datatype.F32, dev, { HIDDEN2, OUTPUT_DIM })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)HIDDEN2));

	Tensor b2 = tensor::empty(Datatype.F32, dev, { OUTPUT_DIM })!!.enable_grad();
	b2.fill(0.0);

	// === Input Placeholders ===

	Tensor input = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, INPUT_DIM })!!;

	Tensor target = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, OUTPUT_DIM })!!;

	// === Forward Pass Graph ===

	Tensor z0 = (input.matmul(w0)!! + b0).stable();
	Tensor a0 = z0.relu()!!;

	Tensor z1 = a0.matmul(w1)!! + b1;
	Tensor a1 = z1.relu()!!;

	Tensor logits = a1.matmul(w2)!! + b2;

	// === Loss ===
	Tensor loss_t = loss::softmax_cross_entropy(logits, target)!!.stable();

	return {
		.weights0 = w0, .biases0 = b0,
		.weights1 = w1, .biases1 = b1,
		.weights2 = w2, .biases2 = b2,
		.input = input,
		.target = target,
		.logits = logits,
		.loss_tensor = loss_t,
	};
}

// ============================================================================
// Accuracy computation
// ============================================================================

fn usz count_correct(float[] predictions, float[] labels, usz batch_size, usz num_classes)
{
	usz correct = 0;
	for (usz b = 0; b < batch_size; b++)
	{
		usz start = b * num_classes;
		usz pred_class = 0;
		float pred_max = predictions[start];
		for (usz i = 1; i < num_classes; i++)
		{
			if (predictions[start + i] > pred_max)
			{
				pred_max = predictions[start + i];
				pred_class = i;
			}
		}
		usz true_class = 0;
		float label_max = labels[start];
		for (usz i = 1; i < num_classes; i++)
		{
			if (labels[start + i] > label_max)
			{
				label_max = labels[start + i];
				true_class = i;
			}
		}
		if (pred_class == true_class) correct++;
	}
	return correct;
}

fn float compute_accuracy(MnistModel* model, DatasetReader* reader)
{
	usz correct = 0;
	usz total = 0;

	reader.reset();
	while (dataset::Batch* batch = reader.next())
	{
		model.input.bind(batch.images);
		model.logits.collect()!!;

		float[BATCH_SIZE * OUTPUT_DIM] logit_buf;
		model.logits.get(&logit_buf);
		float* lp = &logit_buf[0];

		correct += count_correct(
			lp[:BATCH_SIZE * OUTPUT_DIM],
			batch.labels,
			batch.size,
			(usz)OUTPUT_DIM);
		total += batch.size;
	}

	if (total == 0) return 0.0;
	return (float)correct / (float)total;
}

fn double ns_to_ms(NanoDuration ns)
{
	return (double)ns / 1_000_000.0;
}

// ============================================================================
// Main
// ============================================================================

fn void main()
{
	HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	sampling::seed(42);

	String data_dir = "data";

	io::printfn("\n=== MNIST Training Example ===");
	io::printfn("Batch size: %d", BATCH_SIZE);
	io::printfn("Learning rate: %f", LEARNING_RATE);
	io::printfn("Epochs: %d\n", NUM_EPOCHS);

	// Load MNIST data
	DatasetReader train_reader = dataset::init_mnist(data_dir, true, (usz)BATCH_SIZE)!!;
	defer train_reader.free();

	DatasetReader test_reader = dataset::init_mnist(data_dir, false, (usz)BATCH_SIZE)!!;
	defer test_reader.free();

	// Build model
	MnistModel model = build_model(dev);
	io::printfn("Model initialized");
	io::printfn("  Architecture: %d -> %d -> %d -> %d",
		INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM);

	// Build backward graph
	model.loss_tensor.backward();

	// Create optimizer
	Tensor[6] params = {
		model.weights0, model.biases0,
		model.weights1, model.biases1,
		model.weights2, model.biases2,
	};
	Optimizer optim = optimizer::init({ .lr = LEARNING_RATE });
	optim.sgd(&params, {});
	defer optim.deinit();

	usz num_batches = train_reader.num_batches();
	io::printfn("Training batches: %d\n", num_batches);

	// Use stable persistence so backward pass can reuse forward intermediates.
	graph::set_default_persistence(STABLE);

	// Training loop with timing
	io::printfn("Training...");
	io::printfn("----------------------------------------");

	NanoDuration total_train_ns = 0;
	NanoDuration total_forward_ns = 0;
	NanoDuration total_backward_ns = 0;

	for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
	{
		Clock epoch_clock = time::clock::now();
		NanoDuration epoch_forward_ns = 0;
		NanoDuration epoch_backward_ns = 0;
		float epoch_loss = 0.0;

		train_reader.reset();
		while (dataset::Batch* batch = train_reader.next())
		{
			model.input.bind(batch.images);
			model.target.bind(batch.labels);

			Clock step_clock = time::clock::now();
			model.loss_tensor.collect()!!;
			epoch_forward_ns += step_clock.mark();

			model.loss_tensor.get(&epoch_loss);

			step_clock = time::clock::now();
			optim.step();
			epoch_backward_ns += step_clock.mark();
		}

		NanoDuration epoch_ns = epoch_clock.mark();
		double epoch_ms = ns_to_ms(epoch_ns);

		if (epoch < WARMUP_EPOCHS)
		{
			io::printfn("Epoch %d: loss=%.6f [%.1f ms] (warmup)",
				epoch + 1, epoch_loss, epoch_ms);
		}
		else
		{
			total_train_ns += epoch_ns;
			total_forward_ns += epoch_forward_ns;
			total_backward_ns += epoch_backward_ns;
			io::printfn("Epoch %d: loss=%.6f [%.1f ms]",
				epoch + 1, epoch_loss, epoch_ms);
		}
	}

	double total_train_ms = ns_to_ms(total_train_ns);
	usz measured_epochs = NUM_EPOCHS - WARMUP_EPOCHS;
	usz total_samples = measured_epochs * num_batches * (usz)BATCH_SIZE;
	double samples_per_sec = (double)total_samples / (total_train_ms / 1000.0);

	io::printfn("----------------------------------------");
	io::printfn("Training complete!");
	io::printfn("  Total time: %.1f ms", total_train_ms);
	io::printfn("  Samples: %d", total_samples);
	io::printfn("  Throughput: %.1f samples/sec", samples_per_sec);

	// Per-step timing breakdown
	usz total_steps = measured_epochs * num_batches;
	double n = (double)total_steps;
	double fwd_ms = ns_to_ms(total_forward_ns);
	double bwd_ms = ns_to_ms(total_backward_ns);
	double total_step_ms = fwd_ms + bwd_ms;

	io::printfn("\n=== Step Timing (avg per batch) ===");
	io::printfn("  Forward:   %8.3f ms (%5.1f%%)", fwd_ms / n, fwd_ms / total_step_ms * 100.0);
	io::printfn("  Backward:  %8.3f ms (%5.1f%%)", bwd_ms / n, bwd_ms / total_step_ms * 100.0);
	io::printfn("  Total:     %8.3f ms", total_step_ms / n);

	// Evaluation
	io::printfn("\nEvaluation...");

	Clock eval_clock = time::clock::now();
	float train_acc = compute_accuracy(&model, &train_reader);
	double eval_train_ms = ns_to_ms(eval_clock.mark());

	eval_clock = time::clock::now();
	float test_acc = compute_accuracy(&model, &test_reader);
	double eval_test_ms = ns_to_ms(eval_clock.mark());

	io::printfn("  Train accuracy: %.2f%% (n=%d) [%.1f ms]",
		train_acc * 100.0f, train_reader.num_samples(), eval_train_ms);
	io::printfn("  Test accuracy:  %.2f%% (n=%d) [%.1f ms]",
		test_acc * 100.0f, test_reader.num_samples(), eval_test_ms);
}
