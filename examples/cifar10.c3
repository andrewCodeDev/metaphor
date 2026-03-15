//! CIFAR-10 Training Example
//!
//! Trains a 5-layer MLP on CIFAR-10 images (flattened 3072-dim input).
//!
//! Architecture:
//!   Input [batch, 3072] → Linear(512) → ReLU
//!                       → Linear(256) → ReLU
//!                       → Linear(128) → ReLU
//!                       → Linear(64)  → ReLU
//!                       → Linear(10)  → Output
//!
//!
//! NOTE: Uses placeholder data. Full implementation requires dataset loading.

module metaphor::examples::cifar10;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
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

const ulong BATCH_SIZE = 64;
const ulong INPUT_DIM = 3072;   // 32×32×3
const ulong NUM_CLASSES = 10;
const float LEARNING_RATE = 0.01;
const usz NUM_EPOCHS = 1;

// ============================================================================
// Linear Layer helper
// ============================================================================

struct LinearLayer
{
	Tensor weight;
	Tensor bias;
	Tensor output;
}

fn LinearLayer linear(ulong in_dim, ulong out_dim, Tensor input, DeviceReference dev)
{
	Tensor w = tensor::empty(Datatype.F32, dev, { in_dim, out_dim })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)in_dim));

	Tensor b = tensor::empty(Datatype.F32, dev, { out_dim })!!.enable_grad();
	b.fill(0.0);

	Tensor out = input.matmul(w)!!.add(b)!!;

	return { .weight = w, .bias = b, .output = out };
}

// ============================================================================
// CIFAR-10 Model
// ============================================================================

struct Cifar10Model
{
	LinearLayer[5] layers;
	Tensor input;
	Tensor prediction;
}

fn Cifar10Model cifar10_init(DeviceReference dev)
{
	Cifar10Model model;

	// Input placeholder [batch, 3072]
	model.input = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, INPUT_DIM })!!;
	Tensor x = model.input;

	// Layer 0: 3072 → 512 + ReLU
	model.layers[0] = linear(INPUT_DIM, 512, x, dev);
	Tensor h0 = model.layers[0].output.relu()!!;

	// Layer 1: 512 → 256 + ReLU
	model.layers[1] = linear(512, 256, h0, dev);
	Tensor h1 = model.layers[1].output.relu()!!;

	// Layer 2: 256 → 128 + ReLU
	model.layers[2] = linear(256, 128, h1, dev);
	Tensor h2 = model.layers[2].output.relu()!!;

	// Layer 3: 128 → 64 + ReLU
	model.layers[3] = linear(128, 64, h2, dev);
	Tensor h3 = model.layers[3].output.relu()!!;

	// Layer 4: 64 → 10 (raw logits)
	model.layers[4] = linear(64, NUM_CLASSES, h3, dev);
	model.prediction = model.layers[4].output.stable();

	return model;
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

	io::printfn("\n=== CIFAR-10 Training ===\n");

	Cifar10Model model = cifar10_init(dev);

	usz total_params = (usz)(
		INPUT_DIM * 512 + 512 +
		512 * 256 + 256 +
		256 * 128 + 128 +
		128 * 64 + 64 +
		64 * NUM_CLASSES + NUM_CLASSES);
	io::printfn("Model parameters: %d", total_params);
	io::printfn("Architecture: %d -> 512 -> 256 -> 128 -> 64 -> %d\n",
		INPUT_DIM, NUM_CLASSES);

	// Target + softmax cross-entropy loss
	Tensor target = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, NUM_CLASSES })!!;

	Tensor ce = loss::softmax_cross_entropy(model.prediction, target)!!.stable();
	ce.backward();

	// Collect all trainable parameters
	Tensor[10] params = {
		model.layers[0].weight, model.layers[0].bias,
		model.layers[1].weight, model.layers[1].bias,
		model.layers[2].weight, model.layers[2].bias,
		model.layers[3].weight, model.layers[3].bias,
		model.layers[4].weight, model.layers[4].bias,
	};
	Optimizer optim = optimizer::init({ .lr = LEARNING_RATE });
	optim.sgd(&params, {});
	defer optim.deinit();

	io::printfn("Starting training...");
	io::printfn("Epochs: %d, Batch size: %d, LR: %.4f\n",
		NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);

	graph::Graph* g = graph::get();

	for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
	{
		g.advance_generation();
		ce.collect()!!;
		optim.step();

		char[] lb = ce.value_bytes();
		float loss_val = ((float*)lb.ptr)[0];
		io::printfn("Epoch %d: loss = %.6f", epoch + 1, loss_val);
	}

	io::printfn("\nTraining complete!");
}
