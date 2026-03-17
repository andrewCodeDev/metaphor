//! Deep Neural Network Example
//!
//! Multi-layer perceptron with ReLU activations trained on synthetic data.
//! Architecture: 4 → 16 → 8 → 2
//!
//! Demonstrates:
//! - Graph-build-on-init pattern
//! - Kaiming weight initialization
//! - MSE loss from nn::loss
//! - SGD optimizer from nn::optimizer
//! - Training loop with collect / step
//!

module metaphor::examples::dnn;
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

// Problem dimensions
const ulong BATCH_SIZE = 4;
const ulong INPUT_DIM = 4;
const ulong HIDDEN1 = 16;
const ulong HIDDEN2 = 8;
const ulong OUTPUT_DIM = 2;

// Hyperparameters
const float LEARNING_RATE = 0.01;
const usz NUM_EPOCHS = 100;

// ============================================================================
// Linear Layer
// ============================================================================

struct Linear
{
	Tensor weight;
	Tensor bias;
	Tensor output;
}

fn Linear linear_init(
	ulong in_features,
	ulong out_features,
	Tensor input,
	DeviceReference dev)
{
	Tensor weight = tensor::empty(
		Datatype.F32, dev, { in_features, out_features })!!.enable_grad()
		.fill_random(device_reference::RandType.NORMAL,
			device_reference::kaiming_scale((usz)in_features));

	Tensor bias = tensor::empty(Datatype.F32, dev, { out_features })!!.enable_grad();
	bias.fill(0.0);

	Tensor output = input.matmul(weight)!! + bias;

	return { .weight = weight, .bias = bias, .output = output };
}

// ============================================================================
// DNN Model
// ============================================================================

struct DnnModel
{
	Linear[2] hidden;
	Linear output_layer;
	Tensor prediction;
}

fn DnnModel dnn_init(Tensor input, DeviceReference dev)
{
	DnnModel model;

	// Layer 0: 4 → 16 + ReLU
	model.hidden[0] = linear_init(INPUT_DIM, HIDDEN1, input, dev);
	Tensor h0 = model.hidden[0].output.relu()!!;

	// Layer 1: 16 → 8 + ReLU
	model.hidden[1] = linear_init(HIDDEN1, HIDDEN2, h0, dev);
	Tensor h1 = model.hidden[1].output.relu()!!;

	// Output: 8 → 2 (no activation)
	model.output_layer = linear_init(HIDDEN2, OUTPUT_DIM, h1, dev);
	model.prediction = model.output_layer.output;

	return model;
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

	io::printfn("\n=== DNN Example ===");
	io::printfn("Architecture: %d -> %d -> %d -> %d",
		INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM);
	io::printfn("Learning rate: %f", LEARNING_RATE);
	io::printfn("Epochs: %d\n", NUM_EPOCHS);

	// Input placeholder [batch=4, features=4]
	Tensor x = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, INPUT_DIM })!!;

	// Build the network
	DnnModel model = dnn_init(x, dev);

	// Target [batch=4, output=2]
	Tensor target = tensor::empty(Datatype.F32, dev, { BATCH_SIZE, OUTPUT_DIM })!!;

	// MSE loss
	Tensor mse = loss::mse_loss(model.prediction, target)!!.stable();

	// Backward graph
	mse.backward();

	// Optimizer
	Tensor[6] params = {
		model.hidden[0].weight, model.hidden[0].bias,
		model.hidden[1].weight, model.hidden[1].bias,
		model.output_layer.weight, model.output_layer.bias,
	};
	Optimizer optim = optimizer::init({ .lr = LEARNING_RATE });
	optim.sgd(&params, {});
	defer optim.deinit();

	// Synthetic data: simple classification
	// Class 0: positive sum → [1, 0]
	// Class 1: negative sum → [0, 1]
	float[BATCH_SIZE * INPUT_DIM] x_data = {
		 1.0,  0.5,  0.3,  0.2,
		-0.5, -1.0, -0.3, -0.2,
		 0.8,  0.1,  0.6,  0.4,
		-0.3, -0.7, -0.5, -0.1,
	};
	float[BATCH_SIZE * OUTPUT_DIM] y_data = {
		1.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 1.0,
	};

	// Feed data into tensors
	x.set(x_data);
	target.set(y_data);

	usz total_params =
		(usz)(INPUT_DIM * HIDDEN1 + HIDDEN1 +
		HIDDEN1 * HIDDEN2 + HIDDEN2 +
		HIDDEN2 * OUTPUT_DIM + OUTPUT_DIM);
	io::printfn("Total parameters: %d\n", total_params);

	// Training loop
	io::printfn("Training...");
	io::printfn("----------------------------------------");

	for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
	{
		mse.collect()!!;
		optim.step();
		// Input data is fixed, so manually mark a parameter as dirty to
		// trigger re-evaluation. Normally, calling .set() with new data
		// each iteration does this automatically.
		graph::notify_mutation(model.hidden[0].weight);

		if ((epoch + 1) % 20 == 0 || epoch == 0)
		{
			float loss_val;
			mse.get(&loss_val);
			io::printfn("Epoch %4d: loss = %f", epoch + 1, loss_val);
		}
	}

	io::printfn("----------------------------------------");
	io::printfn("Training complete!\n");

	// Print predictions
	model.prediction.collect()!!;
	float[BATCH_SIZE * OUTPUT_DIM] pp;
	model.prediction.get(&pp);

	io::printfn("Predictions:");
	for (usz i = 0; i < (usz)BATCH_SIZE; i++)
	{
		float p0 = pp[i * (usz)OUTPUT_DIM];
		float p1 = pp[i * (usz)OUTPUT_DIM + 1];
		usz predicted = (p0 > p1) ? (usz)0 : (usz)1;
		usz expected = (i % 2 == 0) ? (usz)0 : (usz)1;
		io::printfn("  Sample %d: [%.4f, %.4f] -> class %d (expected %d)",
			i, p0, p1, predicted, expected);
	}
}
