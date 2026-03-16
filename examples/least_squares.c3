//! Least-Squares Solver Example
//!
//! Fits a single linear layer (weight + bias) to synthetic data using
//! gradient descent on MSE loss.
//!
//! Model:
//!   y_pred = X @ W + b
//!
//! Loss:
//!   L = mean((y_pred - y_target)^2)
//!

module metaphor::examples::least_squares;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::loss;
import metaphor::nn::optimizer;
import metaphor::rtti;
import metaphor::utils::sampling;
import std::math;
import std::io;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;
alias HostDevice = host_device::HostDevice;
alias Datatype = rtti::Datatype;
alias Optimizer = optimizer::Optimizer;
alias Prng = sampling::Prng;

// Problem dimensions
const ulong NUM_SAMPLES = 256;
const ulong INPUT_DIM = 4;
const ulong OUTPUT_DIM = 1;

// Training hyperparameters
const float LEARNING_RATE = 0.01;
const usz NUM_EPOCHS = 200;
const float NOISE_SCALE = 0.1;

// ============================================================================
// Main
// ============================================================================

fn void main()
{
	HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	io::printfn("\n=== Least-Squares Solver ===");
	io::printfn("Samples: %d", NUM_SAMPLES);
	io::printfn("Input dim: %d", INPUT_DIM);
	io::printfn("Output dim: %d", OUTPUT_DIM);
	io::printfn("Learning rate: %f", LEARNING_RATE);
	io::printfn("Epochs: %d\n", NUM_EPOCHS);

	// =====================================================================
	// Generate synthetic data: y = X @ w_true + b_true + noise
	// =====================================================================
	Prng rng = sampling::prng_init(42);

	// True parameters
	float[4] w_true = { 2.0, -1.0, 0.5, 3.0 };
	float b_true = 1.5;

	// Generate random input data X [NUM_SAMPLES, INPUT_DIM]
	float[NUM_SAMPLES * INPUT_DIM] x_data;
	for (usz i = 0; i < NUM_SAMPLES * INPUT_DIM; i++)
	{
		// Box-Muller for normal distribution
		float u1 = rng.float01();
		float u2 = rng.float01();
		if (u1 < 1e-10) u1 = 1e-10;
		float r = math::sqrt(-2.0f * math::ln(u1));
		float theta = 2.0f * (float)math::PI * u2;
		x_data[i] = r * math::cos(theta);
	}

	// Compute y = X @ w_true + b_true + noise
	float[NUM_SAMPLES] y_data;
	for (usz i = 0; i < (usz)NUM_SAMPLES; i++)
	{
		float y = b_true;
		for (usz j = 0; j < (usz)INPUT_DIM; j++)
		{
			y += x_data[i * (usz)INPUT_DIM + j] * w_true[j];
		}
		// Add noise
		float u1 = rng.float01();
		float u2 = rng.float01();
		if (u1 < 1e-10) u1 = 1e-10;
		float noise = math::sqrt(-2.0f * math::ln(u1))
			* math::cos(2.0f * (float)math::PI * u2) * NOISE_SCALE;
		y_data[i] = y + noise;
	}

	io::printfn("True weights: %.2f %.2f %.2f %.2f",
		w_true[0], w_true[1], w_true[2], w_true[3]);
	io::printfn("True bias:    %.2f\n", b_true);

	// =====================================================================
	// Build model: y_pred = X @ W + b
	// =====================================================================

	// Trainable weight [INPUT_DIM, OUTPUT_DIM] — random init
	Tensor weight = tensor::empty(Datatype.F32, dev, { INPUT_DIM, OUTPUT_DIM })!!.enable_grad();
	{
		float[INPUT_DIM * OUTPUT_DIM] w_init;
		for (usz i = 0; i < (usz)(INPUT_DIM * OUTPUT_DIM); i++)
		{
			w_init[i] = (rng.float01() - 0.5) * 0.2;
		}
		weight.set(w_init);
	}

	// Trainable bias [OUTPUT_DIM] — zero init
	Tensor bias = tensor::empty(Datatype.F32, dev, { OUTPUT_DIM })!!.enable_grad();
	bias.fill(0.0);

	// Input data [NUM_SAMPLES, INPUT_DIM]
	Tensor input = tensor::@from_data(dev, { NUM_SAMPLES, INPUT_DIM }, &x_data)!!;

	// Target data [NUM_SAMPLES, OUTPUT_DIM]
	Tensor target = tensor::@from_data(dev, { NUM_SAMPLES, OUTPUT_DIM }, &y_data)!!;

	// Forward: y_pred = X @ W + b
	Tensor pred = input.matmul(weight)!! + bias;

	// Loss: MSE
	Tensor mse = loss::mse_loss(pred, target)!!.stable();

	// Build backward graph
	mse.backward();

	// Create optimizer
	Tensor[2] params = { weight, bias };
	Optimizer optim = optimizer::init({ .lr = LEARNING_RATE });
	optim.sgd(&params, {});
	defer optim.deinit();

	// =====================================================================
	// Training loop
	// =====================================================================

	io::printfn("Training...");
	io::printfn("----------------------------------------");

	for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
	{
		mse.collect()!!;
		optim.step();
		// Input data is fixed, so manually mark a parameter as dirty to
		// trigger re-evaluation. Normally, calling .set() with new data
		// each iteration does this automatically.
		graph::notify_mutation(weight);

		if ((epoch + 1) % 20 == 0 || epoch == 0)
		{
			float loss_val;
			mse.get(&loss_val);
			io::printfn("Epoch %4d: loss = %f", epoch + 1, loss_val);
		}
	}

	io::printfn("----------------------------------------");
	io::printfn("Training complete!\n");

	// Print learned parameters
	float[INPUT_DIM * OUTPUT_DIM] fw;
	weight.get(&fw);
	io::printfn("Learned weights: %.4f %.4f %.4f %.4f",
		fw[0], fw[1], fw[2], fw[3]);

	float bias_val;
	bias.get(&bias_val);
	io::printfn("Learned bias:    %.4f", bias_val);

	float loss_final;
	mse.get(&loss_final);
	io::printfn("Final MSE loss:  %.6f", loss_final);
}
