//! HIP Loss Function Tests
//!
//! Tests nn/loss operations on the HIP device:
//! MSE loss, cross-entropy, softmax cross-entropy, binary cross-entropy.
//! Includes backward pass tests for gradient flow through losses.

module metaphor::testing::device::hip_device::hip_loss_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::loss;
import metaphor::nn::functional;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-4;

// ============================================================================
// MSE Loss
// ============================================================================

fn void test_mse_loss_zero(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[4] data = { 1, 2, 3, 4 };
		ulong[1] dims = { 4 };

		Tensor pred = tensor::@from_data(dev, &dims, &data)!!;
		Tensor target = tensor::@from_data(dev, &dims, &data)!!;

		Tensor l = loss::mse_loss(pred, target)!!;
		l.collect()!!;
		dev.sync();

		float[1] r;
		l.get(&r);
		assert(math::abs(r[0]) < TOL, "MSE of identical = 0");
	};
}

fn void test_mse_loss_nonzero(DeviceReference dev)
{
	graph::@subgraph()
	{
		// pred = [0,0,0,0], target = [1,1,1,1] -> MSE = 1
		float[4] p_data = { 0, 0, 0, 0 };
		float[4] t_data = { 1, 1, 1, 1 };
		ulong[1] dims = { 4 };

		Tensor pred = tensor::@from_data(dev, &dims, &p_data)!!;
		Tensor target = tensor::@from_data(dev, &dims, &t_data)!!;

		Tensor l = loss::mse_loss(pred, target)!!;
		l.collect()!!;
		dev.sync();

		float[1] r;
		l.get(&r);
		assert(math::abs(r[0] - 1.0f) < TOL, "MSE [0,0,0,0] vs [1,1,1,1] = 1");
	};
}

fn void test_mse_loss_values(DeviceReference dev)
{
	graph::@subgraph()
	{
		// pred = [1,2,3], target = [4,5,6] -> MSE = mean(9,9,9) = 9
		float[3] p_data = { 1, 2, 3 };
		float[3] t_data = { 4, 5, 6 };
		ulong[1] dims = { 3 };

		Tensor pred = tensor::@from_data(dev, &dims, &p_data)!!;
		Tensor target = tensor::@from_data(dev, &dims, &t_data)!!;

		Tensor l = loss::mse_loss(pred, target)!!;
		l.collect()!!;
		dev.sync();

		float[1] r;
		l.get(&r);
		assert(math::abs(r[0] - 9.0f) < TOL, "MSE [1,2,3] vs [4,5,6] = 9");
	};
}

// ============================================================================
// Softmax Cross-Entropy
// ============================================================================

fn void test_softmax_cross_entropy(DeviceReference dev)
{
	graph::@subgraph()
	{
		// logits = [2, 1, 0.1], target = [1, 0, 0] (class 0 correct)
		float[3] logit_data = { 2, 1, 0.1 };
		float[3] target_data = { 1, 0, 0 };
		ulong[2] dims = { 1, 3 };

		Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
		Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

		Tensor l = loss::softmax_cross_entropy(logits, target)!!;
		l.collect()!!;
		dev.sync();

		float[1] r;
		l.get(&r);
		assert(r[0] > 0.0f, "cross-entropy > 0");
		assert(r[0] < 2.0f, "cross-entropy reasonable magnitude");
	};
}

fn void test_softmax_cross_entropy_perfect(DeviceReference dev)
{
	graph::@subgraph()
	{
		// Strong logits for correct class -> loss near 0
		float[3] logit_data = { 100, 0, 0 };
		float[3] target_data = { 1, 0, 0 };
		ulong[2] dims = { 1, 3 };

		Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
		Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

		Tensor l = loss::softmax_cross_entropy(logits, target)!!;
		l.collect()!!;
		dev.sync();

		float[1] r;
		l.get(&r);
		assert(r[0] < 0.01f, "perfect prediction: loss ~0");
	};
}

fn void test_softmax_cross_entropy_wrong(DeviceReference dev)
{
	graph::@subgraph()
	{
		// Strong logits for wrong class -> loss large
		float[3] logit_data = { 0, 0, 100 };
		float[3] target_data = { 1, 0, 0 };
		ulong[2] dims = { 1, 3 };

		Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!;
		Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

		Tensor l = loss::softmax_cross_entropy(logits, target)!!;
		l.collect()!!;
		dev.sync();

		float[1] r;
		l.get(&r);
		assert(r[0] > 50.0f, "wrong prediction: loss large");
	};
}

// ============================================================================
// MSE Backward
// ============================================================================

fn void test_mse_backward(DeviceReference dev)
{
	graph::@subgraph()
	{
		// loss = mean((pred - target)^2)
		// dloss/dpred = 2*(pred - target)/n
		// pred = [2,4], target = [1,1] -> diff = [1,3]
		// grad = 2*[1,3]/2 = [1, 3]
		float[2] p_data = { 2, 4 };
		float[2] t_data = { 1, 1 };
		ulong[1] dims = { 2 };

		Tensor pred = tensor::@from_data(dev, &dims, &p_data)!!.enable_grad();
		Tensor target = tensor::@from_data(dev, &dims, &t_data)!!;

		Tensor l = loss::mse_loss(pred, target)!!;
		l.collect()!!;
		l.backward();

		Tensor grad_pred = pred.grad();
		grad_pred.collect()!!;
		dev.sync();

		float[2] g;
		grad_pred.get(&g);
		assert(math::abs(g[0] - 1.0f) < TOL, "MSE grad[0]=1");
		assert(math::abs(g[1] - 3.0f) < TOL, "MSE grad[1]=3");
	};
}

// ============================================================================
// Softmax Cross-Entropy Backward
// ============================================================================

fn void test_softmax_cross_entropy_backward(DeviceReference dev)
{
	graph::@subgraph()
	{
		// For softmax cross-entropy, grad w.r.t. logits = softmax(logits) - target
		// logits = [2, 1, 0], target = [1, 0, 0]
		float[3] logit_data = { 2, 1, 0 };
		float[3] target_data = { 1, 0, 0 };
		ulong[2] dims = { 1, 3 };

		Tensor logits = tensor::@from_data(dev, &dims, &logit_data)!!.enable_grad();
		Tensor target = tensor::@from_data(dev, &dims, &target_data)!!;

		Tensor l = loss::softmax_cross_entropy(logits, target)!!;
		l.collect()!!;
		l.backward();

		Tensor grad_logits = logits.grad();
		grad_logits.collect()!!;
		dev.sync();

		// Gradient should exist and be non-zero
		float[3] g;
		grad_logits.get(&g);

		// For correct class (0), gradient should be negative (softmax(x0) - 1 < 0)
		assert(g[0] < 0, "SCE grad for correct class < 0");
		// For wrong classes, gradient should be positive (softmax(xi) - 0 > 0)
		assert(g[1] > 0, "SCE grad for wrong class > 0");
		assert(g[2] > 0, "SCE grad for wrong class > 0");

		// Sum of gradients should be ~0 (softmax sums to 1, target sums to 1)
		float gsum = g[0] + g[1] + g[2];
		assert(math::abs(gsum) < TOL, "SCE grad sum ~0");
	};
}

// ============================================================================
// Entry point
// ============================================================================

fn void run_all(DeviceReference dev)
{
	test_mse_loss_zero(dev);
	test_mse_loss_nonzero(dev);
	test_mse_loss_values(dev);
	test_softmax_cross_entropy(dev);
	test_softmax_cross_entropy_perfect(dev);
	test_softmax_cross_entropy_wrong(dev);
	test_mse_backward(dev);
	test_softmax_cross_entropy_backward(dev);
}
