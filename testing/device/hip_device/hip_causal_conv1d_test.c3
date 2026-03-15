//! HIP Causal Conv1D Tests
//!
//! Tests the dedicated causal conv1d kernel (forward + backward)
//! against hand-computed expected values.
//!
//! Forward: out[b,e,t] = bias[e] + sum_{k=0}^{K-1} weight[e,0,k] * x[b,e,t-(K-1)+k]
//! (cross-correlation with causal zero-padding)

module metaphor::testing::device::hip_device::hip_causal_conv1d_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::nn::functional;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-4;

// ============================================================================
// Forward Tests
// ============================================================================

/// Single batch, 2 channels, 4 timesteps, kernel_size=2.
/// Hand-compute cross-correlation with causal padding.
fn void test_forward_basic(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [1, 2, 4]
		// Channel 0: [1, 2, 3, 4]
		// Channel 1: [5, 6, 7, 8]
		float[8] x_data = { 1, 2, 3, 4, 5, 6, 7, 8 };
		ulong[3] x_dims = { 1, 2, 4 };

		// weight: [2, 1, 2] (K=2)
		// Channel 0: [0.5, 1.0]
		// Channel 1: [2.0, 0.5]
		float[4] w_data = { 0.5, 1.0, 2.0, 0.5 };
		ulong[3] w_dims = { 2, 1, 2 };

		// bias: [2]
		float[2] b_data = { 0.1, 0.2 };
		ulong[1] b_dims = { 2 };

		Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!;
		Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!;
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor result = functional::causal_conv1d(x, w, b)!!;
		result.collect()!!;
		dev.sync();

		float[8] r;
		result.get(&r);

		// Channel 0, weight=[0.5, 1.0], bias=0.1
		// Cross-correlation: out[t] = w[0]*x[t-1] + w[1]*x[t] + bias
		// t=0: w[0]*0     + w[1]*1   + 0.1 = 0.5*0 + 1.0*1 + 0.1 = 1.1
		// t=1: w[0]*1     + w[1]*2   + 0.1 = 0.5*1 + 1.0*2 + 0.1 = 2.6
		// t=2: w[0]*2     + w[1]*3   + 0.1 = 0.5*2 + 1.0*3 + 0.1 = 4.1
		// t=3: w[0]*3     + w[1]*4   + 0.1 = 0.5*3 + 1.0*4 + 0.1 = 5.6
		assert(math::abs(r[0] - 1.1f) < TOL, "fwd ch0 t0");
		assert(math::abs(r[1] - 2.6f) < TOL, "fwd ch0 t1");
		assert(math::abs(r[2] - 4.1f) < TOL, "fwd ch0 t2");
		assert(math::abs(r[3] - 5.6f) < TOL, "fwd ch0 t3");

		// Channel 1, weight=[2.0, 0.5], bias=0.2
		// t=0: 2.0*0 + 0.5*5 + 0.2 = 2.7
		// t=1: 2.0*5 + 0.5*6 + 0.2 = 13.2
		// t=2: 2.0*6 + 0.5*7 + 0.2 = 15.7
		// t=3: 2.0*7 + 0.5*8 + 0.2 = 18.2
		assert(math::abs(r[4] - 2.7f) < TOL, "fwd ch1 t0");
		assert(math::abs(r[5] - 13.2f) < TOL, "fwd ch1 t1");
		assert(math::abs(r[6] - 15.7f) < TOL, "fwd ch1 t2");
		assert(math::abs(r[7] - 18.2f) < TOL, "fwd ch1 t3");
	};
}

/// Test with kernel_size=4 (the actual Mamba d_conv).
/// Single batch, 1 channel, 6 timesteps.
fn void test_forward_k4(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [1, 1, 6] = [1, 2, 3, 4, 5, 6]
		float[6] x_data = { 1, 2, 3, 4, 5, 6 };
		ulong[3] x_dims = { 1, 1, 6 };

		// weight: [1, 1, 4] = [1, 1, 1, 1] (uniform, sums the window)
		float[4] w_data = { 1, 1, 1, 1 };
		ulong[3] w_dims = { 1, 1, 4 };

		// bias: [1] = [0]
		float[1] b_data = { 0 };
		ulong[1] b_dims = { 1 };

		Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!;
		Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!;
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor result = functional::causal_conv1d(x, w, b)!!;
		result.collect()!!;
		dev.sync();

		float[6] r;
		result.get(&r);

		// K=4, causal padding: pad 3 zeros before [1,2,3,4,5,6]
		// Effective input: [0,0,0,1,2,3,4,5,6]
		// t=0: 0+0+0+1 = 1
		// t=1: 0+0+1+2 = 3
		// t=2: 0+1+2+3 = 6
		// t=3: 1+2+3+4 = 10
		// t=4: 2+3+4+5 = 14
		// t=5: 3+4+5+6 = 18
		assert(math::abs(r[0] - 1.0f) < TOL, "fwd k4 t0");
		assert(math::abs(r[1] - 3.0f) < TOL, "fwd k4 t1");
		assert(math::abs(r[2] - 6.0f) < TOL, "fwd k4 t2");
		assert(math::abs(r[3] - 10.0f) < TOL, "fwd k4 t3");
		assert(math::abs(r[4] - 14.0f) < TOL, "fwd k4 t4");
		assert(math::abs(r[5] - 18.0f) < TOL, "fwd k4 t5");
	};
}

/// Test with batch_size=2 to ensure batch dimension is handled.
fn void test_forward_batched(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [2, 1, 3]
		// Batch 0: [1, 2, 3]
		// Batch 1: [4, 5, 6]
		float[6] x_data = { 1, 2, 3, 4, 5, 6 };
		ulong[3] x_dims = { 2, 1, 3 };

		// weight: [1, 1, 2] = [1, 2]
		float[2] w_data = { 1, 2 };
		ulong[3] w_dims = { 1, 1, 2 };

		// bias: [1] = [0.5]
		float[1] b_data = { 0.5 };
		ulong[1] b_dims = { 1 };

		Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!;
		Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!;
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor result = functional::causal_conv1d(x, w, b)!!;
		result.collect()!!;
		dev.sync();

		float[6] r;
		result.get(&r);

		// weight=[1,2], K=2, bias=0.5
		// Batch 0: x=[1,2,3]
		//   t=0: 1*0 + 2*1 + 0.5 = 2.5
		//   t=1: 1*1 + 2*2 + 0.5 = 5.5
		//   t=2: 1*2 + 2*3 + 0.5 = 8.5
		assert(math::abs(r[0] - 2.5f) < TOL, "fwd batch0 t0");
		assert(math::abs(r[1] - 5.5f) < TOL, "fwd batch0 t1");
		assert(math::abs(r[2] - 8.5f) < TOL, "fwd batch0 t2");

		// Batch 1: x=[4,5,6]
		//   t=0: 1*0 + 2*4 + 0.5 = 8.5
		//   t=1: 1*4 + 2*5 + 0.5 = 14.5
		//   t=2: 1*5 + 2*6 + 0.5 = 17.5
		assert(math::abs(r[3] - 8.5f) < TOL, "fwd batch1 t0");
		assert(math::abs(r[4] - 14.5f) < TOL, "fwd batch1 t1");
		assert(math::abs(r[5] - 17.5f) < TOL, "fwd batch1 t2");
	};
}

// ============================================================================
// Backward Tests
// ============================================================================

/// Test gradient flow through causal_conv1d.
/// loss = sum(causal_conv1d(x, w, b))
/// Verify grad_x, grad_w, grad_b against hand-computed values.
fn void test_backward_grad_x(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [1, 1, 3] = [1, 2, 3]
		float[3] x_data = { 1, 2, 3 };
		ulong[3] x_dims = { 1, 1, 3 };

		// weight: [1, 1, 2] = [0.5, 1.0]
		float[2] w_data = { 0.5, 1.0 };
		ulong[3] w_dims = { 1, 1, 2 };

		// bias: [1] = [0]
		float[1] b_data = { 0 };
		ulong[1] b_dims = { 1 };

		Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!.enable_grad();
		Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!;
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor out = functional::causal_conv1d(x, w, b)!!;
		Tensor loss = functional::sum_all(out)!!;
		loss.collect()!!;
		loss.backward();

		// Verify forward first
		Tensor out_val = out;
		out_val.collect()!!;
		dev.sync();
		float[3] fwd;
		out_val.get(&fwd);
		// w=[0.5, 1.0], K=2
		// t=0: 0.5*0 + 1.0*1 = 1.0
		// t=1: 0.5*1 + 1.0*2 = 2.5
		// t=2: 0.5*2 + 1.0*3 = 4.0
		assert(math::abs(fwd[0] - 1.0f) < TOL, "bwd fwd t0");
		assert(math::abs(fwd[1] - 2.5f) < TOL, "bwd fwd t1");
		assert(math::abs(fwd[2] - 4.0f) < TOL, "bwd fwd t2");

		// grad_x: dloss/dx[t] where loss = sum(out)
		// grad_out = [1, 1, 1]
		// x[0] contributes to: out[0] with w[1]=1.0, out[1] with w[0]=0.5
		// x[1] contributes to: out[1] with w[1]=1.0, out[2] with w[0]=0.5
		// x[2] contributes to: out[2] with w[1]=1.0
		// grad_x[0] = 1*w[1] + 1*w[0] = 1.0 + 0.5 = 1.5
		// grad_x[1] = 1*w[1] + 1*w[0] = 1.0 + 0.5 = 1.5
		// grad_x[2] = 1*w[1]           = 1.0
		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[3] gx;
		grad_x.get(&gx);
		assert(math::abs(gx[0] - 1.5f) < TOL, "grad_x[0]=1.5");
		assert(math::abs(gx[1] - 1.5f) < TOL, "grad_x[1]=1.5");
		assert(math::abs(gx[2] - 1.0f) < TOL, "grad_x[2]=1.0");
	};
}

/// Test gradient w.r.t. weight.
fn void test_backward_grad_weight(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [1, 1, 3] = [1, 2, 3]
		float[3] x_data = { 1, 2, 3 };
		ulong[3] x_dims = { 1, 1, 3 };

		// weight: [1, 1, 2] = [0.5, 1.0]
		float[2] w_data = { 0.5, 1.0 };
		ulong[3] w_dims = { 1, 1, 2 };

		// bias: [1] = [0]
		float[1] b_data = { 0 };
		ulong[1] b_dims = { 1 };

		Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!;
		Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!.enable_grad();
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor out = functional::causal_conv1d(x, w, b)!!;
		Tensor loss = functional::sum_all(out)!!;
		loss.collect()!!;
		loss.backward();

		// grad_weight[k] = sum_{t} grad_out[t] * x[t-(K-1)+k]
		// K=2, grad_out=[1,1,1]
		// k=0: x[t-1] for t=0,1,2 -> x[-1]=0, x[0]=1, x[1]=2 -> 0+1+2 = 3
		// k=1: x[t]   for t=0,1,2 -> x[0]=1, x[1]=2, x[2]=3 -> 1+2+3 = 6
		Tensor grad_w = w.grad();
		grad_w.collect()!!;
		dev.sync();

		float[2] gw;
		grad_w.get(&gw);
		assert(math::abs(gw[0] - 3.0f) < TOL, "grad_w[0]=3");
		assert(math::abs(gw[1] - 6.0f) < TOL, "grad_w[1]=6");
	};
}

/// Test gradient w.r.t. bias.
fn void test_backward_grad_bias(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [2, 1, 3] (batch=2)
		float[6] x_data = { 1, 2, 3, 4, 5, 6 };
		ulong[3] x_dims = { 2, 1, 3 };

		// weight: [1, 1, 2] = [1, 1]
		float[2] w_data = { 1, 1 };
		ulong[3] w_dims = { 1, 1, 2 };

		// bias: [1] = [0.5]
		float[1] b_data = { 0.5 };
		ulong[1] b_dims = { 1 };

		Tensor x = tensor::@from_data(dev, &x_dims, &x_data)!!;
		Tensor w = tensor::@from_data(dev, &w_dims, &w_data)!!;
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!.enable_grad();

		Tensor out = functional::causal_conv1d(x, w, b)!!;
		Tensor loss = functional::sum_all(out)!!;
		loss.collect()!!;
		loss.backward();

		// grad_bias[e] = sum_{b,t} grad_out[b,e,t]
		// grad_out = all ones, batch=2, L=3
		// grad_bias[0] = 2 * 3 = 6
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[1] gb;
		grad_b.get(&gb);
		assert(math::abs(gb[0] - 6.0f) < TOL, "grad_bias=6");
	};
}

/// Test all three gradients simultaneously (x, weight, bias all require_grad).
fn void test_backward_all_grads(DeviceReference dev)
{
	graph::@subgraph()
	{
		// x: [1, 2, 3]
		// ch0: [1, 0, -1], ch1: [2, 3, 4]
		float[6] x_data = { 1, 0, -1, 2, 3, 4 };

		// weight: [2, 1, 2]
		// ch0: [1, 2], ch1: [3, 1]
		float[4] w_data = { 1, 2, 3, 1 };

		// bias: [2] = [0.5, -0.5]
		float[2] b_data = { 0.5, -0.5 };

		Tensor x = tensor::@from_data(dev, { 1, 2, 3 }, &x_data)!!.enable_grad();
		Tensor w = tensor::@from_data(dev, { 2, 1, 2 }, &w_data)!!.enable_grad();
		Tensor b = tensor::@from_data(dev, { 2 }, &b_data)!!.enable_grad();

		Tensor out = functional::causal_conv1d(x, w, b)!!;
		Tensor loss = functional::sum_all(out)!!;
		loss.collect()!!;
		loss.backward();

		// --- Forward verification ---
		// Ch0 w=[1,2] bias=0.5, x=[1,0,-1]
		//   t=0: 1*0 + 2*1 + 0.5 = 2.5
		//   t=1: 1*1 + 2*0 + 0.5 = 1.5
		//   t=2: 1*0 + 2*(-1) + 0.5 = -1.5
		// Ch1 w=[3,1] bias=-0.5, x=[2,3,4]
		//   t=0: 3*0 + 1*2 - 0.5 = 1.5
		//   t=1: 3*2 + 1*3 - 0.5 = 8.5
		//   t=2: 3*3 + 1*4 - 0.5 = 12.5
		// loss = 2.5 + 1.5 + (-1.5) + 1.5 + 8.5 + 12.5 = 25.0

		// --- grad_x ---
		// Ch0: grad_out=[1,1,1], w=[1,2]
		//   x[0] -> out[0]*w[1]=2, out[1]*w[0]=1 -> 3
		//   x[1] -> out[1]*w[1]=2, out[2]*w[0]=1 -> 3
		//   x[2] -> out[2]*w[1]=2                 -> 2
		// Ch1: grad_out=[1,1,1], w=[3,1]
		//   x[0] -> out[0]*w[1]=1, out[1]*w[0]=3 -> 4
		//   x[1] -> out[1]*w[1]=1, out[2]*w[0]=3 -> 4
		//   x[2] -> out[2]*w[1]=1                 -> 1
		Tensor grad_x = x.grad();
		grad_x.collect()!!;
		dev.sync();

		float[6] gx;
		grad_x.get(&gx);
		assert(math::abs(gx[0] - 3.0f) < TOL, "all grad_x ch0[0]=3");
		assert(math::abs(gx[1] - 3.0f) < TOL, "all grad_x ch0[1]=3");
		assert(math::abs(gx[2] - 2.0f) < TOL, "all grad_x ch0[2]=2");
		assert(math::abs(gx[3] - 4.0f) < TOL, "all grad_x ch1[0]=4");
		assert(math::abs(gx[4] - 4.0f) < TOL, "all grad_x ch1[1]=4");
		assert(math::abs(gx[5] - 1.0f) < TOL, "all grad_x ch1[2]=1");

		// --- grad_weight ---
		// Ch0 w=[1,2], x=[1,0,-1], grad_out=[1,1,1]
		//   k=0: sum_t grad[t]*x[t-1] = 1*0 + 1*1 + 1*0 = 1
		//   k=1: sum_t grad[t]*x[t]   = 1*1 + 1*0 + 1*(-1) = 0
		// Ch1 w=[3,1], x=[2,3,4], grad_out=[1,1,1]
		//   k=0: sum_t grad[t]*x[t-1] = 1*0 + 1*2 + 1*3 = 5
		//   k=1: sum_t grad[t]*x[t]   = 1*2 + 1*3 + 1*4 = 9
		Tensor grad_w = w.grad();
		grad_w.collect()!!;
		dev.sync();

		float[4] gw;
		grad_w.get(&gw);
		assert(math::abs(gw[0] - 1.0f) < TOL, "all grad_w ch0[0]=1");
		assert(math::abs(gw[1] - 0.0f) < TOL, "all grad_w ch0[1]=0");
		assert(math::abs(gw[2] - 5.0f) < TOL, "all grad_w ch1[0]=5");
		assert(math::abs(gw[3] - 9.0f) < TOL, "all grad_w ch1[1]=9");

		// --- grad_bias ---
		// grad_bias[e] = sum_{b,t} 1 = 3 for each channel (B=1, L=3)
		Tensor grad_b = b.grad();
		grad_b.collect()!!;
		dev.sync();

		float[2] gb;
		grad_b.get(&gb);
		assert(math::abs(gb[0] - 3.0f) < TOL, "all grad_bias[0]=3");
		assert(math::abs(gb[1] - 3.0f) < TOL, "all grad_bias[1]=3");
	};
}

// ============================================================================
// Entry point
// ============================================================================

fn void run_all(DeviceReference dev)
{
	test_forward_basic(dev);
	test_forward_k4(dev);
	test_forward_batched(dev);
	test_backward_grad_x(dev);
	test_backward_grad_weight(dev);
	test_backward_grad_bias(dev);
	test_backward_all_grads(dev);
}
