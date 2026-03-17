//! HIP Test Runner
//!
//! Single entry point for all HIP device tests. Creates one HipDevice,
//! then dispatches to each test module's run_all() with the device reference.
//! Each test wraps itself in graph::@subgraph() for isolation.

module metaphor::testing::device::hip_device::hip_test_runner;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::testing::device::hip_device::hip_basic_ops_test;
import metaphor::testing::device::hip_device::hip_backward_test;
import metaphor::testing::device::hip_device::hip_grad_test;
import metaphor::testing::device::hip_device::hip_fusion_test;
import metaphor::testing::device::hip_device::hip_fusion_depth_test;
import metaphor::testing::device::hip_device::hip_graph_test;
import metaphor::testing::device::hip_device::hip_nn_test;
import metaphor::testing::device::hip_device::hip_loss_test;
import metaphor::testing::device::hip_device::hip_offload_test;
import metaphor::testing::device::hip_device::hip_catstack_grad_test;
import metaphor::testing::device::hip_device::hip_causal_conv1d_test;
import metaphor::testing::device::hip_device::hip_bf16_test;
import metaphor::testing::device::hip_device::hip_transfer_test;
import std::io;

fn int main()
{
	hip_device::HipDevice gpu = hip_device::create();
	device_reference::DeviceReference dev = gpu.reference();

	io::printn("Running HIP basic ops tests...");
	hip_basic_ops_test::run_all(dev);
	io::printn("  basic_ops: OK");

	io::printn("Running HIP graph tests...");
	hip_graph_test::run_all(dev, &gpu);
	io::printn("  graph: OK");

	io::printn("Running HIP backward tests...");
	hip_backward_test::run_all(dev);
	io::printn("  backward: OK");

	io::printn("Running HIP grad tests...");
	hip_grad_test::run_all(dev);
	io::printn("  grad: OK");

	io::printn("Running HIP fusion tests...");
	hip_fusion_test::run_all(dev);
	io::printn("  fusion: OK");

	io::printn("Running HIP fusion depth tests...");
	hip_fusion_depth_test::run_all(dev);
	io::printn("  fusion_depth: OK");

	io::printn("Running HIP NN tests...");
	hip_nn_test::run_all(dev);
	io::printn("  nn: OK");

	io::printn("Running HIP loss tests...");
	hip_loss_test::run_all(dev);
	io::printn("  loss: OK");

	io::printn("Running HIP offload tests...");
	hip_offload_test::run_all(dev);
	io::printn("  offload: OK");

	io::printn("Running HIP catstack grad tests...");
	hip_catstack_grad_test::run_all(dev);
	io::printn("  catstack_grad: OK");

	io::printn("Running HIP causal conv1d tests...");
	hip_causal_conv1d_test::run_all(dev);
	io::printn("  causal_conv1d: OK");

	io::printn("Running HIP BF16 tests...");
	hip_bf16_test::run_all(dev);
	io::printn("  bf16: OK");

	io::printn("Running HIP transfer tests...");
	hip_transfer_test::run_all(dev);
	io::printn("  transfer: OK");

	return 0;
}
