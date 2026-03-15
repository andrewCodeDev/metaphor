//! CUDA Test Runner
//!
//! Single entry point for all CUDA device tests. Creates one CudaDevice,
//! then dispatches to each test module's run_all() with the device reference.
//! Each test wraps itself in graph::@subgraph() for isolation.

module metaphor::testing::device::cuda_device::cuda_test_runner;
import metaphor::device::cuda_device;
import metaphor::device::device_reference;
import metaphor::testing::device::cuda_device::cuda_basic_ops_test;
import metaphor::testing::device::cuda_device::cuda_backward_test;
import metaphor::testing::device::cuda_device::cuda_grad_test;
import metaphor::testing::device::cuda_device::cuda_fusion_test;
import metaphor::testing::device::cuda_device::cuda_fusion_depth_test;
import metaphor::testing::device::cuda_device::cuda_graph_test;
import metaphor::testing::device::cuda_device::cuda_nn_test;
import metaphor::testing::device::cuda_device::cuda_loss_test;
import metaphor::testing::device::cuda_device::cuda_offload_test;
import metaphor::testing::device::cuda_device::cuda_catstack_grad_test;
import metaphor::testing::device::cuda_device::cuda_causal_conv1d_test;
import metaphor::testing::device::cuda_device::cuda_bf16_test;
import metaphor::testing::device::cuda_device::cuda_transfer_test;
import std::io;

fn int main()
{
	cuda_device::CudaDevice gpu = cuda_device::cuda_device_create({ .device_id = 0 });
	device_reference::DeviceReference dev = gpu.reference();

	io::printn("Running basic ops tests...");
	cuda_basic_ops_test::run_all(dev);
	io::printn("  basic ops: OK");

	io::printn("Running backward tests...");
	cuda_backward_test::run_all(dev);
	io::printn("  backward: OK");

	io::printn("Running grad tests...");
	cuda_grad_test::run_all(dev);
	io::printn("  grad: OK");

	io::printn("Running fusion tests...");
	cuda_fusion_test::run_all(dev);
	io::printn("  fusion: OK");

	io::printn("Running CUDA fusion depth limit tests...");
	cuda_fusion_depth_test::run_all(dev);
	io::printn("  fusion depth: OK");

	io::printn("Running CUDA graph capture tests...");
	cuda_graph_test::run_all(dev, &gpu);
	io::printn("  graph capture: OK");

	io::printn("Running CUDA NN tests...");
	cuda_nn_test::run_all(dev);
	io::printn("  nn: OK");

	io::printn("Running CUDA loss tests...");
	cuda_loss_test::run_all(dev);
	io::printn("  loss: OK");

	io::printn("Running CUDA offload tests...");
	cuda_offload_test::run_all(dev);
	io::printn("  offload: OK");

	io::printn("Running CUDA cat/stack grad tests...");
	cuda_catstack_grad_test::run_all(dev);
	io::printn("  cat/stack grad: OK");

	io::printn("Running CUDA causal conv1d tests...");
	cuda_causal_conv1d_test::run_all(dev);
	io::printn("  causal conv1d: OK");

	io::printn("Running CUDA BF16 tests...");
	cuda_bf16_test::run_all(dev);
	io::printn("  bf16: OK");

	io::printn("Running CUDA transfer tests...");
	cuda_transfer_test::run_all(dev);
	io::printn("  transfer: OK");

	io::printn("All CUDA tests passed.");
	return 0;
}
