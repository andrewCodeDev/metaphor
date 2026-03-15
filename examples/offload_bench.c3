//! Offload benchmark — simple chain to verify GPU memory savings
//!
//! Creates N large tensors in a chain: x0 → x1 → x2 → ... → xN
//! Each intermediate is marked .offload().
//! Without offload: all intermediates live on GPU = N * tensor_size
//! With offload: only 1-2 intermediates on GPU at a time

module metaphor::examples::offload_bench;
import metaphor::device::cuda_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor;
import metaphor::rtti;
import std::io;

fn void main(String[] args)
{
	int n_layers = 10;
	if (args.len >= 2)
	{
		n_layers = args[1].to_int()!!;
	}

	cuda_device::CudaDevice gpu = cuda_device::cuda_device_create({ .device_id = 0 });
	defer gpu.deinit();
	device_reference::DeviceReference dev = gpu.reference();

	graph::init();
	defer graph::deinit();

	// Each tensor is 16MB (4M floats)
	ulong[1] dims = { 4 * 1024 * 1024 };

	tensor::Tensor x = tensor::constant(F32, dev, &dims, 1.0)!!;
	tensor::Tensor one = tensor::constant(F32, dev, &dims, 1.0)!!;

	for (int i = 0; i < n_layers; i++)
	{
		x = x.add(one)!!.offload();
	}
	x = x.stable();  // final output stays on GPU

	graph::Graph* g = graph::get();

	// Step 1: cold run
	execution_sequence::reset_stats();
	g.advance_generation();
	x.collect()!!;
	gpu.sync();

	usz peak1 = dev.mem_peak();
	io::printfn("step1: peak=%7.1fMB  offloaded=%d restored=%d",
		(double)peak1 / (1024.0 * 1024.0),
		execution_sequence::stats_offloaded,
		execution_sequence::stats_restored);

	// Step 2: warm run
	execution_sequence::reset_stats();
	g.advance_generation();
	x.collect()!!;
	gpu.sync();

	usz peak2 = dev.mem_peak();
	io::printfn("step2: peak=%7.1fMB  offloaded=%d restored=%d",
		(double)peak2 / (1024.0 * 1024.0),
		execution_sequence::stats_offloaded,
		execution_sequence::stats_restored);

	io::printfn("expected without offload: ~%7.1fMB",
		(double)(n_layers + 2) * 16.0);
}
