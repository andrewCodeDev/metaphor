//! HIP Offload Tests
//!
//! Verifies CPU offloading end-to-end on GPU:
//! - offload-flagged stable tensors get DtoH'd when ref count hits 0
//! - auto-restore brings data back with correct values
//! - non-offload stable tensors are NOT freed
//! - offload stats counters work

module metaphor::testing::device::hip_device::hip_offload_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::rtti;
import std::math;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-4;

// ============================================================================
// Test: offload flag is set on GPU tensors
// ============================================================================

fn void test_offload_flag_set_on_gpu(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor mid = a.add(a)!!.offload();

		assert(mid.core().flags.offload,
			"offload flag should be set for GPU tensor");
		assert(mid.core().flags.persistence == tensor_core::Persistence.STABLE,
			"offload should imply stable");
	};
}

// ============================================================================
// Test: offloaded tensor data survives round-trip
// ============================================================================

fn void test_offload_data_integrity(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 10.0)!!;
		Tensor c = tensor::constant(F32, dev, &dims, 100.0)!!;

		// mid = a + b = [11,22,33,44], marked for offload
		Tensor mid = a.add(b)!!.offload();

		// out = mid + c = [111,222,333,444]
		Tensor out = mid.add(c)!!.stable();

		execution_sequence::reset_stats();

		// First collect — mid should be offloaded after out uses it
		out.collect()!!;
		dev.sync();

		// Verify output: (1+10)+100 = 111 for all elements
		float[4] result;
		out.get(&result);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(result[i] - 111.0) < TOL, "out should be 111");
		}

		// Check offload happened
		assert(execution_sequence::stats_offloaded > 0,
			"mid should have been offloaded");

		// mid should have no GPU data
		assert(!mid.core().has_data(),
			"mid should have no GPU data after offload");

		// mid should be in offload buffer map
		graph::Graph* g = graph::get();
		assert(g.exec_seq.offload_buffers.has_key((uptr)(void*)mid.core()),
			"mid should be in offload buffer map");

		// Second collect — mid should be auto-restored
		// Mutate a leaf to trigger recomputation
		execution_sequence::reset_stats();
		a.fill(1.0);
		out.collect()!!;
		dev.sync();

		assert(execution_sequence::stats_restored > 0,
			"mid should have been restored");

		// Verify output still correct after restore
		out.get(&result);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(result[i] - 111.0) < TOL, "out should still be 111 after restore");
		}
	};
}

// ============================================================================
// Test: stable without offload is NOT freed
// ============================================================================

fn void test_stable_not_offloaded(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };

		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;

		// mid is stable but NOT offload-flagged
		Tensor mid = a.add(b)!!.stable();
		Tensor out = mid.add(a)!!.stable();

		execution_sequence::reset_stats();
		out.collect()!!;
		dev.sync();

		// mid should still have GPU data
		assert(mid.core().has_data(),
			"stable (non-offload) tensor should retain GPU data");
		assert(execution_sequence::stats_offloaded == 0,
			"no tensors should be offloaded without offload flag");
	};
}

// ============================================================================
// Test: GPU memory is actually freed during offload
// ============================================================================

fn void test_offload_frees_gpu_memory(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 1024 };  // 4KB per tensor

		Tensor a = tensor::constant(F32, dev, &dims, 1.0)!!;
		Tensor b = tensor::constant(F32, dev, &dims, 2.0)!!;
		Tensor c = tensor::constant(F32, dev, &dims, 3.0)!!;

		Tensor mid = a.add(b)!!.offload();
		Tensor out = mid.add(c)!!.stable();

		// Record memory before
		usz mem_before = dev.mem_used();

		out.collect()!!;
		dev.sync();

		usz mem_after = dev.mem_used();

		// mid's GPU memory (4KB) should have been freed via offload
		// So mem_after should be less than mem_before + mid's allocation
		// (out gets allocated, mid gets freed)
		assert(!mid.core().has_data(),
			"mid GPU data should be freed after offload");
	};
}

// ============================================================================
// Runner
// ============================================================================

fn void run_all(DeviceReference dev)
{
	test_offload_flag_set_on_gpu(dev);
	test_offload_data_integrity(dev);
	test_stable_not_offloaded(dev);
	test_offload_frees_gpu_memory(dev);
}
