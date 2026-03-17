module metaphor::testing::graph::test_fingerprint;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::device::host_device;
import metaphor::device::device_reference;

alias Tensor = tensor::Tensor;
alias DeviceReference = device_reference::DeviceReference;

// ============================================================================
// Test: second collect with no mutation skips all execution
// ============================================================================

fn void test_no_mutation_skips() @test
{
	host_device::HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		Tensor b = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);
		b.fill(2.0);

		Tensor out = a.matmul(b)!!;

		// First collect: must execute
		execution_sequence::reset_stats();
		out.collect()!!;
		assert(execution_sequence::stats_executed > 0, "first collect should execute");

		// Second collect: no mutation → fingerprints match → skip all
		execution_sequence::reset_stats();
		out.collect()!!;
		assert(execution_sequence::stats_executed == 0,
			"second collect with no mutation should skip all");
	};
}

// ============================================================================
// Test: mutating one leaf triggers recompute
// ============================================================================

fn void test_mutation_triggers_recompute() @test
{
	host_device::HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		Tensor b = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);
		b.fill(2.0);

		Tensor out = a.matmul(b)!!;

		out.collect()!!;

		float val1; out.get(&val1);
		// matmul(1*ones, 2*ones) for 4x4 = 1*2*4 = 8.0
		assert(val1 == 8.0, "first collect should produce 8.0");

		// Mutate leaf a → fingerprint changes → recompute
		a.fill(3.0);

		execution_sequence::reset_stats();
		out.collect()!!;
		assert(execution_sequence::stats_executed > 0,
			"mutation should trigger recompute");

		float val2; out.get(&val2);
		// matmul(3*ones, 2*ones) for 4x4 = 3*2*4 = 24.0
		assert(val2 == 24.0, "after mutation should produce 24.0");
	};
}

// ============================================================================
// Test: selective recompute — only affected paths re-execute
// ============================================================================
//
//   leaf_a ──→ mid_left ──┐
//                          ├──→ out
//   leaf_b ──→ mid_right ─┘
//
// Touch leaf_a → mid_left + out re-execute, mid_right skips.

fn void test_selective_recompute() @test
{
	host_device::HostDevice cpu = host_device::create();
	defer cpu.deinit();
	DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		Tensor leaf_a = tensor::empty(F32, dev, { 4, 4 })!!;
		Tensor leaf_b = tensor::empty(F32, dev, { 4, 4 })!!;
		leaf_a.fill(1.0);
		leaf_b.fill(2.0);

		Tensor mid_left  = leaf_a.matmul(leaf_a)!!.stable();
		Tensor mid_right = leaf_b.matmul(leaf_b)!!.stable();
		Tensor out = mid_left.matmul(mid_right)!!;

		// First collect: all 3 units execute
		execution_sequence::reset_stats();
		out.collect()!!;
		assert(execution_sequence::stats_executed == 3,
			"first collect: expected 3 executed");

		// Second collect (no changes): all skip
		execution_sequence::reset_stats();
		out.collect()!!;
		assert(execution_sequence::stats_executed == 0,
			"second collect: expected 0 executed (all clean)");

		// Touch leaf_a only — mid_left and out should re-execute, mid_right should skip
		leaf_a.fill(1.0);

		execution_sequence::reset_stats();
		out.collect()!!;
		assert(execution_sequence::stats_executed == 2,
			"third collect: expected 2 executed (mid_left + out)");

		// Values should still be correct
		// mid_left = matmul(ones, ones) = 4*ones
		// mid_right = matmul(twos, twos) = 16*ones
		// out = matmul(4*ones, 16*ones) = 256*ones
		float val; out.get(&val);
		assert(val == 256.0, "output value should be 256.0");
	};
}
