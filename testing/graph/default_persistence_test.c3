//! Tests for per-graph default persistence mode.
//!
//! T.1: Auto-promotion respects graph's default_persistence
//! T.2: User .unstable() override in stable mode
//! T.3: User .stable() override survives in unstable mode
//! T.4: .unstable() tensors in stable mode get freed after execution
//! T.6: Subgraph inherits default_persistence from parent

module metaphor::testing::graph::default_persistence_test;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_core;
import metaphor::device::host_device;
import metaphor::device::device_reference;
import metaphor::rtti;

alias Tensor = tensor::Tensor;
alias TensorCore = tensor_core::TensorCore;
alias Persistence = tensor_core::Persistence;
alias Graph = graph::Graph;

// ============================================================================
// T.1: Auto-promotion respects graph default
// ============================================================================

/// In unstable mode, intermediate tensors should be promoted to UNSTABLE.
/// Uses matmul to prevent fusion — each matmul gets its own execution unit.
fn void test_auto_promotion_unstable_mode() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		graph::set_default_persistence(UNSTABLE);

		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);

		Tensor mid = a.matmul(a)!!;
		Tensor out = mid.matmul(a)!!;
		TensorCore* mid_core = mid.core();

		out.collect()!!;

		assert(mid_core.flags.persistence == Persistence.UNSTABLE,
			"T.1: intermediate should be auto-promoted to UNSTABLE");
	};
}

/// In stable mode, intermediate tensors should be promoted to STABLE.
fn void test_auto_promotion_stable_mode() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		graph::set_default_persistence(STABLE);

		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);

		Tensor mid = a.matmul(a)!!;
		Tensor out = mid.matmul(a)!!;
		TensorCore* mid_core = mid.core();

		out.collect()!!;

		assert(mid_core.flags.persistence == Persistence.STABLE,
			"T.1: intermediate should be auto-promoted to STABLE");
	};
}

// ============================================================================
// T.2: User .unstable() override in stable mode
// ============================================================================

fn void test_user_unstable_in_stable_mode() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		graph::set_default_persistence(STABLE);

		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);

		Tensor mid = a.matmul(a)!!.unstable();
		Tensor out = mid.matmul(a)!!;
		TensorCore* mid_core = mid.core();

		assert(mid_core.flags.persistence == Persistence.UNSTABLE,
			"T.2: .unstable() should set UNSTABLE before execution");

		out.collect()!!;
	};
}

// ============================================================================
// T.3: User .stable() override survives in unstable mode
// ============================================================================

fn void test_user_stable_survives_unstable_mode() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		graph::set_default_persistence(UNSTABLE);

		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);

		Tensor mid = a.matmul(a)!!.stable();
		Tensor out = mid.matmul(a)!!;
		TensorCore* mid_core = mid.core();

		out.collect()!!;

		assert(mid_core.flags.persistence == Persistence.STABLE,
			"T.3: .stable() should survive auto-promotion in unstable mode");
	};
}

// ============================================================================
// T.4: Unstable intermediates freed in stable mode
// ============================================================================

fn void test_unstable_intermediate_freed_in_stable_mode() @test
{
	host_device::HostDevice cpu;
	cpu.init();
	defer cpu.deinit();
	device_reference::DeviceReference dev = cpu.reference();

	graph::@subgraph()
	{
		graph::set_default_persistence(STABLE);

		Tensor a = tensor::empty(F32, dev, { 4, 4 })!!;
		a.fill(1.0);

		Tensor mid = a.matmul(a)!!.unstable();
		Tensor out = mid.matmul(a)!!;
		out.collect()!!;

		assert(out.core().has_data(),
			"T.4: stable output should retain data");
	};
}

// ============================================================================
// T.6: Subgraph inherits default_persistence from parent
// ============================================================================

fn void test_subgraph_inherits_stable() @test
{
	graph::@subgraph()
	{
		graph::set_default_persistence(STABLE);

		graph::@subgraph()
		{
			Graph* child = graph::get();
			assert(child.flags.default_persistence == Persistence.STABLE,
				"T.6: subgraph should inherit STABLE from parent");
		};
	};
}

fn void test_subgraph_inherits_unstable() @test
{
	graph::@subgraph()
	{
		graph::set_default_persistence(UNSTABLE);

		graph::@subgraph()
		{
			Graph* child = graph::get();
			assert(child.flags.default_persistence == Persistence.UNSTABLE,
				"T.6: subgraph should inherit UNSTABLE from parent");
		};
	};
}

// ============================================================================
// Persistence.promote() and Persistence.assign() unit tests
// ============================================================================

fn void test_promote_monotonic() @test
{
	Persistence p = Persistence.NONE;

	p.promote(Persistence.UNSTABLE);
	assert(p == Persistence.UNSTABLE, "promote NONE→UNSTABLE");

	p.promote(Persistence.STABLE);
	assert(p == Persistence.STABLE, "promote UNSTABLE→STABLE");

	p.promote(Persistence.UNSTABLE);
	assert(p == Persistence.STABLE, "promote STABLE→UNSTABLE should be no-op");

	p.promote(Persistence.LEAF);
	assert(p == Persistence.LEAF, "promote STABLE→LEAF");
}

fn void test_assign_direct() @test
{
	Persistence p = Persistence.NONE;

	p.assign(Persistence.STABLE);
	assert(p == Persistence.STABLE, "assign STABLE");

	p.assign(Persistence.UNSTABLE);
	assert(p == Persistence.UNSTABLE, "assign UNSTABLE (demotion)");

	p.assign(Persistence.STABLE);
	assert(p == Persistence.STABLE, "assign STABLE again");
}

fn void test_set_default_persistence_setter() @test
{
	graph::@subgraph()
	{
		graph::set_default_persistence(UNSTABLE);
		Graph* g = graph::get();
		assert(g.flags.default_persistence == Persistence.UNSTABLE);

		graph::set_default_persistence(STABLE);
		assert(g.flags.default_persistence == Persistence.STABLE);

		graph::set_default_persistence(UNSTABLE);
		assert(g.flags.default_persistence == Persistence.UNSTABLE);
	};
}
