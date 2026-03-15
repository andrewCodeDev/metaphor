//! CUDA BF16 Tests
//!
//! Tests BF16 (bfloat16) operations on the CUDA device:
//! constant creation, elementwise ops, reduce, cast.

module metaphor::testing::device::cuda_device::cuda_bf16_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;
import std::io;

alias DeviceReference = device_reference::DeviceReference;
alias Tensor = tensor::Tensor;

const float TOL = 1e-2;  // BF16 has ~2 decimal digits of precision

// ============================================================================
// BF16 Constant Creation
// ============================================================================

fn void test_bf16_constant(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims = { 2, 3 };
		Tensor t = tensor::constant(BF16, dev, &dims, 2.5f)!!;
		Tensor f32 = t.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[6] r;
		f32.get(&r);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(r[i] - 2.5f) < TOL, "bf16 constant: expected 2.5");
		}
	};
}

// ============================================================================
// BF16 from_data (native bfloat type)
// ============================================================================

fn void test_bf16_from_data(DeviceReference dev)
{
	graph::@subgraph()
	{
		bfloat[4] data = { (bfloat)1.0, (bfloat)2.0, (bfloat)3.0, (bfloat)4.0 };
		ulong[1] dims = { 4 };

		Tensor t = tensor::@from_data(dev, &dims, &data)!!;
		Tensor f32 = t.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[4] r;
		f32.get(&r);
		assert(math::abs(r[0] - 1.0f) < TOL, "bf16 from_data [0]=1.0");
		assert(math::abs(r[1] - 2.0f) < TOL, "bf16 from_data [1]=2.0");
		assert(math::abs(r[2] - 3.0f) < TOL, "bf16 from_data [2]=3.0");
		assert(math::abs(r[3] - 4.0f) < TOL, "bf16 from_data [3]=4.0");
	};
}

// ============================================================================
// BF16 Elementwise (add, mul)
// ============================================================================

fn void test_bf16_elementwise(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(BF16, dev, &dims, 3.0f)!!;
		Tensor b = tensor::constant(BF16, dev, &dims, 2.0f)!!;

		Tensor sum = a.add(b)!!;
		sum.collect()!!;
		dev.sync();

		bfloat[4] raw;
		char[] sum_bytes = sum.value_bytes();
		io::printfn("  bf16 add: sum bytes len=%d", sum_bytes.len);
		sum.get(&raw);
		io::printfn("  bf16 add (raw): %f %f %f %f", (float)raw[0], (float)raw[1], (float)raw[2], (float)raw[3]);

		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs((float)raw[i] - 5.0f) < TOL, "bf16 add: expected 5.0");
		}
	};

	graph::@subgraph()
	{
		ulong[1] dims = { 4 };
		Tensor a = tensor::constant(BF16, dev, &dims, 3.0f)!!;
		Tensor b = tensor::constant(BF16, dev, &dims, 2.0f)!!;

		Tensor prod = a.mul(b)!!;
		Tensor f32 = prod.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[4] r;
		f32.get(&r);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - 6.0f) < TOL, "bf16 mul: expected 6.0");
		}
	};
}

// ============================================================================
// BF16 Reduce (sum over last dimension) — the main test
// ============================================================================

fn void test_bf16_reduce_2d(DeviceReference dev)
{
	// Simple 2x3 reduce "ij->i" (sum columns)
	graph::@subgraph()
	{
		bfloat[6] data = {
			(bfloat)1.0, (bfloat)2.0, (bfloat)3.0,
			(bfloat)4.0, (bfloat)5.0, (bfloat)6.0
		};
		ulong[2] dims = { 2, 3 };

		Tensor t = tensor::@from_data(dev, &dims, &data)!!;
		Tensor reduced = t.reduce("ij->i")!!;
		Tensor f32 = reduced.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[2] r;
		f32.get(&r);
		// row 0: 1+2+3 = 6, row 1: 4+5+6 = 15
		assert(math::abs(r[0] - 6.0f) < TOL, "bf16 reduce ij->i [0]=6.0");
		assert(math::abs(r[1] - 15.0f) < TOL, "bf16 reduce ij->i [1]=15.0");
	};
}

fn void test_bf16_reduce_3d(DeviceReference dev)
{
	// 3D reduce "ijk->ij" — matches the pattern from mamba2 ("ben->be")
	graph::@subgraph()
	{
		ulong[3] dims = { 1, 4, 8 };
		Tensor t = tensor::constant(BF16, dev, &dims, 1.0f)!!;
		Tensor reduced = t.reduce("ijk->ij")!!;
		Tensor f32 = reduced.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[4] r;
		f32.get(&r);
		// Each row sums 8 ones = 8.0
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - 8.0f) < TOL, "bf16 reduce ijk->ij: expected 8.0");
		}
	};
}

fn void test_bf16_reduce_large(DeviceReference dev)
{
	// Larger reduction to stress-test: [1, 64, 128] → [1, 64]
	// This is closer to the actual mamba2 shapes (though smaller)
	graph::@subgraph()
	{
		ulong[3] dims = { 1, 64, 128 };
		Tensor t = tensor::constant(BF16, dev, &dims, 0.5f)!!;
		Tensor reduced = t.reduce("ijk->ij")!!;
		Tensor f32 = reduced.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[64] r;
		f32.get(&r);
		// Each row sums 128 * 0.5 = 64.0
		for (usz i = 0; i < 64; i++)
		{
			assert(math::abs(r[i] - 64.0f) < 1.0f, "bf16 reduce large: expected 64.0");
		}
	};
}

// ============================================================================
// BF16 Reduce — scalar (sum all)
// ============================================================================

fn void test_bf16_reduce_scalar(DeviceReference dev)
{
	graph::@subgraph()
	{
		bfloat[4] data = { (bfloat)1.0, (bfloat)2.0, (bfloat)3.0, (bfloat)4.0 };
		ulong[1] dims = { 4 };

		Tensor t = tensor::@from_data(dev, &dims, &data)!!;
		Tensor reduced = t.reduce("i->")!!;
		Tensor f32 = reduced.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[1] r;
		f32.get(&r);
		assert(math::abs(r[0] - 10.0f) < TOL, "bf16 reduce scalar: expected 10.0");
	};
}

// ============================================================================
// BF16 Cast roundtrip
// ============================================================================

fn void test_bf16_cast_roundtrip(DeviceReference dev)
{
	graph::@subgraph()
	{
		float[4] f32_data = { 1.5, 2.5, 3.5, 4.5 };

		// F32 → BF16 → F32
		Tensor f32 = tensor::@from_data(dev, { 4 }, &f32_data)!!;
		Tensor bf16 = f32.cast(BF16)!!;
		Tensor back = bf16.cast(F32)!!;
		back.collect()!!;
		dev.sync();

		float[4] r;
		back.get(&r);
		assert(math::abs(r[0] - 1.5f) < TOL, "bf16 roundtrip [0]=1.5");
		assert(math::abs(r[1] - 2.5f) < TOL, "bf16 roundtrip [1]=2.5");
		assert(math::abs(r[2] - 3.5f) < TOL, "bf16 roundtrip [2]=3.5");
		assert(math::abs(r[3] - 4.5f) < TOL, "bf16 roundtrip [3]=4.5");
	};
}

// ============================================================================
// Entry Point
// ============================================================================

fn void run_all(DeviceReference dev)
{
	test_bf16_constant(dev);
	test_bf16_from_data(dev);
	test_bf16_elementwise(dev);
	test_bf16_reduce_2d(dev);
	test_bf16_reduce_3d(dev);
	test_bf16_reduce_large(dev);
	test_bf16_reduce_scalar(dev);
	test_bf16_cast_roundtrip(dev);
}
