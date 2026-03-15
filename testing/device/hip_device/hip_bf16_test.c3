//! HIP BF16 Tests
//!
//! Tests BF16 (bfloat16) operations on the HIP device:
//! constant creation, elementwise ops, reduce, cast.

module metaphor::testing::device::hip_device::hip_bf16_test;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::tensor::tensor_ops;
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
		Tensor f32 = sum.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[4] r;
		f32.get(&r);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - 5.0f) < TOL, "bf16 add: expected 5.0");
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
		ulong[1] dims = { 4 };

		// F32 → BF16 → F32
		Tensor f32 = tensor::@from_data(dev, &dims, &f32_data)!!;
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
// BF16 Permutation (2D transpose via einsum identity)
// ============================================================================

fn void test_bf16_permute_2d(DeviceReference dev)
{
	graph::@subgraph()
	{
		// 2D transpose via einsum with identity: "ij->ji"
		bfloat[6] data = {
			(bfloat)1.0, (bfloat)2.0, (bfloat)3.0,
			(bfloat)4.0, (bfloat)5.0, (bfloat)6.0
		};
		ulong[2] dims = { 2, 3 };

		Tensor t = tensor::@from_data(dev, &dims, &data)!!;
		Tensor p = t.transpose(0, 1)!!;
		Tensor f32 = p.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[6] r;
		f32.get(&r);
		// Transposed: [1,4,2,5,3,6]
		assert(math::abs(r[0] - 1.0f) < TOL, "bf16 permute 2d [0]=1.0");
		assert(math::abs(r[1] - 4.0f) < TOL, "bf16 permute 2d [1]=4.0");
		assert(math::abs(r[2] - 2.0f) < TOL, "bf16 permute 2d [2]=2.0");
		assert(math::abs(r[3] - 5.0f) < TOL, "bf16 permute 2d [3]=5.0");
		assert(math::abs(r[4] - 3.0f) < TOL, "bf16 permute 2d [4]=3.0");
		assert(math::abs(r[5] - 6.0f) < TOL, "bf16 permute 2d [5]=6.0");
	};
}

// ============================================================================
// BF16 Matmul (contraction)
// ============================================================================

fn void test_bf16_matmul(DeviceReference dev)
{
	graph::@subgraph()
	{
		ulong[2] dims_a = { 16, 32 };
		ulong[2] dims_b = { 32, 16 };

		Tensor a = tensor::constant(BF16, dev, &dims_a, 1.0f)!!;
		Tensor b = tensor::constant(BF16, dev, &dims_b, 1.0f)!!;

		Tensor c = a.matmul(b)!!;
		Tensor f32 = c.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[256] r;
		f32.get(&r);
		// dot(ones_32, ones_32) = 32.0
		for (usz i = 0; i < 256; i++)
		{
			assert(math::abs(r[i] - 32.0f) < TOL, "bf16 matmul: expected 32.0");
		}
	};
}

// ============================================================================
// BF16 Cat (concatenation along existing dimension)
// ============================================================================

fn void test_bf16_cat_1d(DeviceReference dev)
{
	graph::@subgraph()
	{
		bfloat[3] a_data = { (bfloat)1.0, (bfloat)2.0, (bfloat)3.0 };
		bfloat[2] b_data = { (bfloat)4.0, (bfloat)5.0 };
		ulong[1] a_dims = { 3 };
		ulong[1] b_dims = { 2 };

		Tensor a = tensor::@from_data(dev, &a_dims, &a_data)!!;
		Tensor b = tensor::@from_data(dev, &b_dims, &b_data)!!;

		Tensor c = tensor_ops::cat({ a, b }, 0)!!;
		Tensor f32 = c.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[5] r;
		f32.get(&r);
		assert(math::abs(r[0] - 1.0f) < TOL, "bf16 cat 1d [0]=1.0");
		assert(math::abs(r[1] - 2.0f) < TOL, "bf16 cat 1d [1]=2.0");
		assert(math::abs(r[2] - 3.0f) < TOL, "bf16 cat 1d [2]=3.0");
		assert(math::abs(r[3] - 4.0f) < TOL, "bf16 cat 1d [3]=4.0");
		assert(math::abs(r[4] - 5.0f) < TOL, "bf16 cat 1d [4]=5.0");
	};
}

fn void test_bf16_cat_2d(DeviceReference dev)
{
	graph::@subgraph()
	{
		// a = [[1, 2], [3, 4]], b = [[5], [6]]
		// cat dim=1 -> [[1, 2, 5], [3, 4, 6]]
		bfloat[4] a_data = { (bfloat)1.0, (bfloat)2.0, (bfloat)3.0, (bfloat)4.0 };
		bfloat[2] b_data = { (bfloat)5.0, (bfloat)6.0 };

		Tensor a = tensor::@from_data(dev, { 2, 2 }, &a_data)!!;
		Tensor b = tensor::@from_data(dev, { 2, 1 }, &b_data)!!;

		Tensor c = tensor_ops::cat({ a, b }, 1)!!;
		Tensor f32 = c.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[6] r;
		f32.get(&r);
		// Row 0: [1, 2, 5], Row 1: [3, 4, 6]
		assert(math::abs(r[0] - 1.0f) < TOL, "bf16 cat 2d [0]=1.0");
		assert(math::abs(r[1] - 2.0f) < TOL, "bf16 cat 2d [1]=2.0");
		assert(math::abs(r[2] - 5.0f) < TOL, "bf16 cat 2d [2]=5.0");
		assert(math::abs(r[3] - 3.0f) < TOL, "bf16 cat 2d [3]=3.0");
		assert(math::abs(r[4] - 4.0f) < TOL, "bf16 cat 2d [4]=4.0");
		assert(math::abs(r[5] - 6.0f) < TOL, "bf16 cat 2d [5]=6.0");
	};
}

// ============================================================================
// BF16 Stack (join along new dimension)
// ============================================================================

fn void test_bf16_stack(DeviceReference dev)
{
	graph::@subgraph()
	{
		// a = [1, 2], b = [3, 4]
		// stack dim=0 -> [[1, 2], [3, 4]]
		bfloat[2] a_data = { (bfloat)1.0, (bfloat)2.0 };
		bfloat[2] b_data = { (bfloat)3.0, (bfloat)4.0 };
		ulong[1] dims = { 2 };

		Tensor a = tensor::@from_data(dev, &dims, &a_data)!!;
		Tensor b = tensor::@from_data(dev, &dims, &b_data)!!;

		Tensor c = tensor_ops::stack({ a, b }, 0)!!;
		Tensor f32 = c.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[4] r;
		f32.get(&r);
		// [[1, 2], [3, 4]]
		assert(math::abs(r[0] - 1.0f) < TOL, "bf16 stack [0]=1.0");
		assert(math::abs(r[1] - 2.0f) < TOL, "bf16 stack [1]=2.0");
		assert(math::abs(r[2] - 3.0f) < TOL, "bf16 stack [2]=3.0");
		assert(math::abs(r[3] - 4.0f) < TOL, "bf16 stack [3]=4.0");
	};
}

// ============================================================================
// BF16 Batched Matmul with unit dimensions (LLaMA attention pattern)
// ============================================================================

fn void test_bf16_batched_matmul_unit_dim(DeviceReference dev)
{
	// This is the exact contraction shape from LLaMA attention:
	// Q @ K^T: [1,32,1,128] @ [1,32,128,2048] → [1,32,1,2048]
	// Then scores @ V: [1,32,1,2048] @ [1,32,2048,128] → [1,32,1,128]
	// The unit dimension (1) at position 2 triggers hipTensor issues on RDNA3.

	// Same contraction WITHOUT unit dimensions: [2,4] @ [2,4,3] → [2,3]
	// Pattern: "be,bed->bd" (b=2 batch, e=4 contracted, d=3 free)
	graph::@subgraph()
	{
		ulong[2] dims_a = { 2, 4 };
		ulong[3] dims_b = { 2, 4, 3 };

		Tensor a = tensor::constant(BF16, dev, &dims_a, 1.0f)!!;
		Tensor b = tensor::constant(BF16, dev, &dims_b, 1.0f)!!;

		Tensor c = a.einsum(b, "be,bed->bd")!!;
		Tensor f32 = c.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[6] r;
		f32.get(&r);
		io::printfn("    bf16 no-unit-dim values: [%f, %f, %f, %f, %f, %f]",
			r[0], r[1], r[2], r[3], r[4], r[5]);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(r[i] - 4.0f) < TOL, "bf16 batched matmul no unit dim: expected 4.0");
		}
	};

	// Same contraction WITH unit dimensions BF16: [1,2,1,4] @ [1,2,4,3] → [1,2,1,3]
	graph::@subgraph()
	{

		Tensor a = tensor::constant(BF16, dev, { 1, 2, 1, 4 }, 1.0f)!!;
		Tensor b = tensor::constant(BF16, dev, { 1, 2, 4, 3 }, 1.0f)!!;

		Tensor c = a.matmul(b)!!;
		Tensor f32 = c.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		float[6] r;
		f32.get(&r);
		io::printfn("    bf16 with-unit-dim values: [%f, %f, %f, %f, %f, %f]",
			r[0], r[1], r[2], r[3], r[4], r[5]);
		for (usz i = 0; i < 6; i++)
		{
			assert(math::abs(r[i] - 4.0f) < TOL, "bf16 batched matmul unit dim: expected 4.0");
		}
	};
}

// ============================================================================
// BF16 LLaMA attention contraction (scores @ V — the shape that crashes hipTensor)
// ============================================================================

fn void test_bf16_attention_contract(DeviceReference dev)
{
	// Q @ K^T: [1,32,1,128] @ [1,32,128,2048] → [1,32,1,2048]  (succeeds)
	// scores @ V: [1,32,1,2048] @ [1,32,2048,128] → [1,32,1,128] (crashes hipTensor on RDNA3)
	// Reproduce with smaller sizes to test the fix without OOM.

	// Q @ K^T direction (succeeds): [1,4,1,8] @ [1,4,8,16] → [1,4,1,16]
	graph::@subgraph()
	{

		Tensor q = tensor::constant(BF16, dev, { 1, 4, 1, 8 }, 1.0f)!!;
		Tensor kt = tensor::constant(BF16, dev, { 1, 4, 8, 16 }, 1.0f)!!;

		Tensor scores = q.matmul(kt)!!;
		Tensor f32 = scores.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		// Each element = dot(ones_8, ones_8) = 8.0
		float[64] r;
		f32.get(&r);
		for (usz i = 0; i < 64; i++)
		{
			assert(math::abs(r[i] - 8.0f) < TOL, "bf16 Q@K^T: expected 8.0");
		}
	};

	// scores @ V direction (crashes at LLaMA scale): [1,32,1,2048] @ [1,32,2048,128] → [1,32,1,128]
	// Use actual LLaMA attention dimensions to reproduce the hipTensor RDNA3 crash.
	graph::@subgraph()
	{

		Tensor s = tensor::constant(BF16, dev, { 1, 32, 1, 2048 }, 0.01f)!!;
		Tensor v = tensor::constant(BF16, dev, { 1, 32, 2048, 128 }, 0.01f)!!;

		Tensor out = s.matmul(v)!!;
		Tensor f32 = out.cast(F32)!!;
		f32.collect()!!;
		dev.sync();

		// Each element = 2048 * 0.01 * 0.01 = 0.2048 (approx due to bf16 precision)
		float[4] r;
		f32.get(&r);
		for (usz i = 0; i < 4; i++)
		{
			assert(math::abs(r[i] - 0.2048f) < 0.05f, "bf16 scores@V: expected ~0.2048");
		}
	};
}

// ============================================================================
// Entry Point
// ============================================================================

fn void run_all(DeviceReference dev)
{
	io::printn("  [1/14] bf16 constant...");
	test_bf16_constant(dev);
	io::printn("  [2/14] bf16 from_data...");
	test_bf16_from_data(dev);
	io::printn("  [3/14] bf16 elementwise...");
	test_bf16_elementwise(dev);
	io::printn("  [4/14] bf16 permute 2d...");
	test_bf16_permute_2d(dev);
	io::printn("  [5/14] bf16 matmul...");
	test_bf16_matmul(dev);
	io::printn("  [6/14] bf16 batched matmul unit dim...");
	test_bf16_batched_matmul_unit_dim(dev);
	io::printn("  [7/14] bf16 reduce 2d...");
	test_bf16_reduce_2d(dev);
	io::printn("  [8/14] bf16 reduce 3d...");
	test_bf16_reduce_3d(dev);
	io::printn("  [9/14] bf16 reduce large...");
	test_bf16_reduce_large(dev);
	io::printn("  [10/14] bf16 reduce scalar...");
	test_bf16_reduce_scalar(dev);
	io::printn("  [11/14] bf16 cast roundtrip...");
	test_bf16_cast_roundtrip(dev);
	io::printn("  [12/14] bf16 cat 1d...");
	test_bf16_cat_1d(dev);
	io::printn("  [13/14] bf16 cat 2d...");
	test_bf16_cat_2d(dev);
	io::printn("  [14/15] bf16 stack...");
	test_bf16_stack(dev);
	io::printn("  [15/15] bf16 attention contract...");
	test_bf16_attention_contract(dev);
}
