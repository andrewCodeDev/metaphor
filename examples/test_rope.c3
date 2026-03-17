//! Minimal RoPE test — verifies apply_rope_batched against apply_rope
//! with small dimensions we can hand-check.
//!
//! Run: c3c build test_rope && ./build/test_rope

module metaphor::examples::test_rope;
import metaphor::device::hip_device;
import metaphor::device::device_reference;
import metaphor::graph;
import metaphor::tensor;
import metaphor::rtti;
import std::math;
import std::io;
import std::core::mem;

alias DeviceReference = device_reference::DeviceReference;
alias TransferDirection = device_reference::TransferDirection;
alias Tensor = tensor::Tensor;
alias Datatype = rtti::Datatype;

fn void read_bf16(DeviceReference dev, Tensor t, float* dst, usz count)
{
	char[] gpu = t.value_bytes();
	usz bytes = count * ushort.sizeof;
	ushort* hbf = (ushort*)mem::malloc(bytes);
	dev.mem_transfer(gpu[:bytes], ((char*)hbf)[:bytes], TransferDirection.DTOH);
	for (usz i = 0; i < count; i++)
	{
		uint bits = (uint)hbf[i] << 16;
		dst[i] = *(float*)&bits;
	}
	mem::free(hbf);
}

fn void write_bf16(DeviceReference dev, Tensor t, float* src, usz count)
{
	usz bytes = count * ushort.sizeof;
	ushort* hbf = (ushort*)mem::malloc(bytes);
	for (usz i = 0; i < count; i++)
	{
		uint bits = *(uint*)&src[i];
		hbf[i] = (ushort)(bits >> 16);
	}
	dev.mem_transfer(((char*)hbf)[:bytes], t.value_bytes()[:bytes], TransferDirection.HTOD);
	mem::free(hbf);
}

fn void dump(float* buf, usz count, char[] label)
{
	io::printf("%s: ", label);
	for (usz i = 0; i < count; i++) { io::printf("%.4f ", buf[i]); }
	io::printn("");
}

/// Decode RoPE: x [B, H, head_dim], cos/sin [1, 1, half]
fn Tensor? apply_rope(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(2) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 2)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 2)!;
	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	Tensor[2] parts = { rot1, rot2 };
	return tensor::cat(&parts, 2);
}

/// Batched RoPE: x [B, S, H, head_dim], cos/sin [1, S, 1, half]
fn Tensor? apply_rope_batched(Tensor x, Tensor cos_pos, Tensor sin_pos)
{
	ulong half = x.shape().get(3) / 2;
	Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 3)!;
	Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 3)!;
	Tensor rot1 = x1 * cos_pos - x2 * sin_pos;
	Tensor rot2 = x2 * cos_pos + x1 * sin_pos;
	Tensor[2] parts = { rot1, rot2 };
	return tensor::cat(&parts, 3);
}

fn int main()
{
	hip_device::HipDevice gpu = hip_device::create();
	DeviceReference dev = gpu.reference();
	Datatype dt = Datatype.BF16;

	// Small dims: B=1, S=2, H=2, head_dim=4, half=2
	usz head_dim = 4;
	usz half = 2;
	usz n_heads = 2;
	usz n_seq = 2;

	// Known input: x = [1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0] per position
	// Position 0: heads 0,1 = [1,2,3,4], [5,6,7,8]
	// Position 1: heads 0,1 = [9,10,11,12], [13,14,15,16]
	float[16] x_data = {
		1, 2, 3, 4,   5, 6, 7, 8,     // pos 0: head 0, head 1
		9, 10, 11, 12, 13, 14, 15, 16, // pos 1: head 0, head 1
	};

	// cos/sin for 2 positions: [1, 2, 1, 2]
	// pos 0: cos=[1.0, 1.0], sin=[0.0, 0.0]  (identity)
	// pos 1: cos=[0.5, 0.8], sin=[0.866, 0.6]
	float[4] cos_data = { 1.0, 1.0,  0.5, 0.8 };
	float[4] sin_data = { 0.0, 0.0,  0.866, 0.6 };

	io::printn("=== RoPE Test ===");
	io::printn("head_dim=4, half=2, n_heads=2, n_seq=2");
	io::printn("");

	// Expected for position 1, head 0: x1=[9,10], x2=[11,12]
	// rot1 = x1*cos - x2*sin = [9*0.5 - 11*0.866, 10*0.8 - 12*0.6] = [4.5-9.526, 8.0-7.2] = [-5.026, 0.8]
	// rot2 = x2*cos + x1*sin = [11*0.5 + 9*0.866, 12*0.8 + 10*0.6] = [5.5+7.794, 9.6+6.0] = [13.294, 15.6]
	// result = [rot1, rot2] = [-5.026, 0.8, 13.294, 15.6]
	io::printn("Expected pos1 head0: -5.026 0.800 13.294 15.600");
	io::printn("");

	graph::@subgraph()
	{
		// --- Batched path: [1, 2, 2, 4] ---
		Tensor x_4d = tensor::empty(dt, dev, { 1, (ulong)n_seq, (ulong)n_heads, (ulong)head_dim })!!;
		x_4d.core().alloc_data();
		write_bf16(dev, x_4d, &x_data[0], 16);

		ulong[4] cs_shape = { 1, (ulong)n_seq, 1, (ulong)half };
		Tensor cos_4d = tensor::empty(dt, dev, &cs_shape)!!;
		cos_4d.core().alloc_data();
		write_bf16(dev, cos_4d, &cos_data[0], 4);

		Tensor sin_4d = tensor::empty(dt, dev, &cs_shape)!!;
		sin_4d.core().alloc_data();
		write_bf16(dev, sin_4d, &sin_data[0], 4);

		// First test: just verify the slice
		ulong uhalf = (ulong)half;
		Tensor x1_4d = x_4d.slice({ { 0, uhalf, 1 } }, offset: 3)!!;
		Tensor x2_4d = x_4d.slice({ { uhalf, 0, 1 } }, offset: 3)!!;
		x1_4d.stable().collect()!!;
		x2_4d.stable().collect()!!;
		dev.sync();

		float[8] s1; // [1,2,2,2] = 8 elements
		float[8] s2;
		read_bf16(dev, x1_4d, &s1[0], 8);
		read_bf16(dev, x2_4d, &s2[0], 8);
		io::printn("x1 (first half of head_dim):");
		dump(&s1[0], 2, "  pos0 head0");
		dump(&s1[2], 2, "  pos0 head1");
		dump(&s1[4], 2, "  pos1 head0");
		dump(&s1[6], 2, "  pos1 head1");
		io::printn("x2 (second half of head_dim):");
		dump(&s2[0], 2, "  pos0 head0");
		dump(&s2[2], 2, "  pos0 head1");
		dump(&s2[4], 2, "  pos1 head0");
		dump(&s2[6], 2, "  pos1 head1");

		// Now full RoPE
		Tensor batched_out = apply_rope_batched(x_4d, cos_4d, sin_4d)!!;
		batched_out.stable().collect()!!;
		dev.sync();

		float[16] out_b;
		read_bf16(dev, batched_out, &out_b[0], 16);
		io::printn("\nBatched RoPE output:");
		dump(&out_b[0], 4, "  pos0 head0");
		dump(&out_b[4], 4, "  pos0 head1");
		dump(&out_b[8], 4, "  pos1 head0");
		dump(&out_b[12], 4, "  pos1 head1");

		// --- Decode path: process position 1 only ---
		// x: [1, 2, 4] — single position, 2 heads
		Tensor x_3d = tensor::empty(dt, dev, { 1, (ulong)n_heads, (ulong)head_dim })!!;
		x_3d.core().alloc_data();
		write_bf16(dev, x_3d, &x_data[8], 8); // pos 1 data: [9..16]

		// cos/sin: [1, 1, 2] — position 1's values
		ulong[3] cs3_shape = { 1, 1, (ulong)half };
		Tensor cos_3d = tensor::empty(dt, dev, &cs3_shape)!!;
		cos_3d.core().alloc_data();
		write_bf16(dev, cos_3d, &cos_data[2], 2); // pos 1 cos

		Tensor sin_3d = tensor::empty(dt, dev, &cs3_shape)!!;
		sin_3d.core().alloc_data();
		write_bf16(dev, sin_3d, &sin_data[2], 2); // pos 1 sin

		Tensor decode_out = apply_rope(x_3d, cos_3d, sin_3d)!!;
		decode_out.stable().collect()!!;
		dev.sync();

		float[8] out_d;
		read_bf16(dev, decode_out, &out_d[0], 8);
		io::printn("");
		dump(&out_d[0], 4, "decode  pos1 head0");
		dump(&out_d[4], 4, "decode  pos1 head1");
	};

	io::printn("\n=== Done ===");
	return 0;
}
