module metaphor::testing::layout_test;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::tensor::dense_core;
import metaphor::tensor::sparse_core;
import metaphor::device::device_reference;
import metaphor::device::sync;
import metaphor::rtti;
import std::io;

fn usz offset_of(void* base, void* field) {
	return (usz)field - (usz)base;
}

fn void tensor_core_layout() @test {
	TensorCore tc;
	void* base = &tc;

	io::printfn("=== TensorCore Layout ===");
	io::printfn("sizeof(TensorCore)       = %d", TensorCore.sizeof);
	io::printfn("sizeof(DeviceReference)  = %d", DeviceReference.sizeof);
	io::printfn("sizeof(Shape)            = %d", Shape.sizeof);
	io::printfn("sizeof(Datatype)         = %d", rtti::Datatype.sizeof);
	io::printfn("sizeof(TensorCoreFlags)  = %d", TensorCoreFlags.sizeof);
	io::printfn("sizeof(SyncPrimitive)    = %d", sync::SyncPrimitive.sizeof);
	io::printfn("sizeof(CoreState)        = %d", CoreState.sizeof);
	io::printfn("sizeof(DenseCore)        = %d", DenseCore.sizeof);
	io::printfn("sizeof(SparseCore)       = %d", SparseCore.sizeof);
	io::printfn("sizeof(device_reference::Device) = %d", device_reference::Device.sizeof);

	io::printfn("");
	io::printfn("--- TensorCore field offsets ---");
	io::printfn("graph              @ %d", offset_of(base, &tc.graph));
	io::printfn("device             @ %d", offset_of(base, &tc.device));
	io::printfn("flags              @ %d", offset_of(base, &tc.flags));
	io::printfn("label              @ %d", offset_of(base, &tc.label));
	io::printfn("sync               @ %d", offset_of(base, &tc.sync));
	io::printfn("consumers          @ %d", offset_of(base, &tc.consumers));
	io::printfn("version            @ %d", offset_of(base, &tc.version));
	io::printfn("creation_context   @ %d", offset_of(base, &tc.creation_context));
	io::printfn("state              @ %d", offset_of(base, &tc.state));
	io::printfn("state.class_       @ %d", offset_of(base, &tc.state.class_));
	io::printfn("state.dense        @ %d", offset_of(base, &tc.state.dense));

	io::printfn("");
	io::printfn("--- DenseCore sub-offsets (from state.dense) ---");
	void* dense_base = &tc.state.dense;
	io::printfn("data               @ %d", offset_of(dense_base, &tc.state.dense.data));
	io::printfn("byte_size          @ %d", offset_of(dense_base, &tc.state.dense.byte_size));
	io::printfn("num_elements       @ %d", offset_of(dense_base, &tc.state.dense.num_elements));
	io::printfn("dtype              @ %d", offset_of(dense_base, &tc.state.dense.dtype));
	io::printfn("shape              @ %d", offset_of(dense_base, &tc.state.dense.shape));
	io::printfn("strides            @ %d", offset_of(dense_base, &tc.state.dense.strides));
}

fn void dense_core_layout_assertions() @test {
	// Verify DenseCore field offsets match the C header (dense_core.h).
	DenseCore dc;
	void* base = &dc;

	assert(offset_of(base, &dc.data)         == 0,   "data should be at offset 0");
	assert(offset_of(base, &dc.byte_size)    == 8,   "byte_size should be at offset 8");
	assert(offset_of(base, &dc.num_elements) == 16,  "num_elements should be at offset 16");
	assert(offset_of(base, &dc.dtype)        == 24,  "dtype should be at offset 24");
	assert(offset_of(base, &dc.shape)        == 32,  "shape should be at offset 32");
	assert(offset_of(base, &dc.strides)      == 104, "strides should be at offset 104");
	assert(DenseCore.sizeof == 168,                   "DenseCore should be 168 bytes total");
}
