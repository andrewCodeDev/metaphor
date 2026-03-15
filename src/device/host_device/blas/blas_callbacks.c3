//! BLAS Callbacks
//!
//! Self-contained callback structures for BLAS operations targeting CBLAS
//! (MKL, OpenBLAS, Accelerate). When the compiler detects an einsum pattern
//! matching a BLAS operation (e.g., "ij,jk->ik"), it injects these callbacks
//! instead of generating JIT code.
//!
//! Each callback implements Executable. Tensor pointers and dimensions are
//! derived from sources/target at execute time.
//!

module metaphor::device::host_device::blas::blas_callbacks;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::rtti;
import metaphor::graph;
import metaphor::graph::execution_sequence;
import metaphor::device::common::operations;
import std::core::mem::allocator;

alias TensorCorePtr = execution_sequence::TensorCorePtr;

faultdef NO_VENDOR_KERNEL;

// CBLAS extern declarations (linked via mkl_rt)

const int CBLAS_ROW_MAJOR = 101;
const int CBLAS_NO_TRANS  = 111;
const int CBLAS_TRANS     = 112;

extern fn void cblas_sgemm(
	int layout, int trans_a, int trans_b,
	int m, int n, int k,
	float alpha, float* a, int lda,
	float* b, int ldb,
	float beta, float* c, int ldc);

extern fn void cblas_dgemm(
	int layout, int trans_a, int trans_b,
	int m, int n, int k,
	double alpha, double* a, int lda,
	double* b, int ldb,
	double beta, double* c, int ldc);

extern fn void cblas_sgemv(
	int layout, int trans,
	int m, int n,
	float alpha, float* a, int lda,
	float* x, int incx,
	float beta, float* y, int incy);

extern fn void cblas_dgemv(
	int layout, int trans,
	int m, int n,
	double alpha, double* a, int lda,
	double* x, int incx,
	double beta, double* y, int incy);

extern fn float cblas_sdot(int n, float* x, int incx, float* y, int incy);
extern fn double cblas_ddot(int n, double* x, int incx, double* y, int incy);

extern fn void cblas_saxpy(int n, float alpha, float* x, int incx, float* y, int incy);
extern fn void cblas_daxpy(int n, double alpha, double* x, int incx, double* y, int incy);

extern fn void cblas_sger(
	int layout, int m, int n,
	float alpha, float* x, int incx,
	float* y, int incy, float* a, int lda);

extern fn void cblas_dger(
	int layout, int m, int n,
	double alpha, double* x, int incx,
	double* y, int incy, double* a, int lda);

// GEMM Callback (Matrix-Matrix Multiply)

/// C = alpha * op(A) * op(B) + beta * C
struct GemmCallback (Executable)
{
	bool trans_a;
	bool trans_b;
	double alpha;
	double beta;
}

fn void GemmCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* input_a = sources[0];
	TensorCore* input_b = sources[1];
	TensorCore* output = target;

	// Derive dimensions from shapes
	Shape* a_shape = input_a.shape();
	Shape* b_shape = input_b.shape();

	ulong a_rows = a_shape.get(0);
	ulong a_cols = a_shape.get(1);
	ulong b_cols = b_shape.get(1);

	ulong m = self.trans_a ? a_cols : a_rows;
	ulong k = self.trans_a ? a_rows : a_cols;
	ulong n = self.trans_b ? b_shape.get(0) : b_cols;
	ulong lda = a_cols;
	ulong ldb = b_cols;
	ulong ldc = n;

	// Update output shape
	output.shape().set(0, m);
	output.shape().set(1, n);
	output.shape().len = 2;
	output.state.dense.num_elements = m * n;
	Strides strides = output.shape().strides();
	for (usz i = 0; i < 8; i++) { output.state.dense.strides[i] = strides.buffer[i]; }

	rtti::Datatype dtype = output.dtype();
	void* a_data = input_a.state.dense.data;
	void* b_data = input_b.state.dense.data;
	void* c_data = output.state.dense.data;
	assert(a_data != null && b_data != null && c_data != null, "BLAS callback: null data pointer");

	int ta = self.trans_a ? CBLAS_TRANS : CBLAS_NO_TRANS;
	int tb = self.trans_b ? CBLAS_TRANS : CBLAS_NO_TRANS;

	switch (dtype)
	{
		case F32:
			cblas_sgemm(
				CBLAS_ROW_MAJOR, ta, tb,
				(int)m, (int)n, (int)k,
				(float)self.alpha, (float*)a_data, (int)lda,
				(float*)b_data, (int)ldb,
				(float)self.beta, (float*)c_data, (int)ldc);
		case F64:
			cblas_dgemm(
				CBLAS_ROW_MAJOR, ta, tb,
				(int)m, (int)n, (int)k,
				self.alpha, (double*)a_data, (int)lda,
				(double*)b_data, (int)ldb,
				self.beta, (double*)c_data, (int)ldc);
		default:
			unreachable("BLAS GEMM: unsupported dtype");
	}
}

fn void GemmCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// Batch GEMM Callback (Strided Batched Matrix-Matrix Multiply)

/// C[b] = alpha * op(A[b]) * op(B[b]) + beta * C[b]
struct BatchGemmCallback (Executable)
{
	bool trans_a;
	bool trans_b;
	double alpha;
	double beta;
}

fn void BatchGemmCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* input_a = sources[0];
	TensorCore* input_b = sources[1];
	TensorCore* output = target;

	Shape* a_shape = input_a.shape();
	Shape* b_shape = input_b.shape();

	ulong batch_size = a_shape.get(0);
	ulong a_rows = a_shape.get(1);
	ulong a_cols = a_shape.get(2);
	ulong b_rows = b_shape.get(1);
	ulong b_cols = b_shape.get(2);

	ulong m = self.trans_a ? a_cols : a_rows;
	ulong k = self.trans_a ? a_rows : a_cols;
	ulong n = self.trans_b ? b_rows : b_cols;
	ulong lda = a_cols;
	ulong ldb = b_cols;
	ulong ldc = n;
	ulong stride_a = a_rows * a_cols;
	ulong stride_b = b_rows * b_cols;
	ulong stride_c = m * n;

	// Update output shape
	output.shape().set(0, batch_size);
	output.shape().set(1, m);
	output.shape().set(2, n);
	output.shape().len = 3;
	output.state.dense.num_elements = batch_size * m * n;
	Strides strides = output.shape().strides();
	for (usz i = 0; i < 8; i++) { output.state.dense.strides[i] = strides.buffer[i]; }

	rtti::Datatype dtype = output.dtype();
	void* a_data = input_a.state.dense.data;
	void* b_data = input_b.state.dense.data;
	void* c_data = output.state.dense.data;
	assert(a_data != null && b_data != null && c_data != null, "BLAS callback: null data pointer");

	int ta = self.trans_a ? CBLAS_TRANS : CBLAS_NO_TRANS;
	int tb = self.trans_b ? CBLAS_TRANS : CBLAS_NO_TRANS;

	// Loop over batches (portable, works with any CBLAS backend)
	switch (dtype)
	{
		case F32:
			for (ulong batch = 0; batch < batch_size; batch++)
			{
				float* a_ptr = (float*)a_data + batch * stride_a;
				float* b_ptr = (float*)b_data + batch * stride_b;
				float* c_ptr = (float*)c_data + batch * stride_c;
				cblas_sgemm(
					CBLAS_ROW_MAJOR, ta, tb,
					(int)m, (int)n, (int)k,
					(float)self.alpha, a_ptr, (int)lda,
					b_ptr, (int)ldb,
					(float)self.beta, c_ptr, (int)ldc);
			}
		case F64:
			for (ulong batch = 0; batch < batch_size; batch++)
			{
				double* a_ptr = (double*)a_data + batch * stride_a;
				double* b_ptr = (double*)b_data + batch * stride_b;
				double* c_ptr = (double*)c_data + batch * stride_c;
				cblas_dgemm(
					CBLAS_ROW_MAJOR, ta, tb,
					(int)m, (int)n, (int)k,
					self.alpha, a_ptr, (int)lda,
					b_ptr, (int)ldb,
					self.beta, c_ptr, (int)ldc);
			}
		default:
			unreachable("BLAS Batch GEMM: unsupported dtype");
	}
}

fn void BatchGemmCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// GEMV Callback (Matrix-Vector Multiply)

/// y = alpha * op(A) * x + beta * y
struct GemvCallback (Executable)
{
	bool trans_a;
	double alpha;
	double beta;
}

fn void GemvCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* input_a = sources[0];
	TensorCore* input_x = sources[1];
	TensorCore* output = target;

	Shape* a_shape = input_a.shape();
	ulong m = a_shape.get(0);
	ulong n = a_shape.get(1);
	ulong lda = a_shape.get(1);

	// Update output shape
	ulong out_len = self.trans_a ? n : m;
	output.shape().set(0, out_len);
	output.shape().len = 1;
	output.state.dense.num_elements = out_len;
	Strides strides = output.shape().strides();
	for (usz i = 0; i < 8; i++) { output.state.dense.strides[i] = strides.buffer[i]; }

	rtti::Datatype dtype = output.dtype();
	void* a_data = input_a.state.dense.data;
	void* x_data = input_x.state.dense.data;
	void* y_data = output.state.dense.data;
	assert(a_data != null && x_data != null && y_data != null, "BLAS callback: null data pointer");

	int ta = self.trans_a ? CBLAS_TRANS : CBLAS_NO_TRANS;

	switch (dtype)
	{
		case F32:
			cblas_sgemv(
				CBLAS_ROW_MAJOR, ta,
				(int)m, (int)n,
				(float)self.alpha, (float*)a_data, (int)lda,
				(float*)x_data, 1,
				(float)self.beta, (float*)y_data, 1);
		case F64:
			cblas_dgemv(
				CBLAS_ROW_MAJOR, ta,
				(int)m, (int)n,
				self.alpha, (double*)a_data, (int)lda,
				(double*)x_data, 1,
				self.beta, (double*)y_data, 1);
		default:
			unreachable("BLAS GEMV: unsupported dtype");
	}
}

fn void GemvCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// DOT Callback (Dot Product)

/// result = x . y
struct DotCallback (Executable)
{
	char _pad;
}

fn void DotCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* input_x = sources[0];
	TensorCore* input_y = sources[1];
	TensorCore* output = target;

	ulong n = input_x.shape().size();

	rtti::Datatype dtype = output.dtype();
	void* x_data = input_x.state.dense.data;
	void* y_data = input_y.state.dense.data;
	void* out_data = output.state.dense.data;
	assert(x_data != null && y_data != null && out_data != null, "BLAS callback: null data pointer");

	switch (dtype)
	{
		case F32:
			float result_f32 = cblas_sdot((int)n, (float*)x_data, 1, (float*)y_data, 1);
			*(float*)out_data = result_f32;
		case F64:
			double result_f64 = cblas_ddot((int)n, (double*)x_data, 1, (double*)y_data, 1);
			*(double*)out_data = result_f64;
		default:
			unreachable("BLAS DOT: unsupported dtype");
	}
}

fn void DotCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// AXPY Callback (Vector Scale and Add)

/// y = alpha * x + y
struct AxpyCallback (Executable)
{
	double alpha;
}

fn void AxpyCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* input_x = sources[0];
	TensorCore* output = target;

	ulong n = input_x.shape().size();

	rtti::Datatype dtype = output.dtype();
	void* x_data = input_x.state.dense.data;
	void* y_data = output.state.dense.data;
	assert(x_data != null && y_data != null, "BLAS callback: null data pointer");

	switch (dtype)
	{
		case F32:
			cblas_saxpy((int)n, (float)self.alpha, (float*)x_data, 1, (float*)y_data, 1);
		case F64:
			cblas_daxpy((int)n, self.alpha, (double*)x_data, 1, (double*)y_data, 1);
		default:
			unreachable("BLAS AXPY: unsupported dtype");
	}
}

fn void AxpyCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// GER Callback (Outer Product)

/// A = alpha * x * y^T + A
struct GerCallback (Executable)
{
	double alpha;
}

fn void GerCallback.execute(&self, TensorCorePtr[] sources, TensorCorePtr target) @dynamic
{
	TensorCore* input_x = sources[0];
	TensorCore* input_y = sources[1];
	TensorCore* output = target;

	ulong m = input_x.shape().size();
	ulong n = input_y.shape().size();
	ulong lda = n;

	// Update output shape
	output.shape().set(0, m);
	output.shape().set(1, n);
	output.shape().len = 2;
	output.state.dense.num_elements = m * n;
	Strides strides = output.shape().strides();
	for (usz i = 0; i < 8; i++) { output.state.dense.strides[i] = strides.buffer[i]; }

	rtti::Datatype dtype = output.dtype();
	void* x_data = input_x.state.dense.data;
	void* y_data = input_y.state.dense.data;
	void* out_data = output.state.dense.data;
	assert(x_data != null && y_data != null && out_data != null, "BLAS callback: null data pointer");

	switch (dtype)
	{
		case F32:
			cblas_sger(
				CBLAS_ROW_MAJOR, (int)m, (int)n,
				(float)self.alpha, (float*)x_data, 1,
				(float*)y_data, 1, (float*)out_data, (int)lda);
		case F64:
			cblas_dger(
				CBLAS_ROW_MAJOR, (int)m, (int)n,
				self.alpha, (double*)x_data, 1,
				(double*)y_data, 1, (double*)out_data, (int)lda);
		default:
			unreachable("BLAS GER: unsupported dtype");
	}
}

fn void GerCallback.destroy(&self) @dynamic { allocator::free(&graph::graph_alloc, self); }

// VendorKernelResult (returned from lookup)

struct VendorKernelResult
{
	any operation;
	TensorCore*[2] sources;
	char source_count;
	TensorCore* target;
}

// Factory Functions

fn VendorKernelResult? make_gemm(
	TensorCore*[2] children,
	TensorCore* result,
	BinaryOp inner,
	BinaryOp outer,
	rtti::Datatype dtype,
	bool transpose_a,
	bool transpose_b)
{
	if (inner != BinaryOp.MUL || outer != BinaryOp.ADD) return NO_VENDOR_KERNEL~;
	if (dtype != F32 && dtype != F64) return NO_VENDOR_KERNEL~;

	TensorCore* a = children[0];
	TensorCore* b = children[1];
	Shape* a_shape = a.shape();
	Shape* b_shape = b.shape();

	if (a_shape.len != 2 || b_shape.len != 2) return NO_VENDOR_KERNEL~;

	ulong a_rows = a_shape.get(0);
	ulong a_cols = a_shape.get(1);
	ulong b_rows = b_shape.get(0);
	ulong b_cols = b_shape.get(1);

	ulong m = transpose_a ? a_cols : a_rows;
	ulong k_a = transpose_a ? a_rows : a_cols;
	ulong k_b = transpose_b ? b_cols : b_rows;
	ulong n = transpose_b ? b_rows : b_cols;

	if (k_a != k_b) return NO_VENDOR_KERNEL~;

	GemmCallback* cb = allocator::new(&graph::graph_alloc, GemmCallback);
	*cb = {
		.trans_a = transpose_a,
		.trans_b = transpose_b,
		.alpha = 1.0,
		.beta = 0.0,
	};

	return {
		.operation = (any)cb,
		.sources = { a, b },
		.source_count = 2,
		.target = result,
	};
}

fn VendorKernelResult? make_batch_gemm(
	TensorCore*[2] children,
	TensorCore* result,
	BinaryOp inner,
	BinaryOp outer,
	rtti::Datatype dtype,
	bool transpose_a,
	bool transpose_b)
{
	if (inner != BinaryOp.MUL || outer != BinaryOp.ADD) return NO_VENDOR_KERNEL~;
	if (dtype != F32 && dtype != F64) return NO_VENDOR_KERNEL~;

	TensorCore* a = children[0];
	TensorCore* b = children[1];
	Shape* a_shape = a.shape();
	Shape* b_shape = b.shape();

	if (a_shape.len != 3 || b_shape.len != 3) return NO_VENDOR_KERNEL~;
	if (a_shape.get(0) != b_shape.get(0)) return NO_VENDOR_KERNEL~;

	ulong a_rows = a_shape.get(1);
	ulong a_cols = a_shape.get(2);
	ulong b_rows = b_shape.get(1);
	ulong b_cols = b_shape.get(2);

	ulong k_a = transpose_a ? a_rows : a_cols;
	ulong k_b = transpose_b ? b_cols : b_rows;

	if (k_a != k_b) return NO_VENDOR_KERNEL~;

	BatchGemmCallback* cb = allocator::new(&graph::graph_alloc, BatchGemmCallback);
	*cb = {
		.trans_a = transpose_a,
		.trans_b = transpose_b,
		.alpha = 1.0,
		.beta = 0.0,
	};

	return {
		.operation = (any)cb,
		.sources = { a, b },
		.source_count = 2,
		.target = result,
	};
}

fn VendorKernelResult? make_gemv(
	TensorCore*[2] children,
	TensorCore* result,
	BinaryOp inner,
	BinaryOp outer,
	rtti::Datatype dtype)
{
	if (inner != BinaryOp.MUL || outer != BinaryOp.ADD) return NO_VENDOR_KERNEL~;
	if (dtype != F32 && dtype != F64) return NO_VENDOR_KERNEL~;

	TensorCore* a = children[0];
	TensorCore* x = children[1];
	Shape* a_shape = a.shape();
	Shape* x_shape = x.shape();

	if (a_shape.len != 2 || x_shape.len != 1) return NO_VENDOR_KERNEL~;
	if (x_shape.get(0) != a_shape.get(1)) return NO_VENDOR_KERNEL~;

	GemvCallback* cb = allocator::new(&graph::graph_alloc, GemvCallback);
	*cb = {
		.trans_a = false,
		.alpha = 1.0,
		.beta = 0.0,
	};

	return {
		.operation = (any)cb,
		.sources = { a, x },
		.source_count = 2,
		.target = result,
	};
}

fn VendorKernelResult? make_dot(
	TensorCore*[2] children,
	TensorCore* result,
	BinaryOp inner,
	BinaryOp outer,
	rtti::Datatype dtype)
{
	if (inner != BinaryOp.MUL || outer != BinaryOp.ADD) return NO_VENDOR_KERNEL~;
	if (dtype != F32 && dtype != F64) return NO_VENDOR_KERNEL~;

	TensorCore* x = children[0];
	TensorCore* y = children[1];
	Shape* x_shape = x.shape();
	Shape* y_shape = y.shape();

	if (x_shape.len != 1 || y_shape.len != 1) return NO_VENDOR_KERNEL~;
	if (x_shape.get(0) != y_shape.get(0)) return NO_VENDOR_KERNEL~;

	DotCallback* cb = allocator::new(&graph::graph_alloc, DotCallback);
	*cb = {};

	return {
		.operation = (any)cb,
		.sources = { x, y },
		.source_count = 2,
		.target = result,
	};
}

fn VendorKernelResult? make_ger(
	TensorCore*[2] children,
	TensorCore* result,
	BinaryOp inner,
	rtti::Datatype dtype)
{
	if (inner != BinaryOp.MUL) return NO_VENDOR_KERNEL~;
	if (dtype != F32 && dtype != F64) return NO_VENDOR_KERNEL~;

	TensorCore* x = children[0];
	TensorCore* y = children[1];
	Shape* x_shape = x.shape();
	Shape* y_shape = y.shape();

	if (x_shape.len != 1 || y_shape.len != 1) return NO_VENDOR_KERNEL~;

	GerCallback* cb = allocator::new(&graph::graph_alloc, GerCallback);
	*cb = {
		.alpha = 1.0,
	};

	return {
		.operation = (any)cb,
		.sources = { x, y },
		.source_count = 2,
		.target = result,
	};
}
