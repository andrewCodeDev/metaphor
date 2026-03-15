module metaphor::device::cuda_device::error_checking;

// C Interop - CUDA error types and functions

typedef CudaError = int;
typedef CublasStatus = int;
typedef CudnnStatus = int;
typedef CutensorStatus = int;

extern fn char* cudaGetErrorString(CudaError err);
extern fn char* cudnnGetErrorString(CudnnStatus status);

// Success constants
const CudaError CUDA_SUCCESS = 0;
const CublasStatus CUBLAS_STATUS_SUCCESS = 0;
const CudnnStatus CUDNN_STATUS_SUCCESS = 0;
const CutensorStatus CUTENSOR_STATUS_SUCCESS = 0;

// cuBLAS error codes
const CublasStatus CUBLAS_STATUS_NOT_INITIALIZED = 1;
const CublasStatus CUBLAS_STATUS_ALLOC_FAILED = 3;
const CublasStatus CUBLAS_STATUS_INVALID_VALUE = 7;
const CublasStatus CUBLAS_STATUS_ARCH_MISMATCH = 8;
const CublasStatus CUBLAS_STATUS_MAPPING_ERROR = 11;
const CublasStatus CUBLAS_STATUS_EXECUTION_FAILED = 13;
const CublasStatus CUBLAS_STATUS_INTERNAL_ERROR = 14;
const CublasStatus CUBLAS_STATUS_NOT_SUPPORTED = 15;
const CublasStatus CUBLAS_STATUS_LICENSE_ERROR = 16;

// cuTENSOR error codes
const CutensorStatus CUTENSOR_STATUS_NOT_INITIALIZED = 1;
const CutensorStatus CUTENSOR_STATUS_ALLOC_FAILED = 3;
const CutensorStatus CUTENSOR_STATUS_INVALID_VALUE = 7;
const CutensorStatus CUTENSOR_STATUS_ARCH_MISMATCH = 8;
const CutensorStatus CUTENSOR_STATUS_MAPPING_ERROR = 11;
const CutensorStatus CUTENSOR_STATUS_EXECUTION_FAILED = 13;
const CutensorStatus CUTENSOR_STATUS_INTERNAL_ERROR = 14;
const CutensorStatus CUTENSOR_STATUS_NOT_SUPPORTED = 15;
const CutensorStatus CUTENSOR_STATUS_LICENSE_ERROR = 16;
const CutensorStatus CUTENSOR_STATUS_CUBLAS_ERROR = 17;
const CutensorStatus CUTENSOR_STATUS_CUDA_ERROR = 18;
const CutensorStatus CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19;
const CutensorStatus CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20;
const CutensorStatus CUTENSOR_STATUS_IO_ERROR = 21;

// Error Checking Functions

/// Check CUDA error and panic with descriptive message if failed.
fn void check_cuda(CudaError err, char* operation) {
	if (err != CUDA_SUCCESS) {
		char* err_str = cudaGetErrorString(err);
		unreachable("CUDA error in %s: %s", operation, err_str);
	}
}

/// Check cuBLAS error and panic with descriptive message if failed.
fn void check_cublas(CublasStatus err, char* operation) {
	if (err != CUBLAS_STATUS_SUCCESS) {
		char* err_str = cublas_status_string(err);
		unreachable("cuBLAS error in %s: %s", operation, err_str);
	}
}

/// Check cuDNN error and panic with descriptive message if failed.
fn void check_cudnn(CudnnStatus err, char* operation) {
	if (err != CUDNN_STATUS_SUCCESS) {
		char* err_str = cudnnGetErrorString(err);
		unreachable("cuDNN error in %s: %s", operation, err_str);
	}
}

/// Check cuTENSOR error and panic with descriptive message if failed.
fn void check_cutensor(CutensorStatus err, char* operation) {
	if (err != CUTENSOR_STATUS_SUCCESS) {
		char* err_str = cutensor_status_string(err);
		unreachable("cuTENSOR error in %s: %s", operation, err_str);
	}
}

// Status String Helpers

fn char* cublas_status_string(CublasStatus status) {
	switch (status) {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
		default: return "UNKNOWN_CUBLAS_ERROR";
	}
}

fn char* cutensor_status_string(CutensorStatus status) {
	switch (status) {
		case CUTENSOR_STATUS_SUCCESS: return "CUTENSOR_STATUS_SUCCESS";
		case CUTENSOR_STATUS_NOT_INITIALIZED: return "CUTENSOR_STATUS_NOT_INITIALIZED";
		case CUTENSOR_STATUS_ALLOC_FAILED: return "CUTENSOR_STATUS_ALLOC_FAILED";
		case CUTENSOR_STATUS_INVALID_VALUE: return "CUTENSOR_STATUS_INVALID_VALUE";
		case CUTENSOR_STATUS_ARCH_MISMATCH: return "CUTENSOR_STATUS_ARCH_MISMATCH";
		case CUTENSOR_STATUS_MAPPING_ERROR: return "CUTENSOR_STATUS_MAPPING_ERROR";
		case CUTENSOR_STATUS_EXECUTION_FAILED: return "CUTENSOR_STATUS_EXECUTION_FAILED";
		case CUTENSOR_STATUS_INTERNAL_ERROR: return "CUTENSOR_STATUS_INTERNAL_ERROR";
		case CUTENSOR_STATUS_NOT_SUPPORTED: return "CUTENSOR_STATUS_NOT_SUPPORTED";
		case CUTENSOR_STATUS_LICENSE_ERROR: return "CUTENSOR_STATUS_LICENSE_ERROR";
		case CUTENSOR_STATUS_CUBLAS_ERROR: return "CUTENSOR_STATUS_CUBLAS_ERROR";
		case CUTENSOR_STATUS_CUDA_ERROR: return "CUTENSOR_STATUS_CUDA_ERROR";
		case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE: return "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
		case CUTENSOR_STATUS_INSUFFICIENT_DRIVER: return "CUTENSOR_STATUS_INSUFFICIENT_DRIVER";
		case CUTENSOR_STATUS_IO_ERROR: return "CUTENSOR_STATUS_IO_ERROR";
		default: return "UNKNOWN_CUTENSOR_ERROR";
	}
}
