module metaphor::device::hip_device::error_checking;

// C Interop - HIP error types and functions

typedef HipError    = int;
typedef RocblasStatus = int;
typedef MiopenStatus  = int;
typedef HipblasltStatus = int;

extern fn char* hipGetErrorString(HipError err);
extern fn char* miopenGetErrorString(MiopenStatus status);

// Success constants
const HipError      HIP_SUCCESS           = 0;
const RocblasStatus ROCBLAS_STATUS_SUCCESS = 0;
const MiopenStatus  MIOPEN_STATUS_SUCCESS  = 0;
const HipblasltStatus HIPBLASLT_STATUS_SUCCESS = 0;

// rocBLAS error codes
const RocblasStatus ROCBLAS_STATUS_INVALID_HANDLE   = 1;
const RocblasStatus ROCBLAS_STATUS_NOT_IMPLEMENTED  = 2;
const RocblasStatus ROCBLAS_STATUS_INVALID_POINTER  = 3;
const RocblasStatus ROCBLAS_STATUS_INVALID_SIZE     = 4;
const RocblasStatus ROCBLAS_STATUS_MEMORY_ERROR     = 5;
const RocblasStatus ROCBLAS_STATUS_INTERNAL_ERROR   = 6;
const RocblasStatus ROCBLAS_STATUS_PERF_DEGRADED    = 7;
const RocblasStatus ROCBLAS_STATUS_SIZE_QUERY_MISMATCH = 8;
const RocblasStatus ROCBLAS_STATUS_SIZE_INCREASED   = 9;
const RocblasStatus ROCBLAS_STATUS_SIZE_UNCHANGED   = 10;
const RocblasStatus ROCBLAS_STATUS_INVALID_VALUE    = 11;
const RocblasStatus ROCBLAS_STATUS_CONTINUE         = 12;
const RocblasStatus ROCBLAS_STATUS_CHECK_NUMERICS_FAIL = 13;

// MIOpen error codes (matches miopenStatus_t)
const MiopenStatus MIOPEN_STATUS_NOT_INITIALIZED    = 1;
const MiopenStatus MIOPEN_STATUS_INVALID_VALUE      = 2;
const MiopenStatus MIOPEN_STATUS_BAD_PARAM          = 3;
const MiopenStatus MIOPEN_STATUS_ALLOC_FAILED       = 4;
const MiopenStatus MIOPEN_STATUS_INTERNAL_ERROR     = 5;
const MiopenStatus MIOPEN_STATUS_NOT_IMPLEMENTED    = 6;
const MiopenStatus MIOPEN_STATUS_UNKNOWN_ERROR      = 7;
const MiopenStatus MIOPEN_STATUS_VERSION_MISMATCH   = 8;

// Error Checking Functions

/// Check HIP error and panic with descriptive message if failed.
fn void check_hip(HipError err, char* operation) {
	if (err != HIP_SUCCESS) {
		char* err_str = hipGetErrorString(err);
		unreachable("HIP error in %s: %s", operation, err_str);
	}
}

/// Check rocBLAS error and panic with descriptive message if failed.
fn void check_rocblas(RocblasStatus err, char* operation) {
	if (err != ROCBLAS_STATUS_SUCCESS) {
		char* err_str = rocblas_status_string(err);
		unreachable("rocBLAS error in %s: %s", operation, err_str);
	}
}

/// Check MIOpen error and panic with descriptive message if failed.
fn void check_miopen(MiopenStatus err, char* operation) {
	if (err != MIOPEN_STATUS_SUCCESS) {
		char* err_str = miopenGetErrorString(err);
		unreachable("MIOpen error in %s: %s", operation, err_str);
	}
}

// Status String Helpers

fn char* rocblas_status_string(RocblasStatus status) {
	switch (status) {
		case ROCBLAS_STATUS_SUCCESS:              return "ROCBLAS_STATUS_SUCCESS";
		case ROCBLAS_STATUS_INVALID_HANDLE:       return "ROCBLAS_STATUS_INVALID_HANDLE";
		case ROCBLAS_STATUS_NOT_IMPLEMENTED:      return "ROCBLAS_STATUS_NOT_IMPLEMENTED";
		case ROCBLAS_STATUS_INVALID_POINTER:      return "ROCBLAS_STATUS_INVALID_POINTER";
		case ROCBLAS_STATUS_INVALID_SIZE:         return "ROCBLAS_STATUS_INVALID_SIZE";
		case ROCBLAS_STATUS_MEMORY_ERROR:         return "ROCBLAS_STATUS_MEMORY_ERROR";
		case ROCBLAS_STATUS_INTERNAL_ERROR:       return "ROCBLAS_STATUS_INTERNAL_ERROR";
		case ROCBLAS_STATUS_PERF_DEGRADED:        return "ROCBLAS_STATUS_PERF_DEGRADED";
		case ROCBLAS_STATUS_SIZE_QUERY_MISMATCH:  return "ROCBLAS_STATUS_SIZE_QUERY_MISMATCH";
		case ROCBLAS_STATUS_SIZE_INCREASED:       return "ROCBLAS_STATUS_SIZE_INCREASED";
		case ROCBLAS_STATUS_SIZE_UNCHANGED:       return "ROCBLAS_STATUS_SIZE_UNCHANGED";
		case ROCBLAS_STATUS_INVALID_VALUE:        return "ROCBLAS_STATUS_INVALID_VALUE";
		case ROCBLAS_STATUS_CONTINUE:             return "ROCBLAS_STATUS_CONTINUE";
		case ROCBLAS_STATUS_CHECK_NUMERICS_FAIL:  return "ROCBLAS_STATUS_CHECK_NUMERICS_FAIL";
		default: return "UNKNOWN_ROCBLAS_ERROR";
	}
}

fn char* miopen_status_string(MiopenStatus status) {
	switch (status) {
		case MIOPEN_STATUS_SUCCESS:           return "MIOPEN_STATUS_SUCCESS";
		case MIOPEN_STATUS_NOT_INITIALIZED:   return "MIOPEN_STATUS_NOT_INITIALIZED";
		case MIOPEN_STATUS_INVALID_VALUE:     return "MIOPEN_STATUS_INVALID_VALUE";
		case MIOPEN_STATUS_BAD_PARAM:         return "MIOPEN_STATUS_BAD_PARAM";
		case MIOPEN_STATUS_ALLOC_FAILED:      return "MIOPEN_STATUS_ALLOC_FAILED";
		case MIOPEN_STATUS_INTERNAL_ERROR:    return "MIOPEN_STATUS_INTERNAL_ERROR";
		case MIOPEN_STATUS_NOT_IMPLEMENTED:   return "MIOPEN_STATUS_NOT_IMPLEMENTED";
		case MIOPEN_STATUS_UNKNOWN_ERROR:     return "MIOPEN_STATUS_UNKNOWN_ERROR";
		case MIOPEN_STATUS_VERSION_MISMATCH:  return "MIOPEN_STATUS_VERSION_MISMATCH";
		default: return "UNKNOWN_MIOPEN_ERROR";
	}
}
