/*
 * interop.h - Shared types for C3 <-> HIP interoperability
 *
 * This file defines the ABI contract between C3 and HIP code.
 * Must be pure C - no C++ features. C3 imports this via cImport.
 */

#ifndef METAPHOR_HIP_INTEROP_H
#define METAPHOR_HIP_INTEROP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define EXTERN_C extern "C"
#else
#define EXTERN_C extern
#endif

/* ============================================================================
 * Primitive Types
 * ============================================================================
 */

typedef uint64_t len_t;

/* ============================================================================
 * Data Type Identifiers
 * ============================================================================
 */

typedef enum {
  DTYPE_F32 = 0,
  DTYPE_F64 = 1,
  DTYPE_U8 = 2,
  DTYPE_F16 = 3,  /* __half - IEEE 754 half precision */
  DTYPE_BF16 = 4  /* hip_bfloat16 - brain float (Ampere+) */
} Dtype;

/* ============================================================================
 * DenseCore - C-interoperable tensor storage
 *
 * Shared struct definition used directly by C3, HIP, and AMD backends.
 * No marshalling needed — pass DenseCore* across the boundary.
 * ============================================================================
 */

#include "dense_core.h"

/* Alias for backwards compatibility with local buffer declarations */
#define HIP_MAX_DIMS METAPHOR_MAX_DIMS

/*
 * Convert DenseCore dtype (rtti::Datatype ordinal) to HIP Dtype.
 * Must stay in sync with rtti::Datatype enum in rtti.c3:
 *   BOOL=0, U8=1, U64=2, F16=3, BF16=4, F32=5, F64=6
 */
static inline Dtype dense_core_dtype(const DenseCore* dc) {
    switch (dc->dtype) {
        case METAPHOR_DTYPE_F32:  return DTYPE_F32;
        case METAPHOR_DTYPE_F64:  return DTYPE_F64;
        case METAPHOR_DTYPE_U8:   return DTYPE_U8;
        case METAPHOR_DTYPE_BOOL: return DTYPE_U8;
        case METAPHOR_DTYPE_F16:  return DTYPE_F16;
        case METAPHOR_DTYPE_BF16: return DTYPE_BF16;
        default:                  return DTYPE_F32;
    }
}

/* ============================================================================
 * Random Distribution Types
 * ============================================================================
 */

typedef enum { RAND_UNIFORM = 0, RAND_NORMAL = 1 } RandType;

/* ============================================================================
 * Binary Operations
 * ============================================================================
 */

typedef enum {
  BINARY_ADD = 0,
  BINARY_MIN = 1,
  BINARY_MAX = 2,
  BINARY_MUL = 3
} BinaryOp;

/* ============================================================================
 * Reduction Types
 * ============================================================================
 */

typedef enum { REDUX_NONE = 0, REDUX_MEAN = 1, REDUX_SUM = 2 } ReductionType;

/* ============================================================================
 * Softmax Types
 * ============================================================================
 */

typedef enum { SMAX_FAST = 0, SMAX_MAX = 1, SMAX_LOG = 2 } SoftmaxType;

/* ============================================================================
 * Opaque Handles
 *
 * Strongly-typed wrappers for pointers crossing the language boundary.
 * Internal structure is hidden - C3 sees these as distinct types.
 * ============================================================================
 */

/* Primary device handle - owns all HIP state */
typedef struct {
  void *ptr;
} HipDeviceHandle;

/* Individual library handles (for standalone use) */
typedef struct {
  void *ptr;
} StreamHandle;
typedef struct {
  void *ptr;
} HipblasHandle;
typedef struct {
  void *ptr;
} MiopenHandle;

/* Memory pointers */
typedef struct {
  void *ptr;
} DevicePtr;
typedef struct {
  void *ptr;
} HostPtr;

/* Event handle for synchronization */
typedef struct {
  void *ptr;
} EventHandle;

/* hipTENSOR plan handle */
typedef struct {
  void *plan;
  len_t scratch_len;
} HiptensorPlanHandle;

/* HipTensor removed — use DenseCore* directly (see dense_core.h) */

/* ============================================================================
 * Unary Map Operations
 * ============================================================================
 */

typedef enum {
  MAP_IDENTITY = 0,
  MAP_NEG,
  MAP_ABS,
  MAP_SQR,
  MAP_SQRT,
  MAP_RECIP,
  MAP_EXP,
  MAP_LOG,
  MAP_SIN,
  MAP_COS,
  MAP_TANH,
  MAP_RELU,
  MAP_HEAVISIDE,
  MAP_SIGMOID,
  MAP_GELU,
  MAP_SILU,
  MAP_SOFTPLUS,
  MAP_NOT,
  MAP_SIGN,
} MapOp;

/* ============================================================================
 * Binary Elementwise Operations (for JIT)
 * ============================================================================
 */

typedef enum {
  BINOP_ADD = 0,
  BINOP_SUB,
  BINOP_MUL,
  BINOP_DIV,
  BINOP_MAX,
  BINOP_MIN,
  BINOP_EQ,
  BINOP_NE,
  BINOP_GT,
  BINOP_LT,
  BINOP_GTE,
  BINOP_LTE,
  BINOP_AND,
  BINOP_OR,
  BINOP_XOR,
} BinOp;

/* ============================================================================
 * Fused Chain Operations (mixed unary/binary)
 *
 * A chain op is either:
 * - Unary: applies MapOp to previous result, no child input needed
 * - Binary: applies BinOp to previous result and a child input
 *
 * Example chain for exp(x) * y:
 *   ops[0] = {.is_unary=true, .unary_op=MAP_EXP}   // exp(in_0)
 *   ops[1] = {.is_unary=false, .binary_op=BINOP_MUL}  // result * in_1
 * ============================================================================
 */

typedef struct {
  bool is_unary;      /* true = unary (MapOp), false = binary (BinOp) */
  union {
    MapOp unary_op;
    BinOp binary_op;
  };
} ChainOp;

#define MAX_FUSED_CHAIN_OPS 8

/* ============================================================================
 * JIT Kernel Handle
 * ============================================================================
 */

typedef struct {
  void *ptr;
} JitKernelHandle;

#endif /* METAPHOR_HIP_INTEROP_H */
