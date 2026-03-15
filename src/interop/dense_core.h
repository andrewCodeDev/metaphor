/*
 * dense_core.h - C-interoperable DenseCore type
 *
 * Matches the C3 DenseCore struct layout exactly.
 * Can be passed directly across the C/CUDA/AMD boundary as DenseCore*.
 */

#ifndef METAPHOR_DENSE_CORE_H
#define METAPHOR_DENSE_CORE_H

#include <stdint.h>
#include "shape.h"

/* Data type enum values (matches rtti::Datatype ordinals) */
typedef enum {
    METAPHOR_DTYPE_BOOL = 0,
    METAPHOR_DTYPE_U8   = 1,
    METAPHOR_DTYPE_U64  = 2,
    METAPHOR_DTYPE_F16  = 3,
    METAPHOR_DTYPE_BF16 = 4,
    METAPHOR_DTYPE_F32  = 5,
    METAPHOR_DTYPE_F64  = 6,
} MetaphorDtype;

typedef struct {
    void*     data;                           /* device/host memory */
    uint64_t  byte_size;                      /* buffer size in bytes */
    uint64_t  num_elements;                   /* product of shape dims */
    uint64_t  dtype;                          /* MetaphorDtype enum value (0-6) */
    Shape     shape;                          /* embedded shape */
    uint64_t  strides[METAPHOR_MAX_DIMS];     /* row-major, in elements */
} DenseCore;

#endif /* METAPHOR_DENSE_CORE_H */
