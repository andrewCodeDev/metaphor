/*
 * shape.h - C-interoperable Shape type
 *
 * Matches the C3 Shape struct layout exactly.
 * Used for cross-language ABI compatibility (C3, CUDA, AMD).
 */

#ifndef METAPHOR_SHAPE_H
#define METAPHOR_SHAPE_H

#include <stdint.h>

#define METAPHOR_MAX_DIMS 8

typedef struct {
    uint64_t buffer[METAPHOR_MAX_DIMS];  /* 1-initialized unused */
    uint64_t len;                         /* active dimension count */
} Shape;

#endif /* METAPHOR_SHAPE_H */
