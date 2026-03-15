/*
 * hiptensor/utils.cpp - hipTENSOR utility types and helpers
 *
 * hipTENSOR 2.x utility types and mappings.
 */

#ifndef __HIPTENSOR_UTILS_H__
#define __HIPTENSOR_UTILS_H__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"

#include <array>
#include <algorithm>

/* ============================================================================
 * Constants
 * ============================================================================ */

static const u32 HIPTENSOR_ALIGNMENT = 128;

/* ============================================================================
 * Type Converters
 * ============================================================================ */

inline hiptensorOperator_t hiptensor_op_from_binary(BinaryOp op) {
    switch (op) {
        case BINARY_ADD: return HIPTENSOR_OP_ADD;
        case BINARY_MIN: return HIPTENSOR_OP_MIN;
        case BINARY_MAX: return HIPTENSOR_OP_MAX;
        case BINARY_MUL: return HIPTENSOR_OP_MUL;
        default:
            SYSTEM_EXIT("Invalid binary operation for hiptensor");
            return HIPTENSOR_OP_ADD;
    }
}

inline hiptensorDataType_t hiptensor_dtype(Dtype id) {
    switch (id) {
        case DTYPE_F16: return HIPTENSOR_R_16F;
        case DTYPE_BF16: return HIPTENSOR_R_16BF;
        case DTYPE_F32: return HIPTENSOR_R_32F;
        case DTYPE_F64: return HIPTENSOR_R_64F;
        default:
            SYSTEM_EXIT("Invalid data type for hiptensor");
            return HIPTENSOR_R_32F;
    }
}

/* Compute descriptor: gfx1100 (RDNA3) matrix cores require native precision
 * for contraction. Permutation, reduction, and elementwise binary override
 * this to F32 compute for half types (hipTensor RDNA3 produces zeros with
 * native half compute for non-contraction operations). */
inline hiptensorComputeDescriptor_t hiptensor_compute_desc(Dtype id) {
    switch (id) {
        case DTYPE_F16: return HIPTENSOR_COMPUTE_DESC_16F;
        case DTYPE_BF16: return HIPTENSOR_COMPUTE_DESC_16BF;
        case DTYPE_F32: return HIPTENSOR_COMPUTE_DESC_32F;
        case DTYPE_F64: return HIPTENSOR_COMPUTE_DESC_64F;
        default:
            SYSTEM_EXIT("Invalid data type for hiptensor");
            return HIPTENSOR_COMPUTE_DESC_32F;
    }
}

/* ============================================================================
 * Bounded Array
 *
 * Fixed-capacity array for dimension/symbol storage without heap allocation.
 * ============================================================================ */

template <typename T, std::size_t N = 8>
struct BoundedArray {
    T data[N];
    std::size_t len = 0;

    BoundedArray() = default;

    template <typename U>
    BoundedArray(const U* vals, std::size_t n) : BoundedArray() {
        this->append(vals, n);
    }

    template <typename U>
    BoundedArray(const U* vals, std::size_t n, bool reverse) : BoundedArray(vals, n) {
        if (reverse) this->reverse();
    }

    T* ptr() { return &this->data[0]; }
    const T* ptr() const { return &this->data[0]; }

    void reverse() {
        std::reverse(this->data, this->data + this->len);
    }

    void sort() {
        std::sort(this->data, this->data + this->len);
    }

    template <typename U>
    void append(const U* vals, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            this->append(vals[i]);
        }
    }

    template <typename U>
    void append(U val) {
        CHECK_INVARIANT(this->len < N, "BoundedArray overflow");
        this->data[this->len] = static_cast<T>(val);
        ++this->len;
    }
};

#endif /* __HIPTENSOR_UTILS_H__ */
