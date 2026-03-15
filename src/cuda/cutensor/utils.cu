/*
 * cutensor/utils.cu - cuTENSOR utility types and helpers
 */

#ifndef __CUTENSOR_UTILS_H__
#define __CUTENSOR_UTILS_H__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"

#include <array>
#include <algorithm>

/* ============================================================================
 * Constants
 * ============================================================================ */

static const u32 CUTENSOR_ALIGNMENT = 128;

/* ============================================================================
 * Type Converters
 * ============================================================================ */

inline cutensorOperator_t cutensor_op_from_binary(BinaryOp op) {
    switch (op) {
        case BINARY_ADD: return CUTENSOR_OP_ADD;
        case BINARY_MIN: return CUTENSOR_OP_MIN;
        case BINARY_MAX: return CUTENSOR_OP_MAX;
        case BINARY_MUL: return CUTENSOR_OP_MUL;
        default:
            SYSTEM_EXIT("Invalid binary operation for cutensor");
            return CUTENSOR_OP_ADD;
    }
}

inline cutensorDataType_t cutensor_dtype(Dtype id) {
    switch (id) {
        case DTYPE_F16: return CUTENSOR_R_16F;
        case DTYPE_BF16: return CUTENSOR_R_16BF;
        case DTYPE_F32: return CUTENSOR_R_32F;
        case DTYPE_F64: return CUTENSOR_R_64F;
        default:
            SYSTEM_EXIT("Invalid data type for cutensor");
            return CUTENSOR_R_32F;
    }
}

inline cutensorComputeDescriptor_t cutensor_compute_desc(Dtype id) {
    switch (id) {
        case DTYPE_F16: return CUTENSOR_COMPUTE_DESC_32F;  // f32 compute for f16 data (better stability)
        case DTYPE_BF16: return CUTENSOR_COMPUTE_DESC_32F; // f32 compute for bf16 data
        case DTYPE_F32: return CUTENSOR_COMPUTE_DESC_32F;
        case DTYPE_F64: return CUTENSOR_COMPUTE_DESC_64F;
        default:
            SYSTEM_EXIT("Invalid data type for cutensor");
            return CUTENSOR_COMPUTE_DESC_32F;
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

#endif /* __CUTENSOR_UTILS_H__ */
