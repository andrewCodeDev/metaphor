/*
 * core/dtype_dispatch.h - Shared dtype dispatch and atomic helpers
 *
 * Consolidates dtype switch/case dispatch and atomic add workarounds
 * for f16/bf16 types that lack native atomicAdd on some AMD GPUs.
 */

#ifndef METAPHOR_HIP_DTYPE_DISPATCH_H
#define METAPHOR_HIP_DTYPE_DISPATCH_H

#include "assert.h"
#include "cast.h"
#include "includes.h"
#include "../interop.h"

/* ============================================================================
 * DISPATCH_DTYPE — eliminates switch/if-else chains for kernel launches
 *
 * Exposes `scalar_t` as the concrete type within the body. Usage:
 *
 *   DISPATCH_DTYPE(dtype,
 *       my_kernel<scalar_t><<<grid, block, 0, stream>>>(
 *           static_cast<scalar_t*>(out->data),
 *           static_cast<const scalar_t*>(in->data), params));
 *
 * ============================================================================ */

#define DISPATCH_DTYPE(dtype, ...)                                           \
    [&] {                                                                    \
        switch (dtype) {                                                     \
        case DTYPE_F32:  { using scalar_t = float;  __VA_ARGS__; break; }   \
        case DTYPE_F64:  { using scalar_t = double; __VA_ARGS__; break; }   \
        case DTYPE_F16:  { using scalar_t = f16;    __VA_ARGS__; break; }   \
        case DTYPE_BF16: { using scalar_t = bf16;   __VA_ARGS__; break; }   \
        default: SYSTEM_EXIT("Unsupported dtype");                           \
        }                                                                    \
    }()

/* ============================================================================
 * typed_atomic_add<T> — single templated device function for atomic addition
 *
 * float, double: forward to native atomicAdd
 * f16 (__half):  CAS loop on u32 (handles alignment), accumulates via float
 * bf16 (hip_bfloat16): CAS loop on u16, accumulates via float
 * ============================================================================ */

template<typename T>
__device__ inline void typed_atomic_add(T* addr, T val) {
    atomicAdd(addr, val);
}

/* f16 (__half) — no native atomicAdd on most AMD GPUs.
 * Operates on the containing u32 word to handle alignment. */
template<>
__device__ inline void typed_atomic_add<f16>(f16* addr, f16 val) {
    unsigned int* base = (unsigned int*)((size_t)addr & ~(size_t)1);
    bool is_upper = ((size_t)addr & 1);
    unsigned int old_val, new_val;
    do {
        old_val = *base;
        unsigned short h;
        if (is_upper) {
            h = (unsigned short)(old_val >> 16);
        } else {
            h = (unsigned short)(old_val & 0xffff);
        }
        f16 old_h = *reinterpret_cast<f16*>(&h);
        f16 new_h = __float2half(__half2float(old_h) + __half2float(val));
        unsigned short new_bits = *reinterpret_cast<unsigned short*>(&new_h);
        if (is_upper) {
            new_val = (old_val & 0x0000ffff) | ((unsigned int)new_bits << 16);
        } else {
            new_val = (old_val & 0xffff0000) | (unsigned int)new_bits;
        }
    } while (atomicCAS(base, old_val, new_val) != old_val);
}

/* bf16 (hip_bfloat16) — CAS loop on u16, accumulates via float. */
template<>
__device__ inline void typed_atomic_add<bf16>(bf16* addr, bf16 val) {
    unsigned short* addr_u16 = reinterpret_cast<unsigned short*>(addr);
    unsigned short old_u16 = *addr_u16;
    unsigned short expected;
    do {
        expected = old_u16;
        bf16 old_bf16;
        memcpy(&old_bf16, &expected, sizeof(unsigned short));
        float sum = (float)old_bf16 + (float)val;
        bf16 result = bf16(sum);
        unsigned short result_u16;
        memcpy(&result_u16, &result, sizeof(unsigned short));
        old_u16 = atomicCAS(addr_u16, expected, result_u16);
    } while (old_u16 != expected);
}

#endif /* METAPHOR_HIP_DTYPE_DISPATCH_H */
