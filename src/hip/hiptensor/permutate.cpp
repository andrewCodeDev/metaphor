/*
 * hiptensor/permutate.cpp - Tensor permutation operations (stateless)
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __HIPTENSOR_PERMUTATE_H__
#define __HIPTENSOR_PERMUTATE_H__

#include "backend.cpp"
#include "../logging.h"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    hiptensorPlan_t plan;
    len_t scratch_len;
} HiptensorPermutatePlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

HiptensorPermutatePlan hiptensor_permutate_plan_create(
    HiptensorHandle wrapper,
    Dtype id,
    const len_t* src_dims,
    const u8* src_syms,
    len_t src_dims_len,
    const len_t* dst_dims,
    const u8* dst_syms,
    len_t dst_dims_len
) {
    LOG_DEBUG("hiptensor_permutate_plan_create: dtype=%d, src_dims_len=%lu, dst_dims_len=%lu",
              id, src_dims_len, dst_dims_len);
    for (size_t i = 0; i < src_dims_len; i++) {
        LOG_DEBUG("  src[%zu]: dim=%lu, sym=%d", i, src_dims[i], src_syms[i]);
    }
    for (size_t i = 0; i < dst_dims_len; i++) {
        LOG_DEBUG("  dst[%zu]: dim=%lu, sym=%d", i, dst_dims[i], dst_syms[i]);
    }

    auto* ct = unwrap_hiptensor(wrapper);
    const auto data_type = hiptensor_dtype(id);

    /* hipTensor requires rank >= 2 for permutation.
     * Pad 0D and 1D tensors to 2D by prepending a dimension of size 1.
     * Use a shared padding symbol that doesn't conflict with existing symbols. */
    len_t src_pad_dims[2];
    u8    src_pad_syms[2];
    len_t dst_pad_dims[2];
    u8    dst_pad_syms[2];
    if (src_dims_len <= 1 || dst_dims_len <= 1) {
        /* Find a symbol not used by either src or dst */
        u8 max_sym = 0;
        for (size_t i = 0; i < src_dims_len; i++)
            if (src_syms[i] >= max_sym) max_sym = src_syms[i] + 1;
        for (size_t i = 0; i < dst_dims_len; i++)
            if (dst_syms[i] >= max_sym) max_sym = dst_syms[i] + 1;
        u8 pad_sym = max_sym;

        if (src_dims_len <= 1) {
            src_pad_dims[0] = 1;
            src_pad_dims[1] = (src_dims_len == 0) ? 1 : src_dims[0];
            src_pad_syms[0] = pad_sym;
            src_pad_syms[1] = (src_dims_len == 0) ? 0 : src_syms[0];
            src_dims = src_pad_dims;
            src_syms = src_pad_syms;
            src_dims_len = 2;
            LOG_DEBUG("  (padded src to 2D [1,%lu])", src_pad_dims[1]);
        }
        if (dst_dims_len <= 1) {
            dst_pad_dims[0] = 1;
            dst_pad_dims[1] = (dst_dims_len == 0) ? 1 : dst_dims[0];
            dst_pad_syms[0] = pad_sym;
            dst_pad_syms[1] = (dst_dims_len == 0) ? 0 : dst_syms[0];
            dst_dims = dst_pad_dims;
            dst_syms = dst_pad_syms;
            dst_dims_len = 2;
            LOG_DEBUG("  (padded dst to 2D [1,%lu])", dst_pad_dims[1]);
        }
    }

    CHECK_INVARIANT(src_dims_len == dst_dims_len,
        "hipTensor permutation requires equal ranks. "
        "Broadcasting/expansion must be handled by the caller via JIT strided copy.");

    /* Build dimension arrays (reversed for column-major) */
    BoundedArray<i64> a_dims(src_dims, src_dims_len, true);
    BoundedArray<i32> a_syms(src_syms, src_dims_len, true);
    BoundedArray<i64> b_dims(dst_dims, dst_dims_len, true);
    BoundedArray<i32> b_syms(dst_syms, dst_dims_len, true);

    /* Create tensor descriptors (temporary) */
    hiptensorTensorDescriptor_t x_desc;
    HIPTENSOR_ASSERT(hiptensorCreateTensorDescriptor(
        ct->handle, &x_desc,
        a_dims.len, a_dims.ptr(), NULL,
        data_type, HIPTENSOR_ALIGNMENT
    ));

    hiptensorTensorDescriptor_t y_desc;
    HIPTENSOR_ASSERT(hiptensorCreateTensorDescriptor(
        ct->handle, &y_desc,
        b_dims.len, b_dims.ptr(), NULL,
        data_type, HIPTENSOR_ALIGNMENT
    ));

    /* Create permutation descriptor.
     * Use F32 compute for half types — hipTensor permutation on RDNA3 produces
     * zeros with native half compute descriptors (16BF/16F). F32 compute works
     * for permutation (unlike contraction which requires native precision). */
    auto compute = (id == DTYPE_BF16 || id == DTYPE_F16)
                 ? HIPTENSOR_COMPUTE_DESC_32F
                 : hiptensor_compute_desc(id);

    hiptensorOperationDescriptor_t op_desc;
    HIPTENSOR_ASSERT(hiptensorCreatePermutation(
        ct->handle, &op_desc,
        x_desc, a_syms.ptr(), HIPTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(),
        compute
    ));

    /* Create plan */
    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_ASSERT(hiptensorCreatePlanPreference(
        ct->handle, &plan_pref,
        HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE
    ));

    len_t scratch_len = 0;
    HIPTENSOR_ASSERT(hiptensorEstimateWorkspaceSize(
        ct->handle, op_desc, plan_pref,
        HIPTENSOR_WORKSPACE_DEFAULT, &scratch_len
    ));

    hiptensorPlan_t plan;
    HIPTENSOR_ASSERT(hiptensorCreatePlan(
        ct->handle, &plan, op_desc, plan_pref, scratch_len
    ));

    /*
     * NOTE: Do NOT destroy tensor descriptors or operation descriptor here.
     * hiptensorCreatePlan stores raw pointers to these — the plan references
     * them during execution (hiptensorPermute reads descA->mType, etc).
     * They are freed when the plan is destroyed.
     */

    return {.plan = plan, .scratch_len = scratch_len};
}

/* ============================================================================
 * Plan Destruction
 * ============================================================================ */

void hiptensor_permutate_plan_destroy(HiptensorPermutatePlan p) {
    if (p.plan) {
        HIPTENSOR_ASSERT(hiptensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void hiptensor_permutate(
    HiptensorHandle wrapper,
    HiptensorPermutatePlan p,
    const void* x_vals,
    void* y_vals,
    void* scratch,
    const void* alpha
) {
    auto* backend = unwrap_hiptensor(wrapper);

    LOG_DEBUG("hiptensor_permutate: plan=%p, x_vals=%p, y_vals=%p, alpha=%p, stream=%p",
              (void*)p.plan, x_vals, y_vals, alpha, (void*)backend->stream);

    if (!p.plan) {
        LOG_ERROR("hiptensor_permutate: plan is NULL!");
    }
    if (!x_vals) {
        LOG_ERROR("hiptensor_permutate: x_vals is NULL!");
    }
    if (!y_vals) {
        LOG_ERROR("hiptensor_permutate: y_vals is NULL!");
    }

    HIPTENSOR_ASSERT(hiptensorPermute(
        backend->handle,
        p.plan,
        alpha, x_vals, y_vals,
        backend->stream
    ));
}

#endif /* __HIPTENSOR_PERMUTATE_H__ */
