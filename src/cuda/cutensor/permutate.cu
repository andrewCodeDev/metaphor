/*
 * cutensor/permutate.cu - Tensor permutation operations (stateless)
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __CUTENSOR_PERMUTATE_H__
#define __CUTENSOR_PERMUTATE_H__

#include "backend.cu"
#include "../logging.h"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    cutensorPlan_t plan;
    len_t scratch_len;
} CutensorPermutatePlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

CutensorPermutatePlan cutensor_permutate_plan_create(
    CutensorHandle wrapper,
    Dtype id,
    const len_t* src_dims,
    const u8* src_syms,
    len_t src_dims_len,
    const len_t* dst_dims,
    const u8* dst_syms,
    len_t dst_dims_len
) {
    LOG_DEBUG("cutensor_permutate_plan_create: dtype=%d, src_dims_len=%lu, dst_dims_len=%lu",
              id, src_dims_len, dst_dims_len);
    for (size_t i = 0; i < src_dims_len; i++) {
        LOG_DEBUG("  src[%zu]: dim=%lu, sym=%d", i, src_dims[i], src_syms[i]);
    }
    for (size_t i = 0; i < dst_dims_len; i++) {
        LOG_DEBUG("  dst[%zu]: dim=%lu, sym=%d", i, dst_dims[i], dst_syms[i]);
    }

    auto* ct = unwrap_cutensor(wrapper);
    const auto data_type = cutensor_dtype(id);

    /* Handle scalar (0D) tensors by treating them as 1D with shape [1] */
    len_t scalar_dim = 1;
    u8 scalar_sym = 0; // Use symbol 0 for the synthetic dimension
    if (src_dims_len == 0) {
        src_dims = &scalar_dim;
        src_syms = &scalar_sym;
        src_dims_len = 1;
        LOG_DEBUG("  (scalar: converted to 1D with shape [1])");
    }

    CHECK_INVARIANT(src_dims_len == dst_dims_len,
        "cuTENSOR permutation requires equal ranks (got src=%lu, dst=%lu). "
        "Broadcasting/expansion must be handled by the caller via JIT strided copy.",
        src_dims_len, dst_dims_len);

    /* Build dimension arrays (reversed for column-major) */
    BoundedArray<i64> a_dims(src_dims, src_dims_len, true);
    BoundedArray<i32> a_syms(src_syms, src_dims_len, true);
    BoundedArray<i64> b_dims(dst_dims, dst_dims_len, true);
    BoundedArray<i32> b_syms(dst_syms, dst_dims_len, true);

    /* Create tensor descriptors (temporary) */
    cutensorTensorDescriptor_t x_desc;
    CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
        ct->handle, &x_desc,
        a_dims.len, a_dims.ptr(), NULL,
        data_type, CUTENSOR_ALIGNMENT
    ));

    cutensorTensorDescriptor_t y_desc;
    CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
        ct->handle, &y_desc,
        b_dims.len, b_dims.ptr(), NULL,
        data_type, CUTENSOR_ALIGNMENT
    ));

    /* Create permutation descriptor */
    cutensorOperationDescriptor_t op_desc;
    CUTENSOR_ASSERT(cutensorCreatePermutation(
        ct->handle, &op_desc,
        x_desc, a_syms.ptr(), CUTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(),
        cutensor_compute_desc(id)
    ));

    /* Create plan */
    cutensorPlanPreference_t plan_pref;
    CUTENSOR_ASSERT(cutensorCreatePlanPreference(
        ct->handle, &plan_pref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE
    ));

    len_t scratch_len = 0;
    CUTENSOR_ASSERT(cutensorEstimateWorkspaceSize(
        ct->handle, op_desc, plan_pref,
        CUTENSOR_WORKSPACE_DEFAULT, &scratch_len
    ));

    cutensorPlan_t plan;
    CUTENSOR_ASSERT(cutensorCreatePlan(
        ct->handle, &plan, op_desc, plan_pref, scratch_len
    ));

    /* Cleanup temporaries */
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(x_desc));
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(y_desc));
    CUTENSOR_ASSERT(cutensorDestroyOperationDescriptor(op_desc));
    CUTENSOR_ASSERT(cutensorDestroyPlanPreference(plan_pref));

    return {.plan = plan, .scratch_len = scratch_len};
}

/* ============================================================================
 * Plan Destruction
 * ============================================================================ */

void cutensor_permutate_plan_destroy(CutensorPermutatePlan p) {
    if (p.plan) {
        CUTENSOR_ASSERT(cutensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void cutensor_permutate(
    CutensorHandle wrapper,
    CutensorPermutatePlan p,
    const void* x_vals,
    void* y_vals,
    void* scratch,
    const void* alpha
) {
    auto* backend = unwrap_cutensor(wrapper);

    LOG_DEBUG("cutensor_permutate: plan=%p, x_vals=%p, y_vals=%p, alpha=%p, stream=%p",
              (void*)p.plan, x_vals, y_vals, alpha, (void*)backend->stream);

    if (!p.plan) {
        LOG_ERROR("cutensor_permutate: plan is NULL!");
    }
    if (!x_vals) {
        LOG_ERROR("cutensor_permutate: x_vals is NULL!");
    }
    if (!y_vals) {
        LOG_ERROR("cutensor_permutate: y_vals is NULL!");
    }

    CUTENSOR_ASSERT(cutensorPermute(
        backend->handle,
        p.plan,
        alpha, x_vals, y_vals,
        backend->stream
    ));
}

#endif /* __CUTENSOR_PERMUTATE_H__ */
