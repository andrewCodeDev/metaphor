/*
 * cutensor/reduce.cu - Tensor reduction operations (stateless)
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __CUTENSOR_REDUCE_H__
#define __CUTENSOR_REDUCE_H__

#include "backend.cu"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    cutensorPlan_t plan;
    len_t scratch_len;
} CutensorReducePlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

CutensorReducePlan cutensor_reduce_plan_create(
    CutensorHandle wrapper,
    Dtype id,
    const len_t* src_dims,
    const u8* src_syms,
    len_t src_dims_len,
    const len_t* dst_dims,
    const u8* dst_syms,
    len_t dst_dims_len,
    BinaryOp op
) {
    CHECK_INVARIANT(dst_dims_len > 0, "Zero length dimensions for reduce output");
    CHECK_INVARIANT(src_dims_len > dst_dims_len, "Reduction requires fewer output dimensions");

    auto* ct = unwrap_cutensor(wrapper);
    const auto data_type = cutensor_dtype(id);
    const auto op_type = cutensor_op_from_binary(op);

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

    /* Create reduction descriptor */
    cutensorOperationDescriptor_t op_desc;
    CUTENSOR_ASSERT(cutensorCreateReduction(
        ct->handle, &op_desc,
        x_desc, a_syms.ptr(), CUTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(), CUTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(),
        op_type, cutensor_compute_desc(id)
    ));

    /* Create plan */
    cutensorPlanPreference_t plan_pref;
    CUTENSOR_ASSERT(cutensorCreatePlanPreference(
        ct->handle, &plan_pref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE
    ));

    len_t scratch_len;
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

void cutensor_reduce_plan_destroy(CutensorReducePlan p) {
    if (p.plan) {
        CUTENSOR_ASSERT(cutensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void cutensor_reduce(
    CutensorHandle wrapper,
    CutensorReducePlan p,
    const void* x_vals,
    void* y_vals,
    void* scratch,
    const void* alpha,
    const void* beta
) {
    auto* backend = unwrap_cutensor(wrapper);

    CUTENSOR_ASSERT(cutensorReduce(
        backend->handle,
        p.plan,
        alpha, x_vals,
        beta, y_vals, y_vals,
        scratch, p.scratch_len,
        backend->stream
    ));
}

#endif /* __CUTENSOR_REDUCE_H__ */
