/*
 * hiptensor/reduce.cpp - Tensor reduction operations (stateless)
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __HIPTENSOR_REDUCE_H__
#define __HIPTENSOR_REDUCE_H__

#include "backend.cpp"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    hiptensorPlan_t plan;
    len_t scratch_len;
} HiptensorReducePlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

HiptensorReducePlan hiptensor_reduce_plan_create(
    HiptensorHandle wrapper,
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

    auto* ct = unwrap_hiptensor(wrapper);
    const auto data_type = hiptensor_dtype(id);
    const auto op_type = hiptensor_op_from_binary(op);

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

    /* Create reduction descriptor.
     * Use F32 compute for half types — hipTensor on RDNA3 produces zeros
     * with native half compute descriptors for reduction. */
    auto compute = (id == DTYPE_BF16 || id == DTYPE_F16)
                 ? HIPTENSOR_COMPUTE_DESC_32F
                 : hiptensor_compute_desc(id);

    hiptensorOperationDescriptor_t op_desc;
    HIPTENSOR_ASSERT(hiptensorCreateReduction(
        ct->handle, &op_desc,
        x_desc, a_syms.ptr(), HIPTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(), HIPTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(),
        op_type, compute
    ));

    /* Create plan */
    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_ASSERT(hiptensorCreatePlanPreference(
        ct->handle, &plan_pref,
        HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE
    ));

    len_t scratch_len;
    HIPTENSOR_ASSERT(hiptensorEstimateWorkspaceSize(
        ct->handle, op_desc, plan_pref,
        HIPTENSOR_WORKSPACE_DEFAULT, &scratch_len
    ));

    hiptensorPlan_t plan;
    HIPTENSOR_ASSERT(hiptensorCreatePlan(
        ct->handle, &plan, op_desc, plan_pref, scratch_len
    ));

    /* Do NOT destroy descriptors or plan preference - plan stores raw pointers
     * to all of them (mOpDesc, mPref). They are freed by PlanCache's destructor
     * when the plan is evicted or the cache is destroyed. */

    return {.plan = plan, .scratch_len = scratch_len};
}

/* ============================================================================
 * Plan Destruction
 * ============================================================================ */

void hiptensor_reduce_plan_destroy(HiptensorReducePlan p) {
    if (p.plan) {
        HIPTENSOR_ASSERT(hiptensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void hiptensor_reduce(
    HiptensorHandle wrapper,
    HiptensorReducePlan p,
    const void* x_vals,
    void* y_vals,
    void* scratch,
    const void* alpha,
    const void* beta
) {
    auto* backend = unwrap_hiptensor(wrapper);

    HIPTENSOR_ASSERT(hiptensorReduce(
        backend->handle,
        p.plan,
        alpha, x_vals,
        beta, y_vals, y_vals,
        scratch, p.scratch_len,
        backend->stream
    ));
}

#endif /* __HIPTENSOR_REDUCE_H__ */
