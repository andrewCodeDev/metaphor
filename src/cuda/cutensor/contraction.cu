/*
 * cutensor/contraction.cu - Tensor contraction operations (stateless)
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __CUTENSOR_CONTRACTION_H__
#define __CUTENSOR_CONTRACTION_H__

#include "backend.cu"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    cutensorPlan_t plan;
    len_t scratch_len;
} CutensorContractionPlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

CutensorContractionPlan cutensor_contraction_plan_create(
    CutensorHandle wrapper,
    Dtype id,
    const len_t* x_dims,
    const u8* x_syms,
    len_t x_dims_len,
    const len_t* y_dims,
    const u8* y_syms,
    len_t y_dims_len,
    const len_t* z_dims,
    const u8* z_syms,
    len_t z_dims_len
) {
    CHECK_INVARIANT(x_dims_len > 0, "Zero length dimensions for x in contraction");
    CHECK_INVARIANT(y_dims_len > 0, "Zero length dimensions for y in contraction");

    auto* ct = unwrap_cutensor(wrapper);
    const auto data_type = cutensor_dtype(id);

    /* Build dimension arrays (reversed for column-major) */
    BoundedArray<i64> a_dims(x_dims, x_dims_len, true);
    BoundedArray<i32> a_syms(x_syms, x_dims_len, true);
    BoundedArray<i64> b_dims(y_dims, y_dims_len, true);
    BoundedArray<i32> b_syms(y_syms, y_dims_len, true);
    BoundedArray<i64> c_dims(z_dims, z_dims_len, true);
    BoundedArray<i32> c_syms(z_syms, z_dims_len, true);

    /* Create tensor descriptors (temporary - destroyed after plan creation) */
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

    cutensorTensorDescriptor_t z_desc;
    CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
        ct->handle, &z_desc,
        c_dims.len, c_dims.ptr(), NULL,
        data_type, CUTENSOR_ALIGNMENT
    ));

    /* Create contraction descriptor */
    cutensorOperationDescriptor_t op_desc;
    CUTENSOR_ASSERT(cutensorCreateContraction(
        ct->handle, &op_desc,
        x_desc, a_syms.ptr(), CUTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(), CUTENSOR_OP_IDENTITY,
        z_desc, c_syms.ptr(), CUTENSOR_OP_IDENTITY,
        z_desc, c_syms.ptr(),
        cutensor_compute_desc(id)
    ));

    /* Create plan preference */
    cutensorPlanPreference_t plan_pref;
    CUTENSOR_ASSERT(cutensorCreatePlanPreference(
        ct->handle, &plan_pref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE
    ));

    /* Estimate workspace */
    len_t scratch_len;
    CUTENSOR_ASSERT(cutensorEstimateWorkspaceSize(
        ct->handle, op_desc, plan_pref,
        CUTENSOR_WORKSPACE_DEFAULT, &scratch_len
    ));

    /* Create plan */
    cutensorPlan_t plan;
    CUTENSOR_ASSERT(cutensorCreatePlan(
        ct->handle, &plan, op_desc, plan_pref, scratch_len
    ));

    /* Cleanup temporaries - plan captures what it needs */
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(x_desc));
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(y_desc));
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(z_desc));
    CUTENSOR_ASSERT(cutensorDestroyOperationDescriptor(op_desc));
    CUTENSOR_ASSERT(cutensorDestroyPlanPreference(plan_pref));

    return {.plan = plan, .scratch_len = scratch_len};
}

/* ============================================================================
 * Plan Destruction
 * ============================================================================ */

void cutensor_contraction_plan_destroy(CutensorContractionPlan p) {
    if (p.plan) {
        CUTENSOR_ASSERT(cutensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void cutensor_contract(
    CutensorHandle wrapper,
    CutensorContractionPlan p,
    const void* x_vals,
    const void* y_vals,
    void* z_vals,
    void* scratch,
    const void* alpha,
    const void* beta
) {
    auto* backend = unwrap_cutensor(wrapper);

    CUTENSOR_ASSERT(cutensorContract(
        backend->handle,
        p.plan,
        alpha, x_vals, y_vals,
        beta, z_vals, z_vals,
        scratch, p.scratch_len,
        backend->stream
    ));
}

#endif /* __CUTENSOR_CONTRACTION_H__ */
