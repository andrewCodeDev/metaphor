/*
 * hiptensor/contraction.cpp - Tensor contraction operations (stateless)
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __HIPTENSOR_CONTRACTION_H__
#define __HIPTENSOR_CONTRACTION_H__

#include "backend.cpp"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    hiptensorPlan_t plan;
    len_t scratch_len;
} HiptensorContractionPlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

HiptensorContractionPlan hiptensor_contraction_plan_create(
    HiptensorHandle wrapper,
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

    auto* ct = unwrap_hiptensor(wrapper);
    const auto data_type = hiptensor_dtype(id);

    /* Build dimension arrays (reversed for column-major) */
    BoundedArray<i64> a_dims(x_dims, x_dims_len, true);
    BoundedArray<i32> a_syms(x_syms, x_dims_len, true);
    BoundedArray<i64> b_dims(y_dims, y_dims_len, true);
    BoundedArray<i32> b_syms(y_syms, y_dims_len, true);
    BoundedArray<i64> c_dims(z_dims, z_dims_len, true);
    BoundedArray<i32> c_syms(z_syms, z_dims_len, true);

    /* Create tensor descriptors (temporary - destroyed after plan creation) */
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

    hiptensorTensorDescriptor_t z_desc;
    HIPTENSOR_ASSERT(hiptensorCreateTensorDescriptor(
        ct->handle, &z_desc,
        c_dims.len, c_dims.ptr(), NULL,
        data_type, HIPTENSOR_ALIGNMENT
    ));

    /* Create contraction descriptor */
    hiptensorOperationDescriptor_t op_desc;
    HIPTENSOR_ASSERT(hiptensorCreateContraction(
        ct->handle, &op_desc,
        x_desc, a_syms.ptr(), HIPTENSOR_OP_IDENTITY,
        y_desc, b_syms.ptr(), HIPTENSOR_OP_IDENTITY,
        z_desc, c_syms.ptr(), HIPTENSOR_OP_IDENTITY,
        z_desc, c_syms.ptr(),
        hiptensor_compute_desc(id)
    ));

    /* Create plan preference */
    hiptensorPlanPreference_t plan_pref;
    HIPTENSOR_ASSERT(hiptensorCreatePlanPreference(
        ct->handle, &plan_pref,
        HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE
    ));

    /* Estimate workspace */
    len_t scratch_len;
    HIPTENSOR_ASSERT(hiptensorEstimateWorkspaceSize(
        ct->handle, op_desc, plan_pref,
        HIPTENSOR_WORKSPACE_DEFAULT, &scratch_len
    ));

    /* Create plan */
    hiptensorPlan_t plan;
    HIPTENSOR_ASSERT(hiptensorCreatePlan(
        ct->handle, &plan, op_desc, plan_pref, scratch_len
    ));

    /* Do NOT destroy descriptors - plan stores raw pointers to them.
     * They will be freed when the plan is destroyed. */

    return {.plan = plan, .scratch_len = scratch_len};
}

/* ============================================================================
 * Plan Destruction
 * ============================================================================ */

void hiptensor_contraction_plan_destroy(HiptensorContractionPlan p) {
    if (p.plan) {
        HIPTENSOR_ASSERT(hiptensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void hiptensor_contract(
    HiptensorHandle wrapper,
    HiptensorContractionPlan p,
    const void* x_vals,
    const void* y_vals,
    void* z_vals,
    void* scratch,
    const void* alpha,
    const void* beta
) {
    auto* backend = unwrap_hiptensor(wrapper);

    HIPTENSOR_ASSERT(hiptensorContract(
        backend->handle,
        p.plan,
        alpha, x_vals, y_vals,
        beta, z_vals, z_vals,
        scratch, p.scratch_len,
        backend->stream
    ));
}

#endif /* __HIPTENSOR_CONTRACTION_H__ */
