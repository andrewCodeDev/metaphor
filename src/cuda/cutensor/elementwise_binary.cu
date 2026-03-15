/*
 * cutensor/elementwise_binary.cu - Elementwise binary operations with einsum broadcasting
 *
 * Uses cutensorElementwiseBinary for operations like D = A * C with einsum-style broadcasting.
 * This handles patterns like "ble,bld->bled" that the JIT kernel cannot handle.
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __CUTENSOR_ELEMENTWISE_BINARY_H__
#define __CUTENSOR_ELEMENTWISE_BINARY_H__

#include "backend.cu"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    cutensorPlan_t plan;
    len_t scratch_len;
} CutensorElementwiseBinaryPlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

CutensorElementwiseBinaryPlan cutensor_elementwise_binary_plan_create(
    CutensorHandle wrapper,
    Dtype id,
    const len_t* a_dims,
    const u8* a_syms,
    len_t a_dims_len,
    const len_t* c_dims,
    const u8* c_syms,
    len_t c_dims_len,
    const len_t* d_dims,
    const u8* d_syms,
    len_t d_dims_len,
    BinaryOp op
) {
    CHECK_INVARIANT(a_dims_len > 0, "Zero length dimensions for elementwise input A");
    CHECK_INVARIANT(c_dims_len > 0, "Zero length dimensions for elementwise input C");
    CHECK_INVARIANT(d_dims_len > 0, "Zero length dimensions for elementwise output D");

    auto* ct = unwrap_cutensor(wrapper);
    const auto data_type = cutensor_dtype(id);
    const auto op_type = cutensor_op_from_binary(op);

    /* Build dimension arrays (reversed for column-major) */
    BoundedArray<i64> a_dims_arr(a_dims, a_dims_len, true);
    BoundedArray<i32> a_syms_arr(a_syms, a_dims_len, true);
    BoundedArray<i64> c_dims_arr(c_dims, c_dims_len, true);
    BoundedArray<i32> c_syms_arr(c_syms, c_dims_len, true);
    BoundedArray<i64> d_dims_arr(d_dims, d_dims_len, true);
    BoundedArray<i32> d_syms_arr(d_syms, d_dims_len, true);

    /* Create tensor descriptors (temporary) */
    cutensorTensorDescriptor_t a_desc;
    CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
        ct->handle, &a_desc,
        a_dims_arr.len, a_dims_arr.ptr(), NULL,
        data_type, CUTENSOR_ALIGNMENT
    ));

    cutensorTensorDescriptor_t c_desc;
    CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
        ct->handle, &c_desc,
        c_dims_arr.len, c_dims_arr.ptr(), NULL,
        data_type, CUTENSOR_ALIGNMENT
    ));

    cutensorTensorDescriptor_t d_desc;
    CUTENSOR_ASSERT(cutensorCreateTensorDescriptor(
        ct->handle, &d_desc,
        d_dims_arr.len, d_dims_arr.ptr(), NULL,
        data_type, CUTENSOR_ALIGNMENT
    ));

    /* Create elementwise binary descriptor */
    cutensorOperationDescriptor_t op_desc;
    CUTENSOR_ASSERT(cutensorCreateElementwiseBinary(
        ct->handle, &op_desc,
        a_desc, a_syms_arr.ptr(), CUTENSOR_OP_IDENTITY,
        c_desc, c_syms_arr.ptr(), CUTENSOR_OP_IDENTITY,
        d_desc, d_syms_arr.ptr(),
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
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(a_desc));
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(c_desc));
    CUTENSOR_ASSERT(cutensorDestroyTensorDescriptor(d_desc));
    CUTENSOR_ASSERT(cutensorDestroyOperationDescriptor(op_desc));
    CUTENSOR_ASSERT(cutensorDestroyPlanPreference(plan_pref));

    return {.plan = plan, .scratch_len = scratch_len};
}

/* ============================================================================
 * Plan Destruction
 * ============================================================================ */

void cutensor_elementwise_binary_plan_destroy(CutensorElementwiseBinaryPlan p) {
    if (p.plan) {
        CUTENSOR_ASSERT(cutensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void cutensor_elementwise_binary(
    CutensorHandle wrapper,
    CutensorElementwiseBinaryPlan p,
    const void* a_vals,
    const void* c_vals,
    void* d_vals,
    void* scratch,
    const void* alpha,
    const void* gamma
) {
    auto* backend = unwrap_cutensor(wrapper);

    CUTENSOR_ASSERT(cutensorElementwiseBinaryExecute(
        backend->handle,
        p.plan,
        alpha, a_vals,
        gamma, c_vals,
        d_vals,
        backend->stream
    ));
}

#endif /* __CUTENSOR_ELEMENTWISE_BINARY_H__ */
