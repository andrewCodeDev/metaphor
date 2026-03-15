/*
 * hiptensor/elementwise_binary.cpp - Elementwise binary operations with einsum broadcasting
 *
 * Uses hiptensorElementwiseBinary for operations like D = A * C with einsum-style broadcasting.
 * This handles patterns like "ble,bld->bled" that the JIT kernel cannot handle.
 *
 * Plan lifecycle: create -> execute -> destroy
 * Caching is handled by the device layer, not here.
 */

#ifndef __HIPTENSOR_ELEMENTWISE_BINARY_H__
#define __HIPTENSOR_ELEMENTWISE_BINARY_H__

#include "backend.cpp"

/* ============================================================================
 * Plan Result
 * ============================================================================ */

typedef struct {
    hiptensorPlan_t plan;
    len_t scratch_len;
} HiptensorElementwiseBinaryPlan;

/* ============================================================================
 * Plan Creation
 * ============================================================================ */

HiptensorElementwiseBinaryPlan hiptensor_elementwise_binary_plan_create(
    HiptensorHandle wrapper,
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

    auto* ct = unwrap_hiptensor(wrapper);
    const auto data_type = hiptensor_dtype(id);
    const auto op_type = hiptensor_op_from_binary(op);

    /* Build dimension arrays (reversed for column-major) */
    BoundedArray<i64> a_dims_arr(a_dims, a_dims_len, true);
    BoundedArray<i32> a_syms_arr(a_syms, a_dims_len, true);
    BoundedArray<i64> c_dims_arr(c_dims, c_dims_len, true);
    BoundedArray<i32> c_syms_arr(c_syms, c_dims_len, true);
    BoundedArray<i64> d_dims_arr(d_dims, d_dims_len, true);
    BoundedArray<i32> d_syms_arr(d_syms, d_dims_len, true);

    /* Create tensor descriptors (temporary) */
    hiptensorTensorDescriptor_t a_desc;
    HIPTENSOR_ASSERT(hiptensorCreateTensorDescriptor(
        ct->handle, &a_desc,
        a_dims_arr.len, a_dims_arr.ptr(), NULL,
        data_type, HIPTENSOR_ALIGNMENT
    ));

    hiptensorTensorDescriptor_t c_desc;
    HIPTENSOR_ASSERT(hiptensorCreateTensorDescriptor(
        ct->handle, &c_desc,
        c_dims_arr.len, c_dims_arr.ptr(), NULL,
        data_type, HIPTENSOR_ALIGNMENT
    ));

    hiptensorTensorDescriptor_t d_desc;
    HIPTENSOR_ASSERT(hiptensorCreateTensorDescriptor(
        ct->handle, &d_desc,
        d_dims_arr.len, d_dims_arr.ptr(), NULL,
        data_type, HIPTENSOR_ALIGNMENT
    ));

    /* Create elementwise binary descriptor.
     * Use F32 compute for half types — hipTensor on RDNA3 produces zeros
     * with native half compute descriptors for elementwise binary. */
    auto compute = (id == DTYPE_BF16 || id == DTYPE_F16)
                 ? HIPTENSOR_COMPUTE_DESC_32F
                 : hiptensor_compute_desc(id);

    hiptensorOperationDescriptor_t op_desc;
    HIPTENSOR_ASSERT(hiptensorCreateElementwiseBinary(
        ct->handle, &op_desc,
        a_desc, a_syms_arr.ptr(), HIPTENSOR_OP_IDENTITY,
        c_desc, c_syms_arr.ptr(), HIPTENSOR_OP_IDENTITY,
        d_desc, d_syms_arr.ptr(),
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

void hiptensor_elementwise_binary_plan_destroy(HiptensorElementwiseBinaryPlan p) {
    if (p.plan) {
        HIPTENSOR_ASSERT(hiptensorDestroyPlan(p.plan));
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

void hiptensor_elementwise_binary(
    HiptensorHandle wrapper,
    HiptensorElementwiseBinaryPlan p,
    const void* a_vals,
    const void* c_vals,
    void* d_vals,
    void* scratch,
    const void* alpha,
    const void* gamma
) {
    auto* backend = unwrap_hiptensor(wrapper);

    HIPTENSOR_ASSERT(hiptensorElementwiseBinaryExecute(
        backend->handle,
        p.plan,
        alpha, a_vals,
        gamma, c_vals,
        d_vals,
        backend->stream
    ));
}

#endif /* __HIPTENSOR_ELEMENTWISE_BINARY_H__ */
