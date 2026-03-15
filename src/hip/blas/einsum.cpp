/*
 * blas/einsum.cpp - Einsum dispatch: route batched GEMM to hipBLAS
 *
 * Classifies einsum dimensions (batch, M, N, K) and dispatches to
 * hipblasGemmStridedBatchedEx. Falls back to hipTensor for patterns
 * that don't decompose to GEMM.
 *
 * Algorithm adapted from PyTorch's sumproduct_pair() in
 * aten/src/ATen/native/Linear.cpp (BSD-3-Clause).
 */

#ifndef __BLAS_EINSUM_CPP__
#define __BLAS_EINSUM_CPP__

#include "../core/assert.h"
#include "../core/cast.h"
#include "../core/includes.h"
#include "../interop.h"
#include "../logging.h"

/* Maximum number of dims in any single category (batch/M/N/K) */
#define EINSUM_MAX_CAT 8

/* ============================================================================
 * Dimension Classification
 *
 * For each symbol in the einsum expression, classify as:
 *   batch: in x AND y AND z  (lro — left, right, output)
 *   M:     in x AND z only   (lo  — left, output)
 *   N:     in y AND z only   (ro  — right, output)
 *   K:     in x AND y only   (sum — contracted)
 * ============================================================================ */

struct EinsumClassification {
    /* Symbol indices and sizes for each category */
    u8     batch_syms[EINSUM_MAX_CAT];
    len_t  batch_sizes[EINSUM_MAX_CAT];
    int    n_batch;

    u8     m_syms[EINSUM_MAX_CAT];
    len_t  m_sizes[EINSUM_MAX_CAT];
    int    n_m;

    u8     n_syms[EINSUM_MAX_CAT];
    len_t  n_sizes[EINSUM_MAX_CAT];
    int    n_n;

    u8     k_syms[EINSUM_MAX_CAT];
    len_t  k_sizes[EINSUM_MAX_CAT];
    int    n_k;

    /* Flattened sizes */
    len_t B, M, N, K;
};

static bool sym_in(u8 sym, const u8* syms, len_t len) {
    for (len_t i = 0; i < len; i++) {
        if (syms[i] == sym) return true;
    }
    return false;
}

static len_t sym_size(u8 sym, const u8* syms, const len_t* dims, len_t len) {
    for (len_t i = 0; i < len; i++) {
        if (syms[i] == sym) return dims[i];
    }
    return 0;
}

static EinsumClassification classify_einsum(
    const DenseCore* x, const u8* x_syms,
    const DenseCore* y, const u8* y_syms,
    const DenseCore* z, const u8* z_syms
) {
    EinsumClassification c = {};

    /* Walk x symbols — each must be in exactly one category */
    for (len_t i = 0; i < x->shape.len; i++) {
        u8 s = x_syms[i];
        bool in_y = sym_in(s, y_syms, y->shape.len);
        bool in_z = sym_in(s, z_syms, z->shape.len);

        if (in_y && in_z) {
            /* batch: in all three */
            c.batch_syms[c.n_batch] = s;
            c.batch_sizes[c.n_batch] = x->shape.buffer[i];
            c.n_batch++;
        } else if (in_z && !in_y) {
            /* M: in x and z only */
            c.m_syms[c.n_m] = s;
            c.m_sizes[c.n_m] = x->shape.buffer[i];
            c.n_m++;
        } else if (in_y && !in_z) {
            /* K: in x and y only (contracted) */
            c.k_syms[c.n_k] = s;
            c.k_sizes[c.n_k] = x->shape.buffer[i];
            c.n_k++;
        } else {
            /* Symbol only in x — not a standard GEMM pattern */
            return c; /* n_k will be 0, caller checks validity */
        }
    }

    /* Walk y symbols for N dims (in y and z only) */
    for (len_t i = 0; i < y->shape.len; i++) {
        u8 s = y_syms[i];
        bool in_x = sym_in(s, x_syms, x->shape.len);
        bool in_z = sym_in(s, z_syms, z->shape.len);

        if (!in_x && in_z) {
            /* N: in y and z only */
            c.n_syms[c.n_n] = s;
            c.n_sizes[c.n_n] = y->shape.buffer[i];
            c.n_n++;
        } else if (!in_x && !in_z) {
            /* Symbol only in y — not standard GEMM */
            c.n_k = 0;
            return c;
        }
        /* batch and K already counted from x walk */
    }

    /* Compute flattened sizes */
    c.B = 1;
    for (int i = 0; i < c.n_batch; i++) c.B *= c.batch_sizes[i];
    c.M = 1;
    for (int i = 0; i < c.n_m; i++) c.M *= c.m_sizes[i];
    c.N = 1;
    for (int i = 0; i < c.n_n; i++) c.N *= c.n_sizes[i];
    c.K = 1;
    for (int i = 0; i < c.n_k; i++) c.K *= c.k_sizes[i];

    return c;
}

/* ============================================================================
 * Layout Validation
 *
 * Check that tensors have the expected memory layout for GEMM dispatch.
 * We need: x = [batch..., M..., K...] contiguous in that order
 *          y = [batch..., K..., N...] OR [batch..., N..., K...] contiguous
 *          z = [batch..., M..., N...] contiguous in that order
 *
 * Rather than requiring specific dim ordering, we compute strides from the
 * DenseCore and check if the memory is compatible with strided batched GEMM.
 * ============================================================================ */

struct GemmLayout {
    bool valid;
    /* After column-major swap (B^T @ A^T = C^T trick): */
    hipblasOperation_t transA;  /* for the hipBLAS "A" (our y) */
    hipblasOperation_t transB;  /* for the hipBLAS "B" (our x) */
    int lda;                    /* leading dim of hipBLAS A */
    int ldb;                    /* leading dim of hipBLAS B */
    int ldc;                    /* leading dim of hipBLAS C */
    long long strideA;          /* batch stride for hipBLAS A (elements) */
    long long strideB;          /* batch stride for hipBLAS B (elements) */
    long long strideC;          /* batch stride for hipBLAS C (elements) */
};

/*
 * Check if a tensor's dims are laid out as [batch..., rows..., cols...] in
 * row-major contiguous order. Returns the batch stride, row stride (leading
 * dim), and whether it's valid.
 *
 * Expected order of symbols in the tensor: batch_syms first, then row_syms,
 * then col_syms. Each group must be contiguous and in the same relative order
 * as they appear in the classification.
 */
static bool check_contiguous_order(
    const DenseCore* t, const u8* t_syms,
    const u8* expect_syms, int n_expect,
    int start_pos
) {
    for (int i = 0; i < n_expect; i++) {
        if (start_pos + i >= (int)t->shape.len) return false;
        if (t_syms[start_pos + i] != expect_syms[i]) return false;
    }
    return true;
}

static bool is_row_major_contiguous(const DenseCore* t) {
    /* Check strides match row-major contiguous layout */
    len_t expected_stride = 1;
    for (int i = (int)t->shape.len - 1; i >= 0; i--) {
        if (t->strides[i] != expected_stride) return false;
        expected_stride *= t->shape.buffer[i];
    }
    return true;
}

static GemmLayout compute_gemm_layout(
    const EinsumClassification& c,
    const DenseCore* x, const u8* x_syms,
    const DenseCore* y, const u8* y_syms,
    const DenseCore* z, const u8* z_syms
) {
    GemmLayout g = {.valid = false};

    /* All tensors must be contiguous */
    if (!is_row_major_contiguous(x) ||
        !is_row_major_contiguous(y) ||
        !is_row_major_contiguous(z)) {
        return g;
    }

    /* Check x layout: [batch..., M..., K...] */
    int pos = 0;
    if (!check_contiguous_order(x, x_syms, c.batch_syms, c.n_batch, pos)) return g;
    pos += c.n_batch;
    if (!check_contiguous_order(x, x_syms, c.m_syms, c.n_m, pos)) return g;
    pos += c.n_m;
    if (!check_contiguous_order(x, x_syms, c.k_syms, c.n_k, pos)) return g;

    /* Check z layout: [batch..., M..., N...] */
    pos = 0;
    if (!check_contiguous_order(z, z_syms, c.batch_syms, c.n_batch, pos)) return g;
    pos += c.n_batch;
    if (!check_contiguous_order(z, z_syms, c.m_syms, c.n_m, pos)) return g;
    pos += c.n_m;
    if (!check_contiguous_order(z, z_syms, c.n_syms, c.n_n, pos)) return g;

    /* Check y layout: either [batch..., K..., N...] or [batch..., N..., K...] */
    pos = 0;
    if (!check_contiguous_order(y, y_syms, c.batch_syms, c.n_batch, pos)) return g;
    pos += c.n_batch;

    bool y_is_KN = check_contiguous_order(y, y_syms, c.k_syms, c.n_k, pos) &&
                   check_contiguous_order(y, y_syms, c.n_syms, c.n_n, pos + c.n_k);
    bool y_is_NK = check_contiguous_order(y, y_syms, c.n_syms, c.n_n, pos) &&
                   check_contiguous_order(y, y_syms, c.k_syms, c.n_k, pos + c.n_n);

    if (!y_is_KN && !y_is_NK) return g;

    /*
     * hipBLAS column-major convention:
     * For row-major C = A @ B, we compute C^T = B^T @ A^T in col-major.
     * Swap A↔B, swap M↔N.
     *
     * Row-major x is [B, M, K] → col-major view is [K, M, B] → "B^T" = [K, M] per batch
     * Row-major y is [B, K, N] → col-major view is [N, K, B] → "A^T" = [N, K] per batch
     * Row-major z is [B, M, N] → col-major view is [N, M, B] → "C^T" = [N, M] per batch
     *
     * hipBLAS call: C^T(N×M) = A^T(N×K) × B^T(K×M)
     *   m_blas = N, n_blas = M, k_blas = K
     *   A_blas = y_ptr, B_blas = x_ptr, C_blas = z_ptr
     *
     * For y = [B, K, N] (row-major): col-major sees [N, K] per batch
     *   This is already N×K with lda=N → transA = N (no transpose)
     * For y = [B, N, K] (row-major): col-major sees [K, N] per batch
     *   We need N×K but have K×N → transA = T
     *
     * For x = [B, M, K] (row-major): col-major sees [K, M] per batch
     *   This is already K×M with ldb=K → transB = N (no transpose)
     */

    if (y_is_KN) {
        /* y=[B,K,N] → col-major [N,K] per batch → already N×K → no transpose */
        g.transA = HIPBLAS_OP_N;
        g.lda = (int)c.N;  /* leading dim = N (innermost stride in col-major) */
    } else {
        /* y=[B,N,K] → col-major [K,N] per batch → need transpose to get N×K */
        g.transA = HIPBLAS_OP_T;
        g.lda = (int)c.K;  /* leading dim of the stored [K,N] matrix */
    }

    /* x=[B,M,K] → col-major [K,M] per batch → already K×M → no transpose */
    g.transB = HIPBLAS_OP_N;
    g.ldb = (int)c.K;  /* leading dim = K */

    /* z=[B,M,N] → col-major [N,M] per batch */
    g.ldc = (int)c.N;

    /* Batch strides in elements */
    g.strideA = (long long)(c.K * c.N);  /* y batch stride when y=[B,K,N] or [B,N,K] */
    g.strideB = (long long)(c.M * c.K);  /* x batch stride */
    g.strideC = (long long)(c.M * c.N);  /* z batch stride */

    g.valid = true;
    return g;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

/*
 * Quick check: can this einsum pattern be dispatched to hipBLAS?
 * Returns true for standard GEMM/batched-GEMM patterns.
 */
static bool hip_einsum_can_dispatch(
    const DenseCore* x, const u8* x_syms,
    const DenseCore* y, const u8* y_syms,
    const DenseCore* z, const u8* z_syms
) {
    EinsumClassification c = classify_einsum(x, x_syms, y, y_syms, z, z_syms);

    /* Must have at least M, N, K dims for a GEMM */
    if (c.n_m == 0 || c.n_n == 0 || c.n_k == 0) return false;

    /* Total dims must account for all symbols */
    int total = c.n_batch + c.n_m + c.n_n + c.n_k;
    if (total != (int)x->shape.len + (int)y->shape.len - c.n_batch - c.n_k) {
        /* Unexpected — repeated symbols or traces */
        return false;
    }

    /* Check layout compatibility */
    GemmLayout g = compute_gemm_layout(c, x, x_syms, y, y_syms, z, z_syms);
    return g.valid;
}

/*
 * Dispatch einsum contraction to hipblasGemmStridedBatchedEx.
 * Caller must verify hip_einsum_can_dispatch() first.
 */
static void hip_einsum_contract(
    hipblasHandle_t handle,
    hipStream_t stream,
    Dtype dtype,
    const DenseCore* x, const u8* x_syms,
    const DenseCore* y, const u8* y_syms,
    DenseCore* z, const u8* z_syms
) {
    EinsumClassification c = classify_einsum(x, x_syms, y, y_syms, z, z_syms);
    GemmLayout g = compute_gemm_layout(c, x, x_syms, y, y_syms, z, z_syms);

    CHECK_INVARIANT(g.valid, "hip_einsum_contract called with invalid layout");

    /* Ensure hipBLAS is on the right stream */
    HIPBLAS_ASSERT(hipblasSetStream(handle, stream));

    /* Map dtype to hipBLAS types */
    hipDataType data_type;
    hipblasComputeType_t compute_type;
    switch (dtype) {
        case DTYPE_BF16:
            data_type = HIP_R_16BF;
            compute_type = HIPBLAS_COMPUTE_32F;
            break;
        case DTYPE_F16:
            data_type = HIP_R_16F;
            compute_type = HIPBLAS_COMPUTE_32F;
            break;
        case DTYPE_F32:
            data_type = HIP_R_32F;
            compute_type = HIPBLAS_COMPUTE_32F;
            break;
        case DTYPE_F64:
            data_type = HIP_R_64F;
            compute_type = HIPBLAS_COMPUTE_64F;
            break;
        default:
            SYSTEM_EXIT("Unsupported dtype for hipBLAS GEMM");
            return;
    }

    /* Alpha/beta scalars — hipBLAS reads these as compute type (always f32 or f64) */
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    const void* alpha = (dtype == DTYPE_F64) ? (const void*)&alpha_d : (const void*)&alpha_f;
    const void* beta  = (dtype == DTYPE_F64) ? (const void*)&beta_d  : (const void*)&beta_f;

    /*
     * Column-major trick: C^T = A_blas @ B_blas
     *   A_blas = y (the right operand), B_blas = x (the left operand)
     *   m_blas = N, n_blas = M
     */
    int m_blas = (int)c.N;
    int n_blas = (int)c.M;
    int k_blas = (int)c.K;
    int batch  = (int)c.B;

    LOG_INFO("einsum dispatch: B=%d M=%d N=%d K=%d transA=%c transB=%c lda=%d ldb=%d ldc=%d",
             batch, (int)c.M, (int)c.N, (int)c.K,
             g.transA == HIPBLAS_OP_N ? 'N' : 'T',
             g.transB == HIPBLAS_OP_N ? 'N' : 'T',
             g.lda, g.ldb, g.ldc);

    HIPBLAS_ASSERT(hipblasGemmStridedBatchedEx(
        handle,
        g.transA,              /* op for A_blas (our y) */
        g.transB,              /* op for B_blas (our x) */
        m_blas,                /* m = N */
        n_blas,                /* n = M */
        k_blas,                /* k = K */
        alpha,
        y->data,               /* A_blas = y */
        data_type,
        g.lda,
        g.strideA,
        x->data,               /* B_blas = x */
        data_type,
        g.ldb,
        g.strideB,
        beta,
        z->data,               /* C_blas = z */
        data_type,
        g.ldc,
        g.strideC,
        batch,
        compute_type,
        HIPBLAS_GEMM_DEFAULT
    ));
}

#endif /* __BLAS_EINSUM_CPP__ */
