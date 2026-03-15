/*
 * nn/ssd_chunk_scan.cpp — Structured State Space Duality (SSD) kernels for Mamba-2 (HIP)
 *
 * Implements the chunked selective scan from:
 *   "Transformers are SSMs: Generalized Models and Efficient Algorithms
 *    Through Structured State Space Duality"
 *   Albert Gu and Tri Dao, 2024
 *   https://arxiv.org/abs/2405.21060
 *   https://github.com/state-spaces/mamba
 *   Licensed under Apache 2.0
 *
 * Ported from the Triton implementation by Tri Dao and Albert Gu.
 *
 * The SSD algorithm splits sequences into chunks and processes:
 *   1. chunk_cumsum:   cumulative sum of dt*A within each chunk
 *   2. bmm_chunk:      C @ B^T within each chunk (intra-chunk attention)
 *   3. chunk_state:    SSM states at chunk boundaries
 *   4. state_passing:  sequential scan across chunk boundaries
 *   5. chunk_scan:     combine intra-chunk and inter-chunk contributions
 *
 * This gives O(n_chunks) sequential steps instead of O(seq_len),
 * with intra-chunk work mapped to parallel matrix operations.
 *
 * Input layouts (row-major):
 *   x:  [batch, seqlen, nheads, headdim]
 *   dt: [batch, seqlen, nheads]
 *   A:  [nheads]
 *   B:  [batch, seqlen, ngroups, dstate]
 *   C:  [batch, seqlen, ngroups, dstate]
 */

#ifndef __NN_SSD_CHUNK_SCAN_H__
#define __NN_SSD_CHUNK_SCAN_H__

#include "utils.cpp"

/* ============================================================================
 * 1. Chunk Cumsum Forward
 *
 * Computes cumulative sum of dt*A within each chunk, with optional softplus.
 *
 * Grid:  dim3(batch, nheads, nchunks)
 * Block: 1 thread
 *
 * Each thread handles one (batch, head, chunk), scanning chunk_size steps.
 *
 * Input:
 *   dt: [batch, seqlen, nheads]
 *   A:  [nheads]
 * Output:
 *   dt_out:     [batch, nheads, nchunks, chunk_size]  (dt values, after softplus)
 *   dA_cumsum:  [batch, nheads, nchunks, chunk_size]  (cumulative dt*A)
 * ============================================================================ */

template<typename T>
__global__ void ssd_chunk_cumsum_forward_kernel(
    const T* __restrict__ dt,          // [batch, seqlen, nheads]
    const T* __restrict__ A,           // [nheads]
    T* __restrict__ dt_out,            // [batch, nheads, nchunks, chunk_size]
    T* __restrict__ dA_cumsum,         // [batch, nheads, nchunks, chunk_size]
    len_t seqlen,
    len_t nheads,
    len_t chunk_size,
    len_t nchunks,
    int dt_softplus
) {
    const len_t b = blockIdx.x;
    const len_t h = blockIdx.y;
    const len_t c = blockIdx.z;

    const float a_val = static_cast<float>(A[h]);
    float cumsum = 0.0f;

    // Output index base: [b, h, c, 0]
    const len_t out_base = b * nheads * nchunks * chunk_size
                         + h * nchunks * chunk_size
                         + c * chunk_size;

    for (len_t l = 0; l < chunk_size; l++) {
        const len_t seq_pos = c * chunk_size + l;
        if (seq_pos >= seqlen) {
            dt_out[out_base + l] = static_cast<T>(0.0f);
            dA_cumsum[out_base + l] = static_cast<T>(cumsum);
            continue;
        }

        // dt input: [b, seq_pos, h]
        float dt_val = static_cast<float>(dt[b * seqlen * nheads + seq_pos * nheads + h]);

        // Optional softplus: log(1 + exp(x)) for x <= 20, else x
        if (dt_softplus) {
            dt_val = (dt_val <= 20.0f) ? log1pf(expf(dt_val)) : dt_val;
        }

        dt_out[out_base + l] = static_cast<T>(dt_val);
        cumsum += dt_val * a_val;
        dA_cumsum[out_base + l] = static_cast<T>(cumsum);
    }
}

/* ============================================================================
 * 2. BMM Chunk Forward (C @ B^T within each chunk)
 *
 * Grid:  flat over all output elements
 * Block: 256 threads
 *
 * Each thread computes one element CB[b,c,g,l,s] = dot(C[b,c*cs+l,g,:], B[b,c*cs+s,g,:])
 *
 * Input:
 *   C: [batch, seqlen, ngroups, dstate]
 *   B: [batch, seqlen, ngroups, dstate]
 * Output:
 *   CB: [batch, nchunks, ngroups, chunk_size, chunk_size]
 * ============================================================================ */

template<typename T>
__global__ void ssd_bmm_chunk_forward_kernel(
    const T* __restrict__ C_in,   // [batch, seqlen, ngroups, dstate]
    const T* __restrict__ B_in,   // [batch, seqlen, ngroups, dstate]
    T* __restrict__ CB,           // [batch, nchunks, ngroups, chunk_size, chunk_size]
    len_t seqlen,
    len_t ngroups,
    len_t dstate,
    len_t chunk_size,
    len_t nchunks
) {
    const len_t total = gridDim.x * (len_t)blockDim.x;
    const len_t idx = blockIdx.x * (len_t)blockDim.x + threadIdx.x;

    // Total output: batch * nchunks * ngroups * chunk_size * chunk_size
    const len_t cs2 = chunk_size * chunk_size;
    const len_t per_batch = nchunks * ngroups * cs2;
    // Use grid-stride loop if needed, but typically we launch enough threads
    const len_t flat_total = gridDim.x * (len_t)blockDim.x;  // approximate

    if (idx >= per_batch * ((seqlen + chunk_size - 1) / chunk_size + 1)) return;

    // Decompose flat index: CB[b, c, g, l, s]
    const len_t s = idx % chunk_size;
    const len_t l = (idx / chunk_size) % chunk_size;
    const len_t g = (idx / cs2) % ngroups;
    const len_t c = (idx / (cs2 * ngroups)) % nchunks;
    const len_t b = idx / (cs2 * ngroups * nchunks);

    const len_t seq_l = c * chunk_size + l;
    const len_t seq_s = c * chunk_size + s;

    if (seq_l >= seqlen || seq_s >= seqlen) {
        CB[idx] = static_cast<T>(0.0f);
        return;
    }

    // C[b, seq_l, g, :] dot B[b, seq_s, g, :]
    const len_t c_base = b * seqlen * ngroups * dstate + seq_l * ngroups * dstate + g * dstate;
    const len_t b_base = b * seqlen * ngroups * dstate + seq_s * ngroups * dstate + g * dstate;

    float acc = 0.0f;
    for (len_t n = 0; n < dstate; n++) {
        acc += static_cast<float>(C_in[c_base + n]) * static_cast<float>(B_in[b_base + n]);
    }

    CB[idx] = static_cast<T>(acc);
}

/* ============================================================================
 * 3. Chunk State Forward
 *
 * Computes SSM states at chunk boundaries:
 *   states[b,c,h,d,n] = sum_l{ B[b,c*cs+l,g,n] *
 *                               exp(dA_cumsum[b,h,c,-1] - dA_cumsum[b,h,c,l]) *
 *                               dt[b,h,c,l] * x[b,c*cs+l,h,d] }
 *
 * Grid:  dim3(batch * nchunks, nheads, ceil(headdim * dstate / 256))
 * Block: 256 threads
 *
 * Input:
 *   B:          [batch, seqlen, ngroups, dstate]
 *   x:          [batch, seqlen, nheads, headdim]
 *   dt:         [batch, nheads, nchunks, chunk_size]
 *   dA_cumsum:  [batch, nheads, nchunks, chunk_size]
 * Output:
 *   states:     [batch, nchunks, nheads, headdim, dstate]
 * ============================================================================ */

template<typename T>
__global__ void ssd_chunk_state_forward_kernel(
    const T* __restrict__ B_in,       // [batch, seqlen, ngroups, dstate]
    const T* __restrict__ x,          // [batch, seqlen, nheads, headdim]
    const T* __restrict__ dt,         // [batch, nheads, nchunks, chunk_size]
    const T* __restrict__ dA_cumsum,  // [batch, nheads, nchunks, chunk_size]
    T* __restrict__ states,           // [batch, nchunks, nheads, headdim, dstate]
    len_t seqlen,
    len_t nheads,
    len_t headdim,
    len_t ngroups,
    len_t dstate,
    len_t chunk_size,
    len_t nchunks
) {
    const len_t bc = blockIdx.x;                    // batch * nchunks
    const len_t b = bc / nchunks;
    const len_t c = bc % nchunks;
    const len_t h = blockIdx.y;
    const len_t dn_idx = blockIdx.z * blockDim.x + threadIdx.x;  // headdim * dstate flat idx

    const len_t total_dn = headdim * dstate;
    if (dn_idx >= total_dn) return;

    const len_t d = dn_idx / dstate;
    const len_t n = dn_idx % dstate;
    const len_t g = h / (nheads / ngroups);  // group index for this head

    // dA_cumsum at end of chunk: [b, h, c, chunk_size-1]
    const len_t last_l = min(chunk_size, seqlen - c * chunk_size) - 1;
    const len_t da_base = b * nheads * nchunks * chunk_size
                        + h * nchunks * chunk_size
                        + c * chunk_size;
    const float dA_end = static_cast<float>(dA_cumsum[da_base + last_l]);

    float acc = 0.0f;
    const len_t chunk_len = min(chunk_size, seqlen - c * chunk_size);

    for (len_t l = 0; l < chunk_len; l++) {
        const len_t seq_pos = c * chunk_size + l;

        // decay = exp(dA_cumsum[end] - dA_cumsum[l])
        float dA_l = static_cast<float>(dA_cumsum[da_base + l]);
        float decay = expf(dA_end - dA_l);

        // dt[b, h, c, l]
        float dt_val = static_cast<float>(dt[da_base + l]);

        // B[b, seq_pos, g, n]
        float b_val = static_cast<float>(B_in[b * seqlen * ngroups * dstate
                                             + seq_pos * ngroups * dstate
                                             + g * dstate + n]);

        // x[b, seq_pos, h, d]
        float x_val = static_cast<float>(x[b * seqlen * nheads * headdim
                                          + seq_pos * nheads * headdim
                                          + h * headdim + d]);

        acc += b_val * decay * dt_val * x_val;
    }

    // states[b, c, h, d, n]
    const len_t state_idx = b * nchunks * nheads * headdim * dstate
                          + c * nheads * headdim * dstate
                          + h * headdim * dstate
                          + d * dstate + n;
    states[state_idx] = static_cast<T>(acc);
}

/* ============================================================================
 * 4. State Passing Forward
 *
 * Sequential scan across chunk boundaries (the core recurrence):
 *   state[c] = exp(dA_cs[c]) * state[c-1] + new_states[c]
 *
 * This is analogous to selective_scan but operates on n_chunks steps
 * (typically 8-32) instead of seq_len steps (typically 2048+).
 *
 * Grid:  dim3(ceil(dim / 256), batch, nheads)
 * Block: 256 threads
 *
 * Each thread handles one element of the flattened state (headdim * dstate),
 * scanning sequentially over nchunks.
 *
 * Input:
 *   states:           [batch, nchunks, nheads, dim]  (dim = headdim * dstate)
 *   dA_chunk_cumsum:  [batch, nheads, nchunks]       (cumsum at chunk end)
 *   initial_states:   [batch, nheads, dim]            (nullable)
 * Output:
 *   out:              [batch, nchunks, nheads, dim]   (passed-through states)
 *   final_states:     [batch, nheads, dim]
 * ============================================================================ */

template<typename T>
__global__ void ssd_state_passing_forward_kernel(
    const T* __restrict__ states,           // [batch, nchunks, nheads, dim]
    const T* __restrict__ dA_chunk_cumsum,  // [batch, nheads, nchunks]
    const T* __restrict__ initial_states,   // [batch, nheads, dim] or nullptr
    T* __restrict__ out,                    // [batch, nchunks, nheads, dim]
    T* __restrict__ final_states,           // [batch, nheads, dim]
    len_t nchunks,
    len_t nheads,
    len_t dim
) {
    const len_t di = blockIdx.x * blockDim.x + threadIdx.x;  // state dim index
    const len_t b = blockIdx.y;
    const len_t h = blockIdx.z;

    if (di >= dim) return;

    // Load initial state
    float running;
    if (initial_states != nullptr) {
        running = static_cast<float>(initial_states[b * nheads * dim + h * dim + di]);
    } else {
        running = 0.0f;
    }

    for (len_t c = 0; c < nchunks; c++) {
        // New state contribution: states[b, c, h, di]
        const len_t state_idx = b * nchunks * nheads * dim
                              + c * nheads * dim
                              + h * dim + di;
        float new_state = static_cast<float>(states[state_idx]);

        // Decay: exp(dA_chunk_cumsum[b, h, c])
        float dA_cs = static_cast<float>(dA_chunk_cumsum[b * nheads * nchunks
                                                        + h * nchunks + c]);
        float decay = expf(dA_cs);

        running = decay * running + new_state;

        // Store passed state for chunk_scan to use
        // out[b, c, h, di] = running (state BEFORE this chunk's contribution
        // is used by chunk_scan — the Triton impl stores the pre-update state)
        // Actually, Triton stores: out[0] = init, out[c] = state after chunk c-1
        // and final_states = state after last chunk
        if (c < nchunks - 1) {
            out[state_idx] = static_cast<T>(running);
        } else {
            // Last chunk: store to final_states
            final_states[b * nheads * dim + h * dim + di] = static_cast<T>(running);
        }
    }
}

/* ============================================================================
 * 5. Chunk Scan Forward
 *
 * Combines intra-chunk and inter-chunk contributions to produce output.
 *
 * For each output position (b, pos, h, d):
 *   c = pos / chunk_size,  l = pos % chunk_size
 *
 *   inter_chunk = sum_n{ C[b,pos,g,n] * prev_states[b,c,h,d,n] } * exp(dA_cs[l])
 *   intra_chunk = sum_{s=0}^{l} CB[b,c,g,l,s] * exp(dA_cs[l] - dA_cs[s])
 *                                * dt[b,h,c,s] * x[b,c*cs+s,h,d]
 *   out[b,pos,h,d] = inter_chunk + intra_chunk [+ D[h] * x[b,pos,h,d]]
 *
 * Grid:  dim3(batch * nchunks, nheads, ceil(headdim / blockDim.x))
 * Block: min(headdim, 256) threads
 *
 * Each thread handles one (b, c, h, d_idx), looping over chunk positions (l)
 * to emit chunk_size output values.
 *
 * Input:
 *   CB:           [batch, nchunks, ngroups, chunk_size, chunk_size]
 *   x:            [batch, seqlen, nheads, headdim]
 *   dt:           [batch, nheads, nchunks, chunk_size]
 *   dA_cumsum:    [batch, nheads, nchunks, chunk_size]
 *   C:            [batch, seqlen, ngroups, dstate]
 *   prev_states:  [batch, nchunks, nheads, headdim, dstate]
 *   D:            [nheads] or nullptr
 * Output:
 *   out:          [batch, seqlen, nheads, headdim]
 * ============================================================================ */

template<typename T>
__global__ void ssd_chunk_scan_forward_kernel(
    const T* __restrict__ CB,            // [batch, nchunks, ngroups, chunk_size, chunk_size]
    const T* __restrict__ x,             // [batch, seqlen, nheads, headdim]
    const T* __restrict__ dt,            // [batch, nheads, nchunks, chunk_size]
    const T* __restrict__ dA_cumsum,     // [batch, nheads, nchunks, chunk_size]
    const T* __restrict__ C_in,          // [batch, seqlen, ngroups, dstate]
    const T* __restrict__ prev_states,   // [batch, nchunks, nheads, headdim, dstate]
    const T* __restrict__ D,             // [nheads] or nullptr
    T* __restrict__ out,                 // [batch, seqlen, nheads, headdim]
    len_t seqlen,
    len_t nheads,
    len_t headdim,
    len_t ngroups,
    len_t dstate,
    len_t chunk_size,
    len_t nchunks
) {
    const len_t bc = blockIdx.x;                    // batch * nchunks
    const len_t b = bc / nchunks;
    const len_t c = bc % nchunks;
    const len_t h = blockIdx.y;
    const len_t d = blockIdx.z * blockDim.x + threadIdx.x;

    if (d >= headdim) return;

    const len_t g = h / (nheads / ngroups);  // group index
    const len_t chunk_len = min(chunk_size, seqlen - c * chunk_size);

    // Stride helpers
    const len_t da_base = b * nheads * nchunks * chunk_size
                        + h * nchunks * chunk_size
                        + c * chunk_size;
    const len_t cb_base = b * nchunks * ngroups * chunk_size * chunk_size
                        + c * ngroups * chunk_size * chunk_size
                        + g * chunk_size * chunk_size;
    const len_t x_batch_stride = seqlen * nheads * headdim;
    const len_t state_base = b * nchunks * nheads * headdim * dstate
                           + c * nheads * headdim * dstate
                           + h * headdim * dstate
                           + d * dstate;

    // D skip connection value
    float d_val = 0.0f;
    if (D != nullptr) {
        d_val = static_cast<float>(D[h]);
    }

    for (len_t l = 0; l < chunk_len; l++) {
        const len_t seq_pos = c * chunk_size + l;
        float dA_cs_l = static_cast<float>(dA_cumsum[da_base + l]);

        // --- Inter-chunk contribution: C[b,pos,g,:] @ prev_states[b,c,h,d,:] * exp(dA_cs[l]) ---
        float inter = 0.0f;
        const len_t c_base = b * seqlen * ngroups * dstate
                           + seq_pos * ngroups * dstate
                           + g * dstate;
        for (len_t n = 0; n < dstate; n++) {
            inter += static_cast<float>(C_in[c_base + n])
                   * static_cast<float>(prev_states[state_base + n]);
        }
        inter *= expf(dA_cs_l);

        // --- Intra-chunk contribution: sum over s <= l ---
        float intra = 0.0f;
        for (len_t s = 0; s <= l; s++) {
            // CB[b, c, g, l, s]
            float cb_val = static_cast<float>(CB[cb_base + l * chunk_size + s]);

            // Causal decay: exp(dA_cs[l] - dA_cs[s])
            float dA_cs_s = static_cast<float>(dA_cumsum[da_base + s]);
            float decay = expf(dA_cs_l - dA_cs_s);

            // dt[b, h, c, s]
            float dt_val = static_cast<float>(dt[da_base + s]);

            // x[b, c*cs+s, h, d]
            const len_t x_idx = b * x_batch_stride
                              + (c * chunk_size + s) * nheads * headdim
                              + h * headdim + d;
            float x_val = static_cast<float>(x[x_idx]);

            intra += cb_val * decay * dt_val * x_val;
        }

        // --- Output ---
        float result = inter + intra;

        // D skip connection: D[h] * x[b, pos, h, d]
        if (D != nullptr) {
            const len_t x_idx = b * x_batch_stride
                              + seq_pos * nheads * headdim
                              + h * headdim + d;
            result += d_val * static_cast<float>(x[x_idx]);
        }

        // out[b, pos, h, d]
        const len_t out_idx = b * seqlen * nheads * headdim
                            + seq_pos * nheads * headdim
                            + h * headdim + d;
        out[out_idx] = static_cast<T>(result);
    }
}

/* ============================================================================
 * 4b. State Passing Backward
 *
 * Reverse scan for gradient flow through the inter-chunk recurrence.
 *
 * Grid:  dim3(ceil(dim / 256), batch, nheads)
 * Block: 256 threads
 *
 * Input:
 *   grad_out:          [batch, nchunks, nheads, dim]   (dL/d(passed_states))
 *   dA_chunk_cumsum:   [batch, nheads, nchunks]
 *   final_grad:        [batch, nheads, dim]             (dL/d(final_state), nullable)
 * Output:
 *   grad_states:       [batch, nchunks, nheads, dim]
 *   grad_initial:      [batch, nheads, dim]             (nullable)
 * ============================================================================ */

template<typename T>
__global__ void ssd_state_passing_backward_kernel(
    const T* __restrict__ grad_out,          // [batch, nchunks, nheads, dim]
    const T* __restrict__ dA_chunk_cumsum,   // [batch, nheads, nchunks]
    const T* __restrict__ final_grad,        // [batch, nheads, dim] or nullptr
    T* __restrict__ grad_states,             // [batch, nchunks, nheads, dim]
    T* __restrict__ grad_initial,            // [batch, nheads, dim] or nullptr
    len_t nchunks,
    len_t nheads,
    len_t dim
) {
    const len_t di = blockIdx.x * blockDim.x + threadIdx.x;
    const len_t b = blockIdx.y;
    const len_t h = blockIdx.z;

    if (di >= dim) return;

    // Start from the end
    float dstate = 0.0f;
    if (final_grad != nullptr) {
        dstate = static_cast<float>(final_grad[b * nheads * dim + h * dim + di]);
    }

    for (len_t c_rev = 0; c_rev < nchunks; c_rev++) {
        const len_t c = nchunks - 1 - c_rev;
        const len_t state_idx = b * nchunks * nheads * dim
                              + c * nheads * dim
                              + h * dim + di;

        // Add gradient from this chunk's output
        if (c < nchunks - 1) {
            dstate += static_cast<float>(grad_out[state_idx]);
        }

        // Store gradient for this chunk's new_state input
        grad_states[state_idx] = static_cast<T>(dstate);

        // Propagate through decay: dstate_prev = decay * dstate
        float dA_cs = static_cast<float>(dA_chunk_cumsum[b * nheads * nchunks
                                                        + h * nchunks + c]);
        dstate = expf(dA_cs) * dstate;
    }

    // Gradient w.r.t. initial state
    if (grad_initial != nullptr) {
        grad_initial[b * nheads * dim + h * dim + di] = static_cast<T>(dstate);
    }
}

/* ============================================================================
 * C Interface — Chunk Cumsum Forward
 * ============================================================================ */

#define LAUNCH_SSD_CUMSUM_FWD(T) do { \
    dim3 grid(batch, nheads, nchunks); \
    ssd_chunk_cumsum_forward_kernel<T><<<grid, 1, 0, stream>>>( \
        static_cast<const T*>(dt), \
        static_cast<const T*>(A), \
        static_cast<T*>(dt_out), \
        static_cast<T*>(dA_cumsum), \
        seqlen, nheads, chunk_size, nchunks, dt_softplus \
    ); \
} while(0)

extern "C" void hip_nn_ssd_chunk_cumsum_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* dt,
    const void* A,
    void* dt_out,
    void* dA_cumsum,
    len_t batch,
    len_t seqlen,
    len_t nheads,
    len_t chunk_size,
    len_t nchunks,
    int dt_softplus
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSD_CUMSUM_FWD(f32);  break;
        case DTYPE_F16:  LAUNCH_SSD_CUMSUM_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSD_CUMSUM_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for ssd_chunk_cumsum_forward");
    }
}

/* ============================================================================
 * C Interface — BMM Chunk Forward
 * ============================================================================ */

#define LAUNCH_SSD_BMM_FWD(T) do { \
    const int threads = 256; \
    const len_t total = batch * nchunks * ngroups * chunk_size * chunk_size; \
    const int blocks = (total + threads - 1) / threads; \
    ssd_bmm_chunk_forward_kernel<T><<<blocks, threads, 0, stream>>>( \
        static_cast<const T*>(C_in), \
        static_cast<const T*>(B_in), \
        static_cast<T*>(CB), \
        seqlen, ngroups, dstate, chunk_size, nchunks \
    ); \
} while(0)

extern "C" void hip_nn_ssd_bmm_chunk_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* C_in,
    const void* B_in,
    void* CB,
    len_t batch,
    len_t seqlen,
    len_t ngroups,
    len_t dstate,
    len_t chunk_size,
    len_t nchunks
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSD_BMM_FWD(f32);  break;
        case DTYPE_F16:  LAUNCH_SSD_BMM_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSD_BMM_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for ssd_bmm_chunk_forward");
    }
}

/* ============================================================================
 * C Interface — Chunk State Forward
 * ============================================================================ */

#define LAUNCH_SSD_CHUNK_STATE_FWD(T) do { \
    const int threads = 256; \
    dim3 grid(batch * nchunks, nheads, (headdim * dstate + threads - 1) / threads); \
    ssd_chunk_state_forward_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(B_in), \
        static_cast<const T*>(x), \
        static_cast<const T*>(dt), \
        static_cast<const T*>(dA_cumsum), \
        static_cast<T*>(states), \
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, nchunks \
    ); \
} while(0)

extern "C" void hip_nn_ssd_chunk_state_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* B_in,
    const void* x,
    const void* dt,
    const void* dA_cumsum,
    void* states,
    len_t batch,
    len_t seqlen,
    len_t nheads,
    len_t headdim,
    len_t ngroups,
    len_t dstate,
    len_t chunk_size,
    len_t nchunks
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSD_CHUNK_STATE_FWD(f32);  break;
        case DTYPE_F16:  LAUNCH_SSD_CHUNK_STATE_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSD_CHUNK_STATE_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for ssd_chunk_state_forward");
    }
}

/* ============================================================================
 * C Interface — State Passing Forward
 * ============================================================================ */

#define LAUNCH_SSD_STATE_PASSING_FWD(T) do { \
    const int threads = 256; \
    dim3 grid((dim + threads - 1) / threads, batch, nheads); \
    ssd_state_passing_forward_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(states), \
        static_cast<const T*>(dA_chunk_cumsum), \
        static_cast<const T*>(initial_states), \
        static_cast<T*>(out), \
        static_cast<T*>(final_states), \
        nchunks, nheads, dim \
    ); \
} while(0)

extern "C" void hip_nn_ssd_state_passing_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* states,
    const void* dA_chunk_cumsum,
    const void* initial_states,
    void* out,
    void* final_states,
    len_t batch,
    len_t nchunks,
    len_t nheads,
    len_t dim
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSD_STATE_PASSING_FWD(f32);  break;
        case DTYPE_F16:  LAUNCH_SSD_STATE_PASSING_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSD_STATE_PASSING_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for ssd_state_passing_forward");
    }
}

/* ============================================================================
 * C Interface — Chunk Scan Forward
 * ============================================================================ */

#define LAUNCH_SSD_CHUNK_SCAN_FWD(T) do { \
    const int threads = 256; \
    dim3 grid(batch * nchunks, nheads, (headdim + threads - 1) / threads); \
    ssd_chunk_scan_forward_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(CB), \
        static_cast<const T*>(x), \
        static_cast<const T*>(dt), \
        static_cast<const T*>(dA_cumsum), \
        static_cast<const T*>(C_in), \
        static_cast<const T*>(prev_states), \
        static_cast<const T*>(D), \
        static_cast<T*>(out), \
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, nchunks \
    ); \
} while(0)

extern "C" void hip_nn_ssd_chunk_scan_forward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* CB,
    const void* x,
    const void* dt,
    const void* dA_cumsum,
    const void* C_in,
    const void* prev_states,
    const void* D,
    void* out,
    len_t batch,
    len_t seqlen,
    len_t nheads,
    len_t headdim,
    len_t ngroups,
    len_t dstate,
    len_t chunk_size,
    len_t nchunks
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSD_CHUNK_SCAN_FWD(f32);  break;
        case DTYPE_F16:  LAUNCH_SSD_CHUNK_SCAN_FWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSD_CHUNK_SCAN_FWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for ssd_chunk_scan_forward");
    }
}

/* ============================================================================
 * C Interface — State Passing Backward
 * ============================================================================ */

#define LAUNCH_SSD_STATE_PASSING_BWD(T) do { \
    const int threads = 256; \
    dim3 grid((dim + threads - 1) / threads, batch, nheads); \
    ssd_state_passing_backward_kernel<T><<<grid, threads, 0, stream>>>( \
        static_cast<const T*>(grad_out), \
        static_cast<const T*>(dA_chunk_cumsum), \
        static_cast<const T*>(final_grad), \
        static_cast<T*>(grad_states), \
        static_cast<T*>(grad_initial), \
        nchunks, nheads, dim \
    ); \
} while(0)

extern "C" void hip_nn_ssd_state_passing_backward(
    Dtype dtype,
    StreamHandle stream_handle,
    const void* grad_out,
    const void* dA_chunk_cumsum,
    const void* final_grad,
    void* grad_states,
    void* grad_initial,
    len_t batch,
    len_t nchunks,
    len_t nheads,
    len_t dim
) {
    hipStream_t stream = cast_stream(stream_handle.ptr);

    switch (dtype) {
        case DTYPE_F32:  LAUNCH_SSD_STATE_PASSING_BWD(f32);  break;
        case DTYPE_F16:  LAUNCH_SSD_STATE_PASSING_BWD(f16);  break;
        case DTYPE_BF16: LAUNCH_SSD_STATE_PASSING_BWD(bf16); break;
        default: SYSTEM_EXIT("Unsupported dtype for ssd_state_passing_backward");
    }
}

#endif /* __NN_SSD_CHUNK_SCAN_H__ */
