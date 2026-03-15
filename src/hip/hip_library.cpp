/*
 * hip_library.cpp - Single compilation unit entry point
 *
 * This file includes all HIP modules to create a single library.
 * All extern "C" functions are exported for C3 cImport.
 *
 * Architecture:
 *   interop.h       - Pure C types for C3 boundary
 *   core/           - Internal helpers (assert, includes, cast)
 *   memory/         - Memory allocation and transfer
 *   devices/        - Device backends (graph, stream)
 *   handles/        - Standalone handle management
 *   hiptensor/       - hipTENSOR operations
 */

/* ============================================================================
 * Core Infrastructure
 * ============================================================================
 */

#include "core/assert.h"
#include "core/cast.h"
#include "core/includes.h"
#include "interop.h"

/* ============================================================================
 * Device Backends
 *
 * graph_device.cpp includes hiptensor internally - those are implementation
 * details not exposed to C3.
 * ============================================================================
 */

#include "devices/graph_device.cpp"

/* ============================================================================
 * Handle Management (streams, events)
 * ============================================================================
 */

#include "handles/stream.cpp"

/* ============================================================================
 * Memory Operations
 * ============================================================================
 */

#include "memory/allocator.cpp"
#include "memory/cast.cpp"
#include "memory/concat_stack.cpp"
#include "memory/fill.cpp"
#include "memory/gather.cpp"
#include "memory/slice.cpp"
#include "memory/transfer.cpp"
#include "memory/virtual_memory.cpp"


/* ============================================================================
 * BLAS Operations (max/min backward only - other ops use hipTENSOR/JIT)
 * ============================================================================
 */

#include "blas/max.cpp"
#include "blas/reduce_backward.cpp"

/* ============================================================================
 * NN Operations (specialized kernels not handled by JIT)
 * ============================================================================
 */

#include "nn/conv2d.cpp"
#include "nn/nll_loss_1d_index.cpp"
#include "nn/optimizers.cpp"
#include "nn/rms_norm.cpp"
#include "nn/selective_scan.cpp"
#include "nn/ssd_chunk_scan.cpp"
#include "nn/causal_conv1d.cpp"
#include "nn/smax_2D_row.cpp"
#include "nn/sampling.cpp"
#include "nn/dropout.cpp"

/* ============================================================================
 * Loss Functions (Graph-Capture Safe)
 * ============================================================================
 */

#include "loss/loss_kernels.cpp"

/* ============================================================================
 * Preprocessing / Scaling Operations
 * ============================================================================
 */

#include "kernels/scaling.cpp"
#include "kernels/scaling_backward.cpp"

/* ============================================================================
 * JIT Compilation (elementwise operations)
 * ============================================================================
 */

#include "jit/elementwise.cpp"
