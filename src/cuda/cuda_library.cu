/*
 * cuda_library.cu - Single compilation unit entry point
 *
 * This file includes all CUDA modules to create a single library.
 * All extern "C" functions are exported for Zig cImport.
 *
 * Architecture:
 *   interop.h       - Pure C types for Zig boundary
 *   core/           - Internal helpers (assert, includes, cast)
 *   memory/         - Memory allocation and transfer
 *   devices/        - Device backends (graph, stream)
 *   handles/        - Standalone handle management
 *   cutensor/       - cuTENSOR operations
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
 * graph_device.cu includes cutensor internally - those are implementation
 * details not exposed to Zig.
 * ============================================================================
 */

#include "devices/graph_device.cu"

/* ============================================================================
 * Handle Management (streams, events)
 * ============================================================================
 */

#include "handles/stream.cu"

/* ============================================================================
 * Memory Operations
 * ============================================================================
 */

#include "memory/allocator.cu"
#include "memory/cast.cu"
#include "memory/concat_stack.cu"
#include "memory/fill.cu"
#include "memory/gather.cu"
#include "memory/slice.cu"
#include "memory/transfer.cu"
#include "memory/virtual_memory.cu"


/* ============================================================================
 * BLAS Operations (max/min backward only - other ops use cuTENSOR/JIT)
 * ============================================================================
 */

#include "blas/max.cu"
#include "blas/reduce_backward.cu"

/* ============================================================================
 * NN Operations (specialized kernels not handled by JIT)
 * ============================================================================
 */

#include "nn/conv2d.cu"
#include "nn/nll_loss_1d_index.cu"
#include "nn/optimizers.cu"
#include "nn/rms_norm.cu"
#include "nn/selective_scan.cu"
#include "nn/ssd_chunk_scan.cu"
#include "nn/causal_conv1d.cu"
#include "nn/smax_2D_row.cu"
#include "nn/sampling.cu"
#include "nn/dropout.cu"

/* ============================================================================
 * Loss Functions (Graph-Capture Safe)
 * ============================================================================
 */

#include "loss/loss_kernels.cu"

/* ============================================================================
 * Preprocessing / Scaling Operations
 * ============================================================================
 */

#include "kernels/scaling.cu"
#include "kernels/scaling_backward.cu"

/* ============================================================================
 * JIT Compilation (elementwise operations)
 * ============================================================================
 */

#include "jit/elementwise.cu"
