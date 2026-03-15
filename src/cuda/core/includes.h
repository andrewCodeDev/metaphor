/*
 * core/includes.h - C++ vendor library includes
 *
 * Consolidates all CUDA/Thrust/vendor includes in one place.
 * C++ only - not for interop.h
 */

#ifndef METAPHOR_CUDA_INCLUDES_H
#define METAPHOR_CUDA_INCLUDES_H

/* CUDA */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>    /* __half - IEEE 754 half precision */
#include <cuda_bf16.h>    /* __nv_bfloat16 - brain float (Ampere+) */

/* cuBLAS */
#include <cublas_v2.h>

/* cuDNN */
#include <cudnn.h>

/* cuRAND */
#include <curand.h>

/* cuTENSOR */
#include <cutensor.h>
#include <cutensor/types.h>

/* Thrust */
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#endif /* METAPHOR_CUDA_INCLUDES_H */
