/*
 * core/includes.h - C++ vendor library includes
 *
 * Consolidates all HIP/Thrust/vendor includes in one place.
 * C++ only - not for interop.h
 */

#ifndef METAPHOR_HIP_INCLUDES_H
#define METAPHOR_HIP_INCLUDES_H

/* HIP */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>    /* __half - IEEE 754 half precision */
#include <hip/hip_bfloat16.h>    /* hip_bfloat16 - brain float (Ampere+) */

/* hipBLAS */
#include <hipblas/hipblas.h>

/* MIOpen */
#include <miopen/miopen.h>

/* hipRAND */
#include <hiprand/hiprand.h>

/* hipTENSOR */
#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>

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

#endif /* METAPHOR_HIP_INCLUDES_H */
