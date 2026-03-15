# Adding BFloat16 Support to hipTensor on gfx1100 (RDNA3)

hipTensor (develop branch) has partial BF16 support — contraction and reduction
work with F32 compute, but permutation has no BF16 at all. On gfx1100 (RDNA3),
the matrix cores only support native-precision compute (16F/16BF), not F32, so
contractions require native BF16 compute too.

This documents everything needed to get BF16 working across all hipTensor
operations on gfx1100.

All paths below are relative to `third_party/hipTensor/`.

---

## Problem 1: BF16 Permutation (completely missing)

The CK backend already has `ck::bhalf_t` support — hipTensor just never wired
it up for permute. Three things were missing:

### 1a. Validation whitelist

**File:** `library/src/elementwise/hiptensor_elementwise_permute.cpp`

The `validDataTypes` array is a hard-coded whitelist of `{typeA, typeScalar}`
pairs. Add two BF16 entries (array size 3 → 5):

```cpp
constexpr std::array<std::array<hiptensorDataType_t, 2>, 5> validDataTypes
    = {{
        {HIPTENSOR_R_16F, HIPTENSOR_R_16F},
        {HIPTENSOR_R_16F, HIPTENSOR_R_32F},
        {HIPTENSOR_R_16BF, HIPTENSOR_R_16BF},   // added
        {HIPTENSOR_R_16BF, HIPTENSOR_R_32F},     // added
        {HIPTENSOR_R_32F, HIPTENSOR_R_32F}}};
```

### 1b. CK instance factory — the critical piece

**File:** `library/src/elementwise/device/hiptensor_elementwise_scale_instances.hpp`

The `GetInstances()` method uses `if constexpr` to dispatch on data types.
It has explicit branches for `ck::Tuple<float>` (ranks 2-4) and
`ck::Tuple<ck::half_t>` (ranks 2-4), each with ~20 tuned block-size configs.
`ck::bhalf_t` doesn't match any branch and **returns an empty map** — so
nothing gets registered despite the instance .cpp files compiling fine.

Fix: add `ck::bhalf_t` to the existing `half_t` conditions. Since both are
2-byte types, they use identical block sizes:

```cpp
// Before:
} else if constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<ck::half_t>> && NumDim == 2) {
// After:
} else if constexpr((std::is_same_v<InDataTypeTuple, ck::Tuple<ck::half_t>> || std::is_same_v<InDataTypeTuple, ck::Tuple<ck::bhalf_t>>) && NumDim == 2) {
```

Apply this to all three half_t branches (NumDim == 2, 3, 4). Ranks 5-6 already
use a generic `InDataTypeTuple::Size() == 1` check that covers bhalf_t.

### 1c. Instance files, declarations, CMake

**Directory:** `library/src/elementwise/instances/`

For each rank 2-6, copy the `_half` file to `_bhalf` and the `_half_noop` to
`_bhalf_noop` (10 new files total):

```bash
cd library/src/elementwise/instances
for rank in 2 3 4 5 6; do
  for suffix in "" "_noop"; do
    src="elementwise_permute_solution_rank${rank}_half${suffix}_instances.cpp"
    dst="elementwise_permute_solution_rank${rank}_bhalf${suffix}_instances.cpp"
    sed -e 's/ck::half_t/ck::bhalf_t/g' -e 's/Half/BHalf/g' "$src" > "$dst"
  done
done
```

**File:** `library/src/elementwise/elementwise_solution_instances.hpp`
— Declare 10 new `...BHalf...` methods.

**File:** `library/src/elementwise/elementwise_solution_instances.cpp`
— Call them in the constructor.

**File:** `library/src/elementwise/CMakeLists.txt`
— Add 10 `_bhalf` source files to `HIPTENSOR_ELEMENTWISE_SOURCES`.

### 1d. Safe-net hash fallbacks

**File:** `library/src/elementwise/device/instance_params.cpp`

The `getHashCodeOfBestPerfInstances()` function generates fallback hash
candidates with known-good block sizes. Add `HIPTENSOR_R_16BF` alongside the
existing `HIPTENSOR_R_16F` checks so BF16 gets the same safe-net entries:

```cpp
// Before:
if (typeIn[0] == HIPTENSOR_R_16F) {
// After:
if (typeIn[0] == HIPTENSOR_R_16F || typeIn[0] == HIPTENSOR_R_16BF) {
```

Apply to ranks 2, 3, and 4.

---

## Problem 2: Rank-1 tensors (hipTensor minimum rank is 2)

hipTensor only has permutation instances for ranks 2-6. LLaMA uses 1D tensors
(e.g., bias vectors) which hit the permutation path.

**File:** `src/hip/hiptensor/permutate.cpp` (our wrapper, not hipTensor)

Pad 0D and 1D tensors to 2D by prepending a dimension of size 1, with a
shared padding symbol that doesn't conflict with existing symbols. Both src
and dst get the same padding symbol so the permutation is well-formed.

---

## Problem 3: Descriptor lifetime (use-after-free)

**File:** `src/hip/hiptensor/permutate.cpp` (our wrapper)

`hiptensorCreatePlan` stores **raw pointers** to tensor descriptors and
operation descriptors. The original code destroyed them immediately after plan
creation, causing use-after-free during execution (`descA->mType` read garbage).

Fix: do NOT destroy descriptors after plan creation. They are freed when the
plan is destroyed.

This also applies to contraction — `hiptensorCreatePlan` stores raw pointers
to both tensor descriptors and operation descriptors via `plan->mOpDesc = desc`.
Destroying them after plan creation caused corrupted `mLengths` data during
`hiptensorContract` execution.

---

## Problem 4: Compute descriptors on gfx1100 (RDNA3)

**File:** `src/hip/hiptensor/utils.cpp` (our wrapper)

gfx1100 matrix cores (WMMA instructions) only support native-precision
compute: `HIPTENSOR_COMPUTE_DESC_16F` for F16, `HIPTENSOR_COMPUTE_DESC_16BF`
for BF16. Using `HIPTENSOR_COMPUTE_DESC_32F` with half types fails:

- **Contraction:** `HIPTENSOR_STATUS_ARCH_MISMATCH` — the `matrixCoreSupport()`
  check in `hip_device.cpp` only allows F32 compute on GFX908/90A/942/950.
- **Reduction:** `HIPTENSOR_STATUS_NOT_SUPPORTED` — BF16→BF16 compute reduction
  instances don't exist (only BF16→F32 compute instances are registered).

Fix: use native compute descriptors:

```cpp
case DTYPE_F16:  return HIPTENSOR_COMPUTE_DESC_16F;
case DTYPE_BF16: return HIPTENSOR_COMPUTE_DESC_16BF;
```

### Current status of each operation with native BF16 compute on gfx1100

| Operation | Status | Notes |
|-----------|--------|-------|
| Permutation | **Working** | After all fixes above |
| Contraction | **Working** | Problems 3, 5, 6, 7 fixed. WMMA 16x16 candidate 0 (Block=64) wins brute force. |
| Reduction (N-D → M-D) | **Working** | hipTensor silently promotes `16BF` compute → `32F` internally; existing `bf16_f32` instances handle it. Only needed descriptor lifetime fix (Problem 3). |
| Reduction (scalar) | **Working** | Custom HIP kernel in `graph_device.cpp` — hipTensor doesn't support 0-D output. |
| Elementwise (binary) | **Untested** | Likely needs similar instance work |

---

## Problem 5: Contraction WMMA tile sizes (gfx1100)

The CK `DeviceContractionMultipleD_Xdl_CShuffle` kernel checks
`is_xdl_wmma_supported<ComputeDataType, ComputeDataType, MPerXDL, NPerXDL>()`
in `IsSupportedArgument()`. On gfx1100 (RDNA3), WMMA requires MPerXDL=16 and
NPerXDL=16. All stock CK contraction instances use 32x32 (MFMA tile size for
CDNA). F32 contractions work because `sizeof(float) > 2` bypasses the WMMA
check entirely, falling back to scalar ALU.

**Root cause path:**
1. `hiptensorCreatePlan` with `HIPTENSOR_ALGO_DEFAULT` → brute force model
2. Brute force tries each registered CK kernel
3. Each kernel's `IsSupportedArgument()` calls `is_xdl_wmma_supported()`
4. Check at `/opt/rocm/include/ck/host_utility/device_prop.hpp:85`:
   `if constexpr((MPerXDL != 16) || (NPerXDL != 16))` → returns false
5. All kernels unsupported → `bestSolution = nullptr` → EXECUTION_FAILED

**Fix:** Create WMMA-compatible contraction instances with MPerXDL=16,
NPerXDL=16, adapted from CK's WMMA convolution instances:

**New file:** `library/src/contraction/device/device_contraction_wmma_instance.hpp`
— Defines `device_contraction_kk_wmma_instance` etc. with 16x16 tiles,
K1=8, SrcScalarPerVector=8 (matching CK's WMMA conv configs).

**New files (4):** `library/src/contraction/device/device_contraction_bilinear_unary_m6_n6_k6_wmma_c_shuffle_bf16_bf16_bf16_bf16_{kknn,knnn,mknn,mnnn}_instance.cpp`

**Modified:** `library/src/contraction/device/hiptensor_contraction_bilinear_unary_ops_instances.hpp`
— Added forward declarations and `GetInstances()` calls for WMMA BF16 instances.

**Modified:** `library/src/contraction/device/CMakeLists.txt`
— Added 4 WMMA instance source files.

---

## Problem 6: Non-unary bilinear factory gap (contraction 0 candidates)

After adding WMMA instances (Problem 5), BF16 contraction still failed with
`HIPTENSOR_STATUS_EXECUTION_FAILED` — brute force found **0 candidate kernels**.

**Root cause:** hipTensor has two parallel factory paths for bilinear contraction:

1. **Unary path** (`BILINEAR_UNARY`): Used when any op is non-identity
   (`HIPTENSOR_OP_UNKNOWN`). Factory: `hiptensor_contraction_bilinear_unary_ops_instances.hpp`.
2. **Non-unary path** (`BILINEAR`): Used when all ops are identity
   (`HIPTENSOR_OP_IDENTITY`). Factory: `hiptensor_contraction_bilinear_instances.hpp`.

Our wrapper (`src/hip/hiptensor/contraction.cpp`) sets all three ops to
`HIPTENSOR_OP_IDENTITY`, which routes to the non-unary `BILINEAR` path. The
Problem 5 WMMA instances were added to the **unary** factory only — the non-unary
factory had no real-valued BF16 instances at all (only complex-valued CF32/CF64).

**Dispatch trace:**
```
hiptensorCreateContraction
  → all ops IDENTITY → hasUnaryOp = false → ContractionOpId_t::BILINEAR
  → contractionInitPlan → allSolutions() → filter by BILINEAR op ID
  → query DeviceOperationInstanceFactory<..., PassThrough, PassThrough, Bilinear, ...>
  → no BF16 specialization → 0 instances → EXECUTION_FAILED
```

**Fix:** Add a second set of WMMA instances to the non-unary bilinear factory,
using `PassThrough/PassThrough/Bilinear` (not the unary `CkHiptensorUnaryOp`/
`CkBilinearUnary` wrapper types):

**New files (4):** `library/src/contraction/device/device_contraction_bilinear_m6_n6_k6_wmma_c_shuffle_bf16_bf16_bf16_bf16_{kknn,knnn,mknn,mnnn}_instance.cpp`
— Same WMMA template instantiation as Problem 5, but with `PassThrough`/`Bilinear`.

**Modified:** `library/src/contraction/device/hiptensor_contraction_bilinear_instances.hpp`
— Added `BF16`/`BF16_Tuple`/`Bilinear` type aliases, 4 forward declarations, and a
new `DeviceOperationInstanceFactory` specialization for real-valued BF16:

```cpp
template <index_t NumDimM, index_t NumDimN, index_t NumDimK, typename ComputeDataT>
struct DeviceOperationInstanceFactory<
    DeviceContractionMultipleD<NumDimM, NumDimN, NumDimK,
        BF16, BF16, BF16_Tuple, BF16,
        PassThrough, PassThrough, Bilinear, ComputeDataT>>
{
    static auto GetInstances() {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<ComputeDataT, BF16>) {
            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6) {
                // 4 WMMA instances (kknn, knnn, mknn, mnnn)
            }
        }
        return op_ptrs;
    }
};
```

**Modified:** `library/src/contraction/device/CMakeLists.txt`
— Added 4 non-unary bilinear WMMA source files.

### Why both factory paths?

The unary path instances (Problem 5) are still needed if anyone calls hipTensor
with non-identity ops. The non-unary instances (this problem) are what our wrapper
actually hits. Both sets use the same underlying WMMA template — only the CK
functor types differ.

---

## Problem 7: Alpha/beta scalar type mismatch (contraction + reduction)

**File:** `src/hip/devices/graph_device.cpp` (our wrapper)

hipTensor's `readVal<ScalarData>` casts the alpha/beta pointer according to
the compute descriptor type. For `HIPTENSOR_COMPUTE_DESC_16BF`, it reads
`*(hip_bfloat16*)value`. Our wrapper was passing `float*` for all types,
causing the wrong 2 bytes to be interpreted as the scalar.

Fix: pass `hip_bfloat16` alpha/beta when dtype is BF16:

```cpp
hip_bfloat16 alpha_bf16 = hip_bfloat16(1.0f), beta_bf16 = hip_bfloat16(0.0f);
if (x_dtype == DTYPE_BF16) {
    alpha = &alpha_bf16; beta = &beta_bf16;
}
```

Applied to both contraction and reduction paths in `graph_device.cpp`.

---

## Problem 8: RPATH vs RUNPATH (transitive dependency resolution)

**File:** `CMakeLists.txt`

`libmetaphor_hip.so` depends on `libhiptensor.so` (our custom build). By
default, cmake links with `--enable-new-dtags` which produces **RUNPATH** on
the .so. RUNPATH does NOT propagate to transitive dependencies — so when the
C3 binary loads `libmetaphor_hip.so`, the runtime linker ignores the
RUNPATH when searching for `libhiptensor.so` and finds the system
`/opt/rocm/lib/libhiptensor.so` instead of our patched build.

Fix: use `--disable-new-dtags` to produce **RPATH** (which does propagate):

```cmake
target_link_options(metaphor_hip PRIVATE -Wl,--disable-new-dtags)
```

---

## Problem 9: Plan cache descriptor leak (segfault at exit)

**File:** `src/hip/devices/graph_device.cpp` (our wrapper)

`hiptensorDestroyPlan` only does `delete plan` — it does NOT free the nested
operation descriptor (`mOpDesc`), tensor descriptors (`mDescA/B/C/D`), or
plan preference (`mPref`). Since we stopped destroying these after plan
creation (Problem 3), they leaked, and the process segfaulted during static
destructor teardown (hipTensor/ROCm globals destroyed while leaked objects
still referenced them).

Fix: the `PlanCache` destructor now explicitly frees the full ownership tree
before calling `hiptensorDestroyPlan`. Uses a `std::set` to deduplicate
descriptor pointers (contraction sets `mDescC == mDescD`).

Requires including `data_types.hpp` (hipTensor internal header) for access
to `plan->mOpDesc` fields — added `library/src/include` to the include path
in `CMakeLists.txt`.

---

## Problem 10: Scalar reduce BF16 kernel

**File:** `src/hip/devices/graph_device.cpp` (our wrapper)

hipTensor doesn't support 0-dimensional output tensors. The wrapper already
had a fast path for scalar reductions using custom HIP kernels
(`scalar_reduce_sum_f32`, `scalar_reduce_sum_f64`), but no BF16 variant.
BF16 input fell through to the F32 kernel, which reinterpreted BF16 data as
float — wrong values and potential out-of-bounds memset.

Fix: added `scalar_reduce_sum_bf16` kernel that reads `hip_bfloat16`,
accumulates in F32 shared memory, and writes back to `hip_bfloat16`. The
dispatch uses a switch statement on `src_dtype`.

---

## Note: F32 contraction is impossible on gfx1100

`matrixCoreSupport()` in `library/src/hip_device.cpp` only allows
`HIPTENSOR_COMPUTE_DESC_32F` on CDNA GPUs (GFX908, GFX90A, GFX942, GFX950).
gfx1100 returns `HIPTENSOR_STATUS_ARCH_MISMATCH` for F32 compute.

This is correct — RDNA3 WMMA instructions only operate on F16/BF16 data types.
The hardware accumulates into F32 registers internally (so precision is not lost
during the matmul), but the API-level compute type must be `16BF` or `16F`.

For ML training, this matches standard practice: PyTorch's `torch.autocast` runs
matmuls in BF16 (with hardware F32 accumulation) while keeping loss/reductions in
F32. This is the default for most modern LLM training.

---

## Build

hipTensor is built separately, then the .so is copied into the main build:

```bash
cd third_party/hipTensor/build
cmake --build . -j$(nproc)          # slow — CK templates take ~20 min
cp lib/libhiptensor.so.0.1 ../../../build/lib/libhiptensor.so.0.1
cd ../../..
cmake --build build -j$(nproc)      # rebuilds metaphor_hip.so + c3c
```

## Files changed summary

### In hipTensor (`third_party/hipTensor/`)

| File | Change |
|------|--------|
| `library/src/elementwise/hiptensor_elementwise_permute.cpp` | Add 2 entries to `validDataTypes` |
| `library/src/elementwise/device/hiptensor_elementwise_scale_instances.hpp` | Add `bhalf_t` to 3 `if constexpr` branches |
| `library/src/elementwise/device/instance_params.cpp` | Add `HIPTENSOR_R_16BF` to safe-net for ranks 2-4 |
| `library/src/elementwise/elementwise_solution_instances.hpp` | Declare 10 `...BHalf...` methods |
| `library/src/elementwise/elementwise_solution_instances.cpp` | Call 10 `...BHalf...` methods |
| `library/src/elementwise/CMakeLists.txt` | Add 10 `_bhalf` source files |
| `library/src/elementwise/instances/*_bhalf_*.cpp` (10 new) | Copy from `_half`, `ck::half_t` → `ck::bhalf_t` |

| `library/src/contraction/device/hiptensor_contraction_bilinear_instances.hpp` | Add BF16 types, forward declarations, factory specialization |
| `library/src/contraction/device/device_contraction_bilinear_m6_n6_k6_wmma_*.cpp` (4 new) | Non-unary bilinear WMMA instances for IDENTITY ops path |

### In metaphor (`src/hip/` and root)

| File | Change |
|------|--------|
| `src/hip/hiptensor/utils.cpp` | BF16 dtype mapping + native compute descriptors |
| `src/hip/hiptensor/permutate.cpp` | 1D→2D padding, descriptor lifetime fix |
| `src/hip/hiptensor/contraction.cpp` | Descriptor lifetime fix (don't destroy after plan) |
| `src/hip/hiptensor/reduce.cpp` | Descriptor lifetime fix (don't destroy after plan) |
| `src/hip/hiptensor/elementwise_binary.cpp` | Descriptor lifetime fix (don't destroy after plan) |
| `src/hip/devices/graph_device.cpp` | Alpha/beta BF16 scalars, PlanCache descriptor cleanup, BF16 scalar reduce kernel |
| `CMakeLists.txt` | `--disable-new-dtags` for RPATH; hipTensor internal include path |
