#!/usr/bin/env python3
"""
Compare Metaphor results against PyTorch test vectors (SafeTensors format).

Expected usage:
1. Run generate_cases.py to create test_vectors/*.safetensors
2. Run Metaphor tests which output to test_results/*.safetensors
3. Run this script to compare and report differences

Result files should mirror test vector format with actual outputs.
"""

import sys
from pathlib import Path
from typing import Tuple
import math
import numpy as np
from safetensors import safe_open

VECTORS_DIR = Path(__file__).parent / "test_vectors"
RESULTS_DIR = Path(__file__).parent / "test_results"

# Tolerances for numerical comparison
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-6  # Absolute tolerance


def load_safetensors(path: Path) -> Tuple[dict, dict]:
    """Load tensors and metadata from a SafeTensors file."""
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, metadata


def compare_arrays(expected: np.ndarray, actual: np.ndarray, path: str) -> Tuple[bool, list]:
    """
    Compare expected vs actual numpy arrays.
    Returns (success, list of error messages).
    """
    errors = []

    if expected.shape != actual.shape:
        errors.append(f"{path}: shape mismatch: expected {expected.shape}, got {actual.shape}")
        return False, errors

    # Flatten for element-wise comparison
    exp_flat = expected.flatten()
    act_flat = actual.flatten()

    all_ok = True
    max_errors = 5  # Limit error reporting

    for i, (e, a) in enumerate(zip(exp_flat, act_flat)):
        # Handle special values
        if math.isnan(e) and math.isnan(a):
            continue
        if math.isinf(e) or math.isinf(a):
            if e == a:
                continue
            if len(errors) < max_errors:
                errors.append(f"{path}[{i}]: inf mismatch: expected {e}, got {a}")
            all_ok = False
            continue

        diff = abs(e - a)
        threshold = ATOL + RTOL * abs(e)
        if diff > threshold:
            all_ok = False
            if len(errors) < max_errors:
                errors.append(
                    f"{path}[{i}]: expected {e:.8f}, got {a:.8f}, "
                    f"diff={diff:.2e}, threshold={threshold:.2e}"
                )

    if not all_ok and len(errors) >= max_errors:
        errors.append(f"  ... (more errors omitted)")

    return all_ok, errors


def compare_case(vector_file: Path, result_file: Path) -> Tuple[bool, dict]:
    """
    Compare a single test case.
    Returns (success, report_dict).
    """
    expected_tensors, expected_meta = load_safetensors(vector_file)
    actual_tensors, actual_meta = load_safetensors(result_file)

    report = {
        "name": expected_meta.get("name", vector_file.stem),
        "operation": expected_meta.get("operation", "unknown"),
        "description": expected_meta.get("description", ""),
        "checks": []
    }

    all_passed = True

    # Compare output tensor
    if "output" in expected_tensors and "output" in actual_tensors:
        ok, errors = compare_arrays(
            expected_tensors["output"],
            actual_tensors["output"],
            "output"
        )
        report["checks"].append({
            "field": "output",
            "passed": ok,
            "errors": errors
        })
        if not ok:
            all_passed = False
    elif "output" in expected_tensors:
        report["checks"].append({
            "field": "output",
            "passed": False,
            "errors": ["Missing output tensor in results"]
        })
        all_passed = False

    # Compare gradient tensors (disabled for now - gradients not yet implemented)
    # for key in expected_tensors:
    #     if not key.startswith("grad."):
    #         continue
    #
    #     if key not in actual_tensors:
    #         report["checks"].append({
    #             "field": key,
    #             "passed": False,
    #             "errors": [f"Missing gradient tensor: {key}"]
    #         })
    #         all_passed = False
    #         continue
    #
    #     ok, errors = compare_arrays(
    #         expected_tensors[key],
    #         actual_tensors[key],
    #         key
    #     )
    #     report["checks"].append({
    #         "field": key,
    #         "passed": ok,
    #         "errors": errors
    #     })
    #     if not ok:
    #         all_passed = False

    # Compare intermediate tensors if present
    for key in expected_tensors:
        if not key.startswith("intermediate."):
            continue

        if key in actual_tensors:
            ok, errors = compare_arrays(
                expected_tensors[key],
                actual_tensors[key],
                key
            )
            report["checks"].append({
                "field": key,
                "passed": ok,
                "errors": errors
            })
            if not ok:
                all_passed = False

    report["passed"] = all_passed
    return all_passed, report


def main():
    if not VECTORS_DIR.exists():
        print(f"Error: Test vectors directory not found: {VECTORS_DIR}")
        print("Run generate_cases.py first.")
        sys.exit(1)

    if not RESULTS_DIR.exists():
        print(f"Error: Test results directory not found: {RESULTS_DIR}")
        print("Run Metaphor tests first to generate results.")
        sys.exit(1)

    vector_files = sorted(VECTORS_DIR.glob("*.safetensors"))
    if not vector_files:
        print("No test vectors found.")
        sys.exit(1)

    print(f"Comparing {len(vector_files)} test cases...")
    print(f"Tolerances: rtol={RTOL}, atol={ATOL}")
    print("=" * 60)

    passed = 0
    failed = 0
    missing = 0
    reports = []

    for vf in vector_files:
        rf = RESULTS_DIR / vf.name

        if not rf.exists():
            print(f"MISSING: {vf.stem}")
            missing += 1
            continue

        ok, report = compare_case(vf, rf)
        reports.append(report)

        if ok:
            print(f"PASS: {report['name']} ({report['operation']})")
            passed += 1
        else:
            print(f"FAIL: {report['name']} ({report['operation']})")
            for check in report["checks"]:
                if not check["passed"]:
                    print(f"  - {check['field']}:")
                    for err in check["errors"][:3]:
                        print(f"      {err}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {missing} missing")

    # Write detailed report as JSON
    import json
    report_file = Path(__file__).parent / "comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "summary": {
                "passed": passed,
                "failed": failed,
                "missing": missing,
                "total": len(vector_files)
            },
            "tolerances": {"rtol": RTOL, "atol": ATOL},
            "cases": reports
        }, f, indent=2)
    print(f"\nDetailed report written to: {report_file}")

    sys.exit(0 if failed == 0 and missing == 0 else 1)


if __name__ == "__main__":
    main()
