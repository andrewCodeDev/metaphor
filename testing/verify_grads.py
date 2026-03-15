#!/usr/bin/env python3
"""Verify gradient calculations using PyTorch."""

import torch

print("=== PyTorch Gradient Verification ===\n")

# Test case: a used in two paths
a = torch.tensor([2.0, 2.0], requires_grad=True)
b = torch.tensor([1.0, 1.0])

# Path 1: c = a + b
c = a + b
print(f"a = {a.data}")
print(f"b = {b.data}")
print(f"c = a + b = {c.data}")

# Path 2: d = sum(a)
d = a.sum()
print(f"d = sum(a) = {d.data}")

# Merge: e = c * d (d is broadcast)
e = c * d
print(f"e = c * d = {e.data}")

# Backward
e.sum().backward()  # Use sum() to get scalar loss

print(f"\n=== Gradients ===")
print(f"a.grad = {a.grad}")

# Verify intermediate gradients by recomputing
a2 = torch.tensor([2.0, 2.0], requires_grad=True)
b2 = torch.tensor([1.0, 1.0])
c2 = a2 + b2
d2 = a2.sum()
e2 = c2 * d2

# For c.grad: de/dc where e = c * d, so de/dc = d
# But we need de_sum/dc = 1 * d = d (since upstream grad from sum is 1)
print(f"\nExpected c.grad = d = {d2.data} (per element)")

# For d.grad: de/dd where e = c * d, so de/dd = sum(c)
print(f"Expected d.grad = sum(c) = {c2.sum().data}")

# For a.grad via c: dc/da = 1, so contribution = c.grad * 1 = d
print(f"\nContribution to a.grad via c: d = {d2.data} (per element)")

# For a.grad via d: dd/da = 1 (for each element), so contribution = d.grad * 1
# But d.grad needs reduction: de/dd = sum(c) = 6
# And dd/da_i = 1, so contribution = 6 per element
print(f"Contribution to a.grad via d: d.grad = sum(c) = {c2.sum().data} (per element)")

print(f"\nTotal a.grad = {d2.data} + {c2.sum().data} = {d2.data + c2.sum().data} (per element)")
print(f"Actual a.grad from PyTorch = {a.grad}")

# Also verify the simple a*a case
print("\n\n=== Simple a*a case ===")
a3 = torch.tensor([3.0, 3.0], requires_grad=True)
y3 = a3 * a3
y3.sum().backward()
print(f"a = {a3.data}")
print(f"y = a*a = {(a3.data * a3.data)}")
print(f"a.grad = {a3.grad}")
print(f"Expected: 2*a = {2 * a3.data}")
