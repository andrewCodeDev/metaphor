# Lazy Execution

All operations in Metaphor defer execution. No operation produces data at registration time. Each operation records a semantic description into the graph — what to compute, not how. Data materializes only on `.collect()`.

## Registration vs Execution

When you write `a.matmul(b)!!`, nothing is computed. The graph records a symbolic node describing a matrix multiply between `a` and `b`, infers the output shape, and returns a handle to the result tensor. The result has no data yet.

```c3
// These lines only build the graph — zero computation
Tensor z0 = input.matmul(w0)!! + b0;
Tensor a0 = z0.relu()!!;
Tensor logits = a0.matmul(w1)!! + b1;
Tensor loss = loss::softmax_cross_entropy(logits, target)!!.stable();

// This triggers the entire chain
loss.collect()!!;
```

## Compilation

On first `.collect()`, the device compiler transforms the symbolic subgraph into executable code. The compiler sees the full computation structure before generating anything, enabling:

- **Kernel fusion**: adjacent operations with `NONE` persistence compile into a single kernel
- **Device-specific optimization**: the same graph produces different code for CPU vs GPU
- **Memory planning**: intermediate buffer sizes are known at compile time

Compiled execution units are cached. Subsequent `.collect()` calls reuse them unless the graph structure has changed.

## Re-execution

The fingerprint system determines whether a cached result is still valid. Each execution unit tracks a fingerprint derived from its leaf dependencies. When leaf data changes (via `.set()`, `.bind()`, or `notify_mutation()`), the fingerprint mismatches and the unit re-executes.

If leaf data hasn't changed since the last execution, `.collect()` returns immediately — no recomputation.

## Execution Order

Execution proceeds via depth-first post-order traversal of dependencies. Diamond patterns (where two paths converge on the same tensor) are handled correctly — each tensor is computed exactly once. Intermediate memory is reclaimed automatically through reference counting as consumers finish.

## Forward and Backward

The same execution model applies to both forward and backward passes. Gradient tensors are ordinary lazy tensors with their own compute chains. Collecting a gradient triggers compilation and execution of the backward subgraph, just like collecting any forward tensor.
