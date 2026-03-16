# Persistence

Persistence controls two things: where the compiler places materialization boundaries, and how long tensor data survives after execution. It is the primary mechanism for managing memory during training and inference.

## Levels

Four levels, lowest to highest:

| Level | Boundary? | Data lifetime | Typical use |
|-------|-----------|---------------|-------------|
| `NONE` | No | Fused into consumer kernel | Intermediate arithmetic |
| `UNSTABLE` | Yes | Freed after last consumer finishes | Forward activations during training |
| `STABLE` | Yes | Preserved across execution boundaries | Loss tensors, KV caches, state |
| `LEAF` | Yes | Never freed by system | Weights, inputs, targets |

**Boundary** means the tensor must be materialized to its own buffer. Non-boundary tensors (`NONE`) can be fused or promoted based on the compiler backend. They never exist in memory as standalone allocations.

## Marking Individual Tensors

```c3
// .stable() — data survives across collect() calls
Tensor loss = loss::mse_loss(pred, target)!!.stable();

// .unstable() — data freed after last consumer
Tensor mid = a.matmul(b)!!.unstable();
```

Both return `self` for chaining. User overrides always take precedence over the graph default.

## Default Persistence Mode

Every graph has a default that controls what happens to intermediate tensors that haven't been explicitly marked.

```c3
graph::set_default_persistence(STABLE);    // preserve intermediates
graph::set_default_persistence(UNSTABLE);  // free after last consumer (default)
```

Only `STABLE` and `UNSTABLE` are valid defaults — `NONE` and `LEAF` are structural properties set per-tensor.

### Unstable mode (default)

Intermediates are freed after their last consumer finishes. You opt in to keeping specific tensors alive with `.stable()`. This minimizes peak memory at the cost of recomputation when the backward pass needs forward activations that were already freed.

### Stable mode

Intermediates are preserved indefinitely. You opt in to freeing tensors with `.unstable()`. This eliminates recomputation and works naturally for stateful models where caches and hidden states must persist between steps, but it holds all intermediates in memory simultaneously.

Both modes produce identical numerical results. The choice is purely a memory-versus-compute tradeoff.

## How Persistence Interacts with Training

In a training loop, the forward pass produces activations that the backward pass needs for gradient computation. The persistence mode determines whether those activations are still in memory when the backward pass runs.

**Stable mode** (simpler, more memory): all forward activations survive, backward pass reads them directly.

**Unstable mode** (less memory, more compute): forward activations are freed, backward pass recomputes them from their inputs. This is automatic — the execution system detects missing data and re-executes the forward subgraph.

## How Persistence Interacts with Compilation

The compiler uses persistence boundaries to decide where to split computation into separate kernels. Two adjacent operations where the intermediate is `NONE` can be fused into a single kernel. An intermediate marked `UNSTABLE` or higher forces a kernel boundary — the first kernel writes the result to memory, the second kernel reads it.

This means persistence affects performance even when memory isn't a concern. Over-marking tensors as `STABLE` can prevent fusion opportunities. Under-marking can cause excessive recomputation.

## LEAF Persistence

Tensors created with `tensor::empty()`, `tensor::zeros()`, or other factory functions are automatically promoted to `LEAF`. Leaf data is never freed by the system — the user owns it.

The mutation tracking system (fingerprinting) only operates on LEAF tensors. When a leaf's data changes via `.set()`, `.bind()`, or `.fill()`, the graph is notified and downstream computations re-execute.

## Subgraph Inheritance

Subgraphs inherit the default persistence from their parent graph. This means a `@subgraph()` inside a stable-mode context will also default to stable.

## Choosing a Mode

| Scenario | Recommended mode | Why |
|----------|-----------------|-----|
| Small models, overfitting demos | STABLE | Simplest; memory isn't a concern |
| Large model training | UNSTABLE + selective `.stable()` | Minimize peak memory |
| Inference / generation | UNSTABLE + selective `.stable()` | State tensors (KV cache) must persist |
