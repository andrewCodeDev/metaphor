# Backward Pass

Gradient computation is split into two phases: graph construction and execution.

## Phase 1: Build the Gradient Graph

Calling `.backward()` on a tensor walks the forward graph in reverse topological order. For each operation, it invokes the corresponding gradient rule to register new lazy computation nodes. No gradient values are computed.

```c3
Tensor loss = loss::mse_loss(pred, target)!!.stable();
loss.backward();   // builds gradient graph, no execution
```

After this call, every tensor in the forward graph that has `.enable_grad()` set will have a corresponding gradient tensor accessible via `.grad()`.

## Phase 2: Execute

Gradients are ordinary lazy tensors. They materialize when collected, either explicitly or through `optim.step()` which collects all parameter gradients internally.

```c3
// Explicit gradient collection
Tensor dw = weight.grad();
dw.collect()!!;

// Or let the optimizer handle it
optim.step();   // collects all registered parameter gradients, then applies updates
```

See [optimizer.md](../core/optimizer.md) for the full optimizer API: SGD/Adam setup, learning rate scheduling, gradient clipping, and gradient accumulation across micro-batches.

## Gradient Rules

Each operation type has a symbolic backward rule that constructs the gradient expression from the forward inputs and the upstream gradient. For example:

- **add**: gradient passes through unchanged
- **mul (element-wise)**: `d/da (a * b) = grad * b`
- **matmul**: gradient is a transposed contraction
- **relu**: `grad * (x > 0)`
- **sigmoid**: `grad * y * (1 - y)`
- **log**: `grad / x`
- **max/min (element-wise)**: `grad * heaviside(a - b)` (gradient flows to the selected input)
- **reduce (sum)**: gradient is broadcast back to input shape

These rules compose. The backward of a fused kernel is built from the chain of individual rules.

## Gradient Accumulation

When multiple forward paths converge on the same tensor, `.backward()` chains their gradients into a symbolic sum.

```c3
// weight is used in two places
Tensor z0 = input.matmul(weight)!!;
Tensor z1 = other.matmul(weight)!!;
Tensor loss = (z0 + z1).reduce("ij->")!!;
loss.backward();
// weight.grad() = gradient from z0 path + gradient from z1 path
```

## Detach

`.detach()` blocks gradient flow through a tensor. The backward pass stops at detached tensors and does not propagate further.

```c3
Tensor frozen = pretrained_output.detach();
Tensor fine_tuned = frozen.matmul(adapter_weight)!!;
// Only adapter_weight receives gradients
```

## Interaction with Persistence

Gradient tensors inherit persistence from their forward counterparts. If a forward tensor is `STABLE`, its gradient is also promoted to `STABLE`. The backward pass needs forward activations to compute gradients. If a forward activation was freed because it was `UNSTABLE`, the system automatically recomputes it.

See [persistence.md](../core/persistence.md) for the full tradeoff discussion.
