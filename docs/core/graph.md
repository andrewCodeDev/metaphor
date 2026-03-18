# Graph

The graph is Metaphor's central coordinator. It owns all tensors, tracks their operations, manages gradients, and orchestrates execution. Every tensor belongs to exactly one graph.

## Lifecycle

The global graph initializes automatically on first use. All tensor operations implicitly use it. For isolated computation scopes, use `@subgraph()` (see below).

## Lazy Execution

No operation produces data at registration time. Each operation records a semantic description of the computation. Data materializes only on `.collect()`.

From `examples/dnn.c3`, the entire forward pass builds a lazy graph:

```c3
// None of this executes yet; just records operations
Tensor z0 = input.matmul(w0)!! + b0;
Tensor a0 = z0.relu()!!;
Tensor z1 = a0.matmul(w1)!! + b1;
Tensor a1 = z1.relu()!!;
Tensor logits = a1.matmul(w2)!! + b2;
Tensor loss = loss::softmax_cross_entropy(logits, target)!!.stable();

// THIS triggers compilation + execution of the full chain
loss.collect()!!;
```

If no compiled kernel exists for the target tensor, the device compiler produces one from the symbolic subgraph, then dispatches it. Subsequent calls reuse the compiled kernel unless inputs have changed.

## Backward Pass

`backward()` walks the forward graph in reverse and registers gradient computation nodes. Gradients are lazy tensors and only execute on `.collect()`.

From `examples/least_squares.c3`:

```c3
// Forward
Tensor pred = input.matmul(weight)!! + bias;
Tensor mse = loss::mse_loss(pred, target)!!.stable();

// Build backward graph (no execution yet)
mse.backward();

// Training loop: forward + backward execute on collect + step
for (usz epoch = 0; epoch < NUM_EPOCHS; epoch++)
{
    // set your input tensors with new data //

    mse.collect()!!;   // executes forward
    optim.step();      // collects gradients, applies updates
}
```

Gradient tensors are ordinary tensors. They participate in fusion, have their own compute chains, and can themselves be differentiated.

## Persistence

Persistence controls compilation boundaries and data lifetime. See [persistence.md](persistence.md) for the full discussion of levels, modes, and tradeoffs.

## Leaf Mutation Tracking

The graph uses fingerprint-based tracking to avoid redundant re-execution. When a leaf tensor's data changes, the graph must be notified so downstream computations re-execute with the new values.

### Automatic notification

`.set()`, `.bind()`, and `.fill()` all call `notify_leaf_mutation()` internally:

```c3
// .set() notifies automatically; downstream ops will re-execute
input_ids.set(token_data);
target.set(label_data);
loss.collect()!!;    // picks up new input values
```

### Manual notification

When data is modified through raw pointers (e.g., the optimizer updating weights directly), call `notify_mutation()` explicitly:

```c3
mse.collect()!!;
optim.step();
// Optimizer wrote weights via raw pointers; graph doesn't know
graph::notify_mutation(input);
```
Only LEAF tensors can be mutation points. Updating a non-leaf requires collecting it first, then collecting any tensor downstream of it.

## Data I/O

### Writing data into tensors

```c3
// Fixed array: copies into tensor, notifies graph
float[4] data = { 1.0, 2.0, 3.0, 4.0 };
tensor.set(data);

// Scalar fill
tensor.set(0.0, count: seq_len);

// Scalar at offset
tensor.set(token_id, offset: pos);
```

### Reading data from tensors

```c3
// Single scalar
float loss_val;
loss.get(&loss_val);

// Fixed array
float[10] buf;
logits.get(&buf);
```

Both `set()` and `get()` handle host-to-device and device-to-host transfers automatically.

### Binding external buffers (zero-copy)

From `examples/mnist.c3`, bind dataset batch directly:

```c3
while (dataset::Batch* batch = train_reader.next())
{
    model.input.bind(batch.images);
    model.target.bind(batch.labels);
    model.loss_tensor.collect()!!;
    optim.step();
}
```

`.bind()` points the tensor at an external buffer without copying and notifies the graph.

## Subgraph Isolation

Subgraphs create a temporary graph scope for isolated computation. All operations inside the scope use the subgraph. On exit, the subgraph is destroyed. Selected results can be exported as leaves on the parent graph.

From `testing/graph/graph_test.c3`:

```c3
Tensor result; // result tensor
graph::@subgraph()
{
    Tensor a = tensor::@from_data(dev, { 3 }, &a_data)!!.enable_grad();
    Tensor b = tensor::@from_data(dev, { 2 }, &b_data)!!;
    Tensor c = tensor_ops::cat({ a, b }, 0)!!;

    c.backward();
    a.grad().collect()!!;

    // export() moves result to root graph as a LEAF (zero-copy)
    result = graph::export(c);
};
// Subgraph destroyed here; exported tensor lives on
```

Use cases:
- **Test isolation**: each test gets a clean graph, no cross-contamination
- **Scoped computation**: build a temporary graph for initialization, export the result
- **Memory control**: subgraph intermediates are freed on scope exit
