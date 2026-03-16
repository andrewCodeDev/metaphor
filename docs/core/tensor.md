# Tensor

Tensor is a lightweight handle wrapping a pointer to a TensorCore. It is the primary user-facing type — you build computation graphs, access data, and control training through tensors.

Multiple handles can reference the same underlying core.

## Return Type Conventions

The API uses two type aliases to signal what an operation does:

- **`GraphTensor`** (alias for `Tensor`) — a *new* tensor node was added to the computation graph. The returned handle points to a different core than the input. Operations like `matmul`, `relu`, `+`, `reshape` return `GraphTensor?`.

- **`Self`** (alias for `Tensor`) — the *same* tensor, returned for method chaining. No graph node was created. Operations like `fill`, `enable_grad`, `set_label`, `stable` return `Self`.

Both are `Tensor` at runtime — the distinction is purely for readability in the API. If a method returns `GraphTensor`, the graph grew. If it returns `Self`, it didn't.

## Creating Tensors

```c3
// Empty (allocated, unitialized memory)
Tensor x = tensor::empty(F32, dev, { 4, 16 })!!;

// Zeros / ones / constant
Tensor z = tensor::zeros(F32, dev, { 64 })!!;
Tensor o = tensor::ones(F32, dev, { 64 })!!;
Tensor c = tensor::constant(F32, dev, { 1 }, 0.5)!!;

// From existing data
float[8] data = { 1, 2, 3, 4, 5, 6, 7, 8 };
Tensor t = tensor::@from_data(dev, { 2, 4 }, &data)!!;

// Random with Kaiming initialization
Tensor w = tensor::random(F32, dev, { 784, 128 },
    RandType.NORMAL, kaiming_scale(784))!!.enable_grad();
```

All factory functions return optionals — use `!!` to force-unwrap or `!` to rethrow.

## Arithmetic

Operator overloads work with tensors and scalars. Broadcasting is automatic. Note: operator overloading panics if an operation fails. If you need to catch a potentially failing operation, use the helper equivalent: `x.add(y)` instead of `x + y`.

```c3
Tensor z = input.matmul(weight)!! + bias;    // linear layer
Tensor scaled = scores * (1.0f / math::sqrt((float)d_k));
Tensor residual = x + attention_output;
Tensor mean_pooled = summed / (float)seq_len;
```

The underlying `.add()`, `.sub()`, `.mul()`, `.div()` methods are still available.

## Activations and Element-wise Ops

```c3
Tensor a = z.relu()!!;
Tensor s = z.sigmoid()!!;
Tensor g = z.silu()!!;           // SiLU (swish)
Tensor t = z.tanh()!!;
Tensor l = z.log()!!;
Tensor e = z.exp()!!;
Tensor r = z.recip()!!;          // 1/x
Tensor sq = z.sqr()!!;           // x^2
Tensor sr = z.sqrt()!!;          // sqrt(x)
Tensor n = z.neg()!!;            // -x (also unary - operator)
Tensor cl = z.clamp(&lower, &upper)!!;
```

## Reductions

```c3
// Sum over sequence dimension: [batch, seq, dim] -> [batch, dim]
Tensor pooled = embeddings.reduce("ijk->ik")!!;
```

## Einsum

General tensor contractions using index notation. See [einsum.md](../reference/einsum.md) for the full pattern reference.

```c3
// Matrix multiply: bsd,hdk->bhsk (Q/K/V projection)
Tensor q = input.einsum(w_q, "bsd,hdk->bhsk")!!;

// Attention scores: bhik,bhjk->bhij (dot product over head dim)
Tensor scores = q.einsum(k, "bhik,bhjk->bhij")!!;

// Output projection: bhsk,hkd->bsd (merge heads)
Tensor output = attn.einsum(w_o, "bhsk,hkd->bsd")!!;
```

## Shape Operations

`reshape`, `squeeze`, and `unsqueeze` are zero-copy — they create a new view of the same underlying data with different shape metadata. No memory is allocated or moved.

```c3
Tensor r = t.reshape({ 2, 3, 4 })!!;    // new shape, same data
Tensor u = t.unsqueeze(0)!!;             // add dim: [4] -> [1, 4]
Tensor s = t.squeeze(2)!!;               // remove dim: [2, 3, 1] -> [2, 3]
Tensor p = t.transpose(0, 1)!!;          // swap dims (may require copy)
Tensor c = t.cast(Datatype.BF16)!!;      // dtype cast (produces new data)
Tensor g = table.gather(0, indices)!!;    // embedding lookup
```

### Slicing

`.slice()` extracts a sub-tensor along one or more dimensions. Each dimension gets a `SliceRange` with `start`, `end`, and `step`. Defaults: `start = 0`, `end = 0` (meaning full extent), `step = 0` (meaning 1).

The `offset` parameter skips leading dimensions — they're left as full-range implicitly. This avoids writing boilerplate `{ 0, 0, 1 }` entries for batch dimensions you don't care about.

From `examples/llama_generate.c3` — splitting a tensor in half along dim 2:

```c3
ulong half = x.shape().get(2) / 2;
Tensor x1 = x.slice({ { 0, half, 1 } }, offset: 2)!;   // first half of dim 2
Tensor x2 = x.slice({ { half, 0, 1 } }, offset: 2)!;    // second half (end=0 means full extent)
```

Without `offset`, you'd need to specify full-range entries for every preceding dimension:

```c3
// Equivalent but verbose — offset: 2 is cleaner
Tensor x1 = x.slice({ { 0, 0, 1 }, { 0, 0, 1 }, { 0, half, 1 } })!;
```

## Data I/O

### Writing

```c3
float[4] data = { 1.0, 2.0, 3.0, 4.0 };
tensor.set(data);                         // fixed array
tensor.set(0.0, count: seq_len);          // scalar fill
tensor.set(token_id, offset: pos);        // scalar at offset
tensor.fill(0.0);                         // fill all elements
```

### Reading

```c3
float loss_val;
loss.get(&loss_val);                      // single scalar

float[10] buf;
logits.get(&buf);                         // fixed array
```

Both handle host-to-device and device-to-host transfers automatically.

### Binding (zero-copy)

```c3
model.input.bind(batch.images);           // point at external buffer
```

The tensor does not own the bound buffer. All three methods (`.set()`, `.fill()`, `.bind()`) automatically notify the graph for re-execution.

## Gradient Control

```c3
Tensor w = tensor::empty(F32, dev, { 4, 4 })!!.enable_grad();
w.requires_grad();    // true

loss.backward();      // build gradient graph (lazy)
Tensor dw = w.grad(); // gradient handle (lazy until collected)

w.detach();           // stop gradient flow through this tensor
w.attach();           // resume gradient flow
```

## Execution

```c3
loss.collect()!!;     // compile + execute the full dependency chain
```

Nothing executes until `.collect()` is called. See [graph.md](graph.md) for details on lazy execution.

## Persistence

```c3
Tensor loss = mse_loss(pred, target)!!.stable();   // keep data alive
Tensor mid = a.matmul(b)!!.unstable();             // free after use
```

See [persistence.md](persistence.md) for the full discussion.

## Metadata

```c3
tensor.shape();                  // Shape* with .get(dim), .len, .size()
tensor.dtype();                  // Datatype enum
tensor.device();                 // DeviceReference
tensor.set_label("my_weight");   // for debug output
tensor.print();                  // numpy-style dump to stderr
```
