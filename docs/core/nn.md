# nn — Neural Network Primitives

The `nn` module provides operations that can't be expressed as simple element-wise or contraction operations on tensors. These exist for three reasons:

1. **Numeric stability.** Algorithms like softmax and log-softmax require careful max-subtraction and log-sum-exp tricks to avoid overflow in float16/bfloat16. The naive `exp(x) / sum(exp(x))` formula produces infinities on half-precision inputs. The `nn` implementations handle this internally with f32 accumulation.

2. **Dedicated kernels.** Operations like RMS normalization, selective scan, and causal convolution have fused CUDA/HIP kernels that are significantly faster than what the JIT compiler would generate from decomposed element-wise ops. The `nn` boundary ops dispatch directly to these optimized implementations.

3. **Backward pass correctness.** Some gradients require access to the forward output (not just the inputs). Sigmoid's gradient is `y * (1 - y)` where `y` is the sigmoid output. Softmax's backward needs the full output distribution. The `nn` boundary ops ensure these values survive for the backward pass, even when default persistence would otherwise free them.

## functional

Higher-level operations built from tensor primitives. Import with `import metaphor::nn::functional`.

### Softmax

```c3
// Numerically stable softmax along dimension
Tensor probs = functional::softmax(logits, 3)!!;     // dim=3 (key positions)
Tensor log_p = functional::log_softmax(logits, 1)!!;  // dim=1 (classes)
```

Implemented as a dedicated boundary op with optimized host/CUDA kernels and correct backward pass.

### Normalization

```c3
// RMS norm with learnable weight (used in transformers)
Tensor normed = functional::rms_norm(x, weight, 1e-5)!!;

// Other normalizations
Tensor l2 = functional::l2_normalize(x, dim, 1e-6)!!;
Tensor zs = functional::zscore(x, dim, 1e-6)!!;
```

`rms_norm` uses dedicated kernels with f32 internal accumulation — critical for bfloat16 training where naive reduction would lose precision.

### Linear

```c3
// y = x @ weight^T + bias
// weight shape: [out_features, in_features]
Tensor y = functional::linear(x, weight, bias)!!;
Tensor y = functional::linear(x, weight)!!;  // no bias
```

### Sequence Operations

```c3
// Fused selective scan (Mamba SSM)
//   h[t] = a_bar[t] * h[t-1] + b_bar_x[t]
Tensor h = functional::selective_scan(a_bar, b_bar_x)!!;

// Fused depthwise causal 1D convolution
//   x: [B, E, L], weight: [E, 1, K], bias: [E]
Tensor y = functional::causal_conv1d(x, weight, bias)!!;
```

These have dedicated CUDA/HIP kernels. The JIT compiler cannot fuse recurrences (selective scan) or sliding-window patterns (causal conv) from decomposed ops.

### Reductions

```c3
Tensor m = functional::mean(x)!!;              // scalar mean
Tensor md = functional::mean_dim(x, 1)!!;      // mean over dim
Tensor s = functional::sum_all(x)!!;            // scalar sum
```

### Dropout

```c3
// Inverted dropout: zeros elements with probability p, scales rest by 1/(1-p)
Tensor dropped = functional::dropout(x, 0.1)!!;
```

Each call generates a unique mask. Skip at inference time — there is no train/eval mode toggle.

## loss

Loss functions with correct gradient behavior. Import with `import metaphor::nn::loss`.

### Classification

```c3
// Softmax + cross-entropy (numerically stable, preferred for classification)
// Takes raw logits, not probabilities
Tensor loss = loss::softmax_cross_entropy(logits, one_hot_target)!!;

// With per-position weights (e.g., masking padding tokens)
Tensor loss = loss::softmax_cross_entropy(logits, target, weights)!!;

// Binary cross-entropy (takes sigmoid output)
Tensor loss = loss::binary_cross_entropy(sigmoid_output, target)!!;
```

`softmax_cross_entropy` is the standard choice for multi-class problems. It fuses the softmax and log into a single numerically stable computation via `log_softmax`.

### Regression

```c3
Tensor loss = loss::mse_loss(pred, target)!!;

// Weighted MSE
Tensor loss = loss::weighted_mse(pred, target, weights)!!;
```

## optimizer

See [optimizer.md](optimizer.md) for the full optimizer documentation.

## When to Use nn vs Raw Tensor Ops

Use `nn::functional` when:
- The operation involves a reduction that must be numerically stable (softmax, rms_norm)
- A fused kernel exists that would be significantly faster (selective_scan, causal_conv1d)
- The backward pass needs the forward output to survive (softmax, sigmoid boundary ops)

Use raw tensor ops when:
- The operation is a straightforward contraction or element-wise computation
- The JIT compiler can handle it efficiently (matmul, relu, add, reshape)
- You need custom behavior not covered by the `nn` API
