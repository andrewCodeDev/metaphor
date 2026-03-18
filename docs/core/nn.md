# nn: Neural Network Primitives

The `nn` module provides compound operations that require specialized implementations. These exist for three reasons:

1. **Numeric stability.** Algorithms like softmax and log-softmax require max-subtraction and log-sum-exp to avoid overflow in float16/bfloat16. The `nn` implementations handle this internally with f32 accumulation.

2. **Dedicated kernels.** Operations like RMS normalization, selective scan, and causal convolution have fused CUDA/HIP kernels. The `nn` boundary ops dispatch directly to these implementations.

3. **Backward pass correctness.** Some gradients require access to the forward output. Sigmoid's gradient is `y * (1 - y)` where `y` is the sigmoid output. Softmax's backward needs the full output distribution. The `nn` boundary ops ensure these values survive for the backward pass.

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

`rms_norm` uses dedicated kernels with f32 internal accumulation for bfloat16 precision.

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

These have dedicated CUDA/HIP kernels for recurrences (selective scan) and sliding-window patterns (causal conv).

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

Each call generates a unique mask. This function is for training only; omit it during inference.

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

`softmax_cross_entropy` fuses the softmax and log into a single numerically stable computation via `log_softmax`.

### Regression

```c3
Tensor loss = loss::mse_loss(pred, target)!!;

// Weighted MSE
Tensor loss = loss::weighted_mse(pred, target, weights)!!;
```

## optimizer

See [optimizer.md](optimizer.md) for the full optimizer documentation.

