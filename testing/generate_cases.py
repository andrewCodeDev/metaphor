#!/usr/bin/env python3
"""
Generate test vectors using PyTorch for Metaphor numerical validation.

Each test case includes:
- Input tensors with requires_grad=True
- Forward pass output
- Backward pass gradients

Output: SafeTensors files in test_vectors/ directory
"""

import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

OUTPUT_DIR = Path(__file__).parent / "test_vectors"


def save_case(
    name: str,
    operation: str,
    description: str,
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    gradients: dict[str, torch.Tensor],
    intermediates: dict[str, torch.Tensor] | None = None,
    extra_metadata: dict[str, str] | None = None,
):
    """
    Save a test case to SafeTensors file.

    Tensor naming convention:
    - input.<name> : Input tensors
    - output       : Forward pass output
    - grad.<name>  : Gradient tensors
    - intermediate.<name> : Intermediate values (optional)
    """
    tensors = {}

    # Store inputs
    for key, tensor in inputs.items():
        tensors[f"input.{key}"] = tensor.detach().contiguous()

    # Store output
    if output.dim() == 0:
        # Scalar: save as 1-element tensor
        tensors["output"] = output.detach().reshape(1).contiguous()
    else:
        tensors["output"] = output.detach().contiguous()

    # Store gradients
    for key, tensor in gradients.items():
        if tensor.dim() == 0:
            tensors[f"grad.{key}"] = tensor.detach().reshape(1).contiguous()
        else:
            tensors[f"grad.{key}"] = tensor.detach().contiguous()

    # Store intermediates if provided
    if intermediates:
        for key, tensor in intermediates.items():
            tensors[f"intermediate.{key}"] = tensor.detach().contiguous()

    # Build metadata
    metadata = {
        "name": name,
        "operation": operation,
        "description": description,
        "input_keys": ",".join(inputs.keys()),
        "grad_keys": ",".join(gradients.keys()),
    }

    # Add shape info for easier debugging
    for key, tensor in inputs.items():
        metadata[f"shape.input.{key}"] = ",".join(map(str, tensor.shape))
    metadata["shape.output"] = ",".join(map(str, output.shape))

    if extra_metadata:
        metadata.update(extra_metadata)

    # Save
    filename = OUTPUT_DIR / f"{name}.safetensors"
    save_file(tensors, filename, metadata=metadata)
    print(f"  Generated: {name}")


# -----------------------------------------------------------------------------
# Matrix Multiplication Tests
# -----------------------------------------------------------------------------

def generate_matmul_cases():
    print("Generating matmul cases...")

    # Basic 2D matmul
    torch.manual_seed(42)
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(3, 4, requires_grad=True)
    out = a @ b
    out.sum().backward()

    save_case(
        name="matmul_2x3_3x4",
        operation="matmul",
        description="Basic 2D matrix multiplication",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Square matrices
    torch.manual_seed(43)
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4, requires_grad=True)
    out = a @ b
    out.sum().backward()

    save_case(
        name="matmul_4x4_4x4",
        operation="matmul",
        description="Square matrix multiplication",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Vector-matrix (1D @ 2D)
    torch.manual_seed(44)
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(3, 4, requires_grad=True)
    out = a @ b
    out.sum().backward()

    save_case(
        name="matmul_vec_mat",
        operation="matmul",
        description="Vector-matrix multiplication (1D @ 2D)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Batched matmul (3D)
    torch.manual_seed(45)
    a = torch.randn(2, 3, 4, requires_grad=True)
    b = torch.randn(2, 4, 5, requires_grad=True)
    out = a @ b
    out.sum().backward()

    save_case(
        name="matmul_batched_2x3x4_2x4x5",
        operation="matmul",
        description="Batched matrix multiplication (3D tensors)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )


# -----------------------------------------------------------------------------
# Linear Transform Tests (weight @ input + bias)
# -----------------------------------------------------------------------------

def generate_linear_cases():
    print("Generating linear transform cases...")

    # Basic linear: y = xW^T + b
    torch.manual_seed(50)
    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    out = F.linear(x, weight, bias)
    out.sum().backward()

    save_case(
        name="linear_2x3_to_2x4",
        operation="linear",
        description="Linear transform with bias: y = xW^T + b",
        inputs={"x": x, "weight": weight, "bias": bias},
        output=out,
        gradients={"x": x.grad, "weight": weight.grad, "bias": bias.grad},
    )

    # Linear without bias
    torch.manual_seed(51)
    x = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    out = F.linear(x, weight, bias=None)
    out.sum().backward()

    save_case(
        name="linear_no_bias_4x8_to_4x16",
        operation="linear",
        description="Linear transform without bias: y = xW^T",
        inputs={"x": x, "weight": weight},
        output=out,
        gradients={"x": x.grad, "weight": weight.grad},
    )

    # Single sample
    torch.manual_seed(52)
    x = torch.randn(1, 5, requires_grad=True)
    weight = torch.randn(3, 5, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)
    out = F.linear(x, weight, bias)
    out.sum().backward()

    save_case(
        name="linear_single_sample",
        operation="linear",
        description="Linear transform with single sample",
        inputs={"x": x, "weight": weight, "bias": bias},
        output=out,
        gradients={"x": x.grad, "weight": weight.grad, "bias": bias.grad},
    )


# -----------------------------------------------------------------------------
# Broadcasting Tests
# -----------------------------------------------------------------------------

def generate_broadcast_cases():
    print("Generating broadcasting cases...")

    # Add: matrix + row vector
    torch.manual_seed(60)
    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    out = a + b
    out.sum().backward()

    save_case(
        name="broadcast_add_matrix_row",
        operation="add",
        description="Broadcasting add: (3,4) + (4,) -> (3,4)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Add: matrix + column vector
    torch.manual_seed(61)
    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, 1, requires_grad=True)
    out = a + b
    out.sum().backward()

    save_case(
        name="broadcast_add_matrix_col",
        operation="add",
        description="Broadcasting add: (3,4) + (3,1) -> (3,4)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Mul: 3D tensor with 2D
    torch.manual_seed(62)
    a = torch.randn(2, 3, 4, requires_grad=True)
    b = torch.randn(3, 4, requires_grad=True)
    out = a * b
    out.sum().backward()

    save_case(
        name="broadcast_mul_3d_2d",
        operation="mul",
        description="Broadcasting mul: (2,3,4) * (3,4) -> (2,3,4)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Scalar broadcast
    torch.manual_seed(63)
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.tensor(2.5, requires_grad=True)
    out = a * b
    out.sum().backward()

    save_case(
        name="broadcast_mul_scalar",
        operation="mul",
        description="Broadcasting mul with scalar: (2,3) * scalar -> (2,3)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )

    # Outer product style: (3,1) * (1,4)
    torch.manual_seed(64)
    a = torch.randn(3, 1, requires_grad=True)
    b = torch.randn(1, 4, requires_grad=True)
    out = a * b
    out.sum().backward()

    save_case(
        name="broadcast_mul_outer",
        operation="mul",
        description="Broadcasting outer product style: (3,1) * (1,4) -> (3,4)",
        inputs={"a": a, "b": b},
        output=out,
        gradients={"a": a.grad, "b": b.grad},
    )


# -----------------------------------------------------------------------------
# Activation Function Tests
# -----------------------------------------------------------------------------

def generate_activation_cases():
    print("Generating activation function cases...")

    # ReLU
    torch.manual_seed(70)
    x = torch.randn(3, 4, requires_grad=True)
    out = F.relu(x)
    out.sum().backward()

    save_case(
        name="activation_relu",
        operation="relu",
        description="ReLU activation function",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )

    # Sigmoid
    torch.manual_seed(71)
    x = torch.randn(3, 4, requires_grad=True)
    out = torch.sigmoid(x)
    out.sum().backward()

    save_case(
        name="activation_sigmoid",
        operation="sigmoid",
        description="Sigmoid activation function",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )

    # Tanh
    torch.manual_seed(72)
    x = torch.randn(3, 4, requires_grad=True)
    out = torch.tanh(x)
    out.sum().backward()

    save_case(
        name="activation_tanh",
        operation="tanh",
        description="Tanh activation function",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )

    # Softmax (along last dim)
    torch.manual_seed(73)
    x = torch.randn(2, 5, requires_grad=True)
    out = F.softmax(x, dim=-1)
    target_weights = torch.randn(2, 5)
    loss = (out * target_weights).sum()
    loss.backward()

    save_case(
        name="activation_softmax",
        operation="softmax",
        description="Softmax activation (dim=-1) with weighted sum backward",
        inputs={"x": x, "target_weights": target_weights},
        output=out,
        gradients={"x": x.grad},
        extra_metadata={"notes": "Backward computed via (out * target_weights).sum()"},
    )

    # GELU (using tanh approximation to match Metaphor implementation)
    torch.manual_seed(74)
    x = torch.randn(3, 4, requires_grad=True)
    out = F.gelu(x, approximate='tanh')
    out.sum().backward()

    save_case(
        name="activation_gelu",
        operation="gelu",
        description="GELU activation function (tanh approximation)",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )

    # SiLU / Swish
    torch.manual_seed(75)
    x = torch.randn(3, 4, requires_grad=True)
    out = F.silu(x)
    out.sum().backward()

    save_case(
        name="activation_silu",
        operation="silu",
        description="SiLU/Swish activation function (x * sigmoid(x))",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )


# -----------------------------------------------------------------------------
# Reduction Tests
# -----------------------------------------------------------------------------

def generate_reduction_cases():
    print("Generating reduction cases...")

    # Sum all
    torch.manual_seed(80)
    x = torch.randn(3, 4, requires_grad=True)
    out = x.sum()
    out.backward()

    save_case(
        name="reduce_sum_all",
        operation="sum",
        description="Sum reduction over all elements",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )

    # Sum along axis 0
    torch.manual_seed(81)
    x = torch.randn(3, 4, requires_grad=True)
    out = x.sum(dim=0)
    out.sum().backward()

    save_case(
        name="reduce_sum_axis0",
        operation="sum",
        description="Sum reduction along axis 0",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
        extra_metadata={"reduce_dim": "0"},
    )

    # Sum along axis 1
    torch.manual_seed(82)
    x = torch.randn(3, 4, requires_grad=True)
    out = x.sum(dim=1)
    out.sum().backward()

    save_case(
        name="reduce_sum_axis1",
        operation="sum",
        description="Sum reduction along axis 1",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
        extra_metadata={"reduce_dim": "1"},
    )

    # Mean
    torch.manual_seed(83)
    x = torch.randn(3, 4, requires_grad=True)
    out = x.mean()
    out.backward()

    save_case(
        name="reduce_mean_all",
        operation="mean",
        description="Mean reduction over all elements",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
    )

    # Max along axis
    torch.manual_seed(84)
    x = torch.randn(3, 4, requires_grad=True)
    out, _ = x.max(dim=1)
    out.sum().backward()

    save_case(
        name="reduce_max_axis1",
        operation="max",
        description="Max reduction along axis 1",
        inputs={"x": x},
        output=out,
        gradients={"x": x.grad},
        extra_metadata={"reduce_dim": "1"},
    )


# -----------------------------------------------------------------------------
# Composite Operations (chained)
# -----------------------------------------------------------------------------

def generate_composite_cases():
    print("Generating composite operation cases...")

    # MLP-style: linear -> relu -> linear
    torch.manual_seed(90)
    x = torch.randn(2, 4, requires_grad=True)
    w1 = torch.randn(8, 4, requires_grad=True)
    b1 = torch.randn(8, requires_grad=True)
    w2 = torch.randn(3, 8, requires_grad=True)
    b2 = torch.randn(3, requires_grad=True)

    h = F.relu(F.linear(x, w1, b1))
    out = F.linear(h, w2, b2)
    out.sum().backward()

    save_case(
        name="composite_mlp_layer",
        operation="composite",
        description="MLP: linear(4->8) -> relu -> linear(8->3)",
        inputs={"x": x, "w1": w1, "b1": b1, "w2": w2, "b2": b2},
        output=out,
        gradients={
            "x": x.grad,
            "w1": w1.grad,
            "b1": b1.grad,
            "w2": w2.grad,
            "b2": b2.grad,
        },
        intermediates={"h": h},
    )

    # Attention-style: softmax(QK^T / sqrt(d)) @ V
    torch.manual_seed(91)
    batch, seq, d = 2, 4, 8
    Q = torch.randn(batch, seq, d, requires_grad=True)
    K = torch.randn(batch, seq, d, requires_grad=True)
    V = torch.randn(batch, seq, d, requires_grad=True)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    out.sum().backward()

    save_case(
        name="composite_attention",
        operation="composite",
        description="Scaled dot-product attention: softmax(QK^T/sqrt(d)) @ V",
        inputs={"Q": Q, "K": K, "V": V},
        output=out,
        gradients={"Q": Q.grad, "K": K.grad, "V": V.grad},
        intermediates={"scores": scores, "attn_weights": attn},
        extra_metadata={"d": "8"},
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating test vectors in: {OUTPUT_DIR}\n")

    generate_matmul_cases()
    generate_linear_cases()
    generate_broadcast_cases()
    generate_activation_cases()
    generate_reduction_cases()
    generate_composite_cases()

    # Count generated files
    files = list(OUTPUT_DIR.glob("*.safetensors"))
    print(f"\nGenerated {len(files)} test cases.")


if __name__ == "__main__":
    main()
