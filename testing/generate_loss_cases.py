#!/usr/bin/env python3
"""
Generate loss function test vectors using PyTorch for Metaphor numerical validation.

Each test case includes:
- Input tensors (predictions, targets)
- Forward pass result (loss value)
- Backward pass result (gradients)

Output: SafeTensors files in test_vectors/ directory
"""

import torch
from pathlib import Path
from safetensors.torch import save_file

OUTPUT_DIR = Path(__file__).parent / "test_vectors"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_loss_case(
    name: str,
    loss_type: str,
    description: str,
    inputs: dict[str, torch.Tensor],
    loss_value: torch.Tensor,
    gradients: dict[str, torch.Tensor],
):
    """
    Save a loss test case to SafeTensors file.

    Tensor naming convention:
    - input.<name>    : Input tensors
    - loss            : Scalar loss value
    - grad.<name>     : Gradient w.r.t. input
    """
    tensors = {}

    # Store inputs
    for key, tensor in inputs.items():
        tensors[f"input.{key}"] = tensor.detach().contiguous()

    # Store loss
    tensors["loss"] = loss_value.detach().contiguous()

    # Store gradients
    for key, tensor in gradients.items():
        tensors[f"grad.{key}"] = tensor.detach().contiguous()

    # Build metadata
    metadata = {
        "name": name,
        "loss_type": loss_type,
        "description": description,
        "input_keys": ",".join(inputs.keys()),
    }

    # Add shape info
    for key, tensor in inputs.items():
        metadata[f"shape.{key}"] = ",".join(map(str, tensor.shape))

    # Save
    filename = OUTPUT_DIR / f"{name}.safetensors"
    save_file(tensors, filename, metadata=metadata)
    print(f"  Generated: {name}")


# -----------------------------------------------------------------------------
# MSE Loss Tests
# -----------------------------------------------------------------------------

def generate_mse_cases():
    print("Generating MSE loss cases...")

    # Basic MSE: simple 2D case
    torch.manual_seed(42)
    pred = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 4, dtype=torch.float64)

    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    save_loss_case(
        name="loss_mse_basic",
        loss_type="mse",
        description="Basic MSE on 4x4 tensors",
        inputs={"pred": pred.detach().clone(), "target": target},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )

    # MSE: batch case
    torch.manual_seed(123)
    pred = torch.randn(8, 10, dtype=torch.float64, requires_grad=True)
    target = torch.randn(8, 10, dtype=torch.float64)

    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    save_loss_case(
        name="loss_mse_batch",
        loss_type="mse",
        description="MSE on batch of 8x10",
        inputs={"pred": pred.detach().clone(), "target": target},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# MAE Loss Tests
# -----------------------------------------------------------------------------

def generate_mae_cases():
    print("Generating MAE loss cases...")

    # Basic MAE
    torch.manual_seed(42)
    pred = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 4, dtype=torch.float64)

    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    save_loss_case(
        name="loss_mae_basic",
        loss_type="mae",
        description="Basic MAE on 4x4 tensors",
        inputs={"pred": pred.detach().clone(), "target": target},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_cross_entropy_cases():
    print("Generating cross-entropy loss cases...")

    # Cross-entropy with softmax probabilities
    torch.manual_seed(42)
    logits = torch.randn(4, 5, dtype=torch.float64)
    pred = torch.softmax(logits, dim=-1).requires_grad_(True)
    # One-hot targets
    target_idx = torch.tensor([0, 2, 1, 4])
    target = torch.zeros(4, 5, dtype=torch.float64)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)

    # Manual cross-entropy: -mean(sum(target * log(pred)))
    eps = 1e-7
    log_pred = torch.log(pred.clamp(min=eps))
    loss = -torch.mean(torch.sum(target * log_pred, dim=-1))
    loss.backward()

    save_loss_case(
        name="loss_cross_entropy_basic",
        loss_type="cross_entropy",
        description="Cross-entropy with softmax probs, one-hot targets",
        inputs={"pred": pred.detach().clone(), "target": target},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Binary Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_bce_cases():
    print("Generating binary cross-entropy loss cases...")

    # BCE basic
    torch.manual_seed(42)
    pred = torch.sigmoid(torch.randn(4, 4, dtype=torch.float64)).requires_grad_(True)
    target = torch.randint(0, 2, (4, 4), dtype=torch.float64)

    loss = torch.nn.functional.binary_cross_entropy(pred, target)
    loss.backward()

    save_loss_case(
        name="loss_bce_basic",
        loss_type="binary_cross_entropy",
        description="Binary cross-entropy on 4x4",
        inputs={"pred": pred.detach().clone(), "target": target},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Softmax Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_softmax_ce_cases():
    print("Generating softmax cross-entropy loss cases...")

    # Softmax CE (from logits)
    torch.manual_seed(42)
    logits = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    target_idx = torch.tensor([0, 2, 1, 4])
    target = torch.zeros(4, 5, dtype=torch.float64)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)

    # PyTorch cross_entropy takes logits directly
    loss = torch.nn.functional.cross_entropy(logits, target_idx)
    loss.backward()

    save_loss_case(
        name="loss_softmax_ce_basic",
        loss_type="softmax_cross_entropy",
        description="Softmax cross-entropy from logits",
        inputs={"logits": logits.detach().clone(), "target": target},
        loss_value=loss,
        gradients={"logits": logits.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Weighted MSE Loss Tests
# -----------------------------------------------------------------------------

def generate_weighted_mse_cases():
    print("Generating weighted MSE loss cases...")

    # Weighted MSE: different weights per element
    torch.manual_seed(42)
    pred = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 4, dtype=torch.float64)
    weight = torch.rand(4, 4, dtype=torch.float64) + 0.1  # Ensure positive weights

    # Manual weighted MSE: sum(weight * (pred - target)^2) / sum(weight)
    diff = pred - target
    weighted_sq = weight * diff ** 2
    loss = weighted_sq.sum() / weight.sum()
    loss.backward()

    save_loss_case(
        name="loss_weighted_mse_basic",
        loss_type="weighted_mse",
        description="Weighted MSE with per-element weights",
        inputs={"pred": pred.detach().clone(), "target": target, "weight": weight},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Weighted MAE Loss Tests
# -----------------------------------------------------------------------------

def generate_weighted_mae_cases():
    print("Generating weighted MAE loss cases...")

    # Weighted MAE: different weights per element
    torch.manual_seed(42)
    pred = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 4, dtype=torch.float64)
    weight = torch.rand(4, 4, dtype=torch.float64) + 0.1  # Ensure positive weights

    # Manual weighted MAE: sum(weight * |pred - target|) / sum(weight)
    diff = torch.abs(pred - target)
    weighted_diff = weight * diff
    loss = weighted_diff.sum() / weight.sum()
    loss.backward()

    save_loss_case(
        name="loss_weighted_mae_basic",
        loss_type="weighted_mae",
        description="Weighted MAE with per-element weights",
        inputs={"pred": pred.detach().clone(), "target": target, "weight": weight},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Weighted Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_weighted_cross_entropy_cases():
    print("Generating weighted cross-entropy loss cases...")

    # Per-sample weighted cross-entropy
    torch.manual_seed(42)
    logits = torch.randn(4, 5, dtype=torch.float64)
    pred = torch.softmax(logits, dim=-1).requires_grad_(True)
    # One-hot targets
    target_idx = torch.tensor([0, 2, 1, 4])
    target = torch.zeros(4, 5, dtype=torch.float64)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)
    # Per-sample weights
    weight = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)

    # Manual weighted CE: -sum(weight * sum(target * log(pred), axis=-1)) / sum(weight)
    eps = 1e-7
    log_pred = torch.log(pred.clamp(min=eps))
    ce_per_sample = torch.sum(target * log_pred, dim=-1)
    weighted_ce = weight * ce_per_sample
    loss = -weighted_ce.sum() / weight.sum()
    loss.backward()

    save_loss_case(
        name="loss_weighted_cross_entropy_basic",
        loss_type="weighted_cross_entropy",
        description="Weighted cross-entropy with per-sample weights",
        inputs={"pred": pred.detach().clone(), "target": target, "weight": weight},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Class-Weighted Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_class_weighted_cross_entropy_cases():
    print("Generating class-weighted cross-entropy loss cases...")

    # Per-class weighted cross-entropy
    torch.manual_seed(42)
    logits = torch.randn(4, 5, dtype=torch.float64)
    pred = torch.softmax(logits, dim=-1).requires_grad_(True)
    # One-hot targets
    target_idx = torch.tensor([0, 2, 1, 4])
    target = torch.zeros(4, 5, dtype=torch.float64)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)
    # Per-class weights (e.g., inverse class frequency)
    class_weight = torch.tensor([1.0, 2.0, 1.5, 0.5, 3.0], dtype=torch.float64)

    # Manual class-weighted CE: -mean(sum(class_weight * target * log(pred), axis=-1))
    eps = 1e-7
    log_pred = torch.log(pred.clamp(min=eps))
    weighted_prod = class_weight.unsqueeze(0) * target * log_pred
    ce_per_sample = torch.sum(weighted_prod, dim=-1)
    loss = -ce_per_sample.mean()
    loss.backward()

    save_loss_case(
        name="loss_class_weighted_cross_entropy_basic",
        loss_type="class_weighted_cross_entropy",
        description="Class-weighted cross-entropy with per-class weights",
        inputs={"pred": pred.detach().clone(), "target": target, "class_weight": class_weight},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Weighted Binary Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_weighted_bce_cases():
    print("Generating weighted binary cross-entropy loss cases...")

    # Weighted BCE
    torch.manual_seed(42)
    pred = torch.sigmoid(torch.randn(4, 4, dtype=torch.float64)).requires_grad_(True)
    target = torch.randint(0, 2, (4, 4), dtype=torch.float64)
    weight = torch.rand(4, 4, dtype=torch.float64) + 0.1  # Ensure positive weights

    # Manual weighted BCE
    eps = 1e-7
    pred_clamped = pred.clamp(min=eps, max=1-eps)
    bce = target * torch.log(pred_clamped) + (1 - target) * torch.log(1 - pred_clamped)
    weighted_bce = weight * bce
    loss = -weighted_bce.sum() / weight.sum()
    loss.backward()

    save_loss_case(
        name="loss_weighted_bce_basic",
        loss_type="weighted_binary_cross_entropy",
        description="Weighted binary cross-entropy with per-element weights",
        inputs={"pred": pred.detach().clone(), "target": target, "weight": weight},
        loss_value=loss,
        gradients={"pred": pred.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Weighted Softmax Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_weighted_softmax_ce_cases():
    print("Generating weighted softmax cross-entropy loss cases...")

    # Per-sample weighted softmax CE
    torch.manual_seed(42)
    logits = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    target_idx = torch.tensor([0, 2, 1, 4])
    target = torch.zeros(4, 5, dtype=torch.float64)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)
    # Per-sample weights
    weight = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)

    # Manual weighted softmax CE with numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_softmax = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    ce_per_sample = torch.sum(target * log_softmax, dim=-1)
    weighted_ce = weight * ce_per_sample
    loss = -weighted_ce.sum() / weight.sum()
    loss.backward()

    save_loss_case(
        name="loss_weighted_softmax_ce_basic",
        loss_type="weighted_softmax_cross_entropy",
        description="Weighted softmax cross-entropy with per-sample weights",
        inputs={"logits": logits.detach().clone(), "target": target, "weight": weight},
        loss_value=loss,
        gradients={"logits": logits.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Class-Weighted Softmax Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------

def generate_class_weighted_softmax_ce_cases():
    print("Generating class-weighted softmax cross-entropy loss cases...")

    # Per-class weighted softmax CE
    torch.manual_seed(42)
    logits = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    target_idx = torch.tensor([0, 2, 1, 4])
    target = torch.zeros(4, 5, dtype=torch.float64)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)
    # Per-class weights
    class_weight = torch.tensor([1.0, 2.0, 1.5, 0.5, 3.0], dtype=torch.float64)

    # Manual class-weighted softmax CE with numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_softmax = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    weighted_prod = class_weight.unsqueeze(0) * target * log_softmax
    ce_per_sample = torch.sum(weighted_prod, dim=-1)
    loss = -ce_per_sample.mean()
    loss.backward()

    save_loss_case(
        name="loss_class_weighted_softmax_ce_basic",
        loss_type="class_weighted_softmax_cross_entropy",
        description="Class-weighted softmax cross-entropy with per-class weights",
        inputs={"logits": logits.detach().clone(), "target": target, "class_weight": class_weight},
        loss_value=loss,
        gradients={"logits": logits.grad.clone()},
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Generating loss function test vectors...")
    print()

    generate_mse_cases()
    generate_mae_cases()
    generate_cross_entropy_cases()
    generate_bce_cases()
    generate_softmax_ce_cases()

    # Weighted variants
    generate_weighted_mse_cases()
    generate_weighted_mae_cases()
    generate_weighted_cross_entropy_cases()
    generate_class_weighted_cross_entropy_cases()
    generate_weighted_bce_cases()
    generate_weighted_softmax_ce_cases()
    generate_class_weighted_softmax_ce_cases()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
