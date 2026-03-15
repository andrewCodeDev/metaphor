#!/usr/bin/env python3
"""
Generate optimizer test vectors using PyTorch for Metaphor numerical validation.

Each test case includes:
- Initial parameter values
- Gradient values (simulating a backward pass)
- Parameter values after N optimizer steps

Output: SafeTensors files in test_vectors/ directory
"""

import torch
from pathlib import Path
from safetensors.torch import save_file

OUTPUT_DIR = Path(__file__).parent / "test_vectors"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_optimizer_case(
    name: str,
    optimizer_type: str,
    description: str,
    initial_params: dict[str, torch.Tensor],
    gradients: list[dict[str, torch.Tensor]],
    final_params: dict[str, torch.Tensor],
    hyperparams: dict[str, float],
    num_steps: int,
):
    """
    Save an optimizer test case to SafeTensors file.

    Tensor naming convention:
    - param.<name>.initial : Initial parameter values
    - grad.<name>.step<N>  : Gradient at step N
    - param.<name>.final   : Final parameter values after all steps
    """
    tensors = {}

    # Store initial params
    for key, tensor in initial_params.items():
        tensors[f"param.{key}.initial"] = tensor.detach().contiguous()

    # Store gradients for each step
    for step_idx, step_grads in enumerate(gradients):
        for key, tensor in step_grads.items():
            tensors[f"grad.{key}.step{step_idx}"] = tensor.detach().contiguous()

    # Store final params
    for key, tensor in final_params.items():
        tensors[f"param.{key}.final"] = tensor.detach().contiguous()

    # Build metadata
    metadata = {
        "name": name,
        "optimizer": optimizer_type,
        "description": description,
        "num_steps": str(num_steps),
        "param_keys": ",".join(initial_params.keys()),
    }

    # Add hyperparams
    for key, value in hyperparams.items():
        metadata[f"hyperparam.{key}"] = str(value)

    # Add shape info
    for key, tensor in initial_params.items():
        metadata[f"shape.{key}"] = ",".join(map(str, tensor.shape))

    # Save
    filename = OUTPUT_DIR / f"{name}.safetensors"
    save_file(tensors, filename, metadata=metadata)
    print(f"  Generated: {name}")


# -----------------------------------------------------------------------------
# SGD Tests
# -----------------------------------------------------------------------------

def generate_sgd_cases():
    print("Generating SGD cases...")

    # Basic SGD: single step
    torch.manual_seed(42)
    param = torch.randn(4, 4, dtype=torch.float64)
    grad = torch.randn(4, 4, dtype=torch.float64)
    lr = 0.01

    # Manual SGD: param = param - lr * grad
    initial = param.clone()
    final = param - lr * grad

    save_optimizer_case(
        name="optimizer_sgd_basic",
        optimizer_type="sgd",
        description="Basic SGD single step on 4x4 matrix",
        initial_params={"w": initial},
        gradients=[{"w": grad}],
        final_params={"w": final},
        hyperparams={"lr": lr},
        num_steps=1,
    )

    # SGD: multiple steps with varying gradients
    torch.manual_seed(123)
    param = torch.randn(3, 3, dtype=torch.float64)
    lr = 0.1
    num_steps = 5

    initial = param.clone()
    grads = []
    current = param.clone()

    for _ in range(num_steps):
        grad = torch.randn(3, 3, dtype=torch.float64)
        grads.append({"w": grad.clone()})
        current = current - lr * grad

    save_optimizer_case(
        name="optimizer_sgd_multistep",
        optimizer_type="sgd",
        description=f"SGD {num_steps} steps on 3x3 matrix",
        initial_params={"w": initial},
        gradients=grads,
        final_params={"w": current},
        hyperparams={"lr": lr},
        num_steps=num_steps,
    )

    # SGD: multiple parameters
    torch.manual_seed(456)
    w1 = torch.randn(2, 3, dtype=torch.float64)
    w2 = torch.randn(3, 2, dtype=torch.float64)
    lr = 0.05

    w1_init = w1.clone()
    w2_init = w2.clone()

    g1 = torch.randn(2, 3, dtype=torch.float64)
    g2 = torch.randn(3, 2, dtype=torch.float64)

    w1_final = w1 - lr * g1
    w2_final = w2 - lr * g2

    save_optimizer_case(
        name="optimizer_sgd_multi_param",
        optimizer_type="sgd",
        description="SGD with multiple parameters",
        initial_params={"w1": w1_init, "w2": w2_init},
        gradients=[{"w1": g1, "w2": g2}],
        final_params={"w1": w1_final, "w2": w2_final},
        hyperparams={"lr": lr},
        num_steps=1,
    )


# -----------------------------------------------------------------------------
# Adam Tests
# -----------------------------------------------------------------------------

def generate_adam_cases():
    print("Generating Adam cases...")

    # Basic Adam: single step
    torch.manual_seed(42)
    param = torch.randn(4, 4, dtype=torch.float64)
    grad = torch.randn(4, 4, dtype=torch.float64)

    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    initial = param.clone()

    # Manual Adam step
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)

    t = 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    final = param - lr * m_hat / (torch.sqrt(v_hat) + eps)

    save_optimizer_case(
        name="optimizer_adam_basic",
        optimizer_type="adam",
        description="Basic Adam single step on 4x4 matrix",
        initial_params={"w": initial},
        gradients=[{"w": grad}],
        final_params={"w": final},
        hyperparams={"lr": lr, "beta1": beta1, "beta2": beta2, "epsilon": eps},
        num_steps=1,
    )

    # Adam: multiple steps
    torch.manual_seed(789)
    param = torch.randn(3, 3, dtype=torch.float64)

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    num_steps = 5

    initial = param.clone()
    current = param.clone()
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)
    grads = []

    for t in range(1, num_steps + 1):
        grad = torch.randn(3, 3, dtype=torch.float64)
        grads.append({"w": grad.clone()})

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        current = current - lr * m_hat / (torch.sqrt(v_hat) + eps)

    save_optimizer_case(
        name="optimizer_adam_multistep",
        optimizer_type="adam",
        description=f"Adam {num_steps} steps on 3x3 matrix",
        initial_params={"w": initial},
        gradients=grads,
        final_params={"w": current},
        hyperparams={"lr": lr, "beta1": beta1, "beta2": beta2, "epsilon": eps},
        num_steps=num_steps,
    )

    # Adam: multiple parameters
    torch.manual_seed(101112)
    w1 = torch.randn(2, 3, dtype=torch.float64)
    w2 = torch.randn(3, 2, dtype=torch.float64)

    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    w1_init = w1.clone()
    w2_init = w2.clone()

    m1 = torch.zeros_like(w1)
    v1 = torch.zeros_like(w1)
    m2 = torch.zeros_like(w2)
    v2 = torch.zeros_like(w2)

    g1 = torch.randn(2, 3, dtype=torch.float64)
    g2 = torch.randn(3, 2, dtype=torch.float64)

    t = 1
    m1 = beta1 * m1 + (1 - beta1) * g1
    v1 = beta2 * v1 + (1 - beta2) * g1 * g1
    m2 = beta1 * m2 + (1 - beta1) * g2
    v2 = beta2 * v2 + (1 - beta2) * g2 * g2

    m1_hat = m1 / (1 - beta1 ** t)
    v1_hat = v1 / (1 - beta2 ** t)
    m2_hat = m2 / (1 - beta1 ** t)
    v2_hat = v2 / (1 - beta2 ** t)

    w1_final = w1 - lr * m1_hat / (torch.sqrt(v1_hat) + eps)
    w2_final = w2 - lr * m2_hat / (torch.sqrt(v2_hat) + eps)

    save_optimizer_case(
        name="optimizer_adam_multi_param",
        optimizer_type="adam",
        description="Adam with multiple parameters",
        initial_params={"w1": w1_init, "w2": w2_init},
        gradients=[{"w1": g1, "w2": g2}],
        final_params={"w1": w1_final, "w2": w2_final},
        hyperparams={"lr": lr, "beta1": beta1, "beta2": beta2, "epsilon": eps},
        num_steps=1,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Generating optimizer test vectors...")
    print()

    generate_sgd_cases()
    generate_adam_cases()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
