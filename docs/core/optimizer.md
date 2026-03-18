# Optimizer

The optimizer manages parameter updates during training. It holds global state (learning rate, step count, scheduler) and a collection of per-parameter units that each run a specific update algorithm with their own state (momentum buffers, master weights).

## Basic Usage

### SGD

From `examples/least_squares.c3`, simple linear regression with SGD:

```c3
Tensor[2] params = { weight, bias };
Optimizer optim = optimizer::init({ .lr = 0.01 });
optim.sgd(&params, {});
defer optim.deinit();

for (usz epoch = 0; epoch < 200; epoch++)
{
    mse.collect()!!;
    optim.step();
    graph::notify_mutation(weight);
}
```

`optim.step()` collects all gradients, applies optional clipping, then updates every registered parameter.

### Adam

From `examples/mamba.c3`, LoRA training with per-group Adam options:

```c3
fn usz register_lora_params(LoRAMambaModel* model, Optimizer* opt)
{
    usz count = 0;

    // Embedding extension, custom betas
    opt.adam({ model.embed_ext },
        { ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
    count += 1;

    // LoRA layers, register in reverse order (closest to loss first)
    for (usz ri = 0; ri < model.num_layers; ri++)
    {
        usz i = model.num_layers - 1 - ri;
        LoRAMambaLayer* l = &model.layers[i];
        if (l.has_in_proj_lora)
        {
            opt.adam({ l.in_proj_lora.lora_a, l.in_proj_lora.lora_b },
                { ...optimizer::DEFAULT_ADAM_OPTIONS, .beta1 = 0.6, .beta2 = 0.95, .clip_grad = 1.0 });
            count += 2;
        }
    }
    return count;
}
```

Different parameter groups can have different hyperparameters. Use `...optimizer::DEFAULT_ADAM_OPTIONS` as a base and override individual fields.

## Learning Rate Scheduling

Six schedulers are available. The most common for LLM training is warmup + cosine:

From `examples/mamba_lora_train.c3`:

```c3
optimizer::Optimizer opt = optimizer::init({
    .lr = tcfg.lr,
    .scheduler = optimizer::scheduler_warmup_cosine(tcfg.warmup_steps, total_steps, 1e-6),
    .global_clip_norm = 1.0,
    .grad_accum_interval = tcfg.grad_accum,
});
```

Available schedulers:
- `scheduler_step(step_size, gamma)`: multiply LR by gamma every N steps
- `scheduler_exponential(gamma)`: multiply LR by gamma every step
- `scheduler_cosine(total_steps, min_lr)`: cosine annealing
- `scheduler_linear_warmup(warmup_steps)`: linear ramp from 0 to base LR
- `scheduler_warmup_cosine(warmup, total, min_lr)`: warmup then cosine
- `scheduler_warmup_linear(warmup, total, min_lr)`: warmup then linear decay

The current LR is available via `opt.lr` after each step.

## Gradient Clipping

Two levels of clipping, usable independently or together.

**Global clipping** caps the total gradient norm across all parameters:

```c3
optimizer::Optimizer opt = optimizer::init({
    .lr = 0.001,
    .global_clip_norm = 1.0,
});
```

**Per-parameter clipping** is set when registering parameter groups:

```c3
opt.adam(&params, { ...optimizer::DEFAULT_ADAM_OPTIONS, .clip_grad = 1.0 });
```

The pre-clipping gradient norm is available via `opt.last_grad_norm` for logging.

## Gradient Accumulation

Accumulate gradients over N micro-batches before updating parameters. Useful when the desired batch size exceeds GPU memory.

```c3
optimizer::Optimizer opt = optimizer::init({
    .lr = 0.001,
    .grad_accum_interval = 8,
});
```

Each call to `step()` scales gradients by `1/N` and accumulates. Parameters only update every Nth call. The LR scheduler advances on actual updates, not micro-batches.

## Mixed Precision

When parameters are stored in half precision (BF16 or F16), the optimizer automatically maintains full-precision master weights and moment buffers. The device-specific factories detect the dtype and set up master weights.
