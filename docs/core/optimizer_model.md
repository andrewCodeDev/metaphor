# Optimizer

The optimizer manages parameter updates during training. It consists of a top-level container holding global state such as learning rate, step count, and scheduler, along with a collection of per-parameter units that each encapsulate a specific update algorithm and its associated state such as momentum buffers or master weights.

Metaphor supports SGD and Adam, both with automatic mixed-precision handling. When parameters are stored in half precision, the optimizer maintains full-precision master weights and moment buffers for numerical stability. Parameter units are created by device-specific factories, so a single optimizer instance can manage parameters across different devices. The step method implements a work-stealing pattern, processing gradient updates as they become ready and applying optional global gradient clipping.

Several learning rate schedulers are available, including step decay, exponential decay, cosine annealing, linear warmup, and combined warmup-cosine or warmup-linear schedules. The scheduler adjusts the effective learning rate each step based on the base rate and current step count.
