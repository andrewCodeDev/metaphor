# Tensor

Tensor is a lightweight handle type that wraps a pointer to a TensorCore. It serves as the primary user-facing API for building computation graphs and accessing data. Every tensor belongs to one graph, and multiple handles can reference the same underlying core.

Operations on tensors are divided into two categories based on their return type. Graph-building operations such as arithmetic, reductions, reshapes, and casts return a new tensor backed by a new node in the computation graph. Self-returning operations such as filling data, enabling gradients, setting labels, and adjusting persistence modify flags or data on the existing tensor without growing the graph. All failable operations return optionals, and callers propagate faults with rethrow or force-unwrap. Factory functions create tensors on the current global graph with various initializations including zeros, constants, random values, or raw data pointers.

Collection triggers compilation and execution, materializing the tensor's data. Backward seeds the loss gradient and constructs the backward computation graph without executing it. Gradient values only exist after collecting the gradient tensor. Data access methods provide raw pointers, byte slices, scalar reads, and pretty-printing, with transfers handled transparently by the device layer.
