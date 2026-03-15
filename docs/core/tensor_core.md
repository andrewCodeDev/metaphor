# TensorCore

TensorCore is the underlying storage object for every tensor, owned by the graph. The user-facing tensor type is a lightweight handle wrapping a pointer to a TensorCore. Multiple handles can reference the same core.

Each TensorCore holds the tensor's shape, datatype, device reference, storage state, lifecycle flags, and a version counter for tracking in-place modifications. Storage is a tagged union of dense and sparse representations, though most tensors are dense. Dense storage consists of a lazily-allocated data pointer, byte size, element count, shape, and row-major strides. Data allocation is deferred until first use and dispatched through the device interface, so it works transparently for both host and GPU memory.

Persistence is a four-level enum controlling compilation boundaries and data lifetime. The lowest level allows fusion into other kernels with immediate data reclamation. The next two levels mark compilation boundaries with different data retention policies. The highest level designates user-managed leaves whose data is never freed by the system. Leaf detection is structural: any tensor with no entry in the graph's symbolic chain map is a leaf. There is no explicit leaf flag.
