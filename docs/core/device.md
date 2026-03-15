# Device

The device layer provides a uniform interface for memory management and operation dispatch across hardware backends. Every tensor holds a reference to a device, enabling runtime backend selection without changing application code.

Metaphor ships two device implementations: a host device for CPU execution with direct memory access, and a CUDA device for GPU execution backed by a C++ layer that manages streams, cuTENSOR plans, and memory pools. A HIP device follows the same pattern for AMD GPUs. All backends implement the same interface, so the tensor API is device-agnostic. The device reference wrapper adds global memory statistics tracking and compatibility checks on top of the underlying backend.

Memory allocated by the host device can be dereferenced directly. Memory on GPU devices resides in device address space and requires explicit transfer operations to move data between host and device. Random initialization supports uniform and normal distributions with configurable scaling, including Kaiming He initialization for neural network weight init.
