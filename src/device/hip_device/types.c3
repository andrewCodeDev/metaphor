//! HIP Compiler Types — re-exports common device types.
module metaphor::device::hip_device::types;
import metaphor::device::common::device_types;

alias OpType = device_types::OpType;
alias ChainOp = device_types::ChainOp;
alias KernelHandle = device_types::KernelHandle;
alias HipOp = device_types::DeviceOp;

const usz MAX_INPUTS = device_types::MAX_INPUTS;
const usz MAX_MULTI_INPUTS = device_types::MAX_MULTI_INPUTS;
const usz MAX_FUSED_CHAIN = device_types::MAX_FUSED_CHAIN;
const usz MAX_JIT_INPUTS = device_types::MAX_JIT_INPUTS;
const usz MAX_PATTERN_LEN = device_types::MAX_PATTERN_LEN;
const usz MAX_SLICE_DIMS = device_types::MAX_SLICE_DIMS;
