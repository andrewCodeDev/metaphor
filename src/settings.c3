//! Global Runtime Settings
//!
//! Mutable flags that control gradient computation and operation tracking
//! at the framework level. These override per-tensor and per-graph flags.
//!

module metaphor::settings;

/// When false, requires_grad() returns false for all tensors regardless
/// of their per-tensor flag. Useful for inference-only mode.
bool grad_enabled = true;

/// When false, no operations are tracked in any graph regardless of
/// the per-graph track_ops flag. Useful for data loading / preprocessing.
bool track_ops = true;

/// When true, gradient cores inherit flags (attached, executable) from
/// their parent forward tensor. This creates fusion boundaries at
/// checkpoint locations during backward pass, mirroring forward-pass
/// execution and persistence boundaries. Default true for layered training.
bool grad_uses_parent_flags = true;
