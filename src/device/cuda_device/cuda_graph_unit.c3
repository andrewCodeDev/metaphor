//! CUDA Graph Unit
//!
//! Implements CUDA graph capture using the "close-steal-reopen" pattern
//! for observable tensors. Manages capture boundaries and replays captured
//! graphs for subsequent executions.
//!
//! The pattern:
//! 1. First execution: operations run normally on the stream (warmup)
//! 2. Device enters capture mode: stream is capturing
//! 3. Unit's collect() closes capture, steals the captured graph, reopens capture
//! 4. Subsequent calls to collect() replay the stolen graph instead of re-executing
//!

module metaphor::device::cuda_device::cuda_graph_unit;
import metaphor::device::cuda_device;
import metaphor::device::cuda_device::utils;

// State Machine

/// Lifecycle state of the graph unit.
enum CaptureState : char
{
	/// Initial state — operations execute normally
	WARMUP,
	/// Device is capturing — next collect() will steal the graph
	CAPTURING,
	/// Graph captured — collect() replays instead of re-executing
	REPLAYING,
}

// CudaGraphUnit

/// A captured CUDA graph segment for observable tensor boundaries.
/// Used by the close-steal-reopen pattern during graph capture mode.
struct CudaGraphUnit
{
	/// Back-reference to owning device
	cuda_device::CudaDevice* device;

	/// Captured graph executable handle (owned, must destroy on deinit)
	cuda_device::GraphExecHandle captured_exec;

	/// Current lifecycle state
	CaptureState state;
}

// Lifecycle

fn CudaGraphUnit cuda_graph_unit_init(cuda_device::CudaDevice* device) {
	return {
		.device = device,
		.captured_exec = { .ptr = null },
		.state = CaptureState.WARMUP,
	};
}

fn void CudaGraphUnit.deinit(&self) {
	if (self.captured_exec.ptr != null) {
		self.device.destroy_exec(self.captured_exec);
		self.captured_exec = { .ptr = null };
	}
}

// Core Operations

/// Called at observable/collect boundaries.
///
/// - WARMUP: no-op (operations already executed on stream)
/// - CAPTURING: close current capture, steal graph, reopen capture for next segment
/// - REPLAYING: replay the captured graph (skip re-execution)
fn void CudaGraphUnit.collect(&self) {
	switch (self.state) {
		case WARMUP:
		{
			// First execution — operations ran normally, nothing to capture yet
		}
		case CAPTURING:
		{
			// Close capture, steal the graph
			self.captured_exec = self.device.capture_to_exec();
			// Reopen capture for the next segment
			self.device.begin_capture();
			self.state = CaptureState.REPLAYING;
		}
		case REPLAYING:
		{
			// Replay the captured graph
			self.device.ensure_scratch();
			self.device.launch_exec(self.captured_exec);
		}
	}
}

/// Transition from WARMUP to CAPTURING.
/// Called when the device enters capture mode.
fn void CudaGraphUnit.start_capture(&self) {
	if (self.state == CaptureState.WARMUP) {
		self.state = CaptureState.CAPTURING;
	}
}

/// Reset back to WARMUP state, discarding any captured graph.
/// Called when graph structure changes (e.g., batch size change).
fn void CudaGraphUnit.reset(&self) {
	if (self.captured_exec.ptr != null) {
		self.device.destroy_exec(self.captured_exec);
		self.captured_exec = { .ptr = null };
	}
	self.state = CaptureState.WARMUP;
}

/// Returns true if this unit has a captured graph ready for replay.
fn bool CudaGraphUnit.has_capture(&self) {
	return self.state == CaptureState.REPLAYING && self.captured_exec.ptr != null;
}
