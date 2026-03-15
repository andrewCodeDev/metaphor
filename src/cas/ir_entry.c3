//! IREntry — device IR produced by the CAS
//!
//! An IR entry for a single device subgraph.
//! The map_graph() bridge that produces these is deferred until
//! symbolic chain map and OpPayload are fully ported.
//!

module metaphor::cas;
import metaphor::tensor::tensor_core;
import metaphor::device::common::op_payload;

struct IREntry
{
	// Output tensor (subgraph root)
	tensor_core::TensorCore* output;
	// The expression tree for this subgraph
	Expr* expr;
	// Input tensors (leaves and device boundaries)
	tensor_core::TensorCore*[] inputs;
	// IndexIds in output dimension order
	IndexId[] output_indices;
	// Operation payload for boundary ops (stub for now)
	op_payload::OpPayload* payload;
}
