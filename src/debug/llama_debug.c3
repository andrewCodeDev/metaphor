//! LLaMA debug tracing module
//!
//! Provides an event hook that logs labeled tensor execution to stderr,
//! including the first 8 element values — directly comparable to
//! reference_llama.py output.
//!
//! Gated entirely behind ENABLE_EVENTS — when disabled, init() is a no-op
//! and everything compiles away.
//!
//! Usage:
//!   import metaphor::debug::llama_debug;
//!   llama_debug::init();   // one call, done
//!
//! Labels on tensors (set_label/fmt_label) are always present and free.
//! This module just decides whether to *observe* them.

module metaphor::debug::llama_debug;
import metaphor::events;
import metaphor::tensor::tensor_core;
import metaphor::tensor::shape;
import metaphor::device::device_reference;
import metaphor::rtti;
import std::io;
import std::core::mem;

const usz DUMP_ELEMS = 8;

// Step counter — increments each time x_embed fires
int g_step = -1;

// Layer filter — only trace this layer (and global tensors). -1 = all layers.
int g_trace_layer = 0;

/// Activate debug tracing. No-op when ENABLE_EVENTS is not set.
fn void init()
{
	$if $feature(ENABLE_EVENTS):
		events::set_global_hook(&trace_hook);
		io::eprintfn("[llama_debug] Event tracing enabled — dumping first %d values per labeled tensor", DUMP_ELEMS);
		io::eprintfn("[llama_debug] Tracing layer %d (set g_trace_layer = -1 for all layers)", g_trace_layer);
	$endif
}

/// Reset step counter (call on RESET/new sequence).
fn void reset_step()
{
	$if $feature(ENABLE_EVENTS):
		g_step = -1;
	$endif
}

/// Set which layer to trace. -1 = all layers.
fn void set_trace_layer(int layer)
{
	$if $feature(ENABLE_EVENTS):
		g_trace_layer = layer;
	$endif
}

fn void trace_hook(tensor_core::TensorCore* tc, events::EventKind kind) @private
{
	$if $feature(ENABLE_EVENTS):
		if (kind != events::EventKind.EXECUTE) return;
		if (tc.label.len == 0) return;
		if (!tc.has_data()) return;

		char[] label = tc.label;

		// Track steps — x_embed signals a new token being processed
		if (label.len >= 7 && label[0] == 'x' && label[1] == '_' &&
			label[2] == 'e' && label[3] == 'm' && label[4] == 'b')
		{
			g_step++;
			io::eprintfn("\n[TRACE] === Step %d ===", g_step);
		}

		// Layer filtering: if g_trace_layer >= 0, only trace that layer
		// Labels like L0_, L1_, L12_ etc. Global labels (x_embed, final_normed, logits) always pass.
		if (g_trace_layer >= 0 && label.len >= 2 && label[0] == 'L')
		{
			// Parse layer number from label
			int lnum = 0;
			usz li = 1;
			bool got_digit = false;
			while (li < label.len && label[li] >= '0' && label[li] <= '9')
			{
				lnum = lnum * 10 + (int)(label[li] - '0');
				got_digit = true;
				li++;
			}
			if (got_digit && lnum != g_trace_layer) return;
		}

		rtti::Datatype dt = tc.dtype();
		usz num_elements = (usz)tc.shape().size();
		usz n = num_elements < DUMP_ELEMS ? num_elements : DUMP_ELEMS;
		usz dtype_size = dt.size_of();
		usz copy_bytes = n * dtype_size;

		// Get raw device bytes
		void* src_data = tc.state.dense.data;
		if (src_data == null) return;

		// Copy first N elements to host
		char[DUMP_ELEMS * 8] host_buf;  // 8 bytes per elem max (f64)
		char[] src_slice = ((char*)src_data)[:copy_bytes];
		char[] dst_slice = host_buf[:copy_bytes];

		if (!tc.device.is_host())
		{
			tc.device.mem_transfer(src_slice, dst_slice,
				device_reference::TransferDirection.DTOH);
			tc.device.sync();
		}
		else
		{
			dst_slice[..] = src_slice[..];
		}

		// Convert to f32 for printing
		float[DUMP_ELEMS] vals;
		switch (dt)
		{
			case rtti::Datatype.F32:
				for (usz i = 0; i < n; i++)
				{
					vals[i] = ((float*)&host_buf)[i];
				}
			case rtti::Datatype.BF16:
				for (usz i = 0; i < n; i++)
				{
					ushort bits = ((ushort*)&host_buf)[i];
					uint f32_bits = (uint)bits << 16;
					vals[i] = *((float*)&f32_bits);
				}
			case rtti::Datatype.F64:
				for (usz i = 0; i < n; i++)
				{
					vals[i] = (float)((double*)&host_buf)[i];
				}
			default:
				// Integer types — just print label+shape, no values
				io::eprintfn("[TRACE] %-25s  %s  (integer type)",
					(ZString)tc.label.ptr, dt.name());
				return;
		}

		// Format shape string
		shape::Shape* s = tc.shape();
		usz rank = s.len;

		// Format: matches reference_llama.py style, with shape info
		// [TRACE] L0_normed         [1,4096]  first 8 = [0.1234, -0.5678, ...]
		io::eprintf("[TRACE] %-25s [", (ZString)tc.label.ptr);
		for (usz i = 0; i < rank; i++)
		{
			if (i > 0) io::eprintf(",");
			io::eprintf("%d", s.get(i));
		}
		io::eprintf("] first %d = [", n);
		for (usz i = 0; i < n; i++)
		{
			if (i > 0) io::eprintf(", ");
			io::eprintf("%.4f", vals[i]);
		}
		io::eprintfn("]");
	$endif
}
