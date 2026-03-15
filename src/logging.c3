//! Logging
//!
//! Opt-in per-module logging infrastructure. All scopes are disabled by
//! default. Call enable() to turn on logging for specific modules.
//!
//!
//! Usage:
//!   import metaphor::logging;
//!   // At init:
//!   logging::enable(logging::Scope.GRAPH, logging::Level.DEBUG);
//!   // In code (uses macros for source location capture):
//!   logging::@debug(GRAPH, "compiling node %d", node_id);
//!   logging::@info(GRAPH, "execution complete");

module metaphor::logging;
import std::io;
import std::core::dstring;

// Log Level

enum Level : char
{
	DEBUG,
	INFO,
	WARN,
	ERR,
}

fn String Level.name(&self)
{
	return self.nameof;
}

// Log Scope — one per module area

enum Scope : char
{
	GRAPH,
	TENSOR,
	DEVICE,
	ALLOCATOR,
	COMPILER,
	EXECUTOR,
	KERNEL,
	CODEGEN,
	NN,
	FUSION,
	OP_PAYLOAD,
	CAS,
	CHAIN_MAP,
	DISPLAY_KERNEL,
}

const usz NUM_SCOPES = Scope.values.len;

fn String Scope.name(&self)
{
	return self.nameof;
}

// Global state — scope enable flags + per-scope minimum level

/// Whether each scope is enabled at all.
bool[NUM_SCOPES] scope_enabled;

/// Minimum log level per scope (only matters if enabled).
Level[NUM_SCOPES] scope_min_level;

// Configuration API

/// Enable logging for a scope at a given minimum level.
fn void enable(Scope scope, Level min_level = Level.DEBUG)
{
	scope_enabled[(char)scope] = true;
	scope_min_level[(char)scope] = min_level;
}

/// Disable logging for a scope.
fn void disable(Scope scope)
{
	scope_enabled[(char)scope] = false;
}

/// Enable all scopes at a given level.
fn void enable_all(Level min_level = Level.DEBUG)
{
	for (usz i = 0; i < NUM_SCOPES; i++)
	{
		scope_enabled[i] = true;
		scope_min_level[i] = min_level;
	}
}

/// Disable all scopes.
fn void disable_all()
{
	for (usz i = 0; i < NUM_SCOPES; i++)
	{
		scope_enabled[i] = false;
	}
}

/// Check if a scope/level combination would produce output.
fn bool is_enabled(Scope scope, Level level)
{
	usz idx = (char)scope;
	return scope_enabled[idx] && (char)level >= (char)scope_min_level[idx];
}

// Formatted log output — macros capture call-site source location

/// Formatted log macros. Capture $$FUNC and $$LINE from the call site.
/// Uses $vasplat for correct variadic forwarding (matches C3 stdlib pattern).
macro void @debug(Scope scope, String fmt, ...)
{
	if (is_enabled(scope, Level.DEBUG))
	{
		io::eprintf("[DEBUG] (%s) <%s:%d> ", scope.name(), $$FUNC, (int)$$LINE);
		io::eprintfn(fmt, $vasplat);
	}
}

macro void @info(Scope scope, String fmt, ...)
{
	if (is_enabled(scope, Level.INFO))
	{
		io::eprintf("[INFO] (%s) <%s:%d> ", scope.name(), $$FUNC, (int)$$LINE);
		io::eprintfn(fmt, $vasplat);
	}
}

macro void @warn(Scope scope, String fmt, ...)
{
	if (is_enabled(scope, Level.WARN))
	{
		io::eprintf("[WARN] (%s) <%s:%d> ", scope.name(), $$FUNC, (int)$$LINE);
		io::eprintfn(fmt, $vasplat);
	}
}

macro void @err(Scope scope, String fmt, ...)
{
	if (is_enabled(scope, Level.ERR))
	{
		io::eprintf("[ERR] (%s) <%s:%d> ", scope.name(), $$FUNC, (int)$$LINE);
		io::eprintfn(fmt, $vasplat);
	}
}

