//! Operation Cache for Idempotent Forward Pass
//!
//! Provides Common Subexpression Elimination (CSE) at graph construction time.
//! When enabled, duplicate operations return cached tensors instead of creating
//! new nodes. This prevents graph bloat from repeated subexpressions (common in
//! backward passes and optimizer updates).
//!

module metaphor::graph::op_cache;
import metaphor::tensor::tensor_core;
import metaphor::device::common::operations;
import metaphor::utils::hash_map;
import std::core::mem::allocator;
import std::hash::fnv64a;

// Cache map — keyed by precomputed ulong hash of (inputs + tag + params)

alias CacheMap = hash_map::HashMap {ulong, tensor_core::TensorCore*};

// FNV-1a constants for key computation

const ulong FNV_PRIME  = 0x00000100000001b3;
const ulong FNV_OFFSET = 0xcbf29ce484222325;

// OpCache

struct OpCache
{
	CacheMap map;
	ulong hits;
	ulong misses;
}

// Lifecycle

fn void OpCache.init(&self, Allocator alloc)
{
	self.map.init(alloc);
	self.hits = 0;
	self.misses = 0;
}

fn void OpCache.free(&self)
{
	self.map.free();
}

fn void OpCache.clear(&self)
{
	self.map.clear();
	self.hits = 0;
	self.misses = 0;
}

// Lookup / Register

/// Look up an operation in the cache.
/// Returns the cached result tensor if found, null otherwise.
fn tensor_core::TensorCore* OpCache.lookup(&self,
	tensor_core::TensorCore*[] inputs,
	operations::OpTag tag,
	ulong params_hash)
{
	ulong key = compute_key(inputs, tag, params_hash);
	if (try result = self.map.get(key))
	{
		self.hits += 1;
		return result;
	}
	self.misses += 1;
	return null;
}

/// Register an operation result in the cache.
fn void OpCache.register(&self,
	tensor_core::TensorCore*[] inputs,
	operations::OpTag tag,
	ulong params_hash,
	tensor_core::TensorCore* result)
{
	ulong key = compute_key(inputs, tag, params_hash);
	self.map.set(key, result);
}

// Key computation — FNV-1a over (tag, params_hash, input pointers)

fn ulong compute_key(
	tensor_core::TensorCore*[] inputs,
	operations::OpTag tag,
	ulong params_hash) @local
{
	ulong h = FNV_OFFSET;

	h ^= (ulong)tag;
	h *= FNV_PRIME;

	h ^= params_hash;
	h *= FNV_PRIME;

	for (usz i = 0; i < inputs.len; i++)
	{
		h ^= (ulong)(uptr)inputs[i];
		h *= FNV_PRIME;
	}

	return h;
}

// Parameter hashing helpers

/// Hash arbitrary bytes (for operation-specific parameters).
fn ulong hash_bytes(char[] data)
{
	return fnv64a::hash(data);
}

/// Combine two hash values.
fn ulong mix(ulong h, ulong v) @inline
{
	h ^= v;
	h *= FNV_PRIME;
	return h;
}
