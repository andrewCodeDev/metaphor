module metaphor::utils::hash;

// FNV-1a constants

const ulong FNV_OFFSET = 14695981039346656037UL;
const ulong FNV_PRIME  = 1099511628211UL;

// Byte-sequence hashing

/// FNV-1a hash of a byte buffer.
fn ulong fnv1a(char[] data) @inline
{
	ulong h = FNV_OFFSET;
	for (usz i = 0; i < data.len; i++)
	{
		h ^= (ulong)data[i];
		h *= FNV_PRIME;
	}
	return h;
}

// Field combiner — for building composite keys from discrete values

/// Combine a running hash with a new field value.
/// Based on boost::hash_combine adapted for 64-bit.
/// Use: h = combine(combine(combine(SEED, field0), field1), field2)
fn ulong combine(ulong h, ulong value) @inline
{
	h ^= value * 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2);
	return h;
}

/// Start a new composite key from a seed value.
const ulong SEED = 0;
