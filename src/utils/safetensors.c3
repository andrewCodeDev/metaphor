//! SafeTensors Format
//!
//! Parses and writes the SafeTensors binary format for named tensor access.
//! Supports deserialization (reading) and serialization (writing).
//!
//! Format: 8-byte LE header_size + JSON header + padding + raw tensor data.
//!

module metaphor::utils::safetensors;
import std::core::bitorder;
import std::core::mem;
import std::io;

import metaphor::utils::cjson;
import metaphor::rtti;

alias CJson = cjson::CJson;

const usz MAX_HEADER_SIZE = 100_000_000;

// Faults

faultdef ST_INVALID_HEADER;
faultdef ST_BUFFER_TOO_SMALL;
faultdef ST_HEADER_OVERFLOW;
faultdef ST_TENSOR_NOT_FOUND;
faultdef ST_INCOMPLETE_BUFFER;
faultdef ST_UNSUPPORTED_DTYPE;
faultdef ST_SHAPE_OVERFLOW;
faultdef ST_SERIALIZE_FAILED;

// Dtype — safetensors element types

enum Dtype : char
{
	BOOL,
	U8,
	I8,
	F16,
	BF16,
	I16,
	U16,
	I32,
	U32,
	F32,
	F64,
	I64,
	U64,
}

fn usz Dtype.byte_size(&self)
{
	switch (*self)
	{
		case BOOL: return 1;
		case U8:   return 1;
		case I8:   return 1;
		case F16:  return 2;
		case BF16: return 2;
		case I16:  return 2;
		case U16:  return 2;
		case I32:  return 4;
		case U32:  return 4;
		case F32:  return 4;
		case F64:  return 8;
		case I64:  return 8;
		case U64:  return 8;
	}
}

fn char[] Dtype.to_string(&self)
{
	switch (*self)
	{
		case BOOL: return "BOOL";
		case U8:   return "U8";
		case I8:   return "I8";
		case F16:  return "F16";
		case BF16: return "BF16";
		case I16:  return "I16";
		case U16:  return "U16";
		case I32:  return "I32";
		case U32:  return "U32";
		case F32:  return "F32";
		case F64:  return "F64";
		case I64:  return "I64";
		case U64:  return "U64";
	}
}

fn Dtype? from_string(char[] s)
{
	if (s == "BOOL") return BOOL;
	if (s == "U8")   return U8;
	if (s == "I8")   return I8;
	if (s == "F16")  return F16;
	if (s == "BF16") return BF16;
	if (s == "I16")  return I16;
	if (s == "U16")  return U16;
	if (s == "I32")  return I32;
	if (s == "U32")  return U32;
	if (s == "F32")  return F32;
	if (s == "F64")  return F64;
	if (s == "I64")  return I64;
	if (s == "U64")  return U64;
	return ST_UNSUPPORTED_DTYPE~;
}

/// Convert from rtti::Datatype to safetensors Dtype.
fn Dtype dtype_from_rtti(rtti::Datatype dt)
{
	switch (dt)
	{
		case BOOL: return BOOL;
		case U8:   return U8;
		case U64:  return U64;
		case F16:  return F16;
		case BF16: return BF16;
		case F32:  return F32;
		case F64:  return F64;
	}
}

// TensorInfo / TensorView / SafeTensorsFile

struct TensorInfo
{
	char[] name;
	Dtype dtype;
	ulong[] shape;
	ulong data_start;
	ulong data_end;
}

struct TensorView
{
	TensorInfo* info;
	char[] data;
}

struct SafeTensorsFile
{
	ulong header_size;
	TensorInfo[] tensors;
	char[] raw_data; // NOT owned
}

fn TensorView? SafeTensorsFile.get(&self, char[] name)
{
	foreach (&info : self.tensors)
	{
		if (info.name == name)
		{
			usz start = (usz)info.data_start + 8 + (usz)self.header_size;
			usz end   = (usz)info.data_end   + 8 + (usz)self.header_size;
			if (end > self.raw_data.len) return ST_INCOMPLETE_BUFFER~;
			if (start == end) { return { .info = info, .data = {} }; }
			return { .info = info, .data = self.raw_data[start:end - start] };
		}
	}
	return ST_TENSOR_NOT_FOUND~;
}

fn void SafeTensorsFile.free(&self)
{
	foreach (&info : self.tensors)
	{
		if (info.name.len > 0) mem::free(info.name.ptr);
		if (info.shape.len > 0) mem::free(info.shape.ptr);
	}
	if (self.tensors.len > 0) mem::free(self.tensors.ptr);
}

fn usz SafeTensorsFile.len(&self)
{
	return self.tensors.len;
}

fn SafeTensorsFile? deserialize(char[] data)
{
	if (data.len < 8) return ST_BUFFER_TOO_SMALL~;

	ulong header_size = bitorder::read(data[0:8], bitorder::ULongLE);
	if (header_size > MAX_HEADER_SIZE) return ST_HEADER_OVERFLOW~;
	ulong header_end = 8 + header_size;
	if ((usz)header_end > data.len) return ST_BUFFER_TOO_SMALL~;

	ulong padded_header_size = (header_size + 7) / 8 * 8;

	TensorInfo[] tensors = parse_header(data[8:(usz)header_size])!;

	return {
		.header_size = padded_header_size,
		.tensors = tensors,
		.raw_data = data,
	};
}

// Header parser

fn TensorInfo[]? parse_header(char[] json)
{
	// Null-terminate JSON for cJSON
	char* cstr = (char*)mem::malloc(json.len + 1);
	assert(cstr != null, "parse_header: allocation failed");
	mem::copy(cstr, json.ptr, json.len);
	cstr[json.len] = 0;

	CJson* root = cjson::cJSON_Parse(cstr);
	mem::free(cstr);
	if (root == null) return ST_INVALID_HEADER~;

	// Count tensors (skip __metadata__)
	usz count = 0;
	CJson* entry = root.child;
	while (entry != null)
	{
		if (entry.string != null && !cstr_eq(entry.string, "__metadata__"))
		{
			count++;
		}
		entry = entry.next;
	}

	if (count == 0)
	{
		cjson::cJSON_Delete(root);
		TensorInfo[] empty;
		return empty;
	}

	TensorInfo* buf = (TensorInfo*)mem::malloc(count * TensorInfo.sizeof);
	assert(buf != null, "parse_header: allocation failed");
	usz idx = 0;

	entry = root.child;
	while (entry != null)
	{
		if (entry.string == null || cstr_eq(entry.string, "__metadata__"))
		{
			entry = entry.next;
			continue;
		}

		// Parse dtype
		CJson* dtype_item = cjson::cJSON_GetObjectItemCaseSensitive(entry, "dtype");
		if (dtype_item == null || cjson::cJSON_IsString(dtype_item) == 0)
		{
			mem::free(buf);
			cjson::cJSON_Delete(root);
			return ST_INVALID_HEADER~;
		}
		char* dtype_cstr = cjson::cJSON_GetStringValue(dtype_item);
		usz dtype_len = cstr_len(dtype_cstr);
		Dtype dtype;
		if (try dt = from_string(dtype_cstr[:dtype_len]))
		{
			dtype = dt;
		}
		else
		{
			// Clean up partially-filled buf entries
			for (usz j = 0; j < idx; j++)
			{
				if (buf[j].name.len > 0) mem::free(buf[j].name.ptr);
				if (buf[j].shape.len > 0) mem::free(buf[j].shape.ptr);
			}
			mem::free(buf);
			cjson::cJSON_Delete(root);
			return ST_UNSUPPORTED_DTYPE~;
		}

		// Parse shape
		CJson* shape_arr = cjson::cJSON_GetObjectItemCaseSensitive(entry, "shape");
		if (shape_arr == null || cjson::cJSON_IsArray(shape_arr) == 0)
		{
			mem::free(buf);
			cjson::cJSON_Delete(root);
			return ST_INVALID_HEADER~;
		}
		int ndims = cjson::cJSON_GetArraySize(shape_arr);
		ulong[] shape;
		if (ndims > 0)
		{
			ulong* shape_ptr = (ulong*)mem::malloc((usz)ndims * ulong.sizeof);
			assert(shape_ptr != null, "parse_header: shape allocation failed");
			for (int d = 0; d < ndims; d++)
			{
				CJson* dim_item = cjson::cJSON_GetArrayItem(shape_arr, d);
				shape_ptr[(usz)d] = (ulong)dim_item.valuedouble;
			}
			shape = shape_ptr[:(usz)ndims];
		}
		else
		{
			shape = {};
		}

		// Parse data_offsets
		CJson* offsets_arr = cjson::cJSON_GetObjectItemCaseSensitive(entry, "data_offsets");
		if (offsets_arr == null || cjson::cJSON_IsArray(offsets_arr) == 0 ||
			cjson::cJSON_GetArraySize(offsets_arr) != 2)
		{
			if (shape.len > 0) mem::free(shape.ptr);
			mem::free(buf);
			cjson::cJSON_Delete(root);
			return ST_INVALID_HEADER~;
		}
		CJson* start_item = cjson::cJSON_GetArrayItem(offsets_arr, 0);
		CJson* end_item = cjson::cJSON_GetArrayItem(offsets_arr, 1);
		ulong start_offset = (ulong)start_item.valuedouble;
		ulong end_offset = (ulong)end_item.valuedouble;

		// Duplicate tensor name
		usz name_len = cstr_len(entry.string);
		char* name_ptr = (char*)mem::malloc(name_len);
		assert(name_ptr != null, "parse_header: name allocation failed");
		mem::copy(name_ptr, entry.string, name_len);

		buf[idx] = {
			.name = name_ptr[:name_len],
			.dtype = dtype,
			.shape = shape,
			.data_start = start_offset,
			.data_end = end_offset,
		};
		idx++;

		entry = entry.next;
	}

	cjson::cJSON_Delete(root);
	return buf[:count];
}

fn usz cstr_len(char* s) @private
{
	if (s == null) return 0;
	usz len = 0;
	while (s[len] != 0) len++;
	return len;
}

fn bool cstr_eq(char* s, char[] target) @private
{
	for (usz i = 0; i < target.len; i++)
	{
		if (s[i] == 0 || s[i] != target[i]) return false;
	}
	return s[target.len] == 0;
}

// Serialization

/// A tensor descriptor for serialization.
struct TensorDescriptor
{
	char[] name;
	Dtype dtype;
	ulong[] shape;
	char[] data; // raw bytes, length must equal product(shape) * dtype.byte_size()
}

/// Serialize tensors into a safetensors-format byte buffer.
/// Returns a malloc'd buffer. Caller must free with mem::free.
fn char[]? serialize_tensors(TensorDescriptor[] tensors)
{
	// Build JSON header using cJSON
	cjson::CJson* root = cjson::cJSON_CreateObject();
	if (root == null) return ST_SERIALIZE_FAILED~;

	ulong offset = 0;

	foreach (&td : tensors)
	{
		assert(td.name.len > 0, "serialize_tensors: tensor name must not be empty");
		assert(td.data.len > 0, "serialize_tensors: tensor data must not be empty");

		// Need null-terminated strings for cJSON
		char* name_z = (char*)mem::malloc(td.name.len + 1);
		if (name_z == null) { cjson::cJSON_Delete(root); return ST_SERIALIZE_FAILED~; }
		for (usz i = 0; i < td.name.len; i++) { name_z[i] = td.name[i]; }
		name_z[td.name.len] = 0;

		cjson::CJson* tensor_obj = cjson::cJSON_CreateObject();
		if (tensor_obj == null) { mem::free(name_z); cjson::cJSON_Delete(root); return ST_SERIALIZE_FAILED~; }

		// dtype
		char* dtype_str = td.dtype.to_string().ptr;
		cjson::CJson* dtype_val = cjson::cJSON_CreateString(dtype_str);
		cjson::cJSON_AddItemToObject(tensor_obj, "dtype", dtype_val);

		// shape
		cjson::CJson* shape_arr = cjson::cJSON_CreateArray();
		foreach (&dim : td.shape)
		{
			cjson::cJSON_AddItemToArray(shape_arr, cjson::cJSON_CreateNumber((double)*dim));
		}
		cjson::cJSON_AddItemToObject(tensor_obj, "shape", shape_arr);

		// data_offsets
		ulong end_offset = offset + (ulong)td.data.len;
		cjson::CJson* offsets_arr = cjson::cJSON_CreateArray();
		cjson::cJSON_AddItemToArray(offsets_arr, cjson::cJSON_CreateNumber((double)offset));
		cjson::cJSON_AddItemToArray(offsets_arr, cjson::cJSON_CreateNumber((double)end_offset));
		cjson::cJSON_AddItemToObject(tensor_obj, "data_offsets", offsets_arr);

		cjson::cJSON_AddItemToObject(root, name_z, tensor_obj);
		mem::free(name_z);

		offset = end_offset;
	}

	// Render JSON to string
	char* json_cstr = cjson::cJSON_PrintUnformatted(root);
	cjson::cJSON_Delete(root);
	if (json_cstr == null) return ST_SERIALIZE_FAILED~;

	// Measure JSON length (find null terminator)
	usz json_len = 0;
	while (json_cstr[json_len] != 0) { json_len++; }

	usz padded_len = (json_len + 7) / 8 * 8; // align to 8 bytes
	usz total_size = 8 + padded_len + (usz)offset;

	char* out = (char*)mem::malloc(total_size);
	if (out == null) { cjson::cJSON_free(json_cstr); return ST_SERIALIZE_FAILED~; }

	// Write header size as LE u64
	ulong hs = (ulong)padded_len;
	out[0] = (char)(hs & 0xFF);
	out[1] = (char)((hs >> 8) & 0xFF);
	out[2] = (char)((hs >> 16) & 0xFF);
	out[3] = (char)((hs >> 24) & 0xFF);
	out[4] = (char)((hs >> 32) & 0xFF);
	out[5] = (char)((hs >> 40) & 0xFF);
	out[6] = (char)((hs >> 48) & 0xFF);
	out[7] = (char)((hs >> 56) & 0xFF);

	// Copy JSON header
	for (usz i = 0; i < json_len; i++) { out[8 + i] = json_cstr[i]; }
	cjson::cJSON_free(json_cstr);

	// Pad with spaces
	for (usz i = json_len; i < padded_len; i++) { out[8 + i] = ' '; }

	// Append tensor data
	usz data_pos = 8 + padded_len;
	foreach (&td : tensors)
	{
		for (usz i = 0; i < td.data.len; i++) { out[data_pos + i] = td.data[i]; }
		data_pos += td.data.len;
	}

	return out[:total_size];
}

/// Free a buffer returned by serialize_tensors.
fn void free_serialized(char[] buf)
{
	if (buf.len > 0) mem::free(buf.ptr);
}

// Sharded SafeTensors — loads models split across multiple files

faultdef ST_SHARD_NOT_FOUND;
faultdef ST_INDEX_PARSE_ERROR;

const usz MAX_SHARDS = 32;
const usz MAX_WEIGHT_MAP_ENTRIES = 4096;

struct WeightMapEntry
{
	char* tensor_name;
	usz tensor_name_len;
	usz shard_idx;   // index into shard_files[]
}

struct LoadedShard
{
	SafeTensorsFile file;
	char* data;       // malloc'd file data (owned)
	usz data_len;
	bool loaded;
}

struct ShardedSafeTensors
{
	// Base directory (owned, null-terminated)
	char* base_dir;
	usz base_dir_len;

	// Shard filenames and loaded data
	char*[MAX_SHARDS] shard_names;    // owned, null-terminated
	usz[MAX_SHARDS] shard_name_lens;
	LoadedShard[MAX_SHARDS] shards;
	usz num_shards;

	// Weight map: tensor name → shard index (heap-allocated)
	WeightMapEntry* weight_map;
	usz num_entries;
	usz weight_map_capacity;
}

/// Open a sharded model from a directory containing model.safetensors.index.json.
fn ShardedSafeTensors? sharded_open(char[] dir_path)
{
	ShardedSafeTensors st;
	st.num_shards = 0;
	st.num_entries = 0;
	st.weight_map = (WeightMapEntry*)mem::malloc(WeightMapEntry.sizeof * MAX_WEIGHT_MAP_ENTRIES);
	assert(st.weight_map != null, "Failed to allocate weight map");
	st.weight_map_capacity = MAX_WEIGHT_MAP_ENTRIES;
	for (usz i = 0; i < MAX_SHARDS; i++) { st.shards[i].loaded = false; }

	// Copy base dir
	st.base_dir = (char*)mem::malloc(dir_path.len + 1);
	assert(st.base_dir != null);
	mem::copy(st.base_dir, dir_path.ptr, dir_path.len);
	st.base_dir[dir_path.len] = 0;
	st.base_dir_len = dir_path.len;

	// Read index file
	String index_path = string::tformat("%s/model.safetensors.index.json", (ZString)dir_path.ptr);
	char[] index_data = read_file(index_path)!;
	defer mem::free(index_data.ptr);

	// Parse index JSON with cJSON
	char* cstr = (char*)mem::malloc(index_data.len + 1);
	assert(cstr != null);
	mem::copy(cstr, index_data.ptr, index_data.len);
	cstr[index_data.len] = 0;

	CJson* root = cjson::cJSON_Parse(cstr);
	mem::free(cstr);
	if (root == null) { return ST_INDEX_PARSE_ERROR~; }

	CJson* weight_map_obj = cjson::cJSON_GetObjectItemCaseSensitive(root, "weight_map");
	if (weight_map_obj == null)
	{
		cjson::cJSON_Delete(root);
		return ST_INDEX_PARSE_ERROR~;
	}

	// Iterate weight_map entries
	CJson* entry = weight_map_obj.child;
	while (entry != null)
	{
		if (entry.string != null && cjson::cJSON_IsString(entry) != 0)
		{
			char* tensor_name_cstr = entry.string;
			usz tname_len = cstr_len(tensor_name_cstr);

			char* shard_name_cstr = cjson::cJSON_GetStringValue(entry);
			usz sname_len = cstr_len(shard_name_cstr);

			// Find or create shard index
			usz shard_idx = find_or_add_shard(&st, shard_name_cstr[:sname_len]);

			// Add weight map entry
			assert(st.num_entries < MAX_WEIGHT_MAP_ENTRIES, "Too many weight map entries");
			char* owned_name = (char*)mem::malloc(tname_len);
			mem::copy(owned_name, tensor_name_cstr, tname_len);
			st.weight_map[st.num_entries] = {
				.tensor_name = owned_name,
				.tensor_name_len = tname_len,
				.shard_idx = shard_idx,
			};
			st.num_entries++;
		}
		entry = entry.next;
	}

	cjson::cJSON_Delete(root);

	io::eprintfn("  Sharded model: %d tensors across %d shards", st.num_entries, st.num_shards);
	return st;
}

fn usz find_or_add_shard(ShardedSafeTensors* st, char[] name)
{
	// Check existing
	for (usz i = 0; i < st.num_shards; i++)
	{
		if (st.shard_name_lens[i] == name.len)
		{
			bool match = true;
			for (usz j = 0; j < name.len; j++)
			{
				if (st.shard_names[i][j] != name[j]) { match = false; break; }
			}
			if (match) { return i; }
		}
	}

	// Add new
	assert(st.num_shards < MAX_SHARDS, "Too many shards");
	usz idx = st.num_shards;
	char* owned = (char*)mem::malloc(name.len + 1);
	mem::copy(owned, name.ptr, name.len);
	owned[name.len] = 0;
	st.shard_names[idx] = owned;
	st.shard_name_lens[idx] = name.len;
	st.num_shards++;
	return idx;
}

/// Get a tensor by name, lazy-loading the shard if needed.
fn TensorView? ShardedSafeTensors.get(&self, char[] name)
{
	// Find shard index for this tensor
	for (usz i = 0; i < self.num_entries; i++)
	{
		if (self.weight_map[i].tensor_name_len == name.len)
		{
			bool match = true;
			for (usz j = 0; j < name.len; j++)
			{
				if (self.weight_map[i].tensor_name[j] != name[j]) { match = false; break; }
			}
			if (match)
			{
				usz shard_idx = self.weight_map[i].shard_idx;
				if (!self.shards[shard_idx].loaded)
				{
					self.load_shard(shard_idx)!;
				}
				return self.shards[shard_idx].file.get(name);
			}
		}
	}
	return ST_TENSOR_NOT_FOUND~;
}

fn void? ShardedSafeTensors.load_shard(&self, usz idx)
{
	ZString base = (ZString)self.base_dir;
	ZString shard = (ZString)self.shard_names[idx];
	String shard_path = string::tformat("%s/%s", base, shard);

	io::eprintfn("  Loading shard: %s", shard_path);

	char[] data = read_file(shard_path)!;
	SafeTensorsFile sf = deserialize(data)!;

	self.shards[idx] = {
		.file = sf,
		.data = data.ptr,
		.data_len = data.len,
		.loaded = true,
	};
}

fn usz ShardedSafeTensors.len(&self)
{
	return self.num_entries;
}

fn void ShardedSafeTensors.free(&self)
{
	// Free weight map names
	for (usz i = 0; i < self.num_entries; i++)
	{
		mem::free(self.weight_map[i].tensor_name);
	}
	mem::free(self.weight_map);

	// Free shard data
	for (usz i = 0; i < self.num_shards; i++)
	{
		mem::free(self.shard_names[i]);
		if (self.shards[i].loaded)
		{
			self.shards[i].file.free();
			mem::free(self.shards[i].data);
		}
	}

	// Free base dir
	mem::free(self.base_dir);
}

/// Check if a path is a directory containing model.safetensors.index.json.
fn bool is_sharded_model(char[] path)
{
	String index_path = string::tformat("%s/model.safetensors.index.json", (ZString)path.ptr);
	return io::file::exists(index_path);
}

/// Read entire file into heap-allocated buffer. Caller must mem::free the .ptr.
fn char[]? read_file(char[] path)
{
	// Heap-copy the path to avoid tformat temp buffer reuse.
	char* buf = (char*)mem::malloc(path.len + 1);
	assert(buf != null);
	mem::copy(buf, path.ptr, path.len);
	buf[path.len] = 0;
	defer mem::free(buf);
	return io::file::load(mem, (String)buf[:path.len]);
}

