//! Tokenizer JSON Parser
//!
//! Parses HuggingFace tokenizer.json format to populate Vocab and MergeTable.
//! Uses cJSON (MIT license, DaveGamble/cJSON) for JSON parsing.

module metaphor::tokenizers::json_parser;
import metaphor::utils::cjson;
import metaphor::tokenizers::vocab;
import std::core::mem;
import std::io;

alias CJson = cjson::CJson;

// Faults

faultdef UNSUPPORTED_TOKENIZER_TYPE;
faultdef INVALID_NUMBER;
faultdef JSON_PARSE_ERROR;

// Public API

/// Parse HuggingFace tokenizer.json and populate vocab & merges.
fn void? parse_tokenizer_json(char[] json_data, vocab::Vocab* v, vocab::MergeTable* merges)
{
	// Null-terminate the JSON data for cJSON (it requires a C string)
	char* cstr = (char*)mem::malloc(json_data.len + 1);
	if (cstr == null) return JSON_PARSE_ERROR~;
	mem::copy(cstr, json_data.ptr, json_data.len);
	cstr[json_data.len] = 0;

	CJson* root = cjson::cJSON_Parse(cstr);
	mem::free(cstr);

	if (root == null)
	{
		return JSON_PARSE_ERROR~;
	}

	// Parse "model" section
	CJson* model = cjson::cJSON_GetObjectItemCaseSensitive(root, "model");
	if (model != null)
	{
		// Parse vocab
		CJson* vocab_obj = cjson::cJSON_GetObjectItemCaseSensitive(model, "vocab");
		if (vocab_obj != null)
		{
			parse_vocab(vocab_obj, v);
		}

		// Parse merges
		CJson* merges_arr = cjson::cJSON_GetObjectItemCaseSensitive(model, "merges");
		if (merges_arr != null)
		{
			parse_merges(merges_arr, v, merges);
		}
	}

	// Parse "added_tokens" section
	CJson* added = cjson::cJSON_GetObjectItemCaseSensitive(root, "added_tokens");
	if (added != null)
	{
		parse_added_tokens(added, v);
	}

	cjson::cJSON_Delete(root);
}

// Internal parsing

fn void parse_vocab(CJson* vocab_obj, vocab::Vocab* v)
{
	// Iterate over object children (linked list via child->next)
	CJson* entry = vocab_obj.child;
	while (entry != null)
	{
		if (entry.string != null && cjson::cJSON_IsNumber(entry) != 0)
		{
			uint id = (uint)entry.valueint;
			// entry.string is the token text (already unescaped by cJSON)
			char* s = entry.string;
			usz len = cstr_len(s);
			v.set_token(id, s[:len]);
			v.check_special_token(s[:len], id);
		}
		entry = entry.next;
	}
}

fn void parse_merges(CJson* merges_arr, vocab::Vocab* v, vocab::MergeTable* merges)
{
	int count = cjson::cJSON_GetArraySize(merges_arr);
	for (int i = 0; i < count; i++)
	{
		CJson* item = cjson::cJSON_GetArrayItem(merges_arr, i);
		if (item == null) continue;

		char[] left_token;
		char[] right_token;

		if (cjson::cJSON_IsString(item) != 0)
		{
			// String format: "left right"
			char* s = cjson::cJSON_GetStringValue(item);
			if (s == null) continue;
			usz slen = cstr_len(s);
			char[] merge_str = s[:slen];
			usz space_idx = find_char(merge_str, ' ');
			if (space_idx >= merge_str.len) continue;
			left_token = merge_str[:space_idx];
			right_token = merge_str[space_idx + 1..];
		}
		else if ((item.type & cjson::CJSON_ARRAY) != 0)
		{
			// Array format: ["left", "right"]
			if (cjson::cJSON_GetArraySize(item) != 2) continue;
			CJson* left_item = cjson::cJSON_GetArrayItem(item, 0);
			CJson* right_item = cjson::cJSON_GetArrayItem(item, 1);
			if (left_item == null || right_item == null) continue;
			if (cjson::cJSON_IsString(left_item) == 0 || cjson::cJSON_IsString(right_item) == 0) continue;
			char* ls = cjson::cJSON_GetStringValue(left_item);
			char* rs = cjson::cJSON_GetStringValue(right_item);
			if (ls == null || rs == null) continue;
			left_token = ls[:cstr_len(ls)];
			right_token = rs[:cstr_len(rs)];
		}
		else { continue; }

		if (try left_id = v.get_id(left_token))
		{
			if (try right_id = v.get_id(right_token))
			{
				usz new_len = left_token.len + right_token.len;
				if (new_len <= 512)
				{
					char[512] buf;
					mem::copy(&buf, left_token.ptr, left_token.len);
					mem::copy(&buf[left_token.len], right_token.ptr, right_token.len);

					if (try result_id = v.get_id(buf[:new_len]))
					{
						merges.add_merge(left_id, right_id, result_id);
					}
				}
			}
		}
	}
}

fn void parse_added_tokens(CJson* added_arr, vocab::Vocab* v)
{
	int count = cjson::cJSON_GetArraySize(added_arr);
	for (int i = 0; i < count; i++)
	{
		CJson* token_obj = cjson::cJSON_GetArrayItem(added_arr, i);
		if (token_obj == null || cjson::cJSON_IsObject(token_obj) == 0) continue;

		CJson* id_item = cjson::cJSON_GetObjectItemCaseSensitive(token_obj, "id");
		CJson* content_item = cjson::cJSON_GetObjectItemCaseSensitive(token_obj, "content");

		if (id_item == null || content_item == null) continue;
		if (cjson::cJSON_IsNumber(id_item) == 0) continue;
		if (cjson::cJSON_IsString(content_item) == 0) continue;

		uint id = (uint)id_item.valueint;
		char* s = cjson::cJSON_GetStringValue(content_item);
		if (s == null) continue;

		usz len = cstr_len(s);
		v.set_token(id, s[:len]);
		v.check_special_token(s[:len], id);

		// Only register tokens marked "special" for literal matching during encode.
		// Non-special added tokens (e.g. whitespace) are handled by BPE normally.
		CJson* special_item = cjson::cJSON_GetObjectItemCaseSensitive(token_obj, "special");
		if (special_item != null && (special_item.type & cjson::CJSON_TRUE) != 0)
		{
			v.add_added_token(id);
		}
	}
}

// Helpers

fn usz find_char(char[] s, char target)
{
	for (usz i = 0; i < s.len; i++)
	{
		if (s[i] == target) { return i; }
	}
	return s.len;  // Not found
}

/// Get length of a null-terminated C string.
fn usz cstr_len(char* s)
{
	if (s == null) return 0;
	usz len = 0;
	while (s[len] != 0) len++;
	return len;
}
