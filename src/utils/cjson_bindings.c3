//! cJSON C3 Bindings
//!
//! Minimal bindings for cJSON (MIT license, DaveGamble/cJSON).
//! Only the functions needed for tokenizer.json parsing are bound.

module metaphor::utils::cjson;

// The cJSON struct — must match C layout exactly
struct CJson
{
	CJson* next;
	CJson* prev;
	CJson* child;
	int type;
	char* valuestring;
	int valueint;
	double valuedouble;
	char* string;  // key name when child of object
}

// cJSON type flags
const int CJSON_INVALID = 0;
const int CJSON_FALSE   = 1;
const int CJSON_TRUE    = 2;
const int CJSON_NULL    = 4;
const int CJSON_NUMBER  = 8;
const int CJSON_STRING  = 16;
const int CJSON_ARRAY   = 32;
const int CJSON_OBJECT  = 64;

// --- Core parsing ---

extern fn CJson* cJSON_Parse(char* value) @extern("cJSON_Parse");
extern fn CJson* cJSON_ParseWithLength(char* value, usz buffer_length) @extern("cJSON_ParseWithLength");
extern fn void cJSON_Delete(CJson* item) @extern("cJSON_Delete");
extern fn char* cJSON_GetErrorPtr() @extern("cJSON_GetErrorPtr");

// --- Tree navigation ---

extern fn CJson* cJSON_GetObjectItemCaseSensitive(CJson* object, char* string) @extern("cJSON_GetObjectItemCaseSensitive");
extern fn int cJSON_GetArraySize(CJson* array) @extern("cJSON_GetArraySize");
extern fn CJson* cJSON_GetArrayItem(CJson* array, int index) @extern("cJSON_GetArrayItem");

// --- Type checks ---

extern fn int cJSON_IsString(CJson* item) @extern("cJSON_IsString");
extern fn int cJSON_IsNumber(CJson* item) @extern("cJSON_IsNumber");
extern fn int cJSON_IsObject(CJson* item) @extern("cJSON_IsObject");
extern fn int cJSON_IsArray(CJson* item) @extern("cJSON_IsArray");

// --- Value accessors ---

extern fn char* cJSON_GetStringValue(CJson* item) @extern("cJSON_GetStringValue");
extern fn double cJSON_GetNumberValue(CJson* item) @extern("cJSON_GetNumberValue");

// --- Creation ---

extern fn CJson* cJSON_CreateObject() @extern("cJSON_CreateObject");
extern fn CJson* cJSON_CreateArray() @extern("cJSON_CreateArray");
extern fn CJson* cJSON_CreateString(char* string) @extern("cJSON_CreateString");
extern fn CJson* cJSON_CreateNumber(double num) @extern("cJSON_CreateNumber");

extern fn int cJSON_AddItemToObject(CJson* object, char* string, CJson* item) @extern("cJSON_AddItemToObject");
extern fn int cJSON_AddItemToArray(CJson* array, CJson* item) @extern("cJSON_AddItemToArray");

// --- Printing ---

extern fn char* cJSON_PrintUnformatted(CJson* item) @extern("cJSON_PrintUnformatted");

// --- Memory ---

extern fn void cJSON_free(void* object) @extern("cJSON_free");
