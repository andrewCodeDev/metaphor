//! Pretokenizer
//!
//! Splits text into chunks before BPE encoding.
//! Supports different splitting strategies: whitespace, GPT-2, none.
//!

module metaphor::tokenizers::pretokenizer;
import std::core::mem;

// Types

enum Mode : char
{
	NONE,        // No pretokenization
	WHITESPACE,  // Split on whitespace
	GPT2,        // GPT-2 style: word boundaries, keep leading space with word
}

struct Pretokenizer
{
	Mode mode;
}

/// Result of pretokenization — growable array of string slices into the original text.
struct SplitResult
{
	char[]* chunks;     // Pointer to array of slices (heap-allocated)
	usz num_chunks;
	usz capacity;
}

fn void SplitResult.init(&self)
{
	self.chunks = null;
	self.num_chunks = 0;
	self.capacity = 0;
}

fn void SplitResult.deinit(&self)
{
	if (self.chunks != null) mem::free(self.chunks);
	self.chunks = null;
}

fn void SplitResult.append(&self, char[] chunk)
{
	if (chunk.len == 0) return;

	if (self.num_chunks >= self.capacity)
	{
		usz new_cap = self.capacity == 0 ? 64 : self.capacity * 2;
		char[]* new_chunks = (char[]*)mem::malloc(char[].sizeof * new_cap);
		if (self.chunks != null)
		{
			mem::copy((void*)new_chunks, (void*)self.chunks, char[].sizeof * self.num_chunks);
			mem::free(self.chunks);
		}
		self.chunks = new_chunks;
		self.capacity = new_cap;
	}
	self.chunks[self.num_chunks] = chunk;
	self.num_chunks++;
}

// Splitting

fn SplitResult Pretokenizer.split(&self, char[] text)
{
	SplitResult result;
	result.init();

	if (text.len == 0) return result;

	switch (self.mode)
	{
		case Mode.NONE:
			result.append(text);

		case Mode.WHITESPACE:
			split_whitespace(text, &result);

		case Mode.GPT2:
			split_gpt2(text, &result);
	}

	return result;
}

fn void split_whitespace(char[] text, SplitResult* result)
{
	usz start = 0;
	for (usz i = 0; i < text.len; i++)
	{
		if (is_whitespace(text[i]))
		{
			if (i > start)
			{
				result.append(text[start:i - start]);
			}
			start = i + 1;
		}
	}
	if (start < text.len)
	{
		result.append(text[start..]);
	}
}

fn void split_gpt2(char[] text, SplitResult* result)
{
	// GPT-2 style: split on category transitions (letter/digit/space/punct).
	// Keeps leading whitespace with the following word: " the" is one chunk.
	// Consecutive whitespace stays together: "   " is one chunk.
	if (text.len == 0) return;

	usz start = 0;
	usz i = 0;

	while (i < text.len)
	{
		if (i > start)
		{
			Category prev = char_category(text[i - 1]);
			Category curr = char_category(text[i]);

			if (should_split(prev, curr))
			{
				result.append(text[start:i - start]);
				start = i;
			}
		}
		i++;
	}

	if (start < text.len)
	{
		result.append(text[start..]);
	}
}

// Character classification (GPT-2 style)

enum Category : char
{
	LETTER,
	DIGIT,
	SPACE,
	PUNCT,
	OTHER,
}

fn Category char_category(char c) @inline
{
	if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return Category.LETTER;
	if (c >= '0' && c <= '9') return Category.DIGIT;
	if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return Category.SPACE;
	if (is_punctuation(c)) return Category.PUNCT;
	return Category.OTHER;
}

fn bool should_split(Category prev, Category curr) @inline
{
	// Split BEFORE a space that follows a non-space (keeps leading space with next word)
	if (prev != Category.SPACE && curr == Category.SPACE) return true;

	// Split punctuation from non-punctuation
	if (prev == Category.PUNCT && curr != Category.PUNCT) return true;
	if (prev != Category.PUNCT && curr == Category.PUNCT) return true;

	// Split on letter/digit transitions
	if (prev == Category.LETTER && curr == Category.DIGIT) return true;
	if (prev == Category.DIGIT && curr == Category.LETTER) return true;

	return false;
}

fn bool is_whitespace(char c) @inline
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

fn bool is_punctuation(char c) @inline
{
	return (c >= '!' && c <= '/') ||
	       (c >= ':' && c <= '@') ||
	       (c >= '[' && c <= '`') ||
	       (c >= '{' && c <= '~');
}
