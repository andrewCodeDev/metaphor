//! Tokenizer Interface
//!
//! Uniform interface for text tokenization across different tokenizer backends.
//! Any struct that implements the required methods (encode, decode, vocab_size,
//! eos_id, bos_id, get_token) automatically satisfies this interface.
//!
//! Current implementations:
//!   - bpe::Bpe — HuggingFace BPE tokenizer (GPT-2 / SentencePiece / LLaMA)

module metaphor::tokenizers::tokenizer;

// Tokenizer interface

interface Tokenizer
{
	/// Encode text to token IDs.
	fn uint[]? encode(char[] text);

	/// Decode token IDs back to text.
	fn char[]? decode(uint[] tokens);

	/// Return the total vocabulary size.
	fn uint vocab_size();

	/// Return the end-of-sequence token ID.
	fn uint eos_id();

	/// Return the beginning-of-sequence token ID.
	fn uint bos_id();

	/// Look up the string representation of a token by ID.
	fn char[]? get_token(uint id);
}
