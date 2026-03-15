#!/usr/bin/env python3
"""Generate tokenizer test vectors using HuggingFace tokenizer.

Encodes a set of test strings and writes token IDs to a file
that the C3 tokenizer test can compare against.
"""
import json
import struct
import sys
import os

# Add the venv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../testing/.venv/lib/python3.12/site-packages'))

from transformers import AutoTokenizer

TOKENIZER_PATH = '/home/andrew/migration/archived/models/mamba-2.8b-hf'
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'test_vectors.bin')
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'test_vectors.json')

# Test strings covering various edge cases
TEST_STRINGS = [
    "hello",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Question: What is 2+2?\nAnswer: 4\n<END>",
    "Python:\nprint('hello world')\n<END>",
    " leading space",
    "tabs\there",
    "newlines\n\n\nmultiple",
    "",  # empty
    "a",  # single char
    "   ",  # just spaces
    "!@#$%^&*()",  # punctuation
    "café résumé naïve",  # accented chars
    "x" * 100,  # long repetition
]

def main():
    print(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    results = []
    all_ids = []
    
    for i, text in enumerate(TEST_STRINGS):
        ids = tokenizer.encode(text, add_special_tokens=False)
        results.append({
            "index": i,
            "text": text,
            "token_ids": ids,
            "num_tokens": len(ids),
        })
        all_ids.append(ids)
        decoded = tokenizer.decode(ids)
        print(f"  [{i:2d}] {repr(text):60s} -> {len(ids):3d} tokens -> {repr(decoded)}")
        
        # Verify HF roundtrip
        if text and decoded != text:
            print(f"       WARNING: HF decode mismatch: {repr(decoded)}")
    
    # Write JSON for human inspection
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")
    
    # Write binary format: for each test string:
    #   u32: num_tokens
    #   u32 * num_tokens: token IDs
    # Header: u32 num_test_strings
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(struct.pack('<I', len(TEST_STRINGS)))
        for ids in all_ids:
            f.write(struct.pack('<I', len(ids)))
            for id in ids:
                f.write(struct.pack('<I', id))
    print(f"Wrote {OUTPUT_PATH}")
    
    total_tokens = sum(len(ids) for ids in all_ids)
    print(f"\n{len(TEST_STRINGS)} test strings, {total_tokens} total tokens")

if __name__ == '__main__':
    main()
