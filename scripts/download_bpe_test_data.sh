#!/bin/bash
# Download a large Wikipedia article as plain text for tokenizer benchmarks.
# Uses the Wikipedia REST API to get the plain-text extract.
#
# Usage: ./download_test_data.sh [output_file]
# Default output: metaphor/testing/tokenizers/wiki_test_data.txt

set -euo pipefail

OUTPUT="${1:-$(dirname "$0")/../../testing/tokenizers/wiki_test_data.txt}"

# "History of the United States" — a large, text-dense article (~150KB of text)
TITLE="History_of_the_United_States"

echo "Downloading Wikipedia article: ${TITLE}..."

# Use the REST API to get plain text extract (no HTML)
curl -sS \
  "https://en.wikipedia.org/w/api.php?action=query&titles=${TITLE}&prop=extracts&explaintext=1&format=json" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
pages = data['query']['pages']
for page_id in pages:
    text = pages[page_id].get('extract', '')
    # Repeat the text a few times to get a larger corpus (~500KB+)
    sys.stdout.write(text)
    sys.stdout.write('\n\n--- REPEAT ---\n\n')
    sys.stdout.write(text)
    sys.stdout.write('\n\n--- REPEAT ---\n\n')
    sys.stdout.write(text)
" > "${OUTPUT}"

SIZE=$(wc -c < "${OUTPUT}")
LINES=$(wc -l < "${OUTPUT}")
echo "Saved ${SIZE} bytes (${LINES} lines) to ${OUTPUT}"
