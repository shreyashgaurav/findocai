"""Search the index from the command line.
Usage: uv run python scripts/search.py "your question"
"""

import sys

from findocai.retrieve import retrieve

question = sys.argv[1]
for hit in retrieve(question):
    preview = hit["text"].replace("\n", " ")
    print(f"{hit['doc_key']:18} p{hit['page']:<3} score={hit['score']}  {preview}")
