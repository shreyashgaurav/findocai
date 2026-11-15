# ask a questiona nd print the cited answer

import sys

from findocai.generate import ask

result = ask(sys.argv[1])
print(result["answer"])
print()
for c in result["citations"]:  # one line per cited source
    print(f"  [{c['tag']}] {c['doc_key']}, page {c['page']}")
print(f"\nanswered={result['answered']}  model={result['model_version']}")
