# FinDocAI

Multimodal retrieval-augmented generation (RAG) over SEC filings and earnings decks. Ask a question about a company's 10-Q, 10-K or earnings presentation and get an answer with page-level citations, or an explicit "not found" when the documents don't support one.

> **Status:** being rebuilt from scratch, one milestone at a time. Each milestone ends with a working, measured and deployed system. The earlier prototype is preserved at tag [`v0-prototype`](https://github.com/shreyashgaurav/findocai/tree/v0-prototype).

## Roadmap

| Milestone | What it delivers | Status |
|---|---|---|
| M1 | Deployed RAG over one company's 10-Q and earnings deck, with page citations and baseline metrics | In progress |
| M2 | Multimodal retrieval: slides retrieved as images, click-through citations | Planned |
| M3 | Compliant EDGAR acquisition for two companies; filtering by company and fiscal period | Planned |
| M4 | Full evaluation: 40-question gold set, per-category metrics, no-retrieval control | Planned |
| M5 | Hybrid retrieval (BM25 + dense), reranking, structure-aware chunking | Planned |
| M6 | Measuring where text-only RAG fails on charts | Planned |
| M7 | Late-interaction visual retrieval | Planned |
| M8 | Hardening: prompt-injection tests, cost and latency report | Planned |

## License

[MIT](LICENSE)