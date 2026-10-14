# Production RAG

Evaluation-driven RAG system: every improvement to the pipeline
(chunking, hybrid retrieval, reranking…) is measured before being kept.

## Goal

Build a production-grade RAG system over internal company documentation.

## Stack

- Python
- FastAPI
- Qdrant
- BM25
- Embeddings
- LLM
- Docker

## Roadmap

- [ ] **Phase 1 — Foundations**: ingestion, dense retrieval, grounded generation, first metrics *(in progress: real corpus + eval set)*
- [ ] **Phase 2 — Evaluation harness**: reproducible experiments, retrieval & generation metrics
- [ ] **Phase 3 — Measured improvements**: chunking, hybrid search, reranking, query rewriting
- [ ] **Phase 4 — Production readiness**: API, Docker, CI, observability, demo

See [docs/ROADMAP.md](docs/ROADMAP.md) for details and experiment results.
