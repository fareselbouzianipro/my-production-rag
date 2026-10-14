# Roadmap

This project is built **evaluation-first**: every change to the RAG pipeline is
measured against a fixed evaluation set before being kept. The goal is not only
a working RAG system, but a documented record of *why* each technical choice was
made, backed by numbers.

## Principles

- **Baseline first.** Start with the simplest pipeline and measure it.
- **One change at a time.** Each improvement is a separate experiment.
- **Measure, then decide.** A change is kept only if the metrics justify it.
- **Tests ≠ evaluation.** Tests (pytest) check that the code is correct and are
  deterministic. Evaluation measures how *good* the system is and produces scores.

## Phase 1 — Foundations

- [x] Document loading and fixed-size chunking
- [x] Local embeddings (`fastembed`) and vector store (Qdrant)
- [x] Dense retrieval
- [x] Grounded generation with a local LLM (Ollama)
- [x] First retrieval metrics (recall@k, MRR)
- [x] `src/` layout, centralized configuration, test suite
- [ ] Real public corpus (hundreds to thousands of documents)
- [ ] Evaluation set of 50–100 questions with expected source documents
- [ ] Baseline scores recorded

## Phase 2 — Evaluation harness

- [ ] Script that runs a pipeline configuration and outputs all metrics
- [ ] Retrieval metrics: recall@k, MRR, nDCG
- [ ] Generation metrics: faithfulness, answer relevance, citation rate (LLM-as-judge)
- [ ] Experiment runs saved and comparable over time
- [ ] Small UI to browse and compare experiments

## Phase 3 — Measured improvements

Each item is an experiment, compared against the baseline.

- [ ] Chunking strategies: fixed size vs. Markdown sections vs. semantic
- [ ] Hybrid retrieval: BM25 + dense
- [ ] Reranking
- [ ] Query rewriting
- [ ] Citation verification

## Phase 4 — Production readiness

- [ ] Clean FastAPI service
- [ ] Docker image for the API
- [ ] CI: lint + tests on every push
- [ ] Observability: latency, cost, traces
- [ ] Public demo

## Results

Experiment results will be tracked here as they are produced.

| Experiment | recall@5 | MRR | Faithfulness | Notes |
|---|---|---|---|---|
| Baseline (dense, fixed chunks) | – | – | – | Pending real corpus |
