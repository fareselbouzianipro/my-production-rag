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
- [x] Real public corpus: IFAB Laws of the Game + 2 FIFA regulations (PDF, ~1,300 chunks), reproducible download with pinned checksums
- [x] PDF extraction: page furniture removed, passages labelled with chapter and section
- [x] Evaluation set of 85 questions with expected passages (source + section)
- [x] Baseline scores recorded
- [ ] Grow the corpus (more FIFA/UEFA regulations)

## Phase 2 — Evaluation harness

- [ ] Script that runs a pipeline configuration and outputs all metrics
- [ ] Retrieval metrics: recall@k, MRR, nDCG
- [ ] Generation metrics: faithfulness, answer relevance, citation rate (LLM-as-judge)
- [ ] Experiment runs saved and comparable over time
- [ ] Small UI to browse and compare experiments

## Phase 3 — Measured improvements

Each item is an experiment, compared against the baseline.

- [ ] Filter or merge very short chunks
- [ ] Chunking strategies: fixed size vs. sections vs. semantic
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

The system targets English questions. Retrieval scores are computed on the 84
answerable questions of `data/eval/questions.json` (a result counts as relevant
when its source and section match an expected passage).

| Experiment | Recall@3 | MRR@3 | Faithfulness | Notes |
|---|---|---|---|---|
| Baseline | 0.792 | 0.718 | – | Dense retrieval, `bge-small-en-v1.5`, 500-char chunks (50 overlap) |
