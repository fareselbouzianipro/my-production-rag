# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Evaluation-driven RAG over public football regulations (IFAB Laws of the Game + FIFA regulations, English only). Every pipeline change (chunking, hybrid retrieval, reranking…) is an experiment measured against a fixed eval set and kept only if the numbers justify it. Progress and results table live in [docs/ROADMAP.md](docs/ROADMAP.md) — update it when an experiment is run.

## Working rules

- **Experiments**: change one thing at a time. Re-index, run `scripts/evaluate_retrieval.py --name <experiment>`, compare against the baseline run in `experiments/`, then add a row to the results table in `docs/ROADMAP.md`. Keep a change only if the numbers justify it.
- **The eval set is fixed**: never edit `data/eval/questions.json` to improve a score. Fixing a genuinely wrong label is allowed, but it must be a separate commit that says so.
- **Paid API calls**: never run `scripts/generate_questions.py` or anything else calling the Anthropic API without asking first — it spends credits.
- **Secrets**: never read, print or commit `.env`.
- **Commits**: conventional format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`), one topic per commit. Run `uv run ruff check . && uv run pytest` before committing.

## Commands

Package manager is `uv` (Python ≥3.12, `src/` layout, package `production_rag`).

```bash
uv sync                                   # install deps
docker compose up -d                      # start Qdrant on :6333
uv run python scripts/download_corpus.py  # fetch PDFs from data/sources.toml into data/corpus/ (sha256-checked, gitignored)
uv run python scripts/index_documents.py  # extract → chunk → embed → (re)create Qdrant collection
uv run python scripts/evaluate_retrieval.py --name <experiment>  # recall@3 / MRR@3, saves run to experiments/
uv run python scripts/ask.py              # end-to-end RAG answer (needs Ollama running with LLM_MODEL)

uv run pytest                             # all tests
uv run pytest tests/test_ingestion.py::test_split_into_chunks_preserves_all_text  # single test
uv run ruff check . && uv run ruff format .
```

`tests/test_vector_store.py` is an integration test that needs Qdrant running; it monkeypatches `settings.collection_name` to a separate test collection. Embedding tests download the fastembed model on first run. Generation tests mock `httpx.post`.

`scripts/generate_questions.py` uses the Anthropic API (`ANTHROPIC_API_KEY`) to draft eval questions into `data/eval/generated_questions.json`; the curated eval set is `data/eval/questions.json`.

## Architecture

Pipeline: `pdf_extraction` → `ingestion` → `embeddings` + `vector_store` → `rag` → `generation`.

- **Config** — all tunables (chunk size/overlap, embedding model, vector size, Qdrant URL, collection name, Ollama URL/model) are in a single `pydantic-settings` object, [config.py](src/production_rag/config.py), overridable via `.env` (see `.env.example`). Modules import the `settings` singleton directly rather than taking parameters.
- **PDF extraction** ([pdf_extraction.py](src/production_rag/pdf_extraction.py)) — strips page furniture by page-margin fractions, detects bold numbered headings (`18.`, `5bis.`), and resolves the chapter from the IFAB running footer or the PDF table of contents. Output is *passages*: the text of one section on one page, labelled `section = "Chapter > Heading"`.
- **Chunks** carry `chunk_id` (`<source>:<n>`), `source` (corpus id from `sources.toml`), `page`, `section`, `content`; the whole dict is stored as the Qdrant payload, and search returns it with a `score`.
- **Indexing is destructive**: `store_documents` deletes and recreates the collection so chunks from different configs never mix. Re-index after changing any chunking/embedding setting before evaluating.
- **Evaluation relevance is `(source, section)`, not chunk id** ([evaluation.py](src/production_rag/evaluation.py) `passage_key`), so the eval set stays valid across chunking strategies. Questions with empty `expected_passages` are unanswerable and excluded from retrieval metrics (metric functions return `None`).
- **Experiment runs** are JSON files in `experiments/` (`<timestamp>_<name>.json`) holding config, aggregate metrics and per-question retrieved results — the basis for comparing experiments.
- **Generation** calls a local Ollama model over HTTP with temperature 0 / fixed seed, and a prompt that requires citations of the form `[source, p. N]`.
- `api.py` is a FastAPI stub (only `/health`); serving is planned for Phase 4.
