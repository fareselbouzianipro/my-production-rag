---
name: run-experiment
description: Measure a pipeline change the user has already made — re-index, evaluate, compare against the baseline run and report wins/losses. Use when the user says they changed something (chunking, embedding model, filtering, reranking…) and wants to run the experiment, or asks "does my change improve retrieval?". Does not write pipeline code: the user makes the change and makes the keep/reject decision.
---

# Run a retrieval experiment

The user owns the change and the decision. This skill only measures: it never edits pipeline code, never reverts the user's work, and never decides on its own whether a change is kept.

## 1. Identify the change

- Read `git diff` and `git status --short` to see what the user changed. A config-only experiment may instead be given as environment variables (e.g. `CHUNK_SIZE=300`); settings are read from the environment.
- Summarise the change in one line and ask the user for their hypothesis if they have not given one (what should move, and why). Do not invent it.
- Agree on a short kebab-case experiment name, e.g. `chunk-300`.

Preconditions — stop and tell the user if one fails:
- Qdrant is up: `curl -s localhost:6333 >/dev/null && echo ok`
- `uv run ruff check . && uv run pytest` pass on the changed code.
- A baseline run exists in `experiments/` (the latest `*_baseline.json`, unless the user names another run to compare against).

## 2. Re-index and evaluate

Indexing recreates the Qdrant collection, so always re-index before evaluating, with the same environment variables for both commands:

```bash
CHUNK_SIZE=300 uv run python scripts/index_documents.py
CHUNK_SIZE=300 uv run python scripts/evaluate_retrieval.py --name chunk-300
```

Check that the `config` block of the new run file in `experiments/` reflects the intended change. If the change is in code rather than config, say so: the `config` block will not show it, so the run name must.

## 3. Compare with the baseline

Read both run files and report:
- **Aggregate**: each metric in `metrics`, baseline vs experiment, with the delta.
- **Per question**: match questions by text and list those whose `recall` or `mrr` changed — wins first, then losses — with the expected section and what was retrieved instead. Skip unanswerable questions (`recall` is `null`).
- **Reading the numbers**: with 84 answerable questions, one question is ~1.2 points of recall, so a gain of 1–2 questions is likely noise unless the per-question analysis shows a consistent pattern. Point out any pattern you see (e.g. "all losses are in the Glossary"), and whether the result supports the user's hypothesis.

Give an opinion if useful, but end by asking the user whether they keep or reject the change.

## 4. After the user decides

- Add a row to the results table in `docs/ROADMAP.md`, for kept *and* rejected experiments: name, metrics, and the change and verdict in the Notes column.
- If rejected, remind the user to revert their change and re-index with the baseline config so the collection matches the defaults again. Do not revert it yourself.
- If kept, remind the user that a config change should become the new default in `config.py` and `.env.example`.
- Suggest commits: the user's change, then `docs: record <name> experiment` with the run file and the ROADMAP update.

Never edit `data/eval/questions.json` to improve a score.
