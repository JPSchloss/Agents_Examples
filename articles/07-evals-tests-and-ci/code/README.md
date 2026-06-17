# Article 7 · Evals, Tests & CI — code

Complete, runnable project as of Article 7 — the final stage. Full walkthrough in
[`../article.md`](../article.md).

**New since Article 6:**
- **`tests/`** — fast, deterministic unit tests (no API key): the path sandbox, RAG chunker,
  safety regexes, approval logic, and the MCP server tools.
- **`evals/`** — a behavioral evaluation suite (`python -m evals.run`) that runs the real
  agent and asserts on routing, tool use, grounding, and guardrails.
- **`.github/workflows/ci.yml`** — runs lint + unit tests on every push; runs evals on
  `main` when an `OPENAI_API_KEY` secret is present.
- Dev tooling in `pyproject.toml`: `pip install -e ".[dev]"` (pytest + ruff), plus pytest
  and ruff config.

## Run the tests (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest                       # deterministic unit tests
```

## Run the evals (calls the model — needs a key)

```bash
cp .env.example .env         # add OPENAI_API_KEY (set OPENAI_MODEL=gpt-4.1 for cheaper evals)
python -m evals.run
```

Prints a pass/fail table and exits non-zero on failure (so CI can gate on it).

## What's new in the tree

```
tests/                       deterministic unit tests (run in CI, no key)
  test_paths.py              the sandbox: traversal/absolute paths rejected
  test_rag_chunking.py       chunker splits + packs paragraphs
  test_safety.py             secret + dangerous-code regexes
  test_context.py            human-in-the-loop approval logic
  test_mcp_server.py         MCP data-standards tools
evals/                       behavioral evals (run the real agent)
  evalset.py                 cases + harness (routing / tools / grounding / guardrail)
  run.py                     `python -m evals.run` → pass/fail table, exit code
.github/workflows/ci.yml     lint + unit tests on push; evals on main (gated on secret)
pyproject.toml               [dev] extras, pytest + ruff config
```

This is the end of the series — see [`../../00-index.md`](../../00-index.md) for the whole arc.
