# Article 6 · Resilience & Observability — code

Complete, runnable project as of Article 6. Full walkthrough in [`../article.md`](../article.md).

**New since Article 5:**
- **Resilience** (`resilience.py`) — installs a model client with `max_retries` + `timeout`
  so transient API failures (429/5xx/timeouts) are retried with backoff at the *request*
  layer (not by re-running the agent).
- **Observability** (`observability.py`) — a structured `data_agent` logger plus
  `AgentLogHooks` (a `RunHooks` implementation) that logs every agent/tool/handoff step,
  correlated by a per-turn id.
- Silent `except: pass` blocks in `rag.py` now log instead of hiding errors.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
data-agent ingest
data-agent chat --verbose                 # log each step to the console
# or capture a structured trace to a file:
data-agent chat --log-file workspace/agent.log
```

With `--verbose` you'll see lines like:

```
12:01:03 INFO    turn.start turn=ab12cd34
12:01:03 INFO    handoff turn=ab12cd34 from_agent=Triage to="Data Engineer"
12:01:05 INFO    tool.start turn=ab12cd34 agent="Data Engineer" tool=profile_dataset
12:01:06 INFO    tool.end turn=ab12cd34 tool=profile_dataset result_chars=812
```

New `chat` flags: `--verbose`, `--log-file PATH`. Resilience tunables:
`OPENAI_MAX_RETRIES`, `OPENAI_TIMEOUT` (see `.env.example`).

## What's new in the tree

```
src/data_agent/resilience.py     configure_resilience(): retrying model client
src/data_agent/observability.py  configure_logging() + AgentLogHooks (RunHooks)
src/data_agent/app.py            configures both at startup; passes hooks per turn; logs errors
src/data_agent/rag.py            swallowed exceptions now log (no silent failures)
```

**Next article:** [Article 7 — Evals, Tests & CI](../../07-evals-tests-and-ci/article.md).
