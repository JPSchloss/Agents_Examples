# Article 5 · Guardrails & Safety — code

Complete, runnable project as of Article 5. Full walkthrough in [`../article.md`](../article.md).

**New since Article 4:**
- **Human-in-the-loop approval** — `write_file` and `run_python_file` ask for your OK
  before acting (via `ctx.context.approve(...)`, wired through the run context). **On by
  default**; pass `--auto-approve` to skip.
- **Output guardrail** (`safety.py`) — scans the final answer for leaked secrets
  (API keys, tokens, private keys) and blocks it.
- **Tool-input guardrail** (`safety.py`) — scans `write_file` content for dangerous
  operations (shell, network, deletion, eval/exec, sandbox escape) and refuses it, asking
  the model to rewrite.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
data-agent ingest
data-agent                          # approval ON — you'll confirm code execution
# or, for an uninterrupted demo:
data-agent chat --auto-approve
```

When you ask it to clean + load data, you'll see a yellow **"approve this action?"** panel
before each `write_file` and `run_python_file`. Answer `y` to proceed, `n` to decline (the
agent is told and can adapt).

## What's new in the tree

```
src/data_agent/safety.py           output guardrail + tool-input guardrail (regex, no LLM)
src/data_agent/context.py          auto_approve + approver, and ctx.approve(action, detail)
src/data_agent/tools/filesystem.py  write_file/run_python_file call ctx.approve() before acting;
                                    write_file gets the dangerous-code guardrail
src/data_agent/team.py             every agent gets the output (secret-leak) guardrail
src/data_agent/app.py              --auto-approve flag + the interactive approver
```

**Next article:** [Article 6 — Resilience & Observability](../../06-resilience-and-observability/article.md).
