# Article 4 · CLI & Developer Experience — code

Complete, runnable project as of Article 4. Full walkthrough in [`../article.md`](../article.md).

**New since Article 3:** a real CLI — `argparse` subcommands (`chat`, `ingest`, `info`) with
`-h` everywhere, a `rich` REPL (markdown rendering, a thinking spinner, per-turn token
usage), and flags (`--model`, `--reset`, `--no-mcp`, `--no-rag`, `--no-trace`, `--max-turns`).
The standalone `python -m data_agent.ingest` from Article 2 is now `data-agent ingest`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add your OPENAI_API_KEY
data-agent ingest             # build the knowledge base (run once)
data-agent                    # chat (the default subcommand)
```

Explore the CLI:

```bash
data-agent -h                 # top-level help (subcommands)
data-agent chat -h            # all chat flags
data-agent info               # config + component status (RAG/MCP/models)
data-agent chat --model gpt-4.1 --reset --max-turns 12
```

REPL commands: `/help`, `/artifacts`, `/reset`, `/exit`.

> If `gpt-5` 404s on your account, set `OPENAI_MODEL=gpt-4.1` in `.env`
> (or pass `--model gpt-4.1`).

## What's new in the tree

```
src/data_agent/app.py    argparse subcommands + rich REPL (replaces the plain loop;
                         absorbs ingest as `data-agent ingest`, adds an `info` command)
```

**Next article:** [Article 5 — Guardrails & Safety](../../05-guardrails-and-safety/article.md).
