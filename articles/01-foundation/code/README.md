# Article 1 · Foundation — code

This folder is the **complete, runnable project** as of Article 1. The full walkthrough is
in [`../article.md`](../article.md).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # then add your OPENAI_API_KEY
python -m data_agent.app      # or: data-agent
```

Then try, in order:

```
user ▸ What data do I have to work with?
user ▸ Profile data/raw/sales_2024.csv and tell me what's wrong with it.
user ▸ Clean it, then load it into SQLite as a table called sales.
user ▸ Build me a dashboard for the cleaned sales data.
```

After the dashboard step, in a second terminal:

```bash
streamlit run workspace/dashboard.py
```

REPL commands: `/artifacts`, `/reset`, `/exit`.

> If `gpt-5` 404s on your account, set `OPENAI_MODEL=gpt-4.1` in `.env`.

## What's here

```
src/data_agent/
  config.py      models + the folders the agent may touch
  context.py     typed run context passed to every tool
  schemas.py     Pydantic models for structured output
  guardrails.py  the input relevance guardrail
  team.py        the agents + handoffs (the heart of the system)
  app.py         the interactive REPL (the agent-loop runner)
  tools/
    _paths.py    path-safety helper (sandbox enforcement)
    filesystem.py list / read / write files, run a script
    data.py      profile a CSV, load to SQLite, run SQL
data/raw/sales_2024.csv   deliberately messy sample data
```

**Next article:** [Article 2 — RAG](../../02-rag-knowledge-base/article.md) gives the agents
a grounded knowledge base.
