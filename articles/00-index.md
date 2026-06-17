# Building a Robust Agentic AI System — a hands-on series

A practical, build-it-as-you-read series on creating a real multi-agent application with the
**OpenAI Agents SDK**. We build a data assistant that profiles, cleans, and pipelines data,
loads it into a database, and builds a dashboard — then make it grounded, extensible, safe,
observable, and tested.

Each part adds **one meaningful capability** and ships a **complete, runnable code snapshot**
in its `code/` folder. You can start any part by cloning its folder; each builds on the one
before it.

> **Stack:** Python 3.10+ · OpenAI Agents SDK · pandas · SQLite · Streamlit, with ChromaDB
> (RAG) and MCP added along the way.

## The series

| # | Part | Adds | Code |
|---|------|------|------|
| 1 | [Foundation](./01-foundation/article.md) | Agents, tools, handoffs, a triage router, an input guardrail, sessions, structured outputs, tracing — the working multi-agent core | [`code`](./01-foundation/code) |
| 2 | [Grounding with RAG](./02-rag-knowledge-base/article.md) | A ChromaDB knowledge base + a `search_knowledge` retrieval tool, and how to *force* grounding | [`code`](./02-rag-knowledge-base/code) |
| 3 | [Extending with MCP](./03-mcp-extending-with-tools/article.md) | A local Model Context Protocol server + wiring external tool servers into the team | [`code`](./03-mcp-extending-with-tools/code) |
| 4 | [CLI & Developer Experience](./04-cli-and-developer-experience/article.md) | `argparse` subcommands, `-h` help, a `rich` REPL, token-usage display, cost controls | [`code`](./04-cli-and-developer-experience/code) |
| 5 | [Guardrails & Safety](./05-guardrails-and-safety/article.md) | Output guardrails, tool guardrails, and human-in-the-loop approval for code execution | [`code`](./05-guardrails-and-safety/code) |
| 6 | [Resilience & Observability](./06-resilience-and-observability/article.md) | Retries/backoff, structured logging, and exporting traces to your own stack | [`code`](./06-resilience-and-observability/code) |
| 7 | [Evals, Tests & CI](./07-evals-tests-and-ci/article.md) | Unit tests, an agent evaluation suite, and a CI workflow | [`code`](./07-evals-tests-and-ci/code) |

*(Parts 2–7 publish as the series continues.)*

> Each part ships a **cover image** (`articles/0N-*/cover.png`) generated from the project's
> own terminal output — no AI art. Regenerate them anytime with
> `python scripts/generate_covers.py`.

## How to read it

- **New here?** Start at [Part 1 — Foundation](./01-foundation/article.md) and run the code.
- **Want a specific technique?** Jump to its part; each `code/` folder is self-contained.
- **You'll need** an OpenAI API key. Each part's `code/.env.example` shows the variables;
  copy it to `.env`. If `gpt-5` isn't on your account, set `OPENAI_MODEL=gpt-4.1`.

## A throughline

One idea recurs in every part: **keep the model in charge of decisions, keep correctness in
ordinary code.** The model decides *when* to clean data, *what* to retrieve, or *which* tool
to call; deterministic code guarantees *how* those things actually happen. That split is what
makes an agentic system both capable and trustworthy.
