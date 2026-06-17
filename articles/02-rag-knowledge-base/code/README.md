# Article 2 · Grounding with RAG — code

Complete, runnable project as of Article 2. Full walkthrough in [`../article.md`](../article.md).

**New since Article 1:** a ChromaDB knowledge base (`knowledge/`, `rag.py`,
`tools/knowledge.py`), a `search_knowledge` tool on the Data Engineer and Advisor, and the
Advisor configured to *force* retrieval (`tool_choice="required"`).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env              # add your OPENAI_API_KEY
python -m data_agent.ingest       # build the knowledge base (run once)
python -m data_agent.app          # chat
```

Try a grounded question:

```
user ▸ Per our standards, how should I handle negative-quantity rows, and what defines total?
```

The Advisor will call `search_knowledge`, retrieve the rule from `company_data_dictionary.md`
("returns → exclude from revenue", `total = quantity × unit_price`), and cite it.

REPL commands: `/artifacts`, `/reset`, `/exit`.

> Re-run `python -m data_agent.ingest` whenever you edit files in `knowledge/`.
> If `gpt-5` 404s on your account, set `OPENAI_MODEL=gpt-4.1` in `.env`.

## What's new in the tree

```
knowledge/
  company_data_dictionary.md       authoritative business rules (RAG source)
  data_engineering_principles.md   house engineering standards (RAG source)
src/data_agent/
  rag.py                           chunk → ChromaDB embed/store/search
  ingest.py                        `python -m data_agent.ingest`
  tools/knowledge.py               the search_knowledge retrieval tool
```

**Next article:** [Article 3 — MCP](../../03-mcp-extending-with-tools/article.md).
