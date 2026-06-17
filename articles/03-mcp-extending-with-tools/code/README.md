# Article 3 · Extending with MCP — code

Complete, runnable project as of Article 3. Full walkthrough in [`../article.md`](../article.md).

**New since Article 2:** a local **MCP server** (`mcp_servers/reference_server.py`,
"acme-data-standards") exposing deterministic naming/typing helpers, the team refactored into
a `build_team(...)` **factory**, and `app.py` managing the MCP server's lifecycle with an
`AsyncExitStack`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env              # add your OPENAI_API_KEY
python -m data_agent.ingest       # build the knowledge base (run once)
python -m data_agent.app          # chat — the MCP server launches automatically
```

The app spawns `reference_server.py` as a subprocess over stdio; its tools
(`canonical_column_name`, `standard_dtype`, `naming_conventions`) appear to the agents
alongside their native tools. Try:

```
user ▸ Per our standards, what's the canonical snake_case name for 'Order Date', and how should I handle returns?
```

(The Advisor uses MCP for exact naming and RAG for the business rule.)

You can smoke-test the server's tools without the LLM:

```bash
python mcp_servers/reference_server.py   # waits on stdio; Ctrl-C to exit
```

## What's new in the tree

```
mcp_servers/reference_server.py    a FastMCP stdio server (data-standards helpers)
src/data_agent/team.py             now build_team(mcp_servers=..., with_knowledge=...)
src/data_agent/app.py              launches MCP via AsyncExitStack, calls build_team
```

**Next article:** [Article 4 — CLI & Developer Experience](../../04-cli-and-developer-experience/article.md).
