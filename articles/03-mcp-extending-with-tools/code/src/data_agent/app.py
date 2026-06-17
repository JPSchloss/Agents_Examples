"""Interactive command-line entry point.

Run it with:
    python -m data_agent.app

New in this article: we launch an MCP server for the lifetime of the session and build the
agent team around it. Because MCP servers are live connections that must be opened (async)
and closed cleanly, the team is now constructed by a factory, `build_team(...)`, instead of
at import time — and we manage the server with an AsyncExitStack.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack

from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession
from agents.mcp import MCPServerStdio

from . import config, rag
from .context import PipelineContext
from .team import build_team

BANNER = """\
╭──────────────────────────────────────────────────────────────╮
│  Data Pipeline Assistant  ·  OpenAI Agents SDK + RAG + MCP     │
│                                                                │
│  Try:                                                          │
│   • "Profile and clean data/raw/sales_2024.csv, then load it"  │
│   • "Build a dashboard for the cleaned sales data"             │
│   • "Per our standards, how should I handle returns?"          │
│                                                                │
│  Commands:  /artifacts   /reset   /exit                        │
╰──────────────────────────────────────────────────────────────╯
"""


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    config.ensure_dirs()
    context = PipelineContext()
    session = SQLiteSession("cli-session", str(config.MEMORY_DB))

    # Launch MCP servers for the lifetime of the session, then build the team around them.
    async with AsyncExitStack() as stack:
        mcp_servers = []
        mcp_status = "off"
        if config.REFERENCE_MCP_SERVER.exists():
            server = MCPServerStdio(
                params={
                    "command": sys.executable,
                    "args": [str(config.REFERENCE_MCP_SERVER)],
                },
                name="acme-data-standards",
                cache_tools_list=True,
                client_session_timeout_seconds=30,
            )
            await stack.enter_async_context(server)  # connect; auto-cleanup on exit
            mcp_servers.append(server)
            mcp_status = "on (acme-data-standards)"

        triage = build_team(mcp_servers=mcp_servers, with_knowledge=True)

        print(BANNER)
        print(f"(model: {config.MODEL}  ·  mcp: {mcp_status})")
        if not rag.index_exists():
            print(
                "NOTE: knowledge base not indexed — run `python -m data_agent.ingest` so the "
                "agents can ground answers in your docs."
            )
        print()

        await _repl(triage, context, session)


async def _repl(triage, context: PipelineContext, session: SQLiteSession) -> None:
    while True:
        try:
            user = input("user ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return

        if not user:
            continue
        if user in {"/exit", "/quit"}:
            print("bye!")
            return
        if user == "/artifacts":
            print("\n".join(f"  • {a}" for a in context.artifacts) or "  (none yet)")
            continue
        if user == "/reset":
            await session.clear_session()
            context.artifacts.clear()
            print("(conversation memory cleared)")
            continue

        try:
            result = await Runner.run(triage, user, context=context, session=session)
            print(f"\nassistant ▸ {result.final_output}\n")
        except InputGuardrailTripwireTriggered:
            print(
                "\nassistant ▸ That's outside what I do. I'm a data-pipeline assistant — "
                "ask me to ingest, profile, clean, pipeline, or visualize data, or for "
                "guidance on doing so.\n"
            )
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}\n")


def run() -> None:
    """Sync entry point used by the `data-agent` console script and `python -m`."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
