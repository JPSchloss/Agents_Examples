"""Interactive command-line entry point.

Run it with:
    python -m data_agent.app

This ties everything together:
  * a SQLiteSession so the conversation (and which specialist you're talking to) persists
    across turns and even across restarts,
  * a single Runner.run per user turn, always starting at the triage agent,
  * graceful handling of the guardrail tripwire,
  * a tiny streaming-free REPL so the example stays readable.
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession

from . import config, rag
from .context import PipelineContext
from .team import triage_agent

BANNER = """\
╭──────────────────────────────────────────────────────────────╮
│  Data Pipeline Assistant  ·  OpenAI Agents SDK example         │
│                                                                │
│  Try:                                                          │
│   • "What data do I have to work with?"                        │
│   • "Profile and clean data/raw/sales_2024.csv, then load it"  │
│   • "Build a dashboard for the cleaned sales data"             │
│   • "How should I think about idempotent pipelines?"  (advice) │
│                                                                │
│  Commands:  /artifacts   /reset   /exit                        │
╰──────────────────────────────────────────────────────────────╯
"""


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    config.ensure_dirs()

    # The run context every tool receives. One per process is fine for a CLI.
    context = PipelineContext()

    # Persistent conversation memory. Same session id => same thread across restarts.
    session = SQLiteSession("cli-session", str(config.MEMORY_DB))

    print(BANNER)
    print(f"(model: {config.MODEL}  ·  workspace: {config.WORKSPACE_DIR})")
    if not rag.index_exists():
        print(
            "NOTE: knowledge base not indexed — run `python -m data_agent.ingest` so the "
            "agents can ground answers in your docs."
        )
    print()

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
            print("Artifacts created this session:")
            print("\n".join(f"  • {a}" for a in context.artifacts) or "  (none yet)")
            continue
        if user == "/reset":
            await session.clear_session()
            context.artifacts.clear()
            print("(conversation memory cleared)")
            continue

        try:
            # Always enter through triage; the session carries prior turns + which
            # specialist last had the floor, so multi-step flows continue naturally.
            result = await Runner.run(
                triage_agent, user, context=context, session=session
            )
            print(f"\nassistant ▸ {result.final_output}\n")
        except InputGuardrailTripwireTriggered:
            print(
                "\nassistant ▸ That's outside what I do. I'm a data-pipeline assistant — "
                "ask me to ingest, profile, clean, pipeline, or visualize data, or for "
                "guidance on doing so.\n"
            )
        except Exception as e:  # keep the REPL alive on transient API/tool errors
            print(f"\n[error] {type(e).__name__}: {e}\n")


def run() -> None:
    """Sync entry point used by the `data-agent` console script and `python -m`."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
