"""Command-line interface for the data-pipeline assistant.

Subcommands (run any with -h for details):

    data-agent ingest      Build/refresh the RAG knowledge index from knowledge/*.md
    data-agent chat        Start the interactive assistant (this is the default)
    data-agent info        Show configuration + component status and exit

`chat` is the default, so bare `data-agent` (or `python -m data_agent.app`) opens the REPL.
The REPL uses `rich` for readable, markdown-rendered output and reports token usage per
turn. MCP servers are launched for the session and cleaned up on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    SQLiteSession,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from . import __version__, config, rag
from .context import PipelineContext
from .observability import AgentLogHooks, configure_logging, logger, new_turn_id
from .resilience import configure_resilience
from .team import build_team

console = Console()
_COMMANDS = {"chat", "ingest", "info"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _require_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]OPENAI_API_KEY is not set.[/] Copy .env.example to .env and add your key."
        )
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-agent",
        description="A multi-agent data-pipeline assistant — OpenAI Agents SDK + RAG + MCP.",
        epilog=(
            "examples:\n"
            "  data-agent ingest                 build the knowledge base (run once)\n"
            "  data-agent                        start chatting (chat is the default)\n"
            "  data-agent chat --model gpt-4.1   chat with a specific model\n"
            "  data-agent chat --reset           start a fresh conversation\n"
            "  data-agent info                   show config + component status\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"data-agent {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{chat,ingest,info}")

    chat = sub.add_parser(
        "chat",
        help="Start the interactive assistant (default).",
        description="Start the interactive REPL. Memory persists per --session id.",
    )
    chat.add_argument("--model", help="Override OPENAI_MODEL for this run (e.g. gpt-4.1).")
    chat.add_argument(
        "--session", default="cli-session", help="Conversation id for persistent memory."
    )
    chat.add_argument(
        "--reset", action="store_true", help="Clear this session's memory before starting."
    )
    chat.add_argument("--no-mcp", action="store_true", help="Disable MCP servers.")
    chat.add_argument("--no-rag", action="store_true", help="Disable the RAG knowledge tool.")
    chat.add_argument("--no-trace", action="store_true", help="Disable tracing for this run.")
    chat.add_argument(
        "--verbose", action="store_true", help="Log each agent/tool/handoff step to the console."
    )
    chat.add_argument(
        "--log-file", metavar="PATH", help="Write structured per-step logs to this file."
    )
    chat.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip human approval for code execution / file writes (off = approval ON).",
    )
    chat.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max agent-loop steps per request (a cost/runaway guard). Default 20.",
    )

    ingest = sub.add_parser(
        "ingest",
        help="Build/refresh the RAG knowledge index from knowledge/*.md.",
        description="Chunk and embed the markdown files in knowledge/ into a local index.",
    )
    ingest.add_argument(
        "--force", action="store_true", help="Rebuild even if an index already exists."
    )

    sub.add_parser(
        "info", help="Show configuration and component status, then exit."
    )
    return parser


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_ingest(force: bool) -> None:
    _require_key()
    config.ensure_dirs()
    if rag.index_exists() and not force:
        console.print(
            "[yellow]A knowledge index already exists.[/] Re-run with --force to rebuild."
        )
        return
    with console.status("[dim]Embedding knowledge base into Chroma…[/]", spinner="dots"):
        n = rag.build_index()
    console.print(
        f"[green]✓[/] Indexed [bold]{n}[/] chunks from {config.KNOWLEDGE_DIR.name}/ → "
        f"Chroma collection '{config.CHROMA_COLLECTION}' (workspace/{config.CHROMA_DIR.name})"
    )


def cmd_info() -> None:
    config.ensure_dirs()
    t = Table(title="data-agent · configuration & status", show_header=False, box=None)
    t.add_column(style="cyan", justify="right")
    t.add_column()
    t.add_row("API key set", "✓" if os.getenv("OPENAI_API_KEY") else "[red]missing[/]")
    t.add_row("Reasoning model", config.MODEL)
    t.add_row("Guardrail model", config.GUARDRAIL_MODEL)
    t.add_row("Embedding model", config.EMBED_MODEL)
    t.add_row("Knowledge dir", str(config.KNOWLEDGE_DIR))
    docs = len(list(config.KNOWLEDGE_DIR.glob("*.md")))
    if rag.index_exists():
        t.add_row("RAG (Chroma)", f"[green]built[/] from {docs} doc(s) → collection '{config.CHROMA_COLLECTION}'")
    else:
        t.add_row("RAG (Chroma)", f"[yellow]not built[/] — run `data-agent ingest` ({docs} doc(s) ready)")
    t.add_row(
        "MCP server",
        "✓ acme-data-standards" if config.REFERENCE_MCP_SERVER.exists() else "[red]missing[/]",
    )
    t.add_row("Workspace", str(config.WORKSPACE_DIR))
    console.print(t)


async def cmd_chat(args: argparse.Namespace) -> None:
    _require_key()
    config.ensure_dirs()

    # Observability + resilience first, so everything after is logged and retried.
    configure_logging(verbose=args.verbose, log_file=args.log_file)
    configure_resilience()

    if args.model:
        config.MODEL = args.model  # build_team reads config.MODEL at call time
    if args.no_trace:
        set_tracing_disabled(True)

    with_knowledge = not args.no_rag
    context = PipelineContext(
        auto_approve=args.auto_approve,
        approver=None if args.auto_approve else _make_approver(),
    )
    session = SQLiteSession(args.session, str(config.MEMORY_DB))
    if args.reset:
        await session.clear_session()

    # Launch MCP servers for the lifetime of the session, then build the team around them.
    async with AsyncExitStack() as stack:
        mcp_servers = []
        mcp_status = "off"
        if not args.no_mcp:
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
                await stack.enter_async_context(server)
                mcp_servers.append(server)
                mcp_status = "on (acme-data-standards)"
            else:
                mcp_status = "[yellow]server file missing[/]"

        triage = build_team(mcp_servers=mcp_servers, with_knowledge=with_knowledge)

        _print_banner(mcp_status, with_knowledge, args.auto_approve)
        await _repl(triage, context, session, args.max_turns)


# --------------------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------------------
def _print_banner(mcp_status: str, with_knowledge: bool, auto_approve: bool) -> None:
    rag_status = "on" if with_knowledge else "off"
    if with_knowledge and not rag.index_exists():
        rag_status = "[yellow]on but NOT indexed — run `data-agent ingest`[/]"
    approval = "[yellow]auto (no prompts)[/]" if auto_approve else "on (you confirm)"
    console.print(
        Panel(
            "Talk to me in plain English. I can profile, clean, and pipeline data,\n"
            "build a Streamlit dashboard, and advise you — grounded in your knowledge base.\n\n"
            "[dim]try:[/] \"profile and clean data/raw/sales_2024.csv, then load it\"\n"
            "[dim]    [/] \"build a dashboard for the cleaned sales data\"\n"
            "[dim]    [/] \"how should I handle the negative-quantity rows?\"\n\n"
            "[dim]commands:[/] /help  /artifacts  /reset  /exit",
            title="[bold]Data Pipeline Assistant[/]",
            subtitle="OpenAI Agents SDK · RAG · MCP",
            border_style="cyan",
        )
    )
    console.print(
        f"[dim]model:[/] {config.MODEL}   [dim]rag:[/] {rag_status}   "
        f"[dim]mcp:[/] {mcp_status}   [dim]approval:[/] {approval}\n"
    )


def _make_approver():
    """Return an approver callback the dangerous tools call before acting.

    It renders the proposed action and asks for a yes/no (default: no). The tool itself
    invokes this via `ctx.context.approve(...)`, so the prompt appears inline, mid-run,
    exactly when the model wants to write a file or execute code.
    """

    def approver(action: str, detail: str) -> bool:
        console.print(
            Panel(detail, title=f"[yellow]⚠ approve: {action}?[/]", border_style="yellow")
        )
        return Confirm.ask("  allow it?", default=False)

    return approver


async def _repl(
    triage,
    context: PipelineContext,
    session: SQLiteSession,
    max_turns: int,
) -> None:
    while True:
        try:
            user = console.input("[bold cyan]user ▸[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye!")
            return

        if not user:
            continue
        if user.startswith("/"):
            if user in ("/exit", "/quit"):
                console.print("bye!")
                return
            if user == "/reset":
                await session.clear_session()
                context.artifacts.clear()
                console.print("[dim](conversation memory cleared)[/]")
                continue
            if user == "/artifacts":
                console.print(
                    "\n".join(f"  • {a}" for a in context.artifacts) or "[dim](none yet)[/]"
                )
                continue
            if user == "/help":
                console.print(
                    "[cyan]/artifacts[/] files the agents created · "
                    "[cyan]/reset[/] clear memory · [cyan]/exit[/] quit"
                )
                continue
            console.print(f"[yellow]unknown command:[/] {user}  (try /help)")
            continue

        turn_id = new_turn_id()
        try:
            # No console.status spinner here: a dangerous tool may print an approval
            # prompt mid-run, and a live spinner would clash with reading your input.
            console.print(f"[dim]· working… (turn {turn_id})[/]")
            logger.info("turn.start " + f"turn={turn_id}")
            result = await Runner.run(
                triage,
                user,
                context=context,
                session=session,
                max_turns=max_turns,
                hooks=AgentLogHooks(turn_id),  # structured per-step logging
            )
            console.print(
                Panel(
                    Markdown(str(result.final_output)),
                    title=f"[green]{result.last_agent.name}[/]",
                    border_style="green",
                )
            )
            _print_usage(result)
        except InputGuardrailTripwireTriggered:
            logger.warning("input_guardrail.tripped " + f"turn={turn_id}")
            console.print(
                "[yellow]assistant ▸[/] That's outside what I do. I'm a data-pipeline "
                "assistant — ask me to ingest, profile, clean, pipeline, or visualize data, "
                "or for guidance on doing so."
            )
        except OutputGuardrailTripwireTriggered:
            logger.warning("output_guardrail.tripped " + f"turn={turn_id}")
            console.print(
                "[red]⚠ blocked:[/] the response looked like it contained a secret "
                "(API key / token / private key), so I withheld it."
            )
        except Exception as e:  # keep the REPL alive on transient API/tool errors
            # Log the full traceback for diagnosis; show the user a one-line summary.
            logger.exception("turn.error " + f"turn={turn_id}")
            console.print(f"[red][error][/] {type(e).__name__}: {e}  [dim](turn {turn_id})[/]")


def _print_usage(result) -> None:
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage and getattr(usage, "total_tokens", 0):
        console.print(
            f"[dim]· {usage.total_tokens:,} tokens "
            f"(in {usage.input_tokens:,} / out {usage.output_tokens:,}) "
            f"over {usage.requests} request(s)[/]\n"
        )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def run() -> None:
    """Console-script / `python -m` entry point. Routes to a subcommand (chat default)."""
    argv = sys.argv[1:]
    help_flags = {"-h", "--help", "--version"}
    if not argv:
        argv = ["chat"]
    elif argv[0] not in _COMMANDS and argv[0] not in help_flags:
        # Allow top-level chat flags: `data-agent --model x` -> `data-agent chat --model x`.
        argv = ["chat"] + argv

    args = _build_parser().parse_args(argv)
    command = args.command or "chat"

    if command == "ingest":
        cmd_ingest(args.force)
    elif command == "info":
        cmd_info()
    else:
        try:
            asyncio.run(cmd_chat(args))
        except KeyboardInterrupt:
            console.print("\nbye!")


if __name__ == "__main__":
    run()
