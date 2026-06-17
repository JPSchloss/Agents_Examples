"""Run the agent evaluation suite.

    python -m evals.run

Builds the knowledge index if needed, runs every case in evalset.CASES against the real
model, prints a pass/fail table, and exits non-zero if any case fails (so it can gate CI).

Costs tokens (it calls the model). Tip: set OPENAI_MODEL=gpt-4.1 for cheaper, faster evals.
"""

from __future__ import annotations

import asyncio
import os
import sys

from rich.console import Console
from rich.table import Table

from data_agent import config, rag
from evals.evalset import CASES, run_case

console = Console()


async def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is not set[/] — evals call the model and need a key.")
        return 2

    config.ensure_dirs()
    if not rag.index_exists():
        console.print("[dim]Building knowledge index for grounding evals…[/]")
        rag.build_index()

    console.print(
        f"Running [bold]{len(CASES)}[/] eval case(s) with model [cyan]{config.MODEL}[/]…\n"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("case")
    table.add_column("result")
    table.add_column("detail", overflow="fold")

    failures = 0
    for case in CASES:
        result = await run_case(case)
        if result.passed:
            table.add_row(case.name, "[green]PASS[/]", result.detail)
        else:
            failures += 1
            table.add_row(case.name, "[red]FAIL[/]", result.detail)

    console.print(table)
    total = len(CASES)
    if failures:
        console.print(f"\n[red]{failures}/{total} failed[/]")
        return 1
    console.print(f"\n[green]all {total} passed[/]")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
