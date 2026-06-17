"""Generate cover images for the article series — from the project's *own* output.

No AI image generation: each cover is the signature terminal output of that stage, rendered
through one consistent `rich` theme and exported as an SVG (a clean terminal-window card).
A separate step converts the SVGs to PNG with macOS Quick Look (`qlmanage`).

    python scripts/generate_covers.py

Writes articles/0N-*/cover.svg for each part. The content mirrors the real app's rich
components (banner, info table, approval prompt, log trace, eval table, etc.).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageChops
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.text import Text
from rich.tree import Tree

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "articles"
WIDTH = 100

SERIES = "Building a Robust Agentic AI System"


def header(part: int, title: str) -> Text:
    t = Text()
    t.append(f"{SERIES}\n", style="bold white")
    t.append(f"Part {part} — {title}", style="bold #7CE7C4")
    return t


# --- per-part renderables (mirror the real app output) ----------------------------


def part1() -> Group:
    tree = Tree("[bold]Triage / Orchestrator[/]  [dim](routes the request · input guardrail)[/]")
    de = tree.add("[#7CE7C4]Data Engineer[/]  [dim]profile · clean · run code · load SQLite[/]")
    de.add("[dim]tools: list_datasets, profile_dataset, write_file, run_python_file, query_sqlite[/]")
    tree.add("[#7CE7C4]Frontend Builder[/]  [dim]writes a Streamlit dashboard[/]")
    tree.add("[#7CE7C4]Advisor[/]  [dim]explains & recommends[/]")
    tree.add("[dim]↩ each specialist hands back to Triage[/]")
    return Group(
        header(1, "Foundation"),
        Text(),
        Panel(tree, title="[bold]multi-agent team[/]", border_style="#7C8CF8", expand=True),
        Text("agents · tools · handoffs · guardrail · sessions · structured output · tracing", style="dim"),
    )


def part2() -> Group:
    answer = (
        "Per [italic]company_data_dictionary.md[/]:\n"
        "• Negative quantities are [bold]returns[/] — exclude them from revenue.\n"
        "• [bold]total = quantity × unit_price[/]; recompute when missing.\n"
        "• Canonical regions: North, South, East, West."
    )
    return Group(
        header(2, "Grounding with RAG"),
        Text(),
        Text("user ▸ Per our standards, how should I handle negative-quantity rows?", style="bold cyan"),
        Text("· tool.start  search_knowledge  → ChromaDB  (2 sources)", style="dim"),
        Panel(answer, title="[green]Advisor[/]", border_style="green", expand=True),
    )


def part3() -> Group:
    t = Table(show_header=False, box=None, expand=True)
    t.add_column(style="#7CE7C4", justify="right")
    t.add_column()
    t.add_row("MCP server", "✓ acme-data-standards (stdio)")
    t.add_row("tools", "canonical_column_name · standard_dtype · naming_conventions")
    t.add_row("call", "canonical_column_name('Order Date')  →  order_date")
    t.add_row("call", "standard_dtype('currency')  →  float64 (USD, no symbols)")
    return Group(
        header(3, "Extending with MCP"),
        Text(),
        Panel(t, title="[bold]Model Context Protocol[/]", border_style="#7C8CF8", expand=True),
        Text("external tool servers over one open protocol — no per-integration glue", style="dim"),
    )


def part4() -> Group:
    cmds = Table(show_header=False, box=None, expand=True)
    cmds.add_column(style="#7CE7C4")
    cmds.add_column()
    cmds.add_row("data-agent chat", "start the assistant (default)")
    cmds.add_row("data-agent ingest", "build the RAG knowledge base")
    cmds.add_row("data-agent info", "config + component status")
    status = Text("model: gpt-5   rag: on   mcp: on (acme-data-standards)   approval: on", style="dim")
    usage = Text("· 4,512 tokens (in 3,980 / out 532) over 3 request(s)", style="dim")
    return Group(
        header(4, "CLI & Developer Experience"),
        Text(),
        Panel(Group(cmds, Text(), status, usage), title="[bold]data-agent[/]", border_style="#7C8CF8", expand=True),
    )


def part5() -> Group:
    approve = Panel(
        "path: clean_sales.py\n[dim]--- content preview ---[/]\n"
        "import pandas as pd\ndf = pd.read_csv(RAW)\ndf = df.drop_duplicates('order_id') …",
        title="[yellow]⚠ approve: write_file?[/]",
        border_style="yellow",
        expand=True,
    )
    return Group(
        header(5, "Guardrails & Safety"),
        Text(),
        approve,
        Text("  allow it? [y/n] (n): y", style="bold"),
        Text("⚠ blocked: response looked like it contained a secret — withheld.", style="red"),
        Text("defense in depth: input + tool + output guardrails + human approval", style="dim"),
    )


def part6() -> Group:
    log = Text()
    lines = [
        ("13:07:22 INFO  ", "resilience configured max_retries=5 timeout=60.0"),
        ("13:07:22 INFO  ", "turn.start turn=702aad8c"),
        ("13:07:24 INFO  ", "handoff turn=702aad8c from_agent=Triage to=Advisor"),
        ("13:07:28 INFO  ", "tool.start turn=702aad8c agent=Advisor tool=search_knowledge"),
        ("13:07:29 INFO  ", "tool.end   turn=702aad8c tool=search_knowledge result_chars=3788"),
    ]
    for ts, msg in lines:
        log.append(ts, style="dim")
        log.append(msg + "\n", style="#7CE7C4")
    return Group(
        header(6, "Resilience & Observability"),
        Text(),
        Panel(log, title="[bold]structured run trace[/]", border_style="#7C8CF8", expand=True),
        Text("retry at the request layer · log every step · correlate by turn id", style="dim"),
    )


def part7() -> Group:
    t = Table(show_header=True, header_style="bold", expand=True)
    t.add_column("case")
    t.add_column("result")
    t.add_row("routes_data_questions_to_engineer", "[green]PASS[/]")
    t.add_row("grounds_advice_in_knowledge_base", "[green]PASS[/]")
    t.add_row("refuses_off_topic_requests", "[green]PASS[/]")
    return Group(
        header(7, "Evals, Tests & CI"),
        Text(),
        Panel(t, title="[bold]python -m evals.run[/]", border_style="#7C8CF8", expand=True),
        Text("ruff: all checks passed   ·   pytest: 37 passed   ·   evals: 3/3 passed", style="green"),
    )


COVERS = {
    "01-foundation": part1,
    "02-rag-knowledge-base": part2,
    "03-mcp-extending-with-tools": part3,
    "04-cli-and-developer-experience": part4,
    "05-guardrails-and-safety": part5,
    "06-resilience-and-observability": part6,
    "07-evals-tests-and-ci": part7,
}


def to_png(svg_path: Path, size: int = 2200, pad: int = 48) -> Path | None:
    """Convert an SVG to a tight landscape PNG: qlmanage rasterizes (macOS, square + opaque
    margins), then we crop to the card by detecting the margin color and re-pad with that
    same color so the window's rounded corners blend seamlessly into the border."""
    outdir = svg_path.parent
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", str(size), "-o", str(outdir), str(svg_path)],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None  # not on macOS / no qlmanage — SVG is still written

    tmp = outdir / (svg_path.name + ".png")
    if not tmp.exists():
        return None
    im = Image.open(tmp).convert("RGB")
    margin = im.getpixel((0, 0))  # the (opaque) padding color qlmanage produced
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, margin)).getbbox()
    card = im.crop(bbox) if bbox else im
    canvas = Image.new("RGB", (card.width + 2 * pad, card.height + 2 * pad), margin)
    canvas.paste(card, (pad, pad))
    png = outdir / "cover.png"
    canvas.save(png)
    tmp.unlink(missing_ok=True)
    return png


def main() -> None:
    for folder, render in COVERS.items():
        with open("/dev/null", "w") as devnull:
            console = Console(record=True, width=WIDTH, file=devnull)
            console.print(render())
        svg = ARTICLES / folder / "cover.svg"
        console.save_svg(str(svg), title="data-agent", theme=MONOKAI)
        png = to_png(svg)
        suffix = f" + {png.name}" if png else " (svg only — qlmanage unavailable)"
        print(f"wrote {svg.relative_to(ROOT)}{suffix}")


if __name__ == "__main__":
    main()
