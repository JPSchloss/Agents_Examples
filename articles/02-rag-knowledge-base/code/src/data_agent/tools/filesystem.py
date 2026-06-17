"""Filesystem + code-execution tools.

These give the agents their hands: they can list inputs, read previews, write code or
config to the workspace, and execute a generated Python script. Together they're what lets
the Data Engineer agent author a real cleaning pipeline and the Frontend Builder write a
Streamlit app.

Every tool is decorated with `@function_tool`, which inspects the type hints and docstring
to build the JSON schema the model sees. Write clear docstrings — they ARE the tool's API
documentation as far as the model is concerned.

SECURITY NOTE: `run_python_file` executes code. We constrain it to the workspace, run it in
a subprocess with a timeout, and resolve all paths through `safe_resolve`. That is enough
for a local teaching example, NOT for running untrusted input in production — see the
"Hardening for production" section of the README.
"""

from __future__ import annotations

import subprocess
import sys

from agents import RunContextWrapper, function_tool

from ..context import PipelineContext
from ._paths import PathNotAllowed, safe_resolve

# Cap on how much file content we echo back into the model's context window.
_PREVIEW_LIMIT = 4000
_RUN_TIMEOUT_SECONDS = 180


@function_tool
def list_datasets(ctx: RunContextWrapper[PipelineContext]) -> str:
    """List the data files available to work with, in both the read-only raw data
    directory and the writable workspace. Use this first to discover inputs."""
    c = ctx.context
    lines: list[str] = []

    def describe(label: str, root) -> None:
        lines.append(f"## {label}: {root}")
        if not root.exists():
            lines.append("(missing)")
            return
        found = sorted(p for p in root.rglob("*") if p.is_file())
        if not found:
            lines.append("(empty)")
        for p in found:
            size = p.stat().st_size
            lines.append(f"- {p.relative_to(root)}  ({size:,} bytes)")

    describe("raw (read-only)", c.raw_data_dir)
    describe("workspace (writable)", c.workspace_dir)
    return "\n".join(lines)


@function_tool
def read_file_preview(ctx: RunContextWrapper[PipelineContext], path: str) -> str:
    """Read the first part of a text file (e.g. a CSV or a script) so you can inspect it.

    Args:
        path: Path relative to the workspace. To read a raw input use 'raw/<name>',
            e.g. 'raw/sales_2024.csv'.
    """
    c = ctx.context
    # Allow reading from the workspace OR, via the 'raw/' prefix, the raw data dir.
    if path.startswith("raw/"):
        target = safe_resolve(path[len("raw/") :], c.raw_data_dir)
    else:
        target = safe_resolve(path, c.workspace_dir, c.raw_data_dir)

    if not target.exists():
        return f"ERROR: file not found: {path}"

    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > _PREVIEW_LIMIT:
        return text[:_PREVIEW_LIMIT] + f"\n... [truncated, {len(text):,} chars total]"
    return text


@function_tool
def write_file(ctx: RunContextWrapper[PipelineContext], path: str, content: str) -> str:
    """Write a text file (a Python script, SQL, or config) into the workspace, creating
    parent folders as needed. Overwrites any existing file at that path.

    Args:
        path: Destination path relative to the workspace, e.g. 'clean_sales.py'.
        content: The full file contents to write.
    """
    c = ctx.context
    try:
        target = safe_resolve(path, c.workspace_dir)
    except PathNotAllowed as e:
        return f"ERROR: {e}"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    rel = target.relative_to(c.workspace_dir)
    c.record(f"wrote workspace/{rel}")
    return f"OK: wrote {len(content):,} chars to workspace/{rel}"


@function_tool
def run_python_file(ctx: RunContextWrapper[PipelineContext], path: str) -> str:
    """Execute a Python script that lives in the workspace and return its stdout/stderr.

    Use this to RUN a cleaning or pipeline script you wrote with write_file. The script
    runs with the workspace as its working directory, so it can write outputs with simple
    relative names (e.g. 'sales_clean.csv'). To read a raw input, use the ABSOLUTE path
    that list_datasets prints for the raw directory.

    Args:
        path: Path to the script relative to the workspace, e.g. 'clean_sales.py'.
    """
    c = ctx.context
    try:
        target = safe_resolve(path, c.workspace_dir)
    except PathNotAllowed as e:
        return f"ERROR: {e}"
    if not target.exists():
        return f"ERROR: script not found: {path}"

    try:
        proc = subprocess.run(
            [sys.executable, str(target)],
            cwd=str(c.workspace_dir),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: script timed out after {_RUN_TIMEOUT_SECONDS}s."

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    parts = [f"{status} running workspace/{target.relative_to(c.workspace_dir)}"]
    if out:
        parts.append(f"--- stdout ---\n{out[:_PREVIEW_LIMIT]}")
    if err:
        parts.append(f"--- stderr ---\n{err[:_PREVIEW_LIMIT]}")
    return "\n".join(parts)
