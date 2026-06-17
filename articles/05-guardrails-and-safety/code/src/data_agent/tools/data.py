"""Data tools: profile a dataset, load a CSV into SQLite, and query SQLite.

These are *deterministic* tools — plain pandas/sqlite code, no LLM inside. That's a
deliberate split: anything that must be correct and repeatable (counting nulls, running
SQL) is ordinary code the model merely *orchestrates*. The model decides WHEN to profile
or WHAT SQL to run; the tool guarantees HOW it's computed.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
from agents import RunContextWrapper, function_tool

from ..context import PipelineContext
from ..schemas import ColumnProfile, DataProfile
from ._paths import PathNotAllowed, safe_resolve

_MAX_SQL_ROWS = 50


def _resolve_data_path(c: PipelineContext, path: str):
    """Resolve a data path. 'raw/...' reads the read-only raw dir; anything else is the
    workspace."""
    if path.startswith("raw/"):
        return safe_resolve(path[len("raw/") :], c.raw_data_dir)
    return safe_resolve(path, c.workspace_dir, c.raw_data_dir)


@function_tool
def profile_dataset(ctx: RunContextWrapper[PipelineContext], path: str) -> DataProfile:
    """Profile a CSV file and return a structured data-quality report: row/column counts,
    per-column dtype, null counts, distinct counts, sample values, and heuristic warnings.

    Always profile a dataset before cleaning it so you understand its shape and problems.

    Args:
        path: 'raw/sales_2024.csv' for a raw input, or a workspace-relative path.
    """
    c = ctx.context
    try:
        target = _resolve_data_path(c, path)
    except PathNotAllowed as e:
        # Tools can return any JSON-serializable type; raising would surface as a tool
        # error, but here a structured "empty" profile with the issue is friendlier.
        return DataProfile(
            path=path, row_count=0, column_count=0, duplicate_rows=0,
            columns=[], issues=[str(e)],
        )
    if not target.exists():
        return DataProfile(
            path=path, row_count=0, column_count=0, duplicate_rows=0,
            columns=[], issues=[f"file not found: {path}"],
        )

    # Read everything as strings first so we see the data as-is (currency symbols, blank
    # cells, mixed date formats) instead of letting pandas silently coerce it.
    df = pd.read_csv(target, dtype=str, keep_default_na=False)
    n_rows = len(df)

    columns: list[ColumnProfile] = []
    issues: list[str] = []

    for col in df.columns:
        series = df[col]
        blank = series.str.strip().eq("") | series.str.lower().isin({"na", "nan", "null"})
        null_count = int(blank.sum())
        distinct = int(series[~blank].nunique())
        samples = [s for s in series[~blank].unique().tolist()[:5]]
        columns.append(
            ColumnProfile(
                name=col,
                dtype="string (raw)",
                null_count=null_count,
                null_pct=round(100 * null_count / n_rows, 1) if n_rows else 0.0,
                distinct_count=distinct,
                sample_values=samples,
            )
        )
        if null_count:
            issues.append(f"'{col}' has {null_count} missing value(s).")

    dupes = int(df.duplicated().sum())
    if dupes:
        issues.append(f"{dupes} fully duplicated row(s).")

    # A couple of cheap cross-column heuristics that make the report feel "smart".
    for col in df.columns:
        sample = df[col].str.strip()
        if sample.str.contains(r"^\$", regex=True, na=False).any():
            issues.append(f"'{col}' mixes currency symbols into a numeric field.")
        if col.lower().endswith("date"):
            formats = sample[sample != ""].str.replace(r"\d", "9", regex=True).nunique()
            if formats > 1:
                issues.append(f"'{col}' uses {formats} different date formats.")

    return DataProfile(
        path=str(target.name),
        row_count=n_rows,
        column_count=len(df.columns),
        duplicate_rows=dupes,
        columns=columns,
        issues=issues,
    )


@function_tool
def load_csv_to_sqlite(
    ctx: RunContextWrapper[PipelineContext],
    csv_path: str,
    table_name: str,
    db_name: str = "analytics.db",
) -> str:
    """Load a (preferably cleaned) CSV from the workspace into a SQLite table so it can be
    queried and powered by a dashboard.

    Args:
        csv_path: Workspace-relative path to the CSV, e.g. 'sales_clean.csv'.
        table_name: Destination table name, e.g. 'sales'.
        db_name: SQLite database filename in the workspace (default 'analytics.db').
    """
    c = ctx.context
    try:
        source = safe_resolve(csv_path, c.workspace_dir)
        db_path = safe_resolve(db_name, c.workspace_dir)
    except PathNotAllowed as e:
        return f"ERROR: {e}"
    if not source.exists():
        return f"ERROR: CSV not found in workspace: {csv_path}"

    df = pd.read_csv(source)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    c.record(f"loaded {len(df)} rows into table '{table_name}' in {db_name}")
    return (
        f"OK: loaded {len(df)} rows / {len(df.columns)} columns into table "
        f"'{table_name}' in workspace/{db_name}. Columns: {', '.join(df.columns)}"
    )


@function_tool
def query_sqlite(
    ctx: RunContextWrapper[PipelineContext], sql: str, db_name: str = "analytics.db"
) -> str:
    """Run a read-only SQL query against a workspace SQLite database and return the rows
    as a markdown table. Use this to validate a load or to answer analytical questions.

    Args:
        sql: A single SELECT statement.
        db_name: SQLite database filename in the workspace (default 'analytics.db').
    """
    c = ctx.context
    if not sql.strip().lower().startswith("select"):
        return "ERROR: only read-only SELECT statements are allowed."
    try:
        db_path = safe_resolve(db_name, c.workspace_dir)
    except PathNotAllowed as e:
        return f"ERROR: {e}"
    if not db_path.exists():
        return f"ERROR: database not found: {db_name} (load data first)."

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(sql, conn)
    except Exception as e:  # surface SQL errors back to the model so it can self-correct
        return f"SQL ERROR: {e}"

    if len(df) > _MAX_SQL_ROWS:
        head = df.head(_MAX_SQL_ROWS).to_markdown(index=False)
        return f"{head}\n... [{len(df)} rows total, showing first {_MAX_SQL_ROWS}]"
    return df.to_markdown(index=False) if len(df) else "(0 rows)"
