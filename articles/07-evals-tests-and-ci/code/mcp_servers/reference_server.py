"""A local MCP (Model Context Protocol) server: "Acme Data Standards".

What is MCP?
------------
MCP is an open protocol that lets an AI app connect to external **tool/data servers** over
a standard interface, instead of hard-coding every integration. Think of it as "USB-C for
tools": write a capability once as an MCP server and ANY MCP-aware client (this app, Claude
Desktop, IDEs, ...) can use it. The Agents SDK is an MCP client — point it at a server and
that server's tools appear to the model alongside its native `@function_tool`s.

This server is intentionally small and runs over **stdio** (the client launches it as a
subprocess and talks over stdin/stdout). It exposes *deterministic reference helpers* about
our data standards — distinct from the RAG knowledge base (which returns prose). Here the
model gets exact, machine-checked answers: canonical column names, the standard type map,
and naming conventions.

Run standalone for a smoke test:  python mcp_servers/reference_server.py
(It will wait on stdio; Ctrl-C to exit. Normally the app launches it for you.)
"""

from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("acme-data-standards")

# The org's canonical type mapping by column "role".
_STANDARD_DTYPES = {
    "id": "int64",
    "date": "datetime64[ns] (serialize as ISO YYYY-MM-DD)",
    "category": "string / pandas category",
    "text": "string",
    "currency": "float64 (USD, no symbols)",
    "count": "int64 (non-negative)",
}


@mcp.tool()
def canonical_column_name(name: str) -> str:
    """Convert a column name to the org's canonical snake_case form.

    Example: 'Order Date' -> 'order_date', 'UnitPrice' -> 'unit_price'.
    """
    # Split camelCase, then normalize separators to underscores, then lowercase.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]", "", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


@mcp.tool()
def standard_dtype(column_role: str) -> str:
    """Return the org's standard data type for a column role.

    Valid roles: id, date, category, text, currency, count.
    """
    role = column_role.strip().lower()
    if role in _STANDARD_DTYPES:
        return f"{role} -> {_STANDARD_DTYPES[role]}"
    return (
        f"Unknown role '{column_role}'. Valid roles: {', '.join(_STANDARD_DTYPES)}."
    )


@mcp.tool()
def naming_conventions() -> str:
    """Return the org's table and column naming conventions."""
    return (
        "Tables: snake_case, singular-domain plural-grain (e.g. 'sales', 'customers').\n"
        "Columns: snake_case; ids end in '_id'; dates end in '_date'; monetary columns are "
        "plain numbers in USD named after the measure (e.g. 'unit_price', 'total').\n"
        "Booleans start with 'is_' or 'has_'. No spaces, no camelCase, no reserved words."
    )


if __name__ == "__main__":
    # Default transport is stdio, which is exactly what MCPServerStdio expects.
    mcp.run()
