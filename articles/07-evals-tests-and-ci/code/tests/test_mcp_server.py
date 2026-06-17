"""Unit tests for the MCP data-standards server tools (pure functions, no transport)."""

import pytest

from mcp_servers.reference_server import (
    canonical_column_name,
    naming_conventions,
    standard_dtype,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Order Date", "order_date"),
        ("UnitPrice", "unit_price"),
        ("  customer  ", "customer"),
        ("total$amount", "totalamount"),
        ("Region-Code", "region_code"),
    ],
)
def test_canonical_column_name(raw, expected):
    assert canonical_column_name(raw) == expected


def test_standard_dtype_known_role():
    assert "float64" in standard_dtype("currency")


def test_standard_dtype_unknown_role_lists_valid_roles():
    out = standard_dtype("nonsense")
    assert "Unknown role" in out and "currency" in out


def test_naming_conventions_mentions_snake_case():
    assert "snake_case" in naming_conventions()
