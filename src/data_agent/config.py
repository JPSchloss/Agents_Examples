"""Central configuration: model names and the folders the agents are allowed to touch.

Everything is read from environment variables (loaded from a local `.env` file) so the
same code runs in development and production without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present. In production you'd inject these
# through your orchestrator / secrets manager instead, and load_dotenv() simply no-ops.
load_dotenv()

# --- Models -----------------------------------------------------------------------
# The main reasoning model used by every "thinking" agent. Override with OPENAI_MODEL.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# A small, fast model used only by the input guardrail (a cheap relevance classifier).
GUARDRAIL_MODEL = os.getenv("OPENAI_GUARDRAIL_MODEL", "gpt-5-mini")


# --- Filesystem boundaries --------------------------------------------------------
# The project root is two levels up from this file: src/data_agent/config.py -> <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source data the agent may read but should treat as read-only.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# The single sandbox the agent may write to: cleaned data, generated scripts,
# SQLite databases, and the Streamlit dashboard all land here. Keeping every write
# inside one folder is what makes the file tools safe to expose to a model.
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# Where conversation memory (the SQLiteSession) is stored between runs.
MEMORY_DB = WORKSPACE_DIR / ".memory.db"


def ensure_dirs() -> None:
    """Create the folders the app depends on. Safe to call repeatedly."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
