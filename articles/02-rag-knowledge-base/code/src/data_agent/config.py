"""Central configuration: model names and the folders the agents are allowed to touch.

Everything is read from environment variables (loaded from a local `.env` file) so the
same code runs in development and production without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Models -----------------------------------------------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
GUARDRAIL_MODEL = os.getenv("OPENAI_GUARDRAIL_MODEL", "gpt-5-mini")

# Embedding model used to build the RAG knowledge index.
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# --- Filesystem boundaries --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
MEMORY_DB = WORKSPACE_DIR / ".memory.db"

# RAG: curated reference documents (read-only) and the Chroma vector DB built from them.
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
CHROMA_DIR = WORKSPACE_DIR / "chroma"        # persistent on-disk Chroma database
CHROMA_COLLECTION = "knowledge"


def ensure_dirs() -> None:
    """Create the folders the app depends on. Safe to call repeatedly."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
