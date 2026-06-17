"""Build (or rebuild) the RAG knowledge index.

Run it with:
    python -m data_agent.ingest

It chunks and embeds every .md file in `knowledge/` into a persistent ChromaDB collection.
Re-run it whenever you edit the knowledge docs. (In Article 4 this folds into the main CLI
as `data-agent ingest`; for now it's a tiny standalone entry point.)
"""

from __future__ import annotations

import os
import sys

from . import config, rag


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)
    config.ensure_dirs()
    print("Embedding knowledge base into Chroma…")
    n = rag.build_index()
    print(
        f"✓ Indexed {n} chunks from {config.KNOWLEDGE_DIR.name}/ → "
        f"Chroma collection '{config.CHROMA_COLLECTION}' (workspace/{config.CHROMA_DIR.name})"
    )


if __name__ == "__main__":
    main()
