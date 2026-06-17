"""A small RAG (Retrieval-Augmented Generation) engine backed by ChromaDB.

RAG in one sentence: instead of hoping the model memorized your facts, you *retrieve* the
most relevant passages from your own documents at query time and hand them to the model as
context. That keeps answers grounded in *your* knowledge (here: the company data dictionary
and house engineering principles) and lets you update knowledge by editing files, not
retraining.

    documents ──chunk──▶ text chunks ──┐
                                        ├─▶ Chroma collection (embeds + stores + indexes)
    query ──────────────────────────────┘            │
                                                       ▼
                          Chroma ANN search ──▶ top-k chunks (with source + score)

Why ChromaDB?
-------------
Earlier this engine stored vectors in a JSON file and did cosine search by hand. That's
great for showing the mechanics, but a real system wants a purpose-built **vector
database**. Chroma gives us, for free:
  * persistent on-disk storage (a real DB under workspace/chroma),
  * an approximate-nearest-neighbour (HNSW) index that scales past brute-force search,
  * embeddings handled by an attached embedding function (we use OpenAI's),
  * metadata storage + filtering alongside each vector.

The retrieval *interface* the agents see (`search(query) -> top-k chunks`) is unchanged —
only the storage/search engine underneath got more capable. Swapping in pgvector/Pinecone
later would be the same shape again.
"""

from __future__ import annotations

import os
import re

import chromadb
from chromadb.utils import embedding_functions

from . import config

_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 150

# Cache the client and collection (built lazily so importing this module never needs a key).
# IMPORTANT: reuse a SINGLE PersistentClient per process. Opening several clients on the
# same on-disk path causes SQLite contention and intermittent failures.
_client_obj = None
_collection = None


def _embedding_function():
    """Chroma calls this to turn text into vectors — both when indexing and querying."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=config.EMBED_MODEL,
    )


def _client() -> "chromadb.api.ClientAPI":
    global _client_obj
    if _client_obj is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client_obj = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client_obj


def _get_collection():
    """Return the live collection (with the OpenAI embedding function attached)."""
    global _collection
    if _collection is None:
        _collection = _client().get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=_embedding_function(),
            # cosine distance suits normalized text embeddings.
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _chunk_text(text: str) -> list[str]:
    """Split a document into overlapping chunks on paragraph boundaries.

    Overlap keeps ideas that straddle a boundary retrievable. We break on blank lines
    first (markdown paragraphs/sections), then pack paragraphs up to the size budget.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= _CHUNK_CHARS:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-_CHUNK_OVERLAP:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
    if current:
        chunks.append(current)
    return chunks


def build_index() -> int:
    """(Re)build the Chroma collection from every .md file in the knowledge dir.

    Chroma embeds each chunk via the attached embedding function as it's added. Returns the
    number of chunks indexed. This is what `data-agent ingest` calls.
    """
    docs = sorted(config.KNOWLEDGE_DIR.glob("*.md"))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {config.KNOWLEDGE_DIR}")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for doc in docs:
        for i, chunk in enumerate(_chunk_text(doc.read_text(encoding="utf-8"))):
            ids.append(f"{doc.name}::{i}")
            documents.append(chunk)
            metadatas.append({"source": doc.name, "chunk": i})

    # Fresh rebuild: drop any existing collection so re-ingesting is idempotent.
    client = _client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass  # collection didn't exist yet
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    global _collection
    _collection = None  # invalidate cache so the next search reopens the fresh collection
    return len(documents)


def index_exists() -> bool:
    """True if a non-empty knowledge collection exists. Avoids constructing the embedding
    function (so it works without an API key, e.g. for `data-agent info`)."""
    if not config.CHROMA_DIR.exists():
        return False
    try:
        client = _client()
        names = [c.name for c in client.list_collections()]
        if config.CHROMA_COLLECTION not in names:
            return False
        return client.get_collection(config.CHROMA_COLLECTION).count() > 0
    except Exception:
        return False


def search(query: str, k: int = 4) -> list[dict]:
    """Return the top-k knowledge chunks most relevant to `query`.

    Each result is {source, text, score}, where score is cosine similarity in [0, 1]
    (Chroma returns distance; we report 1 - distance). Empty list if nothing is indexed.
    """
    if not index_exists():
        return []

    res = _get_collection().query(query_texts=[query], n_results=k)
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    return [
        {"source": m.get("source", "?"), "text": d, "score": 1.0 - float(dist)}
        for d, m, dist in zip(documents, metadatas, distances)
    ]
