"""Unit tests for the RAG chunker (pure text logic, no embeddings / no API)."""

from data_agent.rag import _CHUNK_CHARS, _chunk_text


def test_short_text_is_one_chunk():
    chunks = _chunk_text("A single short paragraph.")
    assert chunks == ["A single short paragraph."]


def test_blank_lines_split_paragraphs_but_pack_within_budget():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = _chunk_text(text)
    # All three short paragraphs fit in one chunk.
    assert len(chunks) == 1
    assert "Para one." in chunks[0] and "Para three." in chunks[0]


def test_long_text_splits_into_multiple_chunks():
    para = "word " * 60  # ~300 chars
    text = "\n\n".join([para] * 6)  # ~1800 chars > one chunk
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= _CHUNK_CHARS + 200 for c in chunks)  # roughly bounded (+ overlap)


def test_empty_text_yields_no_chunks():
    assert _chunk_text("   \n\n   ") == []
