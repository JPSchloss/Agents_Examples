"""The RAG retrieval tool.

This is the bridge between the agents and the knowledge base: the model calls
`search_knowledge(...)` whenever it needs grounded facts (cleaning rules, business
definitions, house principles), and we inject the retrieved passages back into its context.

Giving the model a *retrieval tool* (rather than stuffing all docs into the system prompt)
means it pulls only what's relevant, on demand — cheaper, and it scales to far more
knowledge than fits in a prompt.
"""

from __future__ import annotations

from agents import function_tool

from .. import rag


@function_tool
def search_knowledge(query: str) -> str:
    """Search the company knowledge base (data dictionary + data-engineering principles)
    for grounded facts and rules. Use this before making assumptions about business rules,
    canonical values, or how a column should be cleaned — the answers here are authoritative.

    Args:
        query: A natural-language question, e.g. "canonical region values" or
            "how should negative quantity be handled".
    """
    if not rag.index_exists():
        return (
            "Knowledge base is not indexed yet. Ask the user to run `data-agent ingest` "
            "(or `python -m data_agent.app ingest`) to build it."
        )

    hits = rag.search(query, k=4)
    if not hits:
        return "No relevant passages found in the knowledge base."

    blocks = []
    for h in hits:
        blocks.append(f"[source: {h['source']} · relevance {h['score']:.2f}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)
