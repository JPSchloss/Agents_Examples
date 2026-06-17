"""Path-safety helpers shared by the file tools.

A model that can write files and run code is powerful and dangerous. The single most
important guardrail for that power is: *never* let a tool resolve a path outside the
directories we explicitly allow. `safe_resolve` enforces that — it rejects absolute
paths and any `..` traversal that would escape the sandbox.
"""

from __future__ import annotations

from pathlib import Path


class PathNotAllowed(Exception):
    """Raised when a requested path escapes the allowed roots."""


def safe_resolve(relative_path: str, *allowed_roots: Path) -> Path:
    """Resolve `relative_path` against the first allowed root and verify the result
    stays inside one of the allowed roots.

    Returns the absolute, resolved path. Raises PathNotAllowed otherwise.
    """
    if not allowed_roots:
        raise ValueError("at least one allowed root is required")

    # Interpret the (possibly user/LLM-supplied) path relative to the primary root.
    primary = allowed_roots[0]
    candidate = (primary / relative_path).resolve()

    for root in allowed_roots:
        root_resolved = root.resolve()
        if candidate == root_resolved or root_resolved in candidate.parents:
            return candidate

    allowed = ", ".join(str(r) for r in allowed_roots)
    raise PathNotAllowed(
        f"Path '{relative_path}' resolves outside the allowed directories ({allowed})."
    )
