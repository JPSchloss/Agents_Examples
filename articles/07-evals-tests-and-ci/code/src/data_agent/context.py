"""The run context: a typed object passed to `Runner.run(...)` and handed to every tool.

Why this exists
---------------
Tools should NOT read globals or reach into the filesystem arbitrarily. Instead the SDK
gives every tool call a `RunContextWrapper[PipelineContext]` whose `.context` is this
object. That gives us:

  * one place that defines *where* the agent may read and write,
  * a running log of artifacts the agents have produced (so the UI / advisor can refer
    to them), and
  * a clean seam for dependency injection in tests (swap the dirs for temp folders).

The context is local to a run and is never serialized into the prompt, so it's also a
safe place to keep things you do NOT want the model to see directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class PipelineContext:
    """Shared state for a single conversation / pipeline-building session."""

    raw_data_dir: Path = config.RAW_DATA_DIR
    workspace_dir: Path = config.WORKSPACE_DIR

    # Human-readable log of artifacts the agents created, e.g.
    # "wrote workspace/clean_sales.py", "loaded table 'sales' into analytics.db".
    artifacts: list[str] = field(default_factory=list)

    # --- Human-in-the-loop approval -----------------------------------------------
    # When False, dangerous tools (code execution, file writes) ask `approver` for an
    # OK before acting. `auto_approve=True` skips prompts (the --auto-approve flag).
    # `approver(action, detail) -> bool` is supplied by the CLI; None means non-interactive
    # (e.g. tests), where we default to allowing the action.
    auto_approve: bool = False
    approver: Callable[[str, str], bool] | None = None

    def record(self, note: str) -> None:
        self.artifacts.append(note)

    def approve(self, action: str, detail: str) -> bool:
        """Return True if `action` may proceed. Asks the human approver unless auto-approved
        or running non-interactively."""
        if self.auto_approve or self.approver is None:
            return True
        return self.approver(action, detail)
