"""Observability: structured logging + lifecycle hooks.

`print()` is fine for a demo; a system you operate needs a record you can grep, ship, and
correlate. Two pieces here:

  * `configure_logging()` — one logger ("data_agent") with a console handler (quiet by
    default) and an optional file handler. We log in logfmt-style `key=value` pairs, which
    are human-readable AND machine-greppable.
  * `AgentLogHooks` — a `RunHooks` implementation. The SDK calls these callbacks as a run
    progresses (agent start/end, every tool start/end, every handoff), so we get a
    step-by-step structured trace of *what the agent actually did*, correlated by a per-turn
    id. This is the single most useful thing for debugging agentic behavior offline.

For richer/visual tracing, the SDK also emits spans to the OpenAI Traces dashboard by
default, and you can forward them to your own stack with a custom `TracingProcessor`
(OpenTelemetry, Langfuse, MLflow). See the article for that sketch; hooks + logs cover the
day-to-day.
"""

from __future__ import annotations

import json
import logging
import uuid

from agents import RunHooks

logger = logging.getLogger("data_agent")


def configure_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Configure the 'data_agent' logger. Quiet (WARNING) on the console unless --verbose;
    full INFO detail goes to --log-file when given."""
    logger.setLevel(logging.DEBUG)  # handlers decide what is emitted
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def new_turn_id() -> str:
    """A short id to correlate all log lines belonging to one user turn."""
    return uuid.uuid4().hex[:8]


def kv(**fields) -> str:
    """Render fields as logfmt `key=value` (quoting values that contain spaces)."""
    parts = []
    for key, value in fields.items():
        text = str(value)
        parts.append(f"{key}={json.dumps(text)}" if " " in text else f"{key}={text}")
    return " ".join(parts)


class AgentLogHooks(RunHooks):
    """Logs each lifecycle event of a run as a structured line, tagged with the turn id.

    Passed to `Runner.run(..., hooks=AgentLogHooks(turn_id))`.
    """

    def __init__(self, turn_id: str) -> None:
        self.turn = turn_id

    async def on_agent_start(self, context, agent) -> None:
        logger.info("agent.start " + kv(turn=self.turn, agent=agent.name))

    async def on_agent_end(self, context, agent, output) -> None:
        logger.info("agent.end " + kv(turn=self.turn, agent=agent.name))

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        logger.info("handoff " + kv(turn=self.turn, from_agent=from_agent.name, to=to_agent.name))

    async def on_tool_start(self, context, agent, tool) -> None:
        logger.info("tool.start " + kv(turn=self.turn, agent=agent.name, tool=tool.name))

    async def on_tool_end(self, context, agent, tool, result) -> None:
        logger.info(
            "tool.end " + kv(turn=self.turn, tool=tool.name, result_chars=len(str(result)))
        )
