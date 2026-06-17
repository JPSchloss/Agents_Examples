"""An input guardrail.

A guardrail is a fast, cheap check that runs *alongside* the main agent and can trip a
"tripwire" to stop the run before expensive work happens. Here we use the classic pattern:
a tiny classifier agent decides whether the request is even in scope for a data assistant.

This protects against two things at once:
  * wasted tokens / latency on clearly off-topic requests, and
  * prompt-injection-ish attempts to repurpose the expensive, tool-wielding main agent.

Guardrails run on the *first* agent in the chain. If the tripwire fires, the SDK raises
`InputGuardrailTripwireTriggered`, which we catch in app.py to show a friendly message.
"""

from __future__ import annotations

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    input_guardrail,
)

from . import config
from .schemas import RelevanceCheck

# A minimal, single-purpose agent. Cheap model, structured output, no tools.
_relevance_agent = Agent(
    name="Relevance check",
    instructions=(
        "You decide whether a user message is in scope for a DATA assistant whose job is "
        "to ingest, profile, clean, and pipeline data, build dashboards, and give "
        "engineering guidance on those topics. Greetings, follow-ups, and clarifying "
        "questions about ongoing data work ARE on-topic. Requests to write unrelated "
        "code, general chit-chat, or anything off-topic are NOT."
    ),
    model=config.GUARDRAIL_MODEL,
    output_type=RelevanceCheck,
)


@input_guardrail
async def relevance_guardrail(
    ctx: RunContextWrapper, agent: Agent, user_input
) -> GuardrailFunctionOutput:
    """Trip the wire when the incoming request is off-topic."""
    result = await Runner.run(_relevance_agent, user_input, context=ctx.context)
    check: RelevanceCheck = result.final_output

    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_on_topic,
    )
