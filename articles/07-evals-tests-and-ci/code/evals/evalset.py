"""Evaluation cases + harness for the agent team.

Unlike the unit tests (deterministic, no API), evals exercise the *whole agent* against a
real model and assert on behavior that prompts/models can regress:

  * routing — did triage hand off to the right specialist?
  * tool use — did the agent call the tools it should (e.g. retrieve before advising)?
  * grounding — does the answer reflect our knowledge base?
  * guardrails — is an off-topic request refused?

Each case declares expectations; `run_case` runs the turn and checks them. This is the
"machine-readable verdict" layer of an Agent Development Lifecycle: observability surfaces a
failure, you capture it here as a case, and CI keeps it from coming back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import InputGuardrailTripwireTriggered, Runner

from data_agent.context import PipelineContext
from data_agent.team import build_team


@dataclass
class EvalCase:
    name: str
    prompt: str
    expect_agent: str | None = None          # name of the agent that should produce the answer
    expect_tools: list[str] = field(default_factory=list)  # tools that must be called
    expect_substrings: list[str] = field(default_factory=list)  # case-insensitive, in output
    expect_guardrail: bool = False           # the input guardrail should trip


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    detail: str


CASES: list[EvalCase] = [
    EvalCase(
        name="routes_data_questions_to_engineer",
        prompt="What data do I have to work with?",
        expect_agent="Data Engineer",
        expect_tools=["list_datasets"],
    ),
    EvalCase(
        name="grounds_advice_in_knowledge_base",
        prompt="Per our standards, how should I handle negative-quantity rows?",
        expect_agent="Advisor",
        expect_tools=["search_knowledge"],
        expect_substrings=["return"],  # KB rule: negative qty = returns, excluded from revenue
    ),
    EvalCase(
        name="refuses_off_topic_requests",
        prompt="Write me a long poem about the ocean.",
        expect_guardrail=True,
    ),
]


async def run_case(case: EvalCase) -> EvalResult:
    """Run one case end-to-end and check its expectations."""
    triage = build_team(mcp_servers=[], with_knowledge=True)
    try:
        result = await Runner.run(
            triage, case.prompt, context=PipelineContext(), max_turns=12
        )
    except InputGuardrailTripwireTriggered:
        ok = case.expect_guardrail
        return EvalResult(case, ok, "guardrail tripped" if ok else "guardrail tripped UNEXPECTEDLY")

    if case.expect_guardrail:
        return EvalResult(case, False, "expected guardrail to trip, but the run completed")

    tools = [
        getattr(i.raw_item, "name", None)
        for i in result.new_items
        if i.type == "tool_call_item"
    ]
    tools = [t for t in tools if t]
    agent = result.last_agent.name
    output = str(result.final_output).lower()

    problems: list[str] = []
    if case.expect_agent and agent != case.expect_agent:
        problems.append(f"agent={agent!r} expected {case.expect_agent!r}")
    for tool in case.expect_tools:
        if tool not in tools:
            problems.append(f"missing tool {tool!r} (called: {tools})")
    for sub in case.expect_substrings:
        if sub.lower() not in output:
            problems.append(f"output missing {sub!r}")

    detail = "; ".join(problems) or f"ok (agent={agent}, tools={tools})"
    return EvalResult(case, not problems, detail)
