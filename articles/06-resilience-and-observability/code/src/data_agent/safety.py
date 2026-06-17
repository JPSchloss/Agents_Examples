"""Safety layers: an output guardrail and a tool-input guardrail.

These complement the *input* guardrail from Article 1 and the human-in-the-loop approval
wired into the dangerous tools. Three independent checks, each catching a different failure
mode:

  * input guardrail  (guardrails.py) — is the request even in scope? (runs first)
  * tool-input guardrail (here)      — are the ARGS to a dangerous tool safe? (runs at the
                                       tool boundary, before approval/execution)
  * output guardrail (here)          — did the final answer leak a secret? (runs last)

Defense in depth: no single check is sufficient, so we layer cheap, deterministic checks at
every boundary. None of these call an LLM — they're fast regex scanners, so they add
negligible latency.
"""

from __future__ import annotations

import json
import re

from agents import (
    GuardrailFunctionOutput,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    output_guardrail,
    tool_input_guardrail,
)

# --- Output guardrail: catch leaked secrets ---------------------------------------

_SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"sk-[A-Za-z0-9_\-]{20,}", "OpenAI-style API key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "private key block"),
    (
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9/_\-]{12,}",
        "credential assignment",
    ),
]


@output_guardrail
async def secret_leakage_guardrail(
    ctx: RunContextWrapper, agent, output
) -> GuardrailFunctionOutput:
    """Trip if the final answer appears to contain a secret (key, token, private key)."""
    text = output if isinstance(output, str) else str(output)
    hits = [label for pattern, label in _SECRET_PATTERNS if re.search(pattern, text)]
    return GuardrailFunctionOutput(
        output_info={"matched": hits},
        tripwire_triggered=bool(hits),
    )


# --- Tool-input guardrail: refuse dangerous generated code ------------------------

# Patterns that have no business in a data-cleaning / dashboard script and usually signal
# sandbox escape, destructive ops, or exfiltration. We refuse the *content* and let the
# model revise, rather than hard-crashing the run.
_DANGEROUS_CODE: list[tuple[str, str]] = [
    (r"\bos\.system\s*\(", "os.system() shell call"),
    (r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True", "subprocess with shell=True"),
    (r"\bshutil\.rmtree\s*\(", "recursive delete (shutil.rmtree)"),
    (r"\bos\.remove\s*\(|\bos\.unlink\s*\(", "file deletion"),
    (r"\b(?:requests|urllib|httpx|socket)\b", "network access"),
    (r"\beval\s*\(|\bexec\s*\(", "eval/exec"),
    (r"\bopen\s*\(\s*['\"]/(?!tmp)", "absolute-path file write outside the sandbox"),
]


@tool_input_guardrail
def dangerous_code_guardrail(data) -> ToolGuardrailFunctionOutput:
    """For write_file: scan the file CONTENT for dangerous operations and refuse if found.

    Pairs with `safe_resolve` (which constrains *where* files go) by constraining *what* the
    generated code may do. Returns rejected content (a message the model can act on) rather
    than raising, so the agent can rewrite a safe version.
    """
    try:
        args = json.loads(data.context.tool_arguments or "{}")
    except (TypeError, ValueError):
        return ToolGuardrailFunctionOutput.allow()

    content = args.get("content", "") or ""
    for pattern, label in _DANGEROUS_CODE:
        if re.search(pattern, content):
            return ToolGuardrailFunctionOutput.reject_content(
                message=(
                    f"Refused: the file contains a disallowed operation ({label}). "
                    "Rewrite it without shell, network, deletion, eval/exec, or writes "
                    "outside the workspace — use pandas/sqlite within the workspace only."
                ),
                output_info={"matched": label},
            )
    return ToolGuardrailFunctionOutput.allow()
