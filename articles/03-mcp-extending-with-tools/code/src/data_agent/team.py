"""Define the agent team and wire up handoffs, RAG, and MCP.

The shape of the system
-----------------------
                         ┌──────────────────────┐
        user  ───────────▶  Triage / Orchestrator │  (has the relevance guardrail)
                         └─────────┬────────────┘
              handoff      ┌───────┼───────────┐
                           ▼       ▼           ▼
                     Data Engineer  Frontend    Advisor
                     · native tools Builder    · search_knowledge (RAG)
                     · search_know. (writes a  · data-standards (MCP)
                     · data-stds MCP Streamlit
                                    app)
                           └───────┴───────────┘
                              each can hand BACK to Triage

Why a factory function?
-----------------------
Earlier this module built agents at import time. But MCP servers are live connections that
must be opened (async) before a run and closed after — they can't exist at import time. So
we expose `build_team(...)`, which the app calls once it has connected its MCP servers and
decided whether RAG is enabled. This is also cleaner for testing (build a team with no MCP
and no RAG in one line).

Two capability sources beyond the native `@function_tool`s:
  * RAG  — the `search_knowledge` tool retrieves grounded passages from `knowledge/`.
  * MCP  — `mcp_servers=[...]` attaches external tool servers; their tools appear to the
           model automatically alongside the native ones.
"""

from __future__ import annotations

from agents import Agent, ModelSettings, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from . import config
from .guardrails import relevance_guardrail
from .tools import (
    list_datasets,
    load_csv_to_sqlite,
    profile_dataset,
    query_sqlite,
    read_file_preview,
    run_python_file,
    search_knowledge,
    write_file,
)


def build_team(mcp_servers=None, with_knowledge: bool = True) -> Agent:
    """Construct the full agent team and return the triage (entry-point) agent.

    Args:
        mcp_servers: already-connected MCP server objects to expose to the specialists.
        with_knowledge: whether to give agents the RAG `search_knowledge` tool.
    """
    mcp_servers = list(mcp_servers or [])
    kb = [search_knowledge] if with_knowledge else []

    # --- Data Engineer ------------------------------------------------------------
    data_engineer = Agent(
        name="Data Engineer",
        handoff_description="Profiles, cleans, transforms, and loads data into a database.",
        instructions=prompt_with_handoff_instructions(
            """You are a senior data engineer. You turn messy raw data into clean,
queryable tables by WRITING AND RUNNING real Python, guided by the company's standards.

Ground yourself first:
  * Call search_knowledge for the authoritative business rules and canonical values before
    you assume anything (e.g. valid region/category values, how 'total' is defined, how to
    handle returns/negative quantity). These rules OVERRIDE your defaults.
  * Use the data-standards tools (canonical_column_name, standard_dtype, naming_conventions
    — provided via MCP) to name and type columns correctly.

Then your standard workflow:
  1. list_datasets, then profile_dataset on the raw CSV. Summarize what you found.
  2. Write a documented cleaning script with write_file that reads the raw CSV by its
     ABSOLUTE path, applies the knowledge-base rules (strip currency symbols, parse all
     date formats to ISO, normalize region/category to their canonical sets, drop duplicate
     order_ids, exclude returns, recompute 'total' = quantity*unit_price, coerce types),
     and writes a cleaned CSV into the workspace. Make it print a before/after summary.
  3. Run it with run_python_file; if it fails, read stderr, fix, re-run.
  4. load_csv_to_sqlite, then validate with query_sqlite (row counts, value-domain checks,
     a revenue GROUP BY) following the data-quality checks in the knowledge base.
  5. Report what you did, the table name, and the validation results.

Prefer small, verifiable steps. When the data is clean and loaded — or the user now wants a
UI or general advice — hand back to the Triage agent."""
        ),
        model=config.MODEL,
        tools=[
            list_datasets,
            read_file_preview,
            profile_dataset,
            write_file,
            run_python_file,
            load_csv_to_sqlite,
            query_sqlite,
            *kb,
        ],
        mcp_servers=mcp_servers,
    )

    # --- Frontend Builder ---------------------------------------------------------
    frontend_builder = Agent(
        name="Frontend Builder",
        handoff_description="Builds a Streamlit dashboard on top of the prepared data.",
        instructions=prompt_with_handoff_instructions(
            """You build clean, useful Streamlit dashboards on top of data that has already
been cleaned and loaded into a workspace SQLite database (default 'analytics.db').

Your workflow:
  1. If unsure what's available, use list_datasets and query_sqlite to inspect the tables
     and columns. Never invent column names — verify them.
  2. Write a single self-contained app to 'dashboard.py' with write_file. It should:
       - connect to the workspace SQLite db with sqlite3,
       - show KPIs (st.metric) and at least two charts (st.bar_chart / st.line_chart),
       - include a sidebar filter (e.g. by region or category),
       - be defensive: handle empty results without crashing.
  3. Do NOT run the dashboard yourself (Streamlit is a long-lived server). Tell the user:
         streamlit run workspace/dashboard.py
  4. Summarize what the dashboard shows.

If the data isn't ready, hand back to Triage so the Data Engineer can prepare it."""
        ),
        model=config.MODEL,
        tools=[list_datasets, read_file_preview, query_sqlite, write_file],
    )

    # --- Advisor ------------------------------------------------------------------
    advisor = Agent(
        name="Advisor",
        handoff_description="Explains concepts and recommends approaches; teaches, not builds.",
        instructions=prompt_with_handoff_instructions(
            """You are a pragmatic data-engineering mentor. The user is building a pipeline
and wants guidance, not just execution.

CRITICAL: For ANY question about our data, business rules, standards, or how to build,
you MUST call search_knowledge FIRST — before writing a single sentence of advice — and
base your answer on what it returns, citing the source file (e.g. "per
company_data_dictionary.md"). Do NOT answer such questions from generic knowledge; our
house rules often differ from common practice and the house rules win. Use the
data-standards MCP tools for exact naming/typing. If retrieval returns nothing relevant,
say so explicitly rather than guessing.

Keep answers focused and practical with short examples. When the user is ready to actually
build or run something, hand back to Triage so the right specialist can act."""
        ),
        model=config.MODEL,
        tools=[*kb],
        mcp_servers=mcp_servers,
        # Force a tool call on the Advisor's first step so it actually RETRIEVES instead of
        # answering from (and hallucinating) memory. The SDK resets tool_choice to "auto"
        # after a tool runs (reset_tool_choice defaults True), so the follow-up turn can
        # produce the final grounded answer without looping. Only meaningful when it has
        # tools — i.e. when RAG is enabled.
        model_settings=ModelSettings(tool_choice="required") if kb else ModelSettings(),
    )

    # --- Orchestrator -------------------------------------------------------------
    triage_agent = Agent(
        name="Triage",
        instructions=prompt_with_handoff_instructions(
            """You are the ROUTER of a data-pipeline assistant team. Your only job is to
hand off the conversation to exactly one specialist. You do NOT answer data questions
yourself and you do NOT give advice yourself — you route.

Route to:
  * Data Engineer — ANY request that touches the actual data or files: "what data/files
    do I have", "what's available", listing or inspecting datasets, profiling, cleaning,
    transforming, or loading data. (Only this agent can see the real files, so questions
    about what data exists ALWAYS go here.)
  * Frontend Builder — building or changing a Streamlit dashboard / front end.
  * Advisor — purely conceptual guidance: "how should I...", explanations, trade-offs,
    best practices, where NO action on the user's actual files is needed.

Rules:
  - Always perform a handoff. Do not produce a substantive answer of your own.
  - Do not ask for clarification unless the request is genuinely impossible to route.
  - For multi-step requests, hand off to the FIRST relevant specialist; specialists hand
    back to you to route the next step.
  - Keep any message of your own to one short sentence."""
        ),
        model=config.MODEL,
        # handoff() with explicit tool names keeps display names human-readable
        # ("Data Engineer") without emitting invalid-function-name warnings.
        handoffs=[
            handoff(data_engineer, tool_name_override="transfer_to_data_engineer"),
            handoff(frontend_builder, tool_name_override="transfer_to_frontend_builder"),
            handoff(advisor, tool_name_override="transfer_to_advisor"),
        ],
        input_guardrails=[relevance_guardrail],
    )

    # Let each specialist hand control back to Triage ("Triage" has no space → no override).
    for specialist in (data_engineer, frontend_builder, advisor):
        specialist.handoffs.append(triage_agent)

    return triage_agent
