"""Define the agent team and wire up handoffs.

The shape of the system
-----------------------
                         ┌──────────────────────┐
        user  ───────────▶  Triage / Orchestrator │  (has the relevance guardrail)
                         └─────────┬────────────┘
              handoff      ┌───────┼───────────┐
                           ▼       ▼           ▼
                     Data Engineer  Frontend    Advisor
                     (tools: profile,Builder    (knowledge,
                      write/run code,(writes a   no tools)
                      sqlite)        Streamlit
                                     app)
                           └───────┴───────────┘
                              each can hand BACK to Triage

Why multi-agent instead of one big agent?
  * Smaller, focused instruction sets are more reliable than one mega-prompt.
  * Each specialist only sees the tools it needs (least privilege for the model, too).
  * Handoffs make the control flow legible in traces.

`handoffs` is just a list of agents a given agent is allowed to transfer the conversation
to. We build the specialists first, then the triage agent that points at them, then let
each specialist point back at triage — a small two-step because the references are cyclic.
"""

from __future__ import annotations

from agents import Agent, handoff
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
    write_file,
)

# --- Specialists ------------------------------------------------------------------

data_engineer = Agent(
    name="Data Engineer",
    handoff_description="Profiles, cleans, transforms, and loads data into a database.",
    instructions=prompt_with_handoff_instructions(
        """You are a senior data engineer. You turn messy raw data into clean, queryable
tables by WRITING AND RUNNING real Python.

Your standard workflow:
  1. Call list_datasets to see inputs, then profile_dataset on the raw CSV to understand
     its shape and data-quality issues. Briefly summarize what you found.
  2. Write a documented cleaning script with write_file (e.g. 'clean_sales.py') that reads
     the raw CSV by its ABSOLUTE path (shown by list_datasets), fixes the issues you found
     — strip currency symbols, parse all date formats to ISO, normalize text casing, drop
     duplicates and invalid rows, coerce numeric types, handle missing values sensibly —
     and writes a cleaned CSV into the workspace (e.g. 'sales_clean.csv'). Make the script
     print a short before/after summary.
  3. Run it with run_python_file. If it fails, read the stderr, fix the script, re-run.
  4. Load the cleaned CSV into SQLite with load_csv_to_sqlite, then validate with one or
     two query_sqlite SELECTs (row counts, a GROUP BY sanity check).
  5. Report what you did and the resulting table name and columns.

Prefer small, verifiable steps over one giant script. When the data is clean and loaded,
or if the user now wants a UI or general advice, hand back to the Triage agent."""
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
    ],
)

frontend_builder = Agent(
    name="Frontend Builder",
    handoff_description="Builds a Streamlit dashboard on top of the prepared data.",
    instructions=prompt_with_handoff_instructions(
        """You build clean, useful Streamlit dashboards on top of data that has already
been cleaned and loaded into a workspace SQLite database (default 'analytics.db').

Your workflow:
  1. If you're unsure what's available, use list_datasets and query_sqlite to inspect the
     tables and columns first. Never invent column names — verify them.
  2. Write a single self-contained app to 'dashboard.py' with write_file. The app should:
       - connect to the workspace SQLite db with sqlite3,
       - show KPIs (st.metric) and at least two charts (st.bar_chart / st.line_chart),
       - include a sidebar filter (e.g. by region or category),
       - be defensive: handle empty results without crashing.
  3. Do NOT try to run the dashboard yourself (Streamlit is a long-lived server). Instead
     tell the user the exact command to launch it:
         streamlit run workspace/dashboard.py
  4. Summarize what the dashboard shows.

If the data isn't ready yet, hand back to Triage so the Data Engineer can prepare it."""
    ),
    model=config.MODEL,
    tools=[list_datasets, read_file_preview, query_sqlite, write_file],
)

advisor = Agent(
    name="Advisor",
    handoff_description="Explains concepts and recommends approaches; teaches, not builds.",
    instructions=prompt_with_handoff_instructions(
        """You are a pragmatic data-engineering mentor. The user is building a pipeline and
wants guidance, not just execution. Explain concepts clearly, recommend concrete
approaches with trade-offs, and suggest the next step.

Keep answers focused and practical. Use short examples. When the user is ready to actually
build or run something, hand back to Triage so the right specialist can act. You have no
tools — you teach and advise."""
    ),
    model=config.MODEL,
)

# --- Orchestrator -----------------------------------------------------------------

triage_agent = Agent(
    name="Triage",
    instructions=prompt_with_handoff_instructions(
        """You are the ROUTER of a data-pipeline assistant team. Your only job is to hand
off the conversation to exactly one specialist. You do NOT answer data questions yourself
and you do NOT give advice yourself — you route.

Route to:
  * Data Engineer — ANY request that touches the actual data or files: "what data/files
    do I have", "what's available", listing or inspecting datasets, profiling, cleaning,
    transforming, or loading data. (Only this agent has the tools to see the real files,
    so questions about what data exists ALWAYS go here.)
  * Frontend Builder — building or changing a Streamlit dashboard / front end.
  * Advisor — purely conceptual guidance: "how should I...", explanations, trade-offs,
    best practices, where NO action on the user's actual files is needed.

Rules:
  - Always perform a handoff. Do not produce a substantive answer of your own.
  - Do not ask for clarification unless the request is genuinely impossible to route.
  - For multi-step requests (e.g. "clean the data and build a dashboard"), hand off to the
    FIRST relevant specialist; specialists hand back to you to route the next step.
  - Keep any message of your own to one short sentence."""
    ),
    model=config.MODEL,
    # handoff() with explicit tool names keeps display names human-readable
    # ("Data Engineer") without emitting invalid-tool-name warnings.
    handoffs=[
        handoff(data_engineer, tool_name_override="transfer_to_data_engineer"),
        handoff(frontend_builder, tool_name_override="transfer_to_frontend_builder"),
        handoff(advisor, tool_name_override="transfer_to_advisor"),
    ],
    input_guardrails=[relevance_guardrail],
)

# Let each specialist hand control back to Triage ("Triage" has no space → no override).
for _specialist in (data_engineer, frontend_builder, advisor):
    _specialist.handoffs.append(triage_agent)
