# Building a Robust Agentic AI System with the OpenAI Agents SDK

A complete, runnable teaching example: a **multi-agent data-pipeline assistant**. You talk
to it in plain English and it can **ingest, profile, clean, and pipeline data**, **load it
into a database**, **build a Streamlit dashboard**, and **advise you** on how to build well
— all while you stay in the loop.

This document walks through the system from first principles. Read it top to bottom and
you'll understand not just *what* the code does, but *why* each piece exists and how the
modern agentic patterns fit together.

> **Stack:** Python 3.10+ · [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
> (`openai-agents`) · pandas · SQLite · Streamlit.

---

## Table of contents

1. [What is an "agent," really?](#1-what-is-an-agent-really)
2. [The seven building blocks](#2-the-seven-building-blocks-this-example-teaches)
3. [System architecture](#3-system-architecture)
4. [Project layout](#4-project-layout)
5. [Setup — step by step](#5-setup--step-by-step)
6. [Run it](#6-run-it)
7. [Code walkthrough (the why behind every file)](#7-code-walkthrough)
8. [How one request actually executes](#8-how-one-request-actually-executes)
9. [Token cost: what to expect and how to control it](#9-token-cost-what-to-expect-and-how-to-control-it)
10. [Extending the system](#10-extending-the-system)
11. [Hardening for production](#11-hardening-for-production)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What is an "agent," really?

A plain LLM call is a function: text in, text out. It can't *do* anything in the world.

An **agent** wraps that LLM in a loop and gives it **tools** (functions it can call). On
each turn the model can either produce a final answer *or* ask to call a tool. When it
calls a tool, your code runs the function, feeds the result back, and the model decides
what to do next. That loop — **think → act → observe → repeat** — is what makes the system
*agentic*: it can take multiple steps, react to what it observes, and pursue a goal.

```
            ┌─────────────────────────────────────────────┐
            │                  AGENT LOOP                   │
  user ───▶ │  model thinks ─▶ wants a tool? ─▶ run tool ─┐ │
            │       ▲                                     │ │
            │       └──────── feed result back ◀──────────┘ │
            │                  │ no more tools               │
            └──────────────────┼──────────────────────────--┘
                               ▼
                          final answer ───▶ user
```

The **OpenAI Agents SDK** is a small, production-minded framework that implements this loop
for you and adds the things you need to make it *robust*: handoffs between specialized
agents, guardrails, persistent memory (sessions), structured outputs, and built-in
tracing. This example uses all of them.

---

## 2. The seven building blocks this example teaches

| # | Concept | What it is | Where to see it |
|---|---------|-----------|-----------------|
| 1 | **Agent** | An LLM + instructions + a set of tools/handoffs | `team.py` |
| 2 | **Tools** | Python functions the model can call (`@function_tool`) | `tools/` |
| 3 | **Handoffs** | One agent transferring the conversation to a specialist | `team.py` |
| 4 | **Guardrails** | Fast checks that can stop a run (e.g. off-topic) | `guardrails.py` |
| 5 | **Sessions** | Persistent conversation memory across turns/restarts | `app.py` |
| 6 | **Structured output** | Forcing the model to return typed, parseable data | `schemas.py` |
| 7 | **Tracing** | Automatic, inspectable record of every step | built-in (see §6) |

The design philosophy throughout: **keep the model in charge of decisions, keep correctness
in ordinary code.** The model decides *when* to clean data or *what* SQL to run; the tools
guarantee *how* that work is actually performed.

---

## 3. System architecture

Rather than one giant "do everything" agent, we use a small **team** of focused agents
coordinated by a **triage** (orchestrator) agent. Smaller, single-purpose instruction sets
are more reliable than one mega-prompt, and each specialist only gets the tools it needs.

```
                          ┌────────────────────────┐
        user  ───────────▶│  Triage / Orchestrator  │ ◀── input guardrail
                          │  (routes the request)   │     (is this on-topic?)
                          └───────────┬─────────────┘
                  handoff   ┌─────────┼──────────────┐
                            ▼         ▼              ▼
                  ┌──────────────┐ ┌──────────┐ ┌──────────┐
                  │ Data Engineer│ │ Frontend │ │ Advisor  │
                  │              │ │ Builder  │ │          │
                  │ profile_data │ │ writes a │ │ explains │
                  │ write/run    │ │ Streamlit│ │ & guides │
                  │ code, sqlite │ │ app      │ │ (no      │
                  │              │ │          │ │  tools)  │
                  └──────┬───────┘ └────┬─────┘ └────┬─────┘
                         └──────────────┴────────────┘
                            each can hand BACK to Triage
```

- **Triage** reads your request and hands off to exactly one specialist. It carries the
  input guardrail (so the cheap relevance check happens once, at the front door).
- **Data Engineer** does the real data work: profiles the raw CSV, *writes and runs* a
  cleaning script, loads the result into SQLite, and validates with SQL.
- **Frontend Builder** writes a self-contained Streamlit dashboard on top of the database.
- **Advisor** is your knowledge source — it explains concepts and recommends approaches,
  then hands back so a specialist can act when you're ready.

Specialists **hand back to triage** when their part is done, so a request like *"clean the
data and then build a dashboard"* flows naturally across multiple agents in one
conversation.

---

## 4. Project layout

```
.
├── README.md                ← you are here
├── pyproject.toml           ← installable package (src layout) + `data-agent` script
├── requirements.txt
├── .env.example             ← copy to .env and add your API key
├── data/
│   └── raw/
│       └── sales_2024.csv   ← deliberately MESSY sample data
├── workspace/               ← the agent's sandbox (all writes land here; git-ignored)
└── src/data_agent/
    ├── config.py            ← models + the folders the agent may touch
    ├── context.py           ← typed run context passed to every tool
    ├── schemas.py           ← Pydantic models for structured output
    ├── guardrails.py        ← the input relevance guardrail
    ├── team.py              ← the agents + handoffs (the heart of the system)
    ├── app.py               ← the interactive CLI (the agent loop runner)
    └── tools/
        ├── _paths.py        ← path-safety helper (sandbox enforcement)
        ├── filesystem.py    ← list / read / write files, run a script
        └── data.py          ← profile a CSV, load to SQLite, run SQL
```

The **`data/raw/sales_2024.csv`** is intentionally broken so the cleaning step has real
work to do. It contains: five different date formats, `$` symbols inside the price column,
inconsistent region casing (`North`/`north`/`NORTH`), a fully duplicated row, blank cells,
and a nonsensical negative quantity. Profiling will surface all of these.

---

## 5. Setup — step by step

### Step 1 — Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2 — Install the project

```bash
pip install -e .
# (or: pip install -r requirements.txt)
```

This installs the OpenAI Agents SDK, pandas, Streamlit, and friends, and registers a
`data-agent` command.

### Step 3 — Add your API key

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

`.env` is git-ignored. The app reads it automatically (`config.py` calls `load_dotenv()`).

**Model choice.** The default is `gpt-5`. If your account doesn't have access yet, set
`OPENAI_MODEL=gpt-4.1` in `.env`. The cheap guardrail uses `OPENAI_GUARDRAIL_MODEL`
(`gpt-5-mini` by default; `gpt-4.1-mini` is a fine fallback).

---

## 6. Run it

```bash
python -m data_agent.app      # or just: data-agent
```

You'll get a REPL. Here's a productive first session — try these in order:

```
you ▸ What data do I have to work with?
you ▸ Profile data/raw/sales_2024.csv and tell me what's wrong with it.
you ▸ Clean it, then load it into SQLite as a table called sales.
you ▸ Build me a dashboard for the cleaned sales data.
you ▸ How should I think about making this pipeline idempotent?   (← pure advice)
```

REPL commands: `/artifacts` (list what the agents created), `/reset` (clear memory),
`/exit`.

After the dashboard step, launch the generated front end in a second terminal:

```bash
streamlit run workspace/dashboard.py
```

### Watch it think (tracing)

Every run is automatically traced. Open
**[platform.openai.com/traces](https://platform.openai.com/traces)** to see the full tree:
which agent ran, every handoff, every tool call with its inputs and outputs, and token
usage. This is your single best debugging tool for agentic systems — when behavior
surprises you, the trace shows you exactly which step went sideways. (Disable with
`OPENAI_AGENTS_DISABLE_TRACING=1` if you prefer.)

---

## 7. Code walkthrough

This section goes file by file in the order that builds understanding.

### 7.1 `config.py` — models and boundaries

Two responsibilities: pick the models (from env vars, so the same code runs anywhere) and
define **the only folders the agent may touch** — a read-only `data/raw` and a writable
`workspace/`. Confining all writes to one sandbox folder is the foundation that makes it
safe to hand a model file-writing and code-execution tools.

### 7.2 `context.py` — the run context

```python
@dataclass
class PipelineContext:
    raw_data_dir: Path = config.RAW_DATA_DIR
    workspace_dir: Path = config.WORKSPACE_DIR
    artifacts: list[str] = field(default_factory=list)
```

You pass one `PipelineContext` into `Runner.run(..., context=ctx)`. The SDK then hands every
tool a `RunContextWrapper[PipelineContext]` whose `.context` is this object. This is
**dependency injection for tools**: instead of reading globals, a tool asks the context
*where* it's allowed to read and write. It's also where we log artifacts the agents create.

Crucially, the context is **never serialized into the prompt** — it's your private,
local-only state. (Use it for things the model should act on but not necessarily *see*,
like a DB handle or the current user's id.)

### 7.3 `schemas.py` — structured output

Free-form text is hard to build on. With Pydantic models you can force the model (or a
tool) to return typed, validated data. We use it twice:

- `DataProfile` — the profiling tool returns this exact shape, so the row counts, null
  counts, and issue list are always present and parseable.
- `RelevanceCheck` — the guardrail's classifier must answer `{is_on_topic: bool, reasoning}`,
  making the tripwire decision a clean boolean instead of a regex over English.

Structured outputs are one of the biggest reliability wins in agentic systems.

### 7.4 `tools/` — giving the agents hands

A **tool** is just a Python function decorated with `@function_tool`. The SDK reads its type
hints and docstring to build the JSON schema the model sees — **so the docstring is the
tool's API doc as far as the model is concerned. Write it carefully.**

```python
@function_tool
def profile_dataset(ctx: RunContextWrapper[PipelineContext], path: str) -> DataProfile:
    """Profile a CSV ... Always profile before cleaning.
    Args:
        path: 'raw/sales_2024.csv' for a raw input, or a workspace-relative path.
    """
    ...
```

Two categories of tools, split on purpose:

- **`data.py` — deterministic tools.** `profile_dataset`, `load_csv_to_sqlite`,
  `query_sqlite`. Plain pandas/SQLite, no LLM inside. Anything that must be correct and
  repeatable is ordinary code; the model only *orchestrates* it. Note `query_sqlite`
  rejects anything that isn't a `SELECT` — least privilege, even for the model.
- **`filesystem.py` — the agent's hands.** `list_datasets`, `read_file_preview`,
  `write_file`, and `run_python_file`. The last one *executes code the model wrote*. That's
  what makes this a real builder rather than a chatbot — and it's the most dangerous
  capability, which is why every path goes through `safe_resolve`.

#### `_paths.py` — the sandbox

`safe_resolve(relative_path, *allowed_roots)` resolves a (possibly model-supplied) path and
**raises if the result escapes the allowed roots** — rejecting absolute paths and `..`
traversal. This single helper is the difference between "a tool that writes to a sandbox"
and "a tool that can overwrite `~/.ssh/authorized_keys`." `run_python_file` additionally
runs in a subprocess with a timeout and the workspace as its working directory.

> ⚠️ **Honest scope:** for a *local* teaching example this is reasonable. It is **not** a
> security boundary for untrusted input — a determined script can still reach the network or
> the rest of your machine. See [§11](#11-hardening-for-production).

### 7.5 `guardrails.py` — failing fast and safely

A **guardrail** runs alongside the agent and can trip a "tripwire" to abort the run early.
Ours is the classic relevance check: a tiny, cheap classifier agent (`OPENAI_GUARDRAIL_MODEL`)
decides whether the request is even in scope. If not, the SDK raises
`InputGuardrailTripwireTriggered` *before* the expensive, tool-wielding agent ever runs.

```python
@input_guardrail
async def relevance_guardrail(ctx, agent, user_input) -> GuardrailFunctionOutput:
    result = await Runner.run(_relevance_agent, user_input, context=ctx.context)
    check = result.final_output
    return GuardrailFunctionOutput(output_info=check,
                                   tripwire_triggered=not check.is_on_topic)
```

This buys you two things: you don't burn tokens/latency on off-topic requests, and you
reduce the blast radius of attempts to repurpose your powerful agent.

### 7.6 `team.py` — the agents and how they connect

Each agent is `Agent(name, instructions, model, tools=[...], handoffs=[...])`.

- **Instructions** are wrapped with `prompt_with_handoff_instructions(...)`, an SDK helper
  that appends the boilerplate the model needs to use handoffs correctly.
- **`handoff_description`** is a one-liner that tells *other* agents when to route here.
- **Handoffs are cyclic** (triage → specialist → triage), so we build the specialists,
  then triage pointing at them, then append triage back onto each specialist. We wrap the
  targets in `handoff(...)` with a `tool_name_override` because the SDK derives the handoff
  *tool* name from the agent name — and a human-readable name like `"Data Engineer"` (with
  a space) isn't a valid function name. The override keeps display names readable and tool
  names valid:

  ```python
  triage_agent = Agent(
      ...,
      handoffs=[
          handoff(data_engineer, tool_name_override="transfer_to_data_engineer"),
          handoff(frontend_builder, tool_name_override="transfer_to_frontend_builder"),
          handoff(advisor, tool_name_override="transfer_to_advisor"),
      ],
      input_guardrails=[relevance_guardrail],
  )
  for s in (data_engineer, frontend_builder, advisor):
      s.handoffs.append(triage_agent)   # "Triage" has no space, so no override needed
  ```

  A second lesson lives in the **triage instructions**: a router must be told, firmly, to
  *route and not answer*. An early version let triage reply to "what data do I have?" with
  generic advice instead of handing off to the Data Engineer (who can actually list your
  files). The fix was explicit routing rules — including that questions about *what data
  exists* always go to the Data Engineer — plus "always hand off; do not answer yourself."

Notice **least privilege for tools**: only the Data Engineer gets `run_python_file`; the
Advisor gets no tools at all. The Data Engineer's instructions encode a real *workflow*
(profile → write script → run → fix-on-error → load → validate), which is what turns a pile
of tools into reliable, step-wise behavior.

### 7.7 `app.py` — running the loop

The CLI is thin on purpose; the SDK does the heavy lifting:

```python
session = SQLiteSession("cli-session", str(config.MEMORY_DB))   # persistent memory
context = PipelineContext()                                     # tool dependency injection

result = await Runner.run(triage_agent, user, context=context, session=session)
print(result.final_output)
```

- **`Runner.run`** executes the entire agent loop — model calls, tool calls, and handoffs —
  until a final answer is produced, then returns a result object.
- **`SQLiteSession`** persists the conversation to a SQLite file. Same session id ⇒ the same
  thread, even across restarts. This is why follow-ups like *"now build a dashboard"* work:
  the model still sees everything that happened, including which specialist last had the
  floor. `/reset` calls `session.clear_session()`.
- We always **start at triage** every turn; the session carries the state forward.
- We catch `InputGuardrailTripwireTriggered` for a friendly off-topic message, and catch
  generic exceptions so a transient API/tool error doesn't kill the REPL.

---

## 8. How one request actually executes

Trace of: **"Clean data/raw/sales_2024.csv and load it as a table called sales."**

1. **Guardrail** runs on triage. The relevance classifier returns `is_on_topic=True`. No
   tripwire — continue.
2. **Triage** sees a data task and emits a **handoff** to *Data Engineer*. (In the trace
   this is a `transfer_to_data_engineer` tool call.)
3. **Data Engineer** follows its workflow:
   - `list_datasets()` → sees `raw/sales_2024.csv` and its absolute path.
   - `profile_dataset("raw/sales_2024.csv")` → a typed `DataProfile`: 33 rows, 1 duplicate,
     `order_date` uses 5 formats, `unit_price` has currency symbols, missing values in
     several columns.
   - `write_file("clean_sales.py", <code it just authored>)` → a documented pandas script
     that strips `$`, parses every date to ISO, normalizes casing, drops the duplicate and
     the negative-quantity row, and writes `sales_clean.csv`.
   - `run_python_file("clean_sales.py")` → stdout shows a before/after summary. If it had
     errored, the model would read the stderr and rewrite the script — the **observe →
     react** part of the loop.
   - `load_csv_to_sqlite("sales_clean.csv", "sales")` → table created in
     `workspace/analytics.db`.
   - `query_sqlite("SELECT region, COUNT(*) ...")` → validates the load looks sane.
4. **Data Engineer** hands back to **Triage**, which produces the final summary to you.

Every one of those steps is a node you can inspect in the trace. Run `/artifacts` afterward
to see the recorded outputs (`wrote workspace/clean_sales.py`, `loaded N rows into 'sales'`).

---

## 9. Token cost: what to expect and how to control it

Agentic systems cost more per user request than a single chat call, because **one request
triggers many model calls** and each call re-sends a growing context. Understanding *why*
makes the cost predictable — and easy to control.

### Where the tokens actually go

Two effects compound:

1. **Within a turn — the loop re-sends context.** A single "clean and load" request makes
   the Data Engineer run ~8–10 model calls (think → `list_datasets` → `profile_dataset` →
   `write_file` → `run_python_file` → `load_csv_to_sqlite` → `query_sqlite` → summarize →
   hand back). *Every* call re-sends the accumulating transcript: the agent's instructions
   (~400 tokens) + the handoff prefix (~300) + its tool schemas (~600) + **all prior tool
   outputs in this turn**. So by the end you've re-sent the growing context roughly 8 times.
2. **Across turns — the session replays history.** `SQLiteSession` feeds the *entire*
   conversation back on each new turn (that's what makes follow-ups work), so later turns
   carry earlier ones with them.

### Rough estimates for the demo session in §6

| Turn | Model calls | ~Input tokens | ~Output tokens |
|---|---|---|---|
| "What data do I have?" | 2–3 | 3k | 1k |
| Profile + diagnose | 3–4 | 8k | 1.5k |
| **Clean + load** (writes & runs a script) | 8–10 | 30–40k | 5–8k |
| Build dashboard (writes a Streamlit app) | 6–8 | 30k | 5k |
| Ask for advice | 2–3 | 10k | 2k |
| **Full session total** | ~25 | **~120–180k** | **~20–30k** |

Plus one tiny guardrail call per turn (the `*-mini` model, ~300 tokens) — fractions of a
cent in total.

### In dollars

Translate the totals with whatever your current per-token rates are. Using representative
figures:

- **A `gpt-4.1`-class model (~$2 / 1M input, ~$8 / 1M output):** the heavy "clean + load"
  turn ≈ **$0.10–0.15**; a full walkthrough ≈ **$0.40–0.60**.
- **`gpt-5`:** same order of magnitude — but ⚠️ **reasoning tokens are billed as output**,
  so on *high* reasoning effort the output tokens (and cost) can be 2–4× the table above.

**Bottom line: a few cents per turn, comfortably under ~$1 for the whole demo.** Cheap to
learn and teach with.

### The levers, in order of impact

1. **Model + reasoning effort — by far the biggest.** The reasoning model and how hard it
   thinks dominate cost. For teaching, `gpt-4.1` or `gpt-5` at low/minimal reasoning effort
   is plenty. (Set the model in `.env`; tune reasoning via `ModelSettings` if you add it.)
2. **Session accumulation.** Use `/reset` between unrelated demos, or summarize old turns,
   so you aren't re-sending a long history on every call.
3. **Script-error retries.** If a generated cleaning script fails, the agent reads the
   stderr and rewrites it — extra calls. Our sample data is intentionally messy, so budget
   for ~1 retry; cleaner inputs cost less.
4. **Tool-output size.** Already capped on purpose — file previews at 4k chars, SQL results
   at 50 rows — so a chatty tool can't flood the context window.

Two things cost **nothing**: running the generated Streamlit dashboard (it's just code — no
LLM at runtime) and the deterministic tools themselves (pandas/SQLite).

### Measure, don't guess

The estimates above are exactly that. For ground truth, open the **Traces dashboard**
(see §6) — every run reports exact input/output token counts per call. After one real
session you'll have a precise per-turn number for *your* model and data. If you want a hard
ceiling while experimenting, pass `max_turns=...` to `Runner.run(...)` to cap the number of
loop steps per request.

---

## 10. Extending the system

The architecture is meant to grow. Common next steps:

- **Add a tool.** Write a function, decorate it with `@function_tool`, add it to the right
  agent's `tools=[...]`. Example: a `fetch_url` ingestion tool, or `export_to_parquet`.
- **Add an agent.** Define a new `Agent` with a tight instruction set and a
  `handoff_description`, then add it to `triage_agent.handoffs` (and append triage back).
  Example: a *Data Validator* agent that runs Great Expectations-style checks.
- **Add an output guardrail.** Mirror the input guardrail with `@output_guardrail` to, say,
  block a response that leaked PII or raw secrets.
- **Swap memory for a database.** `SQLiteSession` is one implementation; the SDK's session
  interface lets you back memory with Postgres/Redis for a multi-user service.
- **Stream the output.** Use `Runner.run_streamed(...)` and iterate events to render tokens
  and tool calls live in a UI.
- **Wrap it in a service.** Replace the REPL in `app.py` with a FastAPI endpoint; use one
  session id per conversation and one `PipelineContext` per request.

---

## 11. Hardening for production

This is a teaching example. Before anything real, address:

- **Sandbox code execution properly.** `run_python_file` runs arbitrary model-written code
  with your user's permissions. For untrusted input, run it in a container/VM/microVM (gVisor,
  Firecracker, a jailed Docker container) with no network, a read-only base filesystem, CPU
  /memory limits, and a strict timeout. The `safe_resolve` sandbox is necessary, not
  sufficient.
- **Cost & loop controls.** Set `max_turns` on `Runner.run`, add budget/timeout limits, and
  monitor token usage (visible in traces) so a misbehaving loop can't run away.
- **Stronger guardrails.** Add output guardrails (PII/secret leakage, SQL that isn't
  read-only), and validate tool arguments beyond types.
- **Least privilege everywhere.** Scope DB credentials to read-only where possible; give
  each agent the *minimum* tools. Keep the `SELECT`-only rule on `query_sqlite`.
- **Observability.** Tracing is on by default — ship traces/metrics to your stack, add
  structured logging around tool calls, and alert on guardrail trips and tool errors.
- **Idempotency & retries.** Make data loads idempotent (`if_exists="replace"` or upserts
  keyed on a natural id) so re-runs are safe; add retries with backoff around the API.
- **Evals.** Before shipping prompt/model changes, run an eval suite of representative
  requests and check the resulting artifacts — agentic behavior is sensitive to wording.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | `cp .env.example .env` and add your key. |
| `model ... does not exist` / 404 | Set `OPENAI_MODEL=gpt-4.1` (and `OPENAI_GUARDRAIL_MODEL=gpt-4.1-mini`) in `.env`. |
| `ModuleNotFoundError: data_agent` | Activate the venv and `pip install -e .`. |
| Dashboard won't open | Run `streamlit run workspace/dashboard.py`; ensure the data was cleaned + loaded first. |
| Agent says it can't find a file | Use `list_datasets` first; raw inputs are addressed as `raw/<name>`. |
| Want to start fresh | `/reset` in the REPL, and delete `workspace/` contents. |
| Off-topic request gets refused | That's the guardrail working. Ask a data/pipeline/dashboard question. |

---

### Key takeaways

- An agent is an **LLM in a tool-calling loop**; the SDK gives you that loop plus the
  production scaffolding.
- **Decisions belong to the model; correctness belongs to code.** Keep deterministic work
  in plain tools.
- **Compose small, focused agents** with handoffs instead of one mega-prompt.
- **Guardrails, sandboxing, sessions, structured output, and tracing** are what move a demo
  toward something robust.

Now open `src/data_agent/team.py`, read the instructions you're giving each agent, and
start tinkering. The fastest way to learn this is to change a prompt or add a tool and watch
the trace.
