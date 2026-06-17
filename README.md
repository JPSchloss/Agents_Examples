# Building a Robust Agentic AI System with the OpenAI Agents SDK

A complete, runnable teaching example: a **multi-agent data-pipeline assistant**. You talk
to it in plain English and it can **ingest, profile, clean, and pipeline data**, **load it
into a database**, **build a Streamlit dashboard**, and **advise you** on how to build well
— grounded in your own knowledge base (**RAG**) and extended with external tool servers
(**MCP**), all while you stay in the loop.

This document walks through the system from first principles. Read it top to bottom and
you'll understand not just *what* the code does, but *why* each piece exists and how the
modern agentic patterns fit together.

> **Stack:** Python 3.10+ · [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
> (`openai-agents`) · RAG ([ChromaDB](https://www.trychroma.com) + OpenAI embeddings) ·
> [MCP](https://modelcontextprotocol.io) · pandas · SQLite · Streamlit · `rich` CLI.

---

## Table of contents

1. [What is an "agent," really?](#1-what-is-an-agent-really)
2. [The building blocks](#2-the-building-blocks-this-example-teaches)
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

## 2. The building blocks this example teaches

| # | Concept | What it is | Where to see it |
|---|---------|-----------|-----------------|
| 1 | **Agent** | An LLM + instructions + a set of tools/handoffs | `team.py` |
| 2 | **Tools** | Python functions the model can call (`@function_tool`) | `tools/` |
| 3 | **Handoffs** | One agent transferring the conversation to a specialist | `team.py` |
| 4 | **Guardrails** | Fast checks that can stop a run (e.g. off-topic) | `guardrails.py` |
| 5 | **Sessions** | Persistent conversation memory across turns/restarts | `app.py` |
| 6 | **Structured output** | Forcing the model to return typed, parseable data | `schemas.py` |
| 7 | **Tracing** | Automatic, inspectable record of every step | built-in (see §6) |
| 8 | **RAG + vector DB** | Retrieving grounded facts from your docs via ChromaDB | `rag.py`, `tools/knowledge.py` (§7.8) |
| 9 | **MCP** | Connecting external tool servers over a standard protocol | `mcp_servers/`, `team.py` (§7.9) |

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
                  │ profile/clean│ │ Builder  │ │ explains │
                  │ run code,    │ │ writes a │ │ & guides │
                  │ sqlite       │ │ Streamlit│ │          │
                  │ + RAG + MCP  │ │ app      │ │ +RAG +MCP│
                  └──────┬───────┘ └────┬─────┘ └────┬─────┘
                         └──────────────┴────────────┘
                            each can hand BACK to Triage
                         │                          │
              ┌──────────▼───────────┐   ┌──────────▼───────────────┐
              │ RAG · ChromaDB       │   │ MCP server               │
              │ search_knowledge →   │   │ acme-data-standards →     │
              │ knowledge/*.md       │   │ canonical_column_name, …  │
              └──────────────────────┘   └──────────────────────────┘
```

Two capability sources augment the agents' native tools:
- **RAG** grounds answers in *your* documents (`knowledge/*.md`), retrieved from a
  **ChromaDB** vector store via a `search_knowledge` tool — see §7.8.
- **MCP** connects external tool servers over a standard protocol; here a local
  "data-standards" server adds naming/typing helpers — see §7.9.

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
├── knowledge/               ← RAG source docs (the agents' grounded "additional context")
│   ├── company_data_dictionary.md
│   └── data_engineering_principles.md
├── mcp_servers/
│   └── reference_server.py  ← a local MCP server ("acme-data-standards")
├── workspace/               ← the agent's sandbox (all writes land here; git-ignored)
└── src/data_agent/
    ├── config.py            ← models, embeddings + the folders the agent may touch
    ├── context.py           ← typed run context passed to every tool
    ├── schemas.py           ← Pydantic models for structured output
    ├── guardrails.py        ← the input relevance guardrail
    ├── rag.py               ← the RAG engine (chunk → ChromaDB embed/store/search)
    ├── team.py              ← build_team(): the agents + handoffs + RAG + MCP wiring
    ├── app.py               ← the CLI: argparse subcommands + rich REPL
    └── tools/
        ├── _paths.py        ← path-safety helper (sandbox enforcement)
        ├── filesystem.py    ← list / read / write files, run a script
        ├── data.py          ← profile a CSV, load to SQLite, run SQL
        └── knowledge.py     ← search_knowledge: the RAG retrieval tool
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

This installs the OpenAI Agents SDK, the MCP SDK, ChromaDB, pandas/numpy, Streamlit,
`rich`, and friends, and registers a `data-agent` command.

### Step 3 — Add your API key

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

`.env` is git-ignored. The app reads it automatically (`config.py` calls `load_dotenv()`).

**Model choice.** The default is `gpt-5`. If your account doesn't have access yet, set
`OPENAI_MODEL=gpt-4.1` in `.env`. The cheap guardrail uses `OPENAI_GUARDRAIL_MODEL`
(`gpt-5-mini` by default; `gpt-4.1-mini` is a fine fallback). RAG uses
`OPENAI_EMBED_MODEL` (`text-embedding-3-small` by default).

### Step 4 — Build the knowledge base (RAG index)

```bash
data-agent ingest
```

This chunks and embeds the markdown files in `knowledge/` into a **ChromaDB** vector
collection (persisted at `workspace/chroma/`) so the agents can retrieve grounded facts.
Re-run it (with `--force`) whenever you edit those docs.
You can verify everything is wired up with:

```bash
data-agent info        # shows models, RAG index status, MCP server status
```

---

## 6. Run it

The CLI has three subcommands; `chat` is the default. Every command supports `-h`:

```bash
data-agent -h                 # top-level help (lists subcommands)
data-agent chat -h            # all chat flags
data-agent ingest             # build/refresh the RAG index
data-agent info               # config + component status
data-agent                    # start chatting (chat is the default)
```

Useful `chat` flags: `--model gpt-4.1`, `--reset` (fresh conversation), `--max-turns N`
(cost guard), `--no-mcp` / `--no-rag` (toggle those capabilities), `--no-trace`.

Inside the REPL (a `rich` UI that renders markdown and reports token usage per turn), try
this productive first session in order:

```
user ▸ What data do I have to work with?
user ▸ Profile data/raw/sales_2024.csv and tell me what's wrong with it.
user ▸ Clean it, then load it into SQLite as a table called sales.
user ▸ Build me a dashboard for the cleaned sales data.
user ▸ How should I handle the negative-quantity rows?   (← grounded advice via RAG)
```

REPL commands: `/help`, `/artifacts` (files the agents created), `/reset` (clear memory),
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

Two responsibilities: pick the models — reasoning, guardrail, and embedding, all from env
vars so the same code runs anywhere — and define **the only folders the agent may touch**: a
read-only `data/raw` and `knowledge/`, plus a writable `workspace/`. Confining all writes to
one sandbox folder is the foundation that makes it safe to hand a model file-writing and
code-execution tools.

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

`team.py` exposes a **factory**, `build_team(mcp_servers=None, with_knowledge=True)`, rather
than module-level agents. Why? MCP servers are *live connections* that must be opened
(async) before a run and closed after — they can't exist at import time. The app opens its
MCP servers, then calls `build_team(...)` to construct agents around them. (It's also handy
for tests: `build_team(with_knowledge=False)` gives you a no-RAG, no-MCP team in one line.)

Each agent is `Agent(name, instructions, model, tools=[...], handoffs=[...], mcp_servers=[...])`.

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
Frontend Builder gets no MCP/RAG; the Advisor gets *only* the read-only `search_knowledge`
tool. The Data Engineer and Advisor receive `search_knowledge` (RAG) and the
`mcp_servers` list (MCP) because both benefit from grounded facts and the data-standards
helpers. The Data Engineer's instructions encode a real *workflow* (consult knowledge →
profile → write script → run → fix-on-error → load → validate), which is what turns a pile
of tools into reliable, step-wise behavior.

### 7.7 `app.py` — the CLI and the loop

`app.py` is two things: an **`argparse` CLI** (subcommands `chat`/`ingest`/`info`, all with
`-h`) and a **`rich` REPL**. The agentic core is still tiny — the SDK does the heavy lifting:

```python
async with AsyncExitStack() as stack:                # manage MCP server lifecycles
    server = MCPServerStdio(params={"command": sys.executable,
                                    "args": [str(config.REFERENCE_MCP_SERVER)]},
                            name="acme-data-standards", cache_tools_list=True)
    await stack.enter_async_context(server)          # connect; auto-cleanup on exit
    triage = build_team(mcp_servers=[server], with_knowledge=True)

    session = SQLiteSession(args.session, str(config.MEMORY_DB))   # persistent memory
    result = await Runner.run(triage, user, context=context,
                              session=session, max_turns=args.max_turns)
```

- **`Runner.run`** executes the entire agent loop — model calls, tool calls, MCP calls, and
  handoffs — until a final answer is produced, then returns a result object. `max_turns`
  caps loop steps as a cost/runaway guard.
- **`MCPServerStdio` + `AsyncExitStack`** launch the MCP server as a subprocess for the
  session and guarantee it's cleaned up on exit. See §7.9.
- **`SQLiteSession`** persists the conversation to a SQLite file. Same session id ⇒ the same
  thread, even across restarts. This is why follow-ups like *"now build a dashboard"* work.
  `--reset` / `/reset` calls `session.clear_session()`.
- We always **start at triage** every turn; the session carries the state forward.
- After each turn we print **token usage** from `result.context_wrapper.usage` so cost is
  visible while you teach.
- We catch `InputGuardrailTripwireTriggered` for a friendly off-topic message, and catch
  generic exceptions so a transient API/tool error doesn't kill the REPL.

The `--model` flag mutates `config.MODEL` *before* `build_team()` runs (the factory reads it
at call time), so you can swap models without editing `.env`.

### 7.8 `rag.py` + `tools/knowledge.py` — grounded knowledge (RAG)

**RAG (Retrieval-Augmented Generation)** means: instead of hoping the model memorized your
facts, you *retrieve* the most relevant passages from your own documents at query time and
hand them to the model. Answers stay grounded in *your* truth, and you update knowledge by
editing files — no retraining. Storage and search are handled by **ChromaDB**, a persistent
vector database. The flow, all visible in `rag.py`:

```
knowledge/*.md ─chunk─▶ text chunks ──┐
                                       ├─▶ Chroma collection  (embeds + stores + HNSW index)
query ─────────────────────────────────┘                 │   persisted at workspace/chroma/
                                                           ▼
                          Chroma ANN search ──▶ top-k chunks (source + similarity score)
```

- **`build_index()`** (run via `data-agent ingest`) splits each markdown doc into overlapping
  ~900-char chunks and adds them to a Chroma collection. Chroma embeds each chunk via an
  attached **`OpenAIEmbeddingFunction`** (`text-embedding-3-small`) and persists the vectors
  to disk. Re-ingesting drops and rebuilds the collection, so it's idempotent.
- **`search(query, k=4)`** hands the query to `collection.query(...)`; Chroma embeds it,
  runs an approximate-nearest-neighbour search over its HNSW index, and returns the top
  passages with their source and a cosine similarity score.
- **`tools/knowledge.py`** wraps that in the `search_knowledge` `@function_tool`. The agents
  call it *on demand* — pulling only relevant passages — which is cheaper and scales to far
  more knowledge than fits in a prompt. If the collection is empty, the tool returns a
  friendly "run `data-agent ingest`" message instead of erroring.

Why Chroma rather than a JSON file + hand-rolled cosine? A purpose-built vector DB gives you
persistent on-disk storage, an ANN index that scales past brute-force search, embedding
management, and metadata filtering — all behind the same `search(query) → top-k` interface
the agents see. **The storage engine got more capable; the agent-facing contract didn't
change** — which is exactly how you want infrastructure swaps to feel.

The knowledge base here is the **company data dictionary** (canonical region/category values,
the `total = quantity × unit_price` rule, how to treat returns) and the **house engineering
principles**. So when you ask *"how should I handle negative-quantity rows?"*, the Advisor
retrieves the actual rule ("returns → exclude from revenue") and cites the source, rather
than inventing a generic answer.

> **Reliability lesson — force the retrieval.** A strong model will often *think* it already
> knows the answer and skip the tool, then confidently cite a file that doesn't exist. Prompt
> instructions ("you MUST search first") help but aren't reliable on their own. The robust
> fix is `model_settings=ModelSettings(tool_choice="required")` on the Advisor: the SDK makes
> the model call a tool on its first step (so it actually retrieves), then auto-resets
> `tool_choice` to `"auto"` (because `reset_tool_choice` defaults to `True`) so the next turn
> produces the final grounded answer without looping. *Don't trust the model to ground itself
> — make grounding structurally unavoidable.*

> **Scaling note:** Chroma runs embedded (in-process, on-disk) here, which is perfect for
> teaching and small/medium corpora. The same Chroma code talks to a **client/server**
> deployment by switching `PersistentClient` for `HttpClient`. To go further, swap in
> pgvector/Pinecone or OpenAI's hosted vector stores + the SDK's built-in `FileSearchTool`
> — the retrieval *interface* (query → top-k chunks) stays identical.

### 7.9 `mcp_servers/reference_server.py` — an MCP server

**MCP (Model Context Protocol)** is an open standard that lets an AI app connect to external
**tool/data servers** over one common interface, instead of hard-coding each integration —
"USB-C for tools." Write a capability once as an MCP server and *any* MCP-aware client (this
app, Claude Desktop, IDEs, …) can use it. The Agents SDK is an MCP **client**: attach a
server via `mcp_servers=[...]` and its tools appear to the model alongside the native ones.

Our server (`reference_server.py`) is built with `FastMCP` and speaks **stdio** (the client
launches it as a subprocess and talks over stdin/stdout). It exposes three deterministic
*data-standards* helpers — `canonical_column_name`, `standard_dtype`, `naming_conventions`:

```python
mcp = FastMCP("acme-data-standards")

@mcp.tool()
def canonical_column_name(name: str) -> str:
    """Convert a column name to the org's canonical snake_case form."""
    ...

if __name__ == "__main__":
    mcp.run()          # stdio transport — exactly what MCPServerStdio expects
```

This is deliberately *distinct from RAG*: RAG returns prose passages to reason over; MCP
returns exact, machine-checked answers. The Data Engineer uses both — knowledge for the
*rules*, the MCP server for precise *naming/typing*.

**Swapping in third-party servers** is the real-world payoff. Point `MCPServerStdio` (or
`MCPServerStreamableHttp`) at any published server — for example the filesystem, fetch, or
GitHub servers — and those tools become available to your agents with no code changes:

```python
fs = MCPServerStdio(params={"command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]})
```

---

## 8. How one request actually executes

Trace of: **"Clean data/raw/sales_2024.csv and load it as a table called sales."**

1. **Guardrail** runs on triage. The relevance classifier returns `is_on_topic=True`. No
   tripwire — continue.
2. **Triage** sees a data task and emits a **handoff** to *Data Engineer*. (In the trace
   this is a `transfer_to_data_engineer` tool call.)
3. **Data Engineer** follows its workflow:
   - `search_knowledge("region/category canonical values, returns, total rule")` → retrieves
     the data-dictionary rules (RAG), and the MCP `standard_dtype`/`canonical_column_name`
     tools confirm naming and typing. The agent now cleans to *your* standards, not generic
     defaults.
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
LLM at runtime) and the deterministic tools themselves (pandas/SQLite **and the MCP
data-standards calls** — no model involved).

**RAG cost** is tiny and twofold: a one-time embedding pass at `ingest` (a few thousand
tokens on the cheap `text-embedding-3-small` model — fractions of a cent for this corpus),
plus one small query-embedding per `search_knowledge` call. The retrieved passages do add
input tokens to that turn, but far fewer than stuffing all docs into every prompt would.

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
- **Add knowledge (RAG).** Drop a new `.md` file in `knowledge/` and re-run `data-agent
  ingest`. No code change — the agents can now retrieve it. This is the easiest way to
  teach the system new business rules or conventions.
- **Add an MCP server.** Point another `MCPServerStdio`/`MCPServerStreamableHttp` at a
  published server (filesystem, fetch, GitHub, a database server, …) and pass it into
  `build_team(mcp_servers=[...])`. Its tools appear to the agents automatically.
- **Add an agent.** Define a new `Agent` with a tight instruction set and a
  `handoff_description`, then add it to the triage `handoffs` (and append triage back).
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
| Agent says "knowledge base not indexed" | Run `data-agent ingest` (re-run with `--force` after editing `knowledge/`). |
| MCP tools missing / server won't start | Run `data-agent info` to check status; ensure deps are installed; try `--no-mcp` to isolate. |
| Dashboard won't open | Run `streamlit run workspace/dashboard.py`; ensure the data was cleaned + loaded first. |
| Agent says it can't find a file | Use `list_datasets` first; raw inputs are addressed as `raw/<name>`. |
| Want to start fresh | `data-agent chat --reset` (or `/reset` in the REPL), and delete `workspace/` contents. |
| Off-topic request gets refused | That's the guardrail working. Ask a data/pipeline/dashboard question. |
| See it all step by step | `data-agent info` for status; the [Traces dashboard](https://platform.openai.com/traces) for per-call detail. |

---

### Key takeaways

- An agent is an **LLM in a tool-calling loop**; the SDK gives you that loop plus the
  production scaffolding.
- **Decisions belong to the model; correctness belongs to code.** Keep deterministic work
  in plain tools.
- **Compose small, focused agents** with handoffs instead of one mega-prompt.
- **Ground the model in your own truth** with RAG, and **extend its reach** with MCP servers
  — both plug in without changing the agent loop.
- **Guardrails, sandboxing, sessions, structured output, and tracing** are what move a demo
  toward something robust.

Now open `src/data_agent/team.py`, read the instructions you're giving each agent, and
start tinkering. The fastest way to learn this is to change a prompt or add a tool and watch
the trace.
