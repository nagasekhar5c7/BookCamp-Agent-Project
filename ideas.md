# Deep Research Multi-Agent System — Design Ideas

> Status: Design / pre-implementation. Do **not** start coding until explicitly approved.

---

## 1. Overview

A multi-agent deep-research system that takes a plain-text user query (e.g. *"Tell me about MCP"* or *"Explain Playwright in detail"*), performs iterative web research with citations, and returns a polished **Microsoft Word (.docx)** report as the deliverable.

The system follows a **Lead Researcher → Sub-Agent (master–slave)** orchestration pattern built on **LangGraph**, exposed via a **FastAPI** async job API.

---

## 2. Goals & Non-Goals

### Goals

- Accept a single natural-language query as input.
- Produce a structured, well-cited Word document as output.
- Enforce a strict separation of duties: the **Lead Researcher only plans and synthesizes**; it **never** performs research itself.
- Track every factual claim back to a source URL via inline numbered citations + a References section.
- Production-grade code: modular, testable, typed, observable, configurable via env vars.
- Pluggable LLM and search providers (start with Groq + Tavily).

### Non-Goals (v1)

- No UI — API only (a frontend can be bolted on later).
- No multi-turn conversation — single query in, single document out.
- No PDF / HTML / Markdown export — Word only for v1.
- No user authentication (can be layered on later via FastAPI dependencies).

---

## 3. High-Level Architecture

```
        ┌────────────────────────────────────────────────────────┐
        │                     FastAPI Layer                      │
        │  POST /research                → enqueue, return id    │
        │  GET  /research/{id}           → status + progress     │
        │  GET  /research/{id}/review    → fetch pending plan    │
        │  POST /research/{id}/review    → approve / edit plan   │
        │  GET  /research/{id}/document  → download .docx        │
        └───────────────────────┬────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │             LangGraph Orchestration Engine             │
        │                                                        │
        │  ┌──────┐   ┌───────┐   ┌────────┐   ┌──────────┐      │
        │  │ PLAN │──▶│ HUMAN │──▶│RESEARCH│──▶│SYNTHESIZE│──┐   │
        │  │(Lead)│   │REVIEW │   │ (Subs) │   │  (Lead)  │  │   │
        │  └──────┘   └───────┘   └────────┘   └──────────┘  │   │
        │               ⏸ pause                              ▼   │
        │          (interrupt)                        ┌──────────┐│
        │                                             │ DOCUMENT ││
        │                                             │ (writer) ││
        │                                             └──────────┘│
        └────────────────────────────────────────────────────────┘
               │            │             │            │
               ▼            ▼             ▼            ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐
        │ Groq LLM │  │Job Store │  │  Tavily  │  │py-docx  │
        └──────────┘  └──────────┘  └──────────┘  └─────────┘
```

### Fixed pipeline, dynamic sub-tasks, human-in-the-loop gate

The high-level LangGraph stages are **fixed**: `plan → human_review → research → synthesize → document`. Inside the `research` stage, the Lead's approved plan dictates **how many** sub-tasks exist and **what each one is**, and sub-agents execute them **sequentially** (one after another).

**Human-in-the-loop review** sits between `plan` and `research`: after the Lead generates its sub-task list, the graph **pauses** (via LangGraph's `interrupt()`) and waits for the user to approve, edit, or reject the plan before any Tavily calls or sub-agent LLM calls are made. This is the right gate because it catches bad plans before spending money, and the plan is small/readable (3–7 sub-tasks) — easy for a human to eyeball.

---

## 4. Agent Design

### 4.1 Lead Researcher Agent (`nodes/lead.py`)

The Lead has **three distinct responsibilities**, invoked at different stages of the graph. It never touches a search tool.


| Stage         | Input                                 | Output                                                        | Prompt role                                                                                                    |
| ------------- | ------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `plan`        | User query                            | Ordered list of 3–7 research sub-tasks (JSON)                 | "You are a lead researcher. Decompose this query into atomic research tasks. Do NOT answer any task yourself." |
| `synthesize`  | All sub-agent findings + citation map | Structured outline (sections + bullet points + citation refs) | "Synthesize these findings into a coherent report outline. Preserve all citations. Note any failed sub-tasks as limitations." |

> No automated `review` stage in v1 — the Lead does **not** self-critique mid-pipeline. Review is delegated to a human between `plan` and `research` (see §4.4).


**Hard constraint enforced in the prompt:** the Lead must never produce factual claims on its own — only plans and synthesis of material supplied by sub-agents.

### 4.2 Sub-Agent / Researcher Worker (`nodes/researcher.py`)

Stateless worker that receives **one sub-task** and returns findings + citations.

- **Tool:** Tavily Search API (via `langchain_community.tools.tavily_search` or direct HTTP client — TBD).
- **Loop:** search → read top-k results → summarize into structured findings → return.
- **Output schema:**
  ```python
  class Finding(BaseModel):
      task_id: str
      summary: str               # LLM-written synthesis for this task
      key_points: list[str]
      sources: list[Citation]    # every URL actually used
  ```
- Sub-agents are **stateless** — they don't see other sub-tasks or the overall plan, only their assigned task + the original query for context.

### 4.3 Document Writer (`nodes/writer.py`)

Not an LLM agent — a deterministic transformer. Takes the synthesized outline + citation registry and emits a `.docx` using `python-docx`.

### 4.4 Human Review Node (`nodes/human_review.py`)

A control node, not an agent — it uses LangGraph's `interrupt()` primitive to pause the graph and surface the Lead's plan to the user through the API.

**Flow:**

1. `plan` node finishes → writes `state.plan` → transitions to `human_review`.
2. `human_review` node calls `interrupt({ "plan": state.plan })`. The graph halts; LangGraph persists the interrupted state via the checkpointer.
3. Job status flips to `awaiting_review`. FastAPI exposes the pending plan at `GET /research/{id}/review`.
4. The user submits a decision via `POST /research/{id}/review` with one of:
   - `approve` — proceed with the plan as-is.
   - `edit` — supply a modified list of sub-tasks (same schema), proceed with those.
   - `reject` — abort the job with status `cancelled_by_user`.
5. The background runner resumes the graph with `Command(resume=decision)`. Execution continues into `research`.

**Review timeout**: configurable (default **30 minutes**). If no decision arrives, the job is marked `failed` with reason `review_timeout` — prevents abandoned jobs from pinning memory forever.

**v1 note on persistence**: HITL normally relies on LangGraph checkpointing. We're skipping a durable checkpointer for v1, so interrupted state is held in the in-memory `MemorySaver` checkpointer. **Consequence**: a process restart while a job is awaiting review loses that job. Acceptable for local-only v1; called out as a known limitation.

---

## 5. LangGraph State Schema

Single shared state object flowing through the graph (`graph/state.py`):

```python
class ResearchState(TypedDict):
    # Input
    query: str
    job_id: str

    # Planning
    plan: list[SubTask]              # produced by lead.plan
    plan_approved: bool              # set True after human_review resumes
    current_task_index: int          # drives the sequential sub-agent loop

    # Research accumulation
    findings: list[Finding]          # appended by each sub-agent run (status: ok|failed)
    citations: dict[int, Citation]   # global registry, numbered 1..N

    # Synthesis
    outline: ReportOutline           # produced by lead.synthesize
    limitations: list[str]           # notes from failed sub-tasks → "Research Limitations" section

    # Output
    document_path: str               # path to generated .docx

    # Cost & observability
    tokens_used: TokenUsage          # running total, input/output tokens per provider
    cost_estimate_usd: float         # running total against MAX_JOB_COST_USD ceiling
    errors: list[str]
    step_log: list[StepLogEntry]
```

**Reducers**: `findings` and `citations` use additive reducers so that each sub-agent run appends cleanly.

---

## 6. Citation Handling (Critical Requirement)

Citations are a first-class concept, tracked end-to-end.

1. **Capture at source**: every Tavily result consumed by a sub-agent is turned into a `Citation(title, url, accessed_at, snippet)` before the LLM ever sees it.
2. **Deduplicate & number**: a `CitationRegistry` service assigns a stable integer id to each unique URL across all sub-tasks (e.g. `[1]`, `[2]`, …).
3. **Propagate through synthesis**: the Lead's synthesis prompt is given the numbered citation map and is **instructed to preserve the numeric markers verbatim** next to every claim it writes.
4. **Render in Word**:
  - Inline markers like `[1]` appear next to each claim.
  - A final **References** section lists each citation in the form:
  `[1] Title — https://example.com (accessed 2026-04-04)`
5. **Validation step** before document generation: every `[n]` marker in the outline must exist in the registry; orphan markers abort the job with a clear error.

---

## 7. Tech Stack


| Concern        | Choice                                | Notes                                                                                         |
| -------------- | ------------------------------------- | --------------------------------------------------------------------------------------------- |
| Language       | Python 3.11+                          | Modern typing, `match` statements                                                             |
| Orchestration  | **LangGraph**                         | State machine for the fixed pipeline                                                          |
| LLM            | **Groq API**                          | Confirmed: Groq Cloud (not xAI Grok). Default model: `llama-3.3-70b-versatile` (swap via env) |
| Web search     | **Tavily API**                        | Purpose-built for LLM research; returns clean content + URLs                                  |
| Doc generation | `python-docx`                         | Mature, no MS Office dependency                                                               |
| API            | **FastAPI**                           | Async job pattern                                                                             |
| Job store      | `Redis` (prod) / in-memory dict (dev) | Abstracted behind a `JobStore` interface                                                      |
| Config         | `pydantic-settings`                   | Env-var driven, typed                                                                         |
| Logging        | `structlog`                           | JSON logs in prod, pretty in dev                                                              |
| Testing        | `pytest` + `pytest-asyncio`           | Unit + integration                                                                            |
| Lint/format    | `ruff` + `mypy`                       | Enforced in CI                                                                                |
| Packaging      | `uv` or `poetry`                      | TBD — **open question**                                                                       |


---

## 8. Project Structure (Modular, Production-Grade)

```
deep_research/
├── pyproject.toml
├── README.md
├── .env.example
├── src/
│   └── deep_research/
│       ├── __init__.py
│       ├── config.py                  # pydantic-settings, all env vars
│       ├── logging_setup.py
│       │
│       ├── api/                       # FastAPI layer
│       │   ├── __init__.py
│       │   ├── main.py                # app factory
│       │   ├── routes/
│       │   │   ├── research.py        # POST/GET endpoints
│       │   │   └── health.py
│       │   ├── schemas.py             # Pydantic request/response models
│       │   └── dependencies.py
│       │
│       ├── agents/                    # Agent prompts + logic
│       │   ├── lead.py                # plan + synthesize functions
│       │   ├── researcher.py          # sub-agent worker
│       │   └── prompts/               # Prompts as separate .md/.txt files
│       │       ├── lead_plan.md
│       │       ├── lead_synthesize.md
│       │       └── researcher.md
│       │
│       ├── graph/                     # LangGraph definition
│       │   ├── state.py               # ResearchState TypedDict
│       │   ├── nodes.py               # plan_node, research_node, ...
│       │   ├── edges.py               # conditional routing
│       │   └── builder.py             # build_graph() factory
│       │
│       ├── tools/                     # External tool adapters
│       │   ├── llm.py                 # Groq client wrapper (provider-agnostic interface)
│       │   ├── search.py              # Tavily client wrapper
│       │   └── base.py                # Protocols for LLMClient, SearchClient
│       │
│       ├── services/
│       │   ├── citation_registry.py
│       │   ├── job_store.py           # abstract + in-memory + redis impls
│       │   └── document_writer.py     # python-docx rendering
│       │
│       ├── models/                    # Domain models (Pydantic)
│       │   ├── citation.py
│       │   ├── finding.py
│       │   ├── plan.py
│       │   └── outline.py
│       │
│       └── workers/
│           └── runner.py              # background task that executes the graph
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── scripts/
    └── run_local.py                   # CLI entrypoint for dev
```

**Design principles applied:**

- **Dependency inversion**: agents depend on `LLMClient` / `SearchClient` protocols, not concrete Groq/Tavily classes → swappable + easy to mock in tests.
- **Prompts as data**: stored in `agents/prompts/` as files, loaded at runtime. No giant f-strings in Python code.
- **Single responsibility**: graph nodes are thin — they call into `agents/` and `services/`.
- **Config centralization**: everything configurable lives in `config.py` and reads from env.

---

## 9. API Design (Async Job Pattern + HITL)

Job lifecycle states: `queued → planning → awaiting_review → researching → synthesizing → generating_document → done` (terminal) or `failed` / `cancelled_by_user` (terminal).

```
POST /research
  body: { "query": "Tell me about MCP" }
  → 202 Accepted
  → { "job_id": "uuid", "status": "queued" }

GET /research/{job_id}
  → 200 OK
  → {
      "job_id": "...",
      "status": "awaiting_review" | "researching" | "done" | "failed" | ...,
      "progress": { "stage": "research", "task": 3, "total": 5 },
      "cost_so_far_usd": 0.042,
      "error": null
    }

GET /research/{job_id}/review           # only valid when status == awaiting_review
  → 200 OK
  → {
      "plan": [
        { "id": "t1", "title": "...", "description": "...", "search_hints": [...] },
        ...
      ]
    }
  → 409 Conflict if job is not currently awaiting review

POST /research/{job_id}/review
  body: { "decision": "approve" }
        | { "decision": "edit", "plan": [...edited sub-tasks...] }
        | { "decision": "reject" }
  → 200 OK → { "status": "researching" | "cancelled_by_user" }

GET /research/{job_id}/document
  → 200 OK, Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document
  → binary .docx (only when status == "done")

GET /health
  → liveness + dependency checks (Groq, Tavily reachable)
```

**Rate limiting** (v1 requirement): `slowapi` middleware applied to `POST /research` (e.g. 10 req/min per IP) and `POST /research/{id}/review` (e.g. 30 req/min per IP). Read-only endpoints are not rate-limited. Limits are env-configurable.

**Background execution** via FastAPI `BackgroundTasks` for v1 (simple), with a clean path to Celery/ARQ later if we need real queueing.

---

## 10. Configuration (env vars)

```
# LLM
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile

# Search
TAVILY_API_KEY=...

# Research limits
MAX_SUBTASKS=7
MAX_SOURCES_PER_SUBTASK=5
SUBTASK_TIMEOUT_SEC=180
JOB_TIMEOUT_SEC=900

# Cost ceiling (hard abort if exceeded mid-job)
MAX_JOB_COST_USD=1.00

# Human-in-the-loop
HUMAN_REVIEW_TIMEOUT_SEC=1800     # 30 minutes

# Job store (in-memory for v1)
JOB_STORE_BACKEND=memory

# API rate limits (per IP)
RATE_LIMIT_POST_RESEARCH=10/minute
RATE_LIMIT_POST_REVIEW=30/minute

# Output & logging
OUTPUT_DIR=./artifacts
LOG_LEVEL=INFO
LOG_FORMAT=json                    # or "pretty" for dev
```

---

## 11. Error Handling, Retry & Observability

Failures are handled at **three distinct layers**, each with its own policy. This is critical: a Tavily hiccup should not kill a job, but a malformed LLM plan should not silently corrupt the final report either.

### 11.1 Layer 1 — Tool-call retries (Groq / Tavily HTTP)

Implemented via **`tenacity`** inside the `LLMClient` and `SearchClient` adapters, so retry logic lives in one place and every caller benefits automatically.

| Failure class                               | Action          | Attempts | Backoff                   |
| ------------------------------------------- | --------------- | -------- | ------------------------- |
| Network errors, timeouts, connection resets | Retry           | 3        | Exponential + jitter: 1s → 2s → 4s |
| HTTP 5xx (server errors)                    | Retry           | 3        | Exponential + jitter      |
| HTTP 429 (rate limit)                       | Retry           | 5        | Honor `Retry-After` header if present, else exponential |
| HTTP 4xx (auth, bad request, 404)           | **Fail fast**   | 0        | Not retryable — bug or config issue |
| Tavily: empty result set                    | Not an error    | —        | Returned as `SearchResult(results=[])` and handled by sub-agent |

Wall-clock timeout per individual HTTP call: **30s**. Total budget across retries: **90s**.

### 11.2 Layer 2 — Agent-level recovery (LLM output validation)

LLM responses for `plan` and `synthesize` must conform to a Pydantic schema. When parsing fails:

1. **Format-repair retry** (max 2 attempts): re-prompt the LLM with the original output + the parser error, asking it to fix the JSON. This is separate from the Layer 1 retry budget.
2. **Fallback**: if the Lead's `plan` still fails after repair, the job fails hard with status `failed` and reason `plan_generation_failed`. Planning is critical path — no sensible default exists.
3. For sub-agent findings, if the LLM output can't be parsed, the sub-task is marked `failed` with the raw output preserved in state, and the pipeline **continues with the remaining sub-tasks**.

### 11.3 Layer 3 — Sub-task / graph-node resilience

This is the "agent couldn't do the task" case.

- **Per-sub-task isolation**: each sub-task runs in its own try-boundary. A failed sub-task does **not** abort the job — the `Finding` is recorded with `status="failed"` and an error message, and the Lead's `synthesize` stage is told which tasks failed so it can note gaps in the report.
- **Minimum-success threshold**: if more than **50%** of sub-tasks fail, the job is aborted with `insufficient_research`. Writing a doc from mostly-failed research would produce a misleading artifact.
- **Critical-path nodes** (`plan`, `synthesize`, `document`): any failure here terminates the job. There is nothing useful to salvage without these.
- **Node-level wrapping**: every graph node is wrapped in a decorator that catches exceptions, logs them with full context, appends to `state.errors`, and routes to the terminal `failed` state. No half-written documents ever leave the system.
- **Zero-result sub-tasks**: if Tavily returns zero results even after retries, the sub-task is immediately marked `failed` with reason `no_sources_found`. It counts against the 50% failure threshold and feeds the Research Limitations section.
- **Cost ceiling enforcement**: after every LLM call, the runner updates `state.cost_estimate_usd` using a per-model price table. If the running total crosses `MAX_JOB_COST_USD`, the job aborts immediately with status `failed` and reason `cost_ceiling_exceeded`. This prevents a runaway Lead or repeated format-repair loops from burning unbounded money.
- **LangGraph checkpointing**: skipped for v1 (decision recorded in §13). Uses the in-memory `MemorySaver` to support HITL `interrupt()` within a single process lifetime. A process restart loses in-flight jobs — acceptable trade-off for local-only v1.

### 11.4 Timeouts & budgets

| Scope                   | Limit                | Behavior on breach                |
| ----------------------- | -------------------- | --------------------------------- |
| Single HTTP call        | 30s                  | Raise → Layer 1 retry             |
| Single sub-task (total) | 3 min                | Mark sub-task failed, continue    |
| Full job                | 15 min               | Abort, status `failed` (timeout)  |
| Human review pause      | 30 min               | Abort, status `failed` (review_timeout) |
| Cost per job            | `MAX_JOB_COST_USD` ($1.00) | Abort, status `failed` (cost_ceiling_exceeded) |
| Max sub-tasks per job   | `MAX_SUBTASKS` (7)   | Hard cap in Lead prompt + post-validation |
| Max sources per sub-task| `MAX_SOURCES_PER_SUBTASK` (5) | Truncate on read          |

All limits are env-configurable.

### 11.5 Logging — what and how

**Library**: `structlog` configured to emit JSON in prod and pretty-printed key/value in dev.

**Every log line carries a bound context**:
```
job_id, stage (plan|research|synthesize|document), sub_task_id, attempt, provider
```
This context is pushed at the top of each graph node via `structlog.contextvars.bind_contextvars` so that every downstream call — including inside adapters — inherits it automatically.

**What gets logged, at which level:**

| Level   | Events                                                                       |
| ------- | ---------------------------------------------------------------------------- |
| `INFO`  | Job lifecycle (`job_started`, `stage_entered`, `stage_completed`, `job_done`) |
| `INFO`  | Tool-call summaries: provider, model, latency_ms, input_tokens, output_tokens, cost_estimate |
| `INFO`  | Sub-task outcomes: `subtask_completed` / `subtask_failed` + source count      |
| `WARN`  | Layer 1 retries, LLM format-repair retries, empty Tavily results              |
| `WARN`  | Orphan citation markers detected during validation                            |
| `ERROR` | Any unhandled exception, job failure, critical-path failure                   |
| `DEBUG` | Full prompts, full tool-call responses (gated by `LOG_LEVEL=DEBUG`)           |

**Sensitive data handling**: API keys are never logged. Full search result bodies and full prompts are only emitted at `DEBUG`, never in prod. A `redact()` helper runs on every log payload.

**Per-job summary log** at job completion: total duration, per-stage duration, total tokens (in/out), total cost estimate, number of sub-tasks attempted/succeeded/failed, number of unique citations. This single record is the primary input for the `/research/{id}` status endpoint and for any downstream dashboarding.

**Correlation**: the `job_id` is returned in the `POST /research` response, included in every log line, and surfaced in error responses so users and operators can trace any failure end-to-end with one grep.

---

## 12. Testing Strategy

- **Unit tests**: mock `LLMClient` and `SearchClient`; verify each node's behavior in isolation.
- **Integration tests**: run the full graph with a fake LLM that returns scripted responses and a fake search that returns fixture pages. Assert the final .docx contains expected sections and citations.
- **Contract tests**: a tiny suite that hits real Groq + Tavily (gated by env var) to catch API drift.
- **Golden-file tests**: snapshot the generated outline JSON for a known query.

---

## 13. Resolved Decisions

| # | Question                       | Decision                                         |
| - | ------------------------------ | ------------------------------------------------ |
| 1 | Sub-agent execution model      | **Sequential** (one sub-task at a time)          |
| 2 | Review loop                    | **Human-in-the-loop** after `plan`, before `research` (no automated Lead self-review) |
| 3 | Groq model                     | **`llama-3.3-70b-versatile`** (env-overridable)  |
| 4 | Packaging                      | **`uv`**                                         |
| 5 | Job store                      | **In-memory** (v1; loses state on process restart — accepted) |
| 6 | Word document styling          | **Plain-but-clean** (no branding, no cover page) |
| 7 | API rate limiting              | **Required for v1** (`slowapi`)                  |
| 8 | Deployment target              | **Local only** (no Docker, no cloud for v1)      |
| 9 | LangGraph durable checkpointer | **Skipped for v1** (use `MemorySaver` for HITL interrupts only) |
| 10 | Cost ceiling per job          | **Yes** — hard abort at `MAX_JOB_COST_USD` (default `$1.00`) |
| 11 | Failed sub-tasks in output    | **"Research Limitations"** section listing gaps  |
| 12 | Query language                | **English only** for v1                          |
| 13 | Zero Tavily results           | **Mark sub-task failed** (`no_sources_found`)    |
| 14 | HITL review scope             | **Plan-only** — one gate between `plan` and `research`. No gate after `synthesize`. |
| 15 | HITL edit granularity         | **Full freedom** — user may add, remove, reorder, or rewrite sub-tasks. Edited plan is re-validated against the `SubTask` schema before resuming. |
| 16 | Cost estimate source          | **Hardcoded price table** at `services/pricing.py`, keyed by model id. Manually updated when Groq prices change; a comment at the top of the file records the last-updated date. |

All design questions resolved — ready for implementation on your go-ahead.

---

## 14. Suggested Build Order (once approved)

1. Scaffold project structure (`uv init`) + config + `structlog` logging.
2. Implement `LLMClient` (Groq) and `SearchClient` (Tavily) adapters with `tenacity` retries + unit tests.
3. Define domain models (`Citation`, `Finding`, `SubTask`, `ReportOutline`, `TokenUsage`).
4. Implement `CitationRegistry` service + tests.
5. Implement Lead prompts (`plan`, `synthesize`) and the researcher prompt.
6. Wire the LangGraph state machine with `MemorySaver` + HITL `interrupt()` node.
7. Implement `DocumentWriter` with python-docx (plain-but-clean template) + golden-file test.
8. Build FastAPI layer: job endpoints + HITL review endpoints + `slowapi` rate limits + in-memory `JobStore`.
9. Background runner that drives the graph and handles `interrupt`/`resume`.
10. Cost tracker + price table + hard-abort plumbing.
11. End-to-end integration test on a real query (with mocked providers, then one real run).
12. `.env.example`, README, `scripts/run_local.py`.

