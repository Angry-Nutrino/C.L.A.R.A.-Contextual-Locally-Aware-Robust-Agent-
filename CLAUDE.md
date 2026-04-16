# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Development Commands

### Backend (FastAPI + WebSocket)
```bash
# Activate venv
jarvis_v2\Scripts\activate     # Windows

# Run API server (port 8001)
python api.py
```

### Frontend (React + Vite)
```bash
cd interface
npm install
npm run dev      # Dev server on port 5173
npm run build    # Production build
```

### Tests
```bash
python -m pytest tests/test_brief13.py -v
python -m pytest tests/ -v
```

### Environment Variables (core_logic/.env)
- `XAI_API_KEY` — xAI Grok API key (all LLM calls)
- `tavily_api` — Tavily API key (web search tool)

---

## Architecture Overview

**C.L.A.R.A.** (Contextual Locally Aware Robust Agent) — a personal autonomous AI system.
Consumer hardware: RTX 3050 Mobile, 4GB VRAM. All orchestration is custom-built.

### Execution Pipeline (unified — all sources)
```
ANY INPUT (user / system trigger / background / environment)
    ↓
EventQueue (async priority queue)
    ↓
OrchestratorLoop
    ↓
Interpreter (Grok non-reasoning) → structured intent JSON
    ↓
Router → FAST / CHAT / DELIBERATE
    ↓
Execution → response
    ↓
memorize_episode (background thread)
```

**No bypasses allowed.** All tasks regardless of origin go through the same pipeline.

### Three Execution Modes

| Mode | Trigger | LLM calls | Latency |
|------|---------|-----------|---------|
| FAST | tool known, confidence ≥ 0.75, uncertainty ≤ 0.30, no planning | Interpreter (non-reasoning) + format_llm (non-reasoning) | ~2-4s |
| CHAT | tool=null, confidence ≥ 0.75, uncertainty ≤ 0.30, no planning | Interpreter (non-reasoning) + direct stream (**non-reasoning**) | ~1.5-2.5s |
| DELIBERATE | requires_planning=true OR low confidence OR FAST failed | Interpreter + ReAct loop (reasoning, max 8 turns) | ~5-30s |

FAST escalates to DELIBERATE on failure, injecting failure context as an assistant block.
CHAT streams directly via `_run_chat()` — no ReAct loop, no tool calls.

### Key Modules

| Module | Path | Role |
|--------|------|------|
| API server | `api.py` | FastAPI + concurrent WebSocket (fire-and-forget, message_id) |
| Agent | `core_logic/agent.py` | process_request, route(), _run_fast, _run_chat, run_task |
| Interpreter | `core_logic/interpreter.py` | Grok non-reasoning → structured intent JSON |
| Orchestrator | `core_logic/orchestrator.py` | OrchestratorLoop, task dispatch, retry architecture |
| TaskGraph | `core_logic/task_graph.py` | SQLite-backed task state machine with crash recovery |
| EventQueue | `core_logic/event_queue.py` | Async priority queue, drain_blocking=0.1s |
| Memory CRUD | `core_logic/crud.py` | get_smart_context, episodic log, vault |
| System prompt | `core_logic/system_prompt.py` | PERSONA + CHAT_SYSTEM_PROMPT + SYSTEM_PROMPT |
| Tools | `core_logic/tools.py` | All tool implementations + Grok vision |
| Background tasks | `core_logic/background_tasks.py` | health_check, memory_maintenance, context_warmup |
| Environment watcher | `core_logic/environment.py` | File watch, memory growth, interaction density triggers |
| Conflict | `core_logic/conflict.py` | ConflictDetector + ArbitrationEngine |
| Tracer | `core_logic/tracer.py` | JSONL observability (orchestrator_tick events) |
| Session logger | `core_logic/session_logger.py` | Per-session timestamped logs in logs/ |
| Bench logger | `core_logic/bench_logger.py` | Per-request latency log in benchmarks/ |
| STT | `core_logic/ears.py` | Thin wrapper — VoiceCoordinator not yet active |
| TTS | `core_logic/kokoro_mouth.py` | Thin wrapper — VoiceCoordinator not yet active |

---

## Interpreter + Router (Brief 13)

The Interpreter replaced the old Gatekeeper (MiniLM + Phi3 + boost pattern).

### Interpreter Output Schema
```json
{
  "intent": "string",
  "tool": "tool_name or null",
  "args": {},
  "confidence": 0.0-1.0,
  "uncertainty": 0.0-1.0,
  "requires_planning": true/false
}
```

### Route Logic
```python
if confidence >= 0.75 and uncertainty <= 0.30 and requires_planning == False:
    if tool is not None → FAST
    if tool is None    → CHAT
else:
    DELIBERATE
```

### FAST Failure Escalation
When FAST fails, before calling run_task():
- Tool attempted, args, error, and any partial result are injected into `llm` as an assistant block
- DELIBERATE sees what was tried and adapts — does not repeat the same failed approach

---

## Memory System

Stored in `core_logic/memory.json`:
- **Episodic log** — interaction summaries with timestamps, written after every request
- **Long-term vault** — permanent facts, deduplicated at 0.85 cosine similarity threshold
- **User profile** — name, role, preferences

### Smart Context Retrieval (`get_smart_context`)
- Filters out `[AUTONOMOUS]`, `[TASK FAILED]`, `[TASK RETRY]` prefixed entries entirely
- Returns: last 3 **user-facing** episodic entries (recency) + top 2 semantic hits (MiniLM)
- Vault always included
- Deduplicates via set union — max ~5 episodic entries total
- Injected as assistant message with `[MEMORY_CONTEXT_BLOCK]` tags

### Memory Consolidation
Runs in `asyncio.to_thread` after every response (never blocks main path):
- Disposable non-reasoning Grok instance extracts `summary` + `facts`
- New episodic embedding encoded with MiniLM and appended to `episodic_embeddings` list (CPU)
- Chat snapshot filters out `[MEMORY_CONTEXT_BLOCK]` to prevent circular contamination

### Vault Dedup (threading.Lock)
`_vault_lock` (threading.Lock) wraps all vault writes inside `memorize_episode`.
This prevents the race condition where two concurrent requests both read an empty vault,
pass the cosine check independently, and write the same fact twice.
Dedup order: (1) exact string match fast-path, (2) cosine similarity ≥ 0.85.
Both checks happen inside the lock against the live vault state.
`add_long_term_fact()` in `crud.py` also has an exact string guard as a second layer of defence.

### Vault Fact Qualifications
The consolidation prompt only extracts facts that are **truly permanent**:
- Personal attributes (name, relationship, confirmed preference, personality trait)
- Stable project decisions or architectural constraints
- Real-world facts about people/places that won't change

**Excluded from vault:** file paths, file counts, file sizes, screenshot metadata,
directory listings, timestamps, tool outputs, anything time-sensitive or transient.

### Episodic Embeddings Sync
Every entry written to `episodic_log` must have a corresponding embedding in
`episodic_embeddings` — otherwise `get_smart_context()` disables semantic retrieval.

- **User entries** (via `memorize_episode`): encoded with MiniLM, appended to list.
- **System/autonomous entries** (via `log_system_episode` in `agent.py`): a zero-vector
  (384-dim) is appended. Zero vectors are never retrieved (system entries are filtered
  by `[AUTONOMOUS]`/`[TASK *]` prefix) but maintain array index alignment.
- All `add_episodic_log()` calls in `orchestrator.py` have been replaced with
  `self._agent.log_system_episode()` to enforce this invariant.
- `_context_warmup` in `background_tasks.py` self-repairs if drift is ever detected:
  re-encodes all summaries and replaces `episodic_embeddings` entirely.

### Memory Growth Trigger
`EnvironmentWatcher.check_memory_growth()` only counts **user-facing** episodic entries
toward the threshold (excludes `[AUTONOMOUS]`, `[TASK *]` prefixed entries).
Threshold raised from 5 → 20 to reduce background noise.

### MiniLM Usage
MiniLM (`all-MiniLM-L6-v2`) is kept ONLY for episodic embedding similarity in `get_smart_context`.
It no longer has any routing role. Encodes on CUDA, stored CPU-side.

---

## Persona System

Defined in `core_logic/system_prompt.py`:

```python
PERSONA            # Shared identity — injected into ALL three paths
CHAT_SYSTEM_PROMPT # PERSONA + minimal chat operational line
SYSTEM_PROMPT      # PERSONA + full DELIBERATE operational block
```

**FAST path:** `format_llm` gets `PERSONA + "Format the tool result into a natural response."`
**CHAT path:** `llm` gets `CHAT_SYSTEM_PROMPT`
**DELIBERATE path:** `llm` gets `SYSTEM_PROMPT` (PERSONA + ReAct tools/format/examples)

### Persona Guardrails (Brief 16.3)
Three guardrails added to PERSONA's "How you speak" block:
- Never narrate own architecture (no websocket/memory/routing self-description)
- More detail = more substance about Alkama's world, not more words about self
- End statements with statements — questions only when genuinely needed

System prompt is injected **after** routing — FAST gets no system prompt on `llm` (consolidation only).

---

## Concurrent WebSocket (Brief 12)

Each incoming message gets a `message_id`. The handler fires `asyncio.create_task(handle_message(...))` and immediately loops back to `receive_text()`. Multiple requests can be in-flight simultaneously. Responses are tagged with `message_id` for frontend attribution.

`active_connections: set` tracks live WebSocket connections for broadcasting.

---

## Autonomous System

### Background Scheduler
- `health_check` — every 2 minutes
- `memory_maintenance` — every 5 minutes  
- `context_warmup` — every 10 minutes

### Environment Watcher
Triggers: `file_change`, `memory_growth`, `interaction_density`
All trigger via EventQueue as system-origin tasks.

`file_change` has a **5-second per-path debounce** (`_last_file_change` dict in `EnvironmentWatcher`).
Rapid saves to the same file (e.g., 3 saves in 5s during a coding session) emit exactly 1 event.
Debounce is per-path — two different files changed within 5s both trigger independently.

### SIMPLE_TRIGGERS
Known lightweight system tasks bypass the Interpreter and go directly to `run_background_task`.
Unknown/complex system tasks go through the full Interpreter → Router → Execution pipeline.

### Retry Architecture
`MAX_ATTEMPTS = 3`. On failure: summarize failure context, create new task with
`failure_summary` in context. At max attempts: resolve future with failure message, log to episodic.

---

## Observability

- **Session logs:** `logs/session_YYYY-MM-DD_HH-MM-SS.log` — every request, full response text
  - `>> [FAST] Response:` — full FAST response
  - `>> [CHAT] Response:` — full CHAT response
  - `>> [DELIBERATE] Final Answer:` — full DELIBERATE final answer
  - `>> [MEMORY_CONTEXT] Injecting into Grok:` — full memory context (file only, not console)
- **Bench log:** `benchmarks/bench_YYYY-MM-DD.log` — TOTAL_MS, INTERP_MS, EXEC_MS per request
- **Tracer:** `traces/trace_*.jsonl` — orchestrator_tick JSONL events

**Important:** Session logs before `session_2026-04-14_12-46-14.log` do NOT contain full
response text — they only have memory consolidation summaries. Full response logging was
added on 2026-04-14. Use only logs from that date onward for persona assessment.

---

## Vision Tool

Moondream2 (`core_logic/sight.py`) is replaced by Grok Vision API (`analyze_image_grok` in tools.py).
`sight.py` is kept but not imported. Vision calls use a disposable non-reasoning Grok instance.
The `xai_client_ref` is injected at startup via `set_xai_client(clara.client)` in api.py.

---

## WebSocket Message Protocol

Backend sends:
- `"thought"` — internal reasoning (neural stream panel, keyed by `turn_id`)
- `"stream"` — response tokens
- `"status"` — system status updates
- `"final_answer"` — complete response with `message_id`
- `"speaking_start"` / `"speaking_stop"` — voice waveform animation (voice phase, not yet active)

---

## Conventions

### LLM Models in Use
- `grok-4-1-fast-non-reasoning` — Interpreter, format_llm (FAST), **CHAT stream**, memory consolidation
- `grok-4-1-fast-reasoning` — DELIBERATE ReAct loop only

CHAT was switched from reasoning → non-reasoning (Brief 16.3). TTFT drops from 3-8s → ~0.5s.
Reasoning is reserved for DELIBERATE where the ReAct loop quality justifies the cost.

### Action Format (DELIBERATE)
```
Action: [{"tool": "tool_name", "query": "input"}]
```
Multiple tools batched in one array = parallel execution via `asyncio.gather`.
Parser in `parse_actions()`: 3-layer (direct JSON, bracket-counting, old-format fallback).

### LLM Instance Pattern
`process_request` creates a local `llm` variable per request (not `self.llm`).
This isolates concurrent requests — each request has its own conversation context.
`self.llm` is a legacy reference kept only for the CLI `run()` path; `process_request` never touches it.
`_run_fast`, `_run_chat`, and `run_task` all accept and use the passed `llm` parameter.
`run_task` falls back to `self.llm` only if `llm=None` (legacy CLI path).

### Frontend
- React 19 + Vite + Tailwind CSS 4
- No StrictMode — causes double WebSocket connections
- Dark theme, emerald (`#10b981`) accent
- Three-panel: sidebar (identity/vitals), center (chat), right (neural stream)
- Messages persisted to localStorage (`clara_messages`)
- WebSocket reconnects with exponential backoff (1s → 30s cap)
- Quote feature: highlight text → QUOTE button → injects `> [Clara]:` or `> [Alkama]:` prefix

### Response Style Persistence
When Alkama says "you're too verbose" or "give more detail", `memorize_episode` extracts a
`style_update` field from the consolidation output and writes to `user_profile.preferences`:
- `response_style`: "concise" | "detailed" | "default"
- `style_note`: brief reason string

`get_smart_context` injects `RESPONSE STYLE: concise (reason)` into every context when
non-default. This reaches all three paths. Updates persist in `memory.json` until changed.

### File System Awareness
`user_profile.environment.known_locations` in `memory.json` holds key path mappings.
`get_smart_context` injects a `[KNOWN LOCATIONS]` block into every context string.
Add entries manually to `memory.json` when new paths need to be known. Format:
```json
"environment": {
  "known_locations": {
    "Screenshots": "C:\\Users\\alkam\\OneDrive\\Pictures\\Screenshots",
    "AGENT_ZERO (Clara)": "E:\\ML PROJECTS\\AGENT_ZERO"
  }
}
```

### Interpreter Logging
Full raw JSON output now logged: `>> [Interpreter] Raw output:\n{full_json}`
Parsed summary: `>> [Interpreter] Parsed → tool=X | confidence=X | uncertainty=X | requires_planning=X | intent=X`
Use these to diagnose routing decisions.

### Vault Write Protection
`_vault_lock = threading.Lock()` in `Clara_Agent.__init__`. The entire vault write block in
`memorize_episode` runs inside `with self._vault_lock`. Re-reads `existing_facts` fresh inside
the lock. Exact string equality fast-path before cosine check prevents concurrent duplicate writes.

### Vault Facts Criteria
Only extract as permanent facts:
- Personal attributes of Alkama (name, relationship, confirmed preference, personality trait)
- Stable project decisions or architectural constraints
- Real-world facts about a person/place/thing that won't change
- Something Alkama explicitly stated as a standing preference or rule

Never extract: file paths, counts, sizes, screenshot metadata, timestamps, tool outputs,
anything stale within days or weeks.

### Files That Are Dead / Legacy
- `core_logic/sight.py` — Moondream2, replaced by Grok Vision, no longer imported
- `core_logic/tool_descriptions.json` — was MiniLM embedding source, Interpreter replaced this role
- `core_logic/ears.py` — thin wrapper only, VoiceCoordinator not yet implemented
- `core_logic/kokoro_mouth.py` — thin wrapper only, VoiceCoordinator not yet implemented
- Architecture PNG (`Clara_Architecture_Fixed_And_Updated.png`) — outdated, does not reflect current system

### Branch
All work on `features/stream-and-functionality`. Never merge to main until full system validated.
