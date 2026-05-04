# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## Timeline Tracking (Global Instruction)

**IMPORTANT:** Every feature, fix, update, refactor, or enhancement must be logged to `timeline.md`.

**When:** After implementation is complete and tested.

**How:** Add entry following this format:
```
## YYYY-MM-DD

[FEATURE|FIX|UPDATE|REFACTOR|ENHANCEMENT] Title
One or more lines describing what changed and why.
Include relevant brief numbers, affected modules, and key behavioral changes.
```

**Guidelines:**
- Use only 5 markers: `[FEATURE]`, `[FIX]`, `[UPDATE]`, `[REFACTOR]`, `[ENHANCEMENT]`
- Group multiple entries on same date under one date header
- Be specific and factual — this is a trace log, not marketing material
- Include affected files/modules if significant
- Reference brief numbers (e.g., "Brief 22") when applicable
- Multi-line descriptions are encouraged for clarity
- No fabrication — if unclear what changed, describe what you verified

**Timeline file:** `timeline.md` (top-level, tracks all project history)

---

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
| Voice | `core_logic/voice.py` | VoiceCoordinator — Whisper STT, Kokoro TTS, PTT, acknowledgments (active) |

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

### Persona Guardrails (Brief 16.3 + Session eval 2026-04-16)
Five guardrails in PERSONA's "How you speak" block:
- Never narrate own architecture (no websocket/memory/routing self-description)
- More detail = more substance about Alkama's world, not more words about self
- End statements with statements — questions only when genuinely needed
- Personal history = memory context only. If no relevant episode exists, say so. Do not construct illustrative incidents.
- Technical self-claims must be architecturally true. Do not describe capabilities or features that don't exist.

System prompt is injected **after** routing — FAST gets no system prompt on `llm` (consolidation only).

---

## Concurrent WebSocket (Brief 12)

Each incoming message gets a `message_id`. The handler fires `asyncio.create_task(handle_message(...))` and immediately loops back to `receive_text()`. Multiple requests can be in-flight simultaneously. Responses are tagged with `message_id` for frontend attribution.

`active_connections: set` tracks live WebSocket connections for broadcasting.

**send_update guard:** `send_update()` checks `websocket.client_state != WebSocketState.CONNECTED` before attempting any send. If the client has disconnected mid-stream, the function returns immediately. Without this guard, a disconnect during a long DELIBERATE response causes repeated "Cannot call send once a close message has been sent" errors on every subsequent streaming token. `WebSocketState` is imported at module top-level (`from starlette.websockets import WebSocketState`) — not inside the hot-path function. Disconnect events are logged at `DEBUG` level, not `ERROR`.

---

## Autonomous System

### Background Scheduler
- `health_check` — every 2 minutes
- `memory_maintenance` — every 5 minutes  
- `context_warmup` — every 10 minutes

### Environment Watcher
Triggers: `file_change`, `memory_growth`, `interaction_density`, `rag_rebuild`
All trigger via EventQueue as system-origin tasks.

`file_change` has a **5-second per-path debounce** (`_last_file_change` dict in `EnvironmentWatcher`).
Rapid saves to the same file (e.g., 3 saves in 5s during a coding session) emit exactly 1 event.
Debounce is per-path — two different files changed within 5s both trigger independently.

`rag_rebuild` fires instead of `file_change` when the changed path is a RAG source file
(`CLAUDE.md`, `ROADMAP.md`, or any file in `core_logic/docs/`). Triggers full knowledge base
rebuild in background thread + hot-reload of the in-memory FAISS engine.

Watched paths: `core_logic/`, `CLAUDE.md`, `briefs/ROADMAP.md`.

### User Task Serialization (Brief 25)
When a new `user_input` event arrives while a user-origin task is already in `running` state, the new task's priority is set to `0.95` (vs the default `1.0`). The dispatch loop skips it until the first task completes. Background tasks (origin=system) are unaffected and continue running in parallel. On the next tick after the running task completes, the 0.95 task dispatches normally. This ensures user responses arrive in conversational order.

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

### Token Usage Tracking (Brief 26)
Every user request accumulates token usage across all LLM calls:

- **Interpreter call** (always present) — `prompt_tokens`, `completion_tokens`
- **FAST path** — format_llm `sample()` call (single call)
- **CHAT path** — streaming `chat()` call (single stream)
- **DELIBERATE path** — one `stream()` call per ReAct turn, all summed

Usage captured from xAI SDK `Response.usage` object via response object attached to each completion:

- `prompt_tokens` — input tokens
- `completion_tokens` — output tokens
- `total_tokens` — sum
- `cached_prompt_text_tokens` — tokens served from cache (not billed at full rate)

**Emission:**

- Logged to session log: `>> [Tokens] total=N prompt=P completion=C cached=K` after every user request
- Emitted as `token_usage` WebSocket event with `extra` field containing full breakdown dict
- Logged to `benchmarks/bench_YYYY-MM-DD.log` with 4 new columns: PROMPT, COMPLETION, TOTAL, CACHED

**Frontend display:**

- Neural Stream panel shows a token usage pill after each completed response
- Format: "Last query · total tokens · prompt in · completion out · \[cached in green if &gt; 0\]"

**Background tasks:** System-origin requests (health_check, memory_maintenance, etc.) do NOT emit token events — only user-origin requests. This prevents noise in the Neural Stream during autonomous operation.

---

## Vision Tool

Moondream2 (`core_logic/sight.py`) is replaced by Grok Vision API (`analyze_image_grok` in [tools.py](http://tools.py)). `sight.py` is kept but not imported. Vision calls use a disposable non-reasoning Grok instance. The `xai_client_ref` is injected at startup via `set_xai_client(clara.client)` in [api.py](http://api.py).

---

## Voice System (Brief 15)

`core_logic/voice.py` — `VoiceCoordinator` singleton, loaded at startup via `api.py` lifespan.

### Audio Architecture
Two persistent streams opened once at `load()` and closed at `unload()`:
- `self._in_stream` — `sd.InputStream` (mic, 16kHz, callback-based, always open)
- `self._out_stream` — `sd.OutputStream` (speaker, 24kHz, persistent)

**Critical:** No `sd.play()` / `sd.wait()` / `sd.stop()` global calls anywhere. Those disrupt the
mic InputStream on Windows WASAPI via device resets. All playback uses `self._out_stream.write(chunk)`
in 0.2s chunks with `_stop_flag` checked between each. Interruption uses `stream.abort()` + `stream.start()`.

### STT
Faster-Whisper `medium.en` on CUDA. Push-to-talk: `start_recording()` on F4 down, `stop_recording_async()` on F4 up. Audio buffered in `_audio_buf` via callback, written to temp WAV, transcribed, temp file deleted. Returns None on silence.

### TTS
Kokoro ONNX v0.19. `kokoro-onnx` has a broken GPU detection bug (`find_spec("onnxruntime-gpu")` always returns None — hyphens are invalid in Python module names). Fixed at startup by replacing `self._kokoro.sess` with a `CUDAExecutionProvider` InferenceSession directly. First-call ONNX JIT absorbed by a warmup synthesis at startup. Result: ~200ms first-audio latency.

Pipelined playback: synthesizer thread fills `audio_q` (maxsize=3), playback loop writes chunks to `_out_stream`. First audio starts after first-sentence synthesis only (~200ms on CUDA). Long responses stream sentence-by-sentence.

First sentence is sub-split at clause boundary (comma/semicolon/em-dash after 30 chars) so the first synthesis chunk is short and audio starts faster.

### Push-to-Talk (Frontend)
F4 held = record, F4 released = transcribe. If Clara is speaking when F4 pressed = interrupt.
PTT `useEffect` in `useClara.js` has `[]` deps (single mount). Handlers read `voiceActiveRef` and
`claraIsSpeakingRef` (not state) — avoids listener teardown/re-add race that dropped `voice_stop` messages.

### Acknowledgments
Fired via `on_interpreted` callback immediately after routing, before execution:
- FAST + tool: "On it." (non-blocking)
- DELIBERATE (confidence ≥ 0.75): "Give me a moment."
- DELIBERATE (low confidence): "This will take a moment."
- CHAT or FAST without tool: no ack

### WS Message Types (voice-related)
Frontend → Backend: `voice_start`, `voice_stop` (with `message_id`), `voice_interrupt`
Backend → Frontend: `user_transcript` (STT result), `speaking_start`, `speaking_stop`

---

## WebSocket Message Protocol

Backend sends:

- `"thought"` — internal reasoning (neural stream panel, keyed by `turn_id`)
- `"stream"` — response tokens
- `"status"` — system status updates
- `"final_answer"` — complete response with `message_id`
- `"speaking_start"` / `"speaking_stop"` — voice waveform animation (fires when Kokoro TTS is playing)
- `"user_transcript"` — STT result from Whisper, displayed as User bubble in chat

---

## Conventions

### LLM Models in Use

- `grok-4-1-fast-non-reasoning` — Interpreter, format_llm (FAST), **CHAT stream**, memory consolidation
- `grok-4-1-fast-reasoning` — DELIBERATE ReAct loop only

CHAT was switched from reasoning → non-reasoning (Brief 16.3). TTFT drops from 3-8s → \~0.5s. Reasoning is reserved for DELIBERATE where the ReAct loop quality justifies the cost.

### Action Format (DELIBERATE)

```
Action: [{"tool": "tool_name", "query": "input"}]
```

Multiple tools batched in one array = parallel execution via `asyncio.gather`. Parser in `parse_actions()`: 3-layer (direct JSON, bracket-counting, old-format fallback).

### LLM Instance Pattern

`process_request` creates a local `llm` variable per request (not `self.llm`). This isolates concurrent requests — each request has its own conversation context. `self.llm` is a legacy reference kept only for the CLI `run()` path; `process_request` never touches it. `_run_fast`, `_run_chat`, and `run_task` all accept and use the passed `llm` parameter. `run_task` falls back to `self.llm` only if `llm=None` (legacy CLI path).

### Frontend — Interface Redesign (Brief 18)

Full rewrite of `interface/src/Layout.jsx`, `index.css`, `hooks/useClara.js`.

**Zone A (Sidebar):** Identity block, Active Context (live from tasks), skills matrix, animated vitals bars (CPU%/RAM%/VRAM%). Scanline texture overlay.

**Zone B (Chat):** Spring animation on message arrival. CLARA gradient bubbles with glow. Three-dot breathing pre-stream state. Empty state ambient ring. Hover timestamps + copy. Mode chip in header (FAST/CHAT/DELIBERATE). Send button active glow.

**Zone C (Neural Stream):** Split Task Board (top) + Thought Stream (bottom). Task cards with state colors, priority bars, enter/exit animations, shake on failure. Thought stream scoped to latest, older entries dimmed.

**Backend:** `broadcast_task_event()` in [api.py](http://api.py). `_broadcast_task()` in orchestrator fires on pending/running/completed/failed. Soul endpoint now returns cpu%, VRAM GB, version.

**useClara.js:** `tasks` state array, pruned 2s after completion/failure.

### Frontend

- No StrictMode — causes double WebSocket connections
- Dark theme, emerald (`#10b981`) accent
- Three-panel: sidebar (identity/vitals), center (chat), right (neural stream)
- Messages persisted to localStorage (`clara_messages`)
- WebSocket reconnects with exponential backoff (1s → 30s cap)
- Quote feature: highlight text → QUOTE button → injects `> [Clara]:` or `> [Alkama]:` prefix

### Tool Registry + MCP Client (Briefs 21-A, 21-B)

**New modules:**

- `core_logic/tool_registry.py` — central schema store for all tools (native + MCP)
- `core_logic/mcp_client.py` — subprocess lifecycle and JSON-RPC for MCP servers
- `core_logic/tool_executor.py` — unified dispatch replacing the two duplicate blocks

**ToolRegistry lifecycle:**

1. `register_native_tools()` at startup — 6 native tools (web_search, python_repl, date_time, vision_tool, consult_archive, query_task_status)
2. `register_server_tools(server_name, schemas)` after each MCP handshake
3. `rebuild_embeddings(agent._encode)` after all registrations — MiniLM encodes all tool descriptions → (N, 384) CPU tensor
4. `search(q_emb_cpu, top_k=8)` at query time — cosine similarity returns top-k schemas

**MCPClient:** Manages stdio JSON-RPC subprocesses. One server per connection. `connect()` performs MCP handshake (initialize → initialized notification → tools/list). `call()` dispatches tool with direct `await` — never use asyncio.to_thread for async MCP calls.

**Desktop Commander:** Connected at startup via `DC_NODE_PATH` + `DC_CLI_PATH` in `.env`. Uses absolute node.exe + cli.js paths (npx.cmd breaks Windows stdio). Provides 24 tools registered under server name "desktop_commander".

**Pre-Interpreter injection:** Before every `interpret()` call, `tool_registry.search(q_emb_cpu, top_k=8)` runs and top-8 schemas are appended to context under `[DISCOVERED_TOOLS]` tag. Interpreter uses these for accurate tool names and args.

**Mandatory injection (Brief 25):** After cosine top-8 search, `process_request` checks `ENUMERATION_KEYWORDS` (find, list, all, search, what files, directory, folder, etc.) against the query. If matched, `list_directory` and `start_search` schemas are appended to `discovered` if not already present. Max discovered set grows to top_k + 2. Prevents cosine similarity from missing enumeration tools when queries describe the target (image files) rather than the operation (list directory).

**DC description cleaning (Brief 24):** MCP tool descriptions are cleaned at registration time in `register_server_tools()`. Boilerplate (`\nIMPORTANT:`, `\nThis command can be referenced`, etc.) is stripped so each DC tool's embedding reflects its actual function rather than shared boilerplate. Raw descriptions from the MCP handshake are not retained. `format_tool_schemas_for_context()` truncates to 150 chars in the injected context to keep token cost low.

**TOOL_ARG_DEFAULTS (Brief 24):** `_build_args_from_query()` in `tool_executor.py` applies default values for known multi-required-arg DC tools after mapping the primary arg: `start_process → timeout_ms: 10000`, `read_process_output → timeout_ms: 5000`, `interact_with_process → timeout_ms: 8000`, `list_directory → depth: 0`. Only fills args not already present — never overwrites explicit values. `list_directory` default is 0 (immediate contents only) — prevents silent chunk-limit overflow when the model omits the depth arg on dense directories.

**tool_search architectural note (Brief 23):** `tool_search` is NOT in the tool registry and is NOT embedded or returned by `registry.search()`. It is injected directly into the DELIBERATE \[SYSTEM MODE: TASK\] prompt. This prevents it from appearing in \[DISCOVERED_TOOLS\] and being mistakenly selected by the Interpreter as an action tool for arbitrary queries. `VALID_TOOLS` in `parse_action` is built dynamically from the registry at call time (| {"tool_search"}) so all MCP tools are always valid without manual maintenance. DELIBERATE can always call `tool_search` to discover filesystem/process/MCP capabilities by semantic query.

**Tool executor routing:**

- `execute_fast(tool_name, args_dict, ...)` — FAST path, structured args from Interpreter
- `execute_deliberate(tool_name, query_str, ...)` — DELIBERATE path, flat string from ReAct Action
- Both routes to native Python functions or `await mcp_client.call(server, tool, args)` based on `registry.get_server(tool_name)`
- `_build_args_from_query` maps flat DELIBERATE query string to MCP tool's required args (transitional until Pattern B streaming migration)

*fs\_ tool names:*\* Changed from `fs_read_file`/`fs_write_file`/`fs_list_directory`/`fs_run_command` to DC's native names (`read_file`, `write_file`, `list_directory`, `start_process`). Old names return "not found" → FAST escalates → DELIBERATE calls tool_search → finds correct DC tool.

**Scaling:** Every new MCP server: `mcp_client.connect()` → `tool_registry.register_server_tools()` → `tool_registry.rebuild_embeddings()`. Zero changes to TOOL_ARG_SCHEMAS or system prompt.

**Config:** `core_logic/.env` requires `DC_NODE_PATH` and `DC_CLI_PATH`. Registry + MCP init in `api.py` lifespan after `orchestrator._broadcast_fn` injection. `clara.tool_registry` and `clara.mcp_client` injected after `rebuild_embeddings()` completes.

### ReAct Loop Format Enforcement

Rules 11-16 in SYSTEM_PROMPT:

- Rule 11: After a Glint that answers the question, next output MUST be Thought → Final Answer. No prose dumps, no markdown headers before Final Answer.
- Rule 12: Never simulate or fabricate metrics/statistics/telemetry. python_repl must not be used to generate random numbers presented as real data.
- Rule 13: FILESYSTEM RESOLUTION — when given a filename, use `start_search` first (confirms existence + returns exact path in one call, no chunk-limit risk). Only fall back to `list_directory` (no depth) if search returns nothing. Never use `list_directory` as the first move for a named file. `list_directory` depth: omit or use 0 by default — immediate contents only. Only use depth > 0 when subdirectory structure is explicitly needed AND the directory is known to be sparse. Dense directories (`__pycache__`, model weights, indexes) overflow at depth > 0.
- Rule 14: ACTION FORMAT IS MANDATORY — every Action must be a valid JSON array. No markdown, no prose, no code fences. A malformatted Action cannot be parsed and wastes the turn.
- Rule 15: TOOL DISCOVERY — for filesystem, process, or MCP-backed operations not in the core tools list, call tool_search first with a semantic query. Use returned schemas exactly. One retry with refined query allowed; do not repeat the same query.
- Rule 16: COMPLETION CHECK — before writing Final Answer, Thought must confirm every sub-task is complete or genuinely impossible. Partial results do not constitute a complete answer.

**Chunk-limit error class (Rule 4):** When a tool returns "chunk exceed the limit" or "Separator is not found", the response is too large for the stdio transport. Recovery: retry the SAME tool on the SAME path with reduced scope — omit depth, use a narrower subpath, or read a specific file by name instead of listing a directory. Do NOT change the path or assume the error means the file/directory doesn't exist.

Safety net in run_task: if a turn contains no Thought/Action/Final Answer markers but has content, it is treated as an implicit Final Answer and returned immediately. Prevents the loop from burning remaining turns on idle noise after an off-format response.

**Hallucination detection (two forms):**

1. **Bare Glint** — model emits a `Glint:` line without a preceding Action. Loop detects it, strips fabricated content, appends truncated assistant message, injects corrective system message ("Glints can ONLY come from actual tool execution"), increments turn counter, and `continue`s — forcing a real tool call on the next turn.

2. **Inline fabrication** — model writes `Action: [...]` then immediately writes a fabricated `Glint:` in the same turn (before the system executes anything). Loop splits on `"Glint:"`, keeps only the `pre_glint` portion (the real Action), appends it as the assistant turn, injects a corrective message, and `continue`s — the system then executes the Action normally on the next turn.

`pre_glint` is computed once before the if/elif/else branch to avoid duplication. Turn budget applies to both cases. The custom `Glint:` token (replacing "Observation") reduces hallucination pressure from training bigrams on the word "Observation".

### Vision Tool Improvements

`analyze_image_grok` in [tools.py](http://tools.py):

- Auto-selects detail level: questions containing "read", "text", "code", "exact" etc → "high"; all others → "low". 3-4× faster for layout/visual queries.
- Compresses images to JPEG 85% quality, resizes to ≤1280px wide before encoding. \~5-10× smaller payload, saves 2-5s on network round-trip. Requires Pillow (falls back to raw bytes if unavailable).
- detail="auto" is the new default (was "high").

### Response Style Persistence

When Alkama says "you're too verbose" or "give more detail", `memorize_episode` extracts a `style_update` field from the consolidation output and writes to `user_profile.preferences`:

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

### Interpreter write_file Routing
`write_file` where content must be **generated** (code, structured text, analysis, class drafts) → `requires_planning=true`, even if the path is clear. Generating content is always multi-step: compose first, then write.
`write_file` where content IS the query (e.g. "write 'hello world' to file.txt") → `requires_planning=false`.

Without this distinction the Interpreter routes content-generation tasks to CHAT, which has no tool call capability — the file is never written.

### Interpreter Personal Memory Routing (Brief 25)
Questions about people Alkama has mentioned, past conversations, or anything phrased as "do you remember X" / "did I tell you about X" → `tool=null, requires_planning=false`. The answer lives in `[MEMORY_CONTEXT_BLOCK]` already injected. `consult_archive` is explicitly excluded from personal memory lookups — it searches FAISS-indexed documentation (CLAUDE.md, ROADMAP.md, resume), not conversation history.

### Interpreter web_search Guidance
`web_search` is only assigned when the answer requires live or post-training data:
- Current prices, rates, scores, news, events after mid-2025
- Anything explicitly marked "latest", "current", "today", "now"

NOT assigned for stable knowledge answerable from training data:
- Historical facts, scientific concepts, definitions, capitals
- Well-established technical knowledge (Python, algorithms, best practices)
- Explanations, creative tasks, reasoning, analysis

Rule of thumb: if the answer could have been in a textbook 5 years ago, do not search.
This was added after session eval 2026-04-16 showed Q1 (Australia capital) and Q4 (Python
mistakes) were routed to web_search unnecessarily, adding ~5s latency with no benefit.

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

### Archive Context Injection (Brief 18)

`get_archive_context()` in `tools.py` runs before the Interpreter on every request.
Uses the same MiniLM embedding already computed for episodic retrieval — zero extra encode calls.
If FAISS cosine similarity ≥ 0.35 against any chunk, top 3 chunks are appended to context
under `[ARCHIVE CONTEXT]` tag. Below threshold → empty string, no injection, no overhead.

This runs in `asyncio.to_thread` (FAISS search is CPU-bound, ~<10ms).
Both the Interpreter and the LLM (via `MEMORY_CONTEXT_BLOCK`) receive the full context.

`consult_archive` tool still exists and coexists — passive injection handles the common case,
explicit tool call handles deeper digs in DELIBERATE.

### RAG Knowledge Base (Brief 17)

`consult_archive` tool uses a FAISS vector index built by `core_logic/rag_db_builder.py`.

**Indexed sources:**
- `core_logic/docs/` — all `.pdf`, `.md`, `.txt`, `.py` files (resume and any future docs)
- `CLAUDE.md` — current architecture reference (always included)
- `briefs/ROADMAP.md` — implementation history and status (always included)

**Build behavior:**
- Full rebuild every time — incremental FAISS updates are fragile at this scale
- Runs at startup via `lifespan` in `api.py` (non-blocking, `asyncio.to_thread`)
- Auto-rebuild triggered by `rag_rebuild` event when any source file changes
- Hot-reload via `reload_rag_engine()` in `tools.py` — updates the global `RAG_ENGINE` in place
  without restarting the server

**Chunk settings:** `chunk_size=800`, `chunk_overlap=80`, markdown-aware separators
(`\n## `, `\n### `, `\n\n`, `\n`).

To add a new permanent document: drop it into `core_logic/docs/` and restart (or wait for
auto-rebuild if the file lands in a watched path).

### Files That Are Dead / Legacy
- `core_logic/sight.py` — Moondream2, replaced by Grok Vision, no longer imported
- `core_logic/tool_descriptions.json` — was MiniLM embedding source, Interpreter replaced this role
- `core_logic/ears.py` — superseded by `core_logic/voice.py`, no longer imported
- `core_logic/kokoro_mouth.py` — superseded by `core_logic/voice.py`, no longer imported
- Architecture PNG (`Clara_Architecture_Fixed_And_Updated.png`) — outdated, does not reflect current system

### Branch
All work on `features/stream-and-functionality`. Never merge to main until full system validated.
