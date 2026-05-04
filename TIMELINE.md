# CLARA Project Timeline

## 2026-05-03

[FIX] Voice recording breaks after 4-5 prompts — persistent OutputStream replaces sd.play/wait/stop
Root cause: `sd.play()` + `sd.wait()` are global sounddevice calls. On Windows WASAPI, when
`sd.wait()` hung indefinitely (output stream held open after audio finished), the eventual
device state reset disrupted the mic `InputStream` callback, leaving `_audio_buf` empty on
subsequent recordings. `stop_recording()` returned None with no STT despite "Recording started"
being logged.
Fix: replaced `sd.play/wait/stop` entirely with a persistent `sd.OutputStream` (self._out_stream)
opened once at `load()` alongside the mic stream. Audio is written in 0.2s chunks via
`stream.write()` with `_stop_flag` checked between chunks. Interruption uses `stream.abort()`
scoped to the output stream only — mic InputStream is now completely isolated from TTS activity.
Also eliminated the `_waiter` daemon thread and all `sd.wait()` / deadline workarounds.

[FIX] TTS latency 5-6 seconds → ~200ms — Kokoro CUDA upgrade
Root cause: `kokoro-onnx` GPU detection is broken. It checks `importlib.util.find_spec("onnxruntime-gpu")`
which always returns None because hyphens are invalid in Python module names. Kokoro always ran
on CPU despite CUDA being available, causing ~3-5s synthesis per sentence.
Fix: after `Kokoro(onnx_path, voices_path)` initializes, replace `self._kokoro.sess` with a new
`ort.InferenceSession` built with `CUDAExecutionProvider`. Added ONNX warmup call at startup to
absorb JIT compilation cost (~2s once), so first real query synthesizes in ~200ms. Also added
sub-sentence splitting at clause boundaries (comma/semicolon/em-dash after 30 chars) so the first
TTS chunk is shorter and starts even faster.

[FIX] PTT keyup missed — voice_stop not sent after several queries
Root cause: F4 PTT effect in useClara.js had `[voiceActive, claraIsSpeaking]` as dependencies.
On keydown, `setVoiceActive(true)` triggered a React re-render which tore down and re-added event
listeners. If F4 was released during the brief listener-swap window, keyup fired with no handler
and `voice_stop` was never sent to the server — recording started but transcription never ran.
Fix: moved voiceActive and claraIsSpeaking to refs (voiceActiveRef, claraIsSpeakingRef) alongside
their state counterparts. PTT useEffect uses `[]` deps (single mount), handlers read from refs
which are always current. No listener teardown/re-add during the session.

## 2026-05-02

[FEATURE] Voice Phase 1 — Push-to-talk STT + TTS (Brief 15)
Implemented full voice I/O pipeline for CLARA:
- New `core_logic/voice.py` — VoiceCoordinator owning Faster-Whisper (medium.en, CUDA) and Kokoro ONNX TTS lifecycle. Push-to-talk STT with asyncio.to_thread transcription. Thread-safe speak() via _speak_lock. Event.wait-based playback instead of polling. Unbounded buffer guard (60s auto-stop). Module-level singleton (get_voice/set_voice).
- `api.py` — VoiceCoordinator wired into lifespan startup/shutdown (non-fatal on load failure). New `_broadcast()` helper consolidates dead-connection pruning for all WS broadcasts. `broadcast_task_event` and `_broadcast_speaking` both use it. WS receive loop handles voice_start/voice_stop/voice_interrupt message types. TTS response gated on `via_voice=True` — text-input responses never trigger TTS.
- `core_logic/orchestrator.py` — `on_interpreted` callback added to `submit_user_event` signature and injected into task context in `_handle_user_input`.
- `core_logic/agent.py` — `on_interpreted(interpreted, mode)` called immediately after routing decision in `process_request`.
- `interface/src/hooks/useClara.js` — `voiceActive` and `claraIsSpeaking` state. F4 push-to-talk key handler (keydown/keyup). speaking_start/speaking_stop WS message handlers.
- `interface/src/Layout.jsx` + `index.css` — Emerald waveform animation when CLARA speaks. Red recording indicator when F4 held.
- Restored `core_logic/agent.py` from git history after accidental deletion in d8fe364 (Cleanups commit).

---

**Purpose:** Track all features, updates, fixes, refactors, and enhancements chronologically. For tracing what changed, when, and why — not for motivation.

**Markers:**

- `[FEATURE]` — New capability added
- `[FIX]` — Bug fixed or reliability improved
- `[UPDATE]` — Existing feature modified or improved
- `[REFACTOR]` — Internal restructuring, no functional change
- `[ENHANCEMENT]` — Performance or efficiency improvement

---

## 2025-12-24

\[FEATURE\] Project initialization Initial commit. Clean slate with CLI-only implementation (no Claude Code usage yet).

---

## 2026-02-04

\[FEATURE\] First functional agent + memory system Core agent architecture with episodic logging and long-term memory vault working. Added context retrieval for last 10 interactions. Memory persisted to JSON.

\[FEATURE\] Basic interface foundation Initial React UI connected to FastAPI backend via WebSocket. Message sending/receiving working.

---

## 2026-02-05

\[FEATURE\] Fully functional agent + interface Agent responding to user messages through UI. Core conversation loop complete. Basic prompt routing (CHAT vs TASK mode) operational.

---

## 2026-02-08

\[UPDATE\] UI improvements Added better visual design with improved transitions. Image functionality integrated for vision tasks. First iteration of image analysis support.

---

## 2026-02-10

\[UPDATE\] Documentation and requirements Updated README and requirements.txt to reflect current project state and dependencies.

---

## 2026-03-06

\[FEATURE\] Streaming responses Implemented response streaming from backend to frontend. Changed interface dynamics to display tokens as they arrive instead of waiting for full completion. Rewrote consolidation logic.

---

## 2026-03-08

\[FIX\] Consolidation logic Fixed bug where system prompts were being included in memory consolidation, causing context pollution.

---

## 2026-03-10

\[FEATURE\] Gatekeeper with MiniLM + Phi-3 Mini Replaced simple gatekeeper with semantic routing. MiniLM encodes queries, Phi-3 Mini makes routing decisions (CHAT vs TASK). First structured classifier added to system.

---

## 2026-03-11

\[FIX\] Gatekeeper reliability Phi-3 Mini output parsing was failing (0% pass rate on XML output). Fixed structured output reliability to achieve 100% pass rate on test cases.

---

## 2026-03-13

\[UPDATE\] Gatekeeper redesign Complete rewrite of gatekeeper routing logic. Clara architecture documentation created as PNG diagram. Shows major components and execution flow (now outdated).

---

## 2026-03-29

\[FEATURE\] Parallel tool batching Implemented asyncio.gather() for parallel execution of multiple tools in single ReAct turn. Tools can now be batched via JSON action format: `[{"tool": "X", ...}, {"tool": "Y", ...}]`

\[FEATURE\] Interface redesign Major redesign of React UI. New layout with sidebar (identity), center (chat), right panel (neural stream). Added visual indicators for execution mode, task board, thought stream.

---

## 2026-04-09

\[FIX\] MiniLM embedding issues Fixed PyTorch/HuggingFace version incompatibility in embedding model. Model now loads without errors on CUDA. Enabled episodic semantic retrieval to work reliably.

\[UPDATE\] Persistent browser memory Added localStorage persistence for chat messages on frontend. Messages now survive page refresh. Browser state no longer lost on reload.

\[FEATURE\] Quote feature Added ability to highlight text in chat and quote it with `> [Clara]:` or `> [Alkama]:` prefix. Improves conversation clarity when referencing previous messages.

---

## 2026-04-11 - 2026-04-12

\[FEATURE\] Autonomy foundation architecture (Briefs 0-12) Multi-brief implementation week establishing the autonomous system foundation:

- **MiniLM thread safety:** Added asyncio.Lock around all encoding calls to prevent concurrent access issues
- **TaskGraph:** SQLite-backed task state machine with persistence and crash recovery
- **EventQueue:** Async priority queue for unified event ingestion from all sources
- **OrchestratorLoop:** Continuous decision engine that never sleeps, runs from startup
- **Interrupt model:** Ability to pause, resume, and interrupt running tasks
- **Background execution:** Parallel task execution while main loop continues
- **Conflict detection:** ConflictDetector identifies conflicts between tasks
- **Arbitration engine:** ArbitrationEngine resolves conflicts with priority + reversibility
- **Environmental awareness:** File watcher, memory growth monitoring, interaction density tracking
- **Boost removal:** Removed legacy "boost" pattern from system
- **Episodic logging:** Proper episodic entry creation with timestamps and summaries
- **Observability:** Session logs, benchmark logs, JSONL tracer events
- **MCP tools architecture:** Foundation for pluggable MCP servers
- **Task status awareness:** `query_task_status` tool for task graph introspection
- **Concurrent WebSocket:** Multiple simultaneous requests via message_id tagging

---

## 2026-04-13

\[FEATURE\] Interpreter + Router (Brief 13) Replaced old Gatekeeper. New architecture:

- Interpreter: Grok non-reasoning → structured intent JSON (tool, args, confidence, uncertainty, requires_planning)
- Router: Deterministic rules (confidence ≥ 0.75, uncertainty ≤ 0.30) → FAST or DELIBERATE
- FAST: Direct tool execution with Interpreter args, no LLM reasoning
- DELIBERATE: ReAct loop with reasoning for complex tasks
- FAST escalation: On failure, context injected into DELIBERATE for adaptation

\[FEATURE\] Grok Vision API integration (Brief 14) Replaced Moondream2 with Grok Vision. Auto-detail-selection based on query intent. Image compression (JPEG 85%, ≤1280px width) reduces payload 5-10×. Works with single or multi-image.

\[FEATURE\] Voice Phase 1 (Brief 15) Foundation for voice I/O. Thin wrappers for STT ([ears.py](http://ears.py)) and TTS (kokoro_mouth.py) added. VoiceCoordinator not yet active. Infrastructure in place for future voice support.

\[UPDATE\] Repository cleanup Removed ignored folders and refined directory structure. Updated core logic modules.

---

## 2026-04-15

\[UPDATE\] Vault synchronization (Brief 16.1) Implemented vault write protection using threading.Lock. Prevents duplicate facts from concurrent requests. Exact-match fast-path + cosine dedup (0.85 threshold) inside lock.

\[UPDATE\] Voice prerequisites (Brief 16.2) Prepared groundwork for voice phase. System still using text, but infrastructure ready for voice CoW-time integration.

\[FIX\] Chat latency optimization (Brief 16.3) Switched CHAT path from grok-4-1-fast-reasoning to grok-4-1-fast-non-reasoning. TTFT dropped from 3-8s to \~0.5s. Streaming now more responsive. Persona guardrails added to prevent self-description, fabrication, and technical claims.

\[UPDATE\] Environment noise reduction (Brief 16.4) Memory growth trigger threshold raised from 5 → 20 user-facing episodic entries. Filters out \[AUTONOMOUS\], \[TASK FAILED\], \[TASK RETRY\] prefixed entries from memory threshold.

---

## 2026-04-16

\[FEATURE\] RAG knowledge base rebuild (Brief 17) Implemented FAISS vector index for knowledge base. Indexes: [CLAUDE.md](http://CLAUDE.md), [ROADMAP.md](http://ROADMAP.md), core_logic/docs/ Auto-rebuild on file change via rag_rebuild event. Hot-reload via reload_rag_engine() without restart. Chunk size 800 with 80 overlap, markdown-aware separators.

---

## 2026-04-17

\[FEATURE\] Archive context injection (Brief 18) Passive retrieval: Before Interpreter, query is embedded with MiniLM. If cosine similarity ≥ 0.35 against FAISS chunks, top 3 results injected as \[ARCHIVE CONTEXT\]. Zero overhead if below threshold. Complements active tool `consult_archive` for deeper searches.

---

## 2026-04-18

\[UPDATE\] Tool resolution strategy (Brief 19) Defined routing for tool naming conflicts. fs\_\* tools remapped to Desktop Commander native names. Tool discovery workflow: old name returns "not found" → FAST escalates to DELIBERATE → DELIBERATE calls tool_search → finds correct tool.

---

## 2026-04-22

\[FEATURE\] Tool Registry (Brief 21-A) Central schema store for all tools. Native tools + MCP tools registered at startup. ToolRegistry.search(q_emb, top_k=5) uses cosine similarity for semantic discovery. MiniLM encodes all tool descriptions → (N, 384) tensor stored CPU-side.

\[FEATURE\] MCP Client (Brief 21-A) Manages MCP server subprocesses via JSON-RPC over stdio. MCPClient.connect() performs handshake. Serializes all calls with asyncio.Lock. Works with Desktop Commander + future servers. Absolute paths required for Windows stdio stability (npx.cmd breaks pipe transport).

\[FEATURE\] Tool Registry integration (Brief 21-B) Wired registry into request pipeline. Pre-Interpreter: `tool_registry.search(q_emb, top_k=5)`returns most relevant schemas. Appended as \[DISCOVERED_TOOLS\] in context. Interpreter sees top 5 tools for query, not all 33.

\[FEATURE\] Tool executor (Brief 21-B) Unified dispatcher: execute_fast() and execute_deliberate() route to native Python or MCP. Reads tool.\_server tag to decide dispatch target. Handles arg mapping from flat query string.

\[FEATURE\] tool_search native tool (Brief 21-B) New tool in DELIBERATE ReAct loop. Query returns matching schemas via registry.search(). Enables dynamic tool discovery mid-task. Returns formatted schemas for subsequent calls.

---

## 2026-04-23

\[FEATURE\] Desktop Commander setup and testing (Brief 22) Integrated Desktop Commander MCP server. Connected at startup via configured DC_NODE_PATH + DC_CLI_PATH. 24 DC tools registered. Full test suite passing: registry (7 native), MCP (26 DC), search, format, live.

\[FIX\] Unicode emoji encoding Removed emojis from print statements in [tools.py](http://tools.py) and [crud.py](http://crud.py). Windows console encoding (cp1252) cannot render Unicode emojis — caused silent encoding failures and exception handling issues.

---

## 2026-04-24

\[FIX\] Tool Registry surgical fixes (Brief 23) Fixed three bugs in tool discovery and validation:

1. Removed tool_search from NATIVE_TOOL_SCHEMAS — prevents it from appearing in \[DISCOVERED_TOOLS\] via semantic search
2. Made VALID_TOOLS dynamic in parse_action — built from registry.keys() at runtime, always includes tool_search, handles all MCP tools
3. Updated \[SYSTEM MODE: TASK\] injection — accurate description of 6 core tools + tool_search + \[DISCOVERED_TOOLS\]
4. Updated Rule 13 in system_prompt — corrected tool names (read_file, list_directory) with tool_search fallback guidance Result: Filesystem queries no longer route to tool_search in Interpreter; DELIBERATE can still use it for dynamic discovery.

---

## 2026-04-24 (continued)

\[FIX\] Tool discovery quality + runtime bugs (Brief 24) Seven bugs from session log analysis. Four fix groups:

Group A — Tool discovery quality (root cause of ranking failures):

- Added \_clean_description() to ToolRegistry.register_server_tools() — strips DC boilerplate (\\nIMPORTANT:, \\nThis command can be referenced, etc.) so each tool embeds its actual function
- Increased top_k from 5 → 8 in both process_request ([agent.py](http://agent.py)) and tool_search handler (tool_executor.py) — correct tool was frequently ranking 6-8 under top_k=5
- format_tool_schemas_for_context() truncates descriptions to 150 chars to keep token cost low

Group B — Multi-arg MCP tools (start_process timeout_ms missing):

- Added TOOL_ARG_DEFAULTS in \_build_args_from_query() — fills timeout_ms for start_process (10000ms), read_process_output (5000ms), interact_with_process (8000ms) when not explicitly set

Group C — vision_tool None client crash:

- Added None guard at top of analyze_image_grok() in [tools.py](http://tools.py)
- Added \_xai_client_ref None guards in execute_fast() and execute_deliberate() in tool_executor.py

Group D — Orchestrator background task re-activation warning noise:

- system_trigger handler now silently skips tasks in completed/failed/invalidated state (normal for background tasks that complete and re-fire their scheduler)
- Only warns for tasks in unexpected non-pending states

Also added full \[DISCOVERED_TOOLS\] debug log to session logs ([agent.py](http://agent.py)) — untruncated schema dump after every pre-Interpreter search, enabling tool ranking diagnosis.

---

## 2026-04-24 (continued)

\[FIX\] ReAct integrity, discovery reliability, and runtime fixes (Brief 25) Seven bugs from session log analysis. Five files changed:

Fix A — Hallucinated tool observations (Critical):

- DELIBERATE loop now detects model-fabricated Observations (model generates "Observation:" without calling a tool). Strips content, appends truncated assistant message, injects corrective system message, increments turn counter, and continues. Forces a real tool call on the next turn instead of reasoning from invented data.

Fix B — list_directory missing for enumeration queries:

- Added ENUMERATION_KEYWORDS check in process_request after cosine search. If query contains find/list/all/search/directory/folder/files etc., list_directory and start_search are guaranteed to appear in \[DISCOVERED_TOOLS\] regardless of cosine rank.

Fix C — FAST vision contaminated with episodic memory:

- format_llm in \_run_fast now uses a vision-specific system prompt when tool=vision_tool. Instructs model to describe ONLY visual content from the result — no session history, no memory context. intent string (which carries memory context) not passed for vision calls.

Fix D — consult_archive misused for personal memory queries:

- Added personal memory routing rules to INTERPRETER_SYSTEM_PROMPT. Queries about remembered people/conversations → tool=null, answer from MEMORY_CONTEXT_BLOCK. consult_archive explicitly excluded from personal memory lookups.

Fix E — list_directory depth arg via comma format crashes:

- Added list_directory special-case in \_build_args_from_query. Detects "path,depth" format before JSON parse, splits correctly. Added "list_directory: {depth: 2}" to TOOL_ARG_DEFAULTS.

Fix F — Concurrent user tasks run out of conversational order:

- \_handle_user_input checks for running user tasks. If one exists, new task priority set to 0.95 (vs 1.0) — queues behind the running task. Background tasks unaffected.

Fix G — Orchestrator system_trigger log spam:

- Changed residual slog.warning to slog.debug for already-completed background task re-activation events. Message updated to "already completed (normal for background tasks)".

---

## 2026-04-25

\[FIX\] No-arg tool validation in DELIBERATE parser

- [agent.py](http://agent.py) `_validate_actions()` now checks tool registry schema to determine if a tool requires arguments instead of hardcoding `date_time` as the only exception.
- Allows model to call no-arg tools like `list_searches`, `list_sessions`, `list_processes`, `get_usage_stats`, `give_feedback_to_desktop_commander` without providing empty query errors.
- Uses schema.inputSchema.required length: if empty, tool is no-arg and allows empty query.

\[FIX\] RAG knowledge base and Archive tool session logging

- Replaced all print() calls in rag_db_builder.py with slog calls (info/warning/debug/error).
- Replaced all print() calls in [tools.py](http://tools.py) Archive context injection and RAG operations with slog.
- Added threading.Lock to RAG rebuilds to prevent duplicate loads at startup (concurrent calls from startup thread + EnvironmentWatcher race now serialized).

\[FEATURE\] Token Usage Tracking (Brief 26)

- Added TokenUsage dataclass to [agent.py](http://agent.py) for accumulating tokens across all LLM calls.
- Captures usage from xAI SDK Response.usage on: Interpreter (non-reasoning), FAST format_llm, CHAT stream, and all DELIBERATE turns.
- Updated [interpreter.py](http://interpreter.py) to return (result, usage) tuple.
- Updated \_run_fast, \_run_chat, run_task to capture and return usage.
- Aggregates in process_request, logs to session as `>> [Tokens]` and emits WebSocket token_usage event.
- Bench logger now includes PROMPT, COMPLETION, TOTAL, CACHED columns (4 new tab-separated fields).
- Frontend useClara.js now tracks lastTokenUsage state from token_usage event.
- Layout.jsx displays token usage pill in Neural Stream showing total, in, out, and cached (in green).
- CSS styling for token-usage-pill, token-label, token-stat, token-cached, token-divider.
- Updated [CLAUDE.md](http://CLAUDE.md) with Token Usage Tracking section describing capture, emission, and backend/frontend behavior.
- Background tasks (source != "user") do not emit token events — only user requests.

\[FIX\] CSS dual-plugin conflict causing blank interface after Brief 26

- Root cause: `@tailwindcss/vite` plugin in vite.config.js AND `@tailwindcss/postcss` in postcss.config.js were both processing Tailwind simultaneously, corrupting the CSS output.
- Fix: Removed `@tailwindcss/postcss` from postcss.config.js (Vite plugin is the single source of truth).
- Secondary fix: index.css had `@import "tailwindcss"; @import "tailwindcss/preflight"` double import. Since `@import "tailwindcss"` already includes preflight, the second import was redundant and caused ordering issues. Reduced to single `@import "tailwindcss";`.
- Result: All interface styling (card borders, section backgrounds, input bar, sidebar cards) restored.

\[FIX\] on_step callback missing extra kwarg (Brief 26 token_usage emission)

- [api.py](http://api.py) handle_message's inner `on_step` function only accepted (content, type, turn_id).
- process_request calls on_step_update with extra=token_usage.to_dict() for token_usage events.
- Added `extra=None` parameter and forwarded it to `send_update()` — fixes TypeError that would have silently dropped token_usage WebSocket events on every user request.

\[FIX\] Token tracking accuracy — FAST escalation and total_tokens derivation

- FAST→DELIBERATE escalation path silently returned deliberate_usage_list as fast_usage (a list). Aggregation code did `token_usage.add("fast_execution", list)` — getattr on a list returns 0, so all deliberate turn tokens from escalated FAST requests were counted as zero. Fixed: isinstance(fast_usage, list) check routes escalation tokens through the deliberate loop.
- total_tokens was re-derived as p+c instead of reading SDK's usage.total_tokens field. Fixed: reads total_tokens from the usage object directly, falls back to p+c only if absent.

---

## Known Issues

- **RAG build incompatibility:** PyTorch/HuggingFace version mismatch causes "Cannot copy out of meta tensor" error at startup. Affects archive injection initialization but does not crash core functionality.
- **Voice system:** STT ([ears.py](http://ears.py)) and TTS (kokoro_mouth.py) are thin wrappers. VoiceCoordinator not implemented.
- **Architecture diagram:** PNG diagram from Mar 13 is outdated. Current system includes Tool Registry, MCP Client, Desktop Commander.

---

## 2026-04-27 (continued)

[FIX] DELIBERATE named-param actions failing for single-required-param tools
- Root cause: `_validate_actions` in agent.py only extracted `item.get("query")` for
  single-arg tools. When TEMP_SYSTEM_PROMPT taught the model correct named params
  (e.g. `{"tool": "python_repl", "code": "..."}` instead of `{"tool": "python_repl", "query": "..."}`),
  the query came back empty → "Empty query" error → tool skipped silently.
- Fix 1 (agent.py `_validate_actions`): detect named params via `any(k not in ("tool", "query") for k in item)`.
  When present, serialize full item as JSON — same path the multi-arg branch already used.
  Flat-query tools still use `item.get("query")` unchanged. No-arg tools unchanged.
- Fix 2 (tool_executor.py `execute_deliberate`): added `_extract_param()` helper that
  JSON-parses the query string and extracts the right field by name. Updated all native
  tool handlers: python_repl extracts "code", web_search/consult_archive extract "query",
  query_task_status extracts "keyword", vision_tool extracts "path"+"question".
  Each falls back to raw query string if JSON parse fails (backward compatible).

[UPDATE] DELIBERATE system prompt experiment (TEMP_SYSTEM_PROMPT)
- Removed static tool list from SYSTEM_PROMPT section.
- Replaced with tool_search JSON schema block as the sole tool anchor.
- Replaced 5 concrete examples with 1 pseudo-example showing tool_search → call flow.
- Updated rules to remove specific tool name references (python_repl → "code execution", etc).
- Batching example now uses `<tool_a>`/`<tool_b>` placeholders instead of real tool names.
- [SYSTEM MODE: TASK] user message stripped to single line — system prompt covers the rest.
- Plugged into agent.py as `self.system_prompt = TEMP_SYSTEM_PROMPT` for testing.

---

## 2026-04-28

[FIX] TOOL_ARG_DEFAULTS not applied in JSON parse path (tool_executor.py)
- `TOOL_ARG_DEFAULTS` block was positioned after the JSON early-return, so when the model
  explicitly passed JSON args, defaults were never injected.
- Hoisted `TOOL_ARG_DEFAULTS` dict above the JSON parse branch and applied it inside the
  JSON branch too: fills missing args only, never overwrites explicit values.
- Affected tools: write_file (mode), start_process (timeout_ms), read_process_output (timeout_ms),
  interact_with_process (timeout_ms), list_directory (depth).

[FIX] write_file "w" mode normalization (tool_executor.py)
- Model generates `"mode": "w"` from Python training priors. Desktop Commander requires
  `"rewrite"` | `"append"` — `"w"` causes a silent rejection.
- Added `TOOL_ARG_NORMALIZERS` dict in `_build_args_from_query`. Applied after JSON parse:
  maps `"w"` → `"rewrite"`, `"a"` → `"append"`. Explicit override — fires even when model
  provides the arg, unlike defaults which only fill missing values.

[FIX] Hallucination handler double-increment (agent.py)
- When hallucination detected, the handler did `turn_count += 1` before `continue`, then
  the loop's own `turn_count += 1` fired on the next iteration. Net: 2 turns burned per
  hallucination event (Loop 1 → Loop 3).
- Removed the extra `turn_count += 1` from the hallucination handler. Detection now burns
  exactly 1 turn.

[FIX] Observation → Glint rename — DELIBERATE loop coin token (agent.py)
- Coined custom token "Glint" to replace "Observation" everywhere in the ReAct loop.
- Root cause: "Observation:" is a strongly learned bigram in ReAct training data. Model
  pattern-completes `Action: [...]\nObservation:` from prior, hallucinating tool results
  before the tool actually runs.
- "Glint" has zero training prior as a ReAct token — hallucination pressure near zero.
- Changes in agent.py:
  - Regex in thought extraction: `Observation` → `Glint`
  - Hallucination detector: checks for `Glint:` without preceding `Action:`, corrects with
    updated message referencing Glint
  - Loop variables: `observations` → `glints`, `obs` → `glint`, `combined_observation` → `combined_glints`
  - Log strings: `Obs:` → `Glint:`, `[Observation]` → `[Glint]`
  - Comment: "Feed all observations" → "Feed all Glints"
- Changes in system_prompt.py (both SYSTEM_PROMPT and TEMP_SYSTEM_PROMPT):
  - All `Observation:` lines in execution loop format → `Glint:`
  - `Trust observations.` → `Trust Glints.`
  - `After each observation:` → `After each Glint:`
- Changes in tool_registry.py: `format_tool_schemas_for_observation` → `format_tool_schemas_for_glint`
- Changes in tool_executor.py: import and call updated to `format_tool_schemas_for_glint`

[FIX] Inline hallucination detector — Action + fabricated Glint in same turn (agent.py)
- Stress test (20 queries, 2026-04-28) revealed 4/5 hallucination failures shared one pattern:
  model writes `Action: [...]` then immediately generates fake `Glint:` content in the same
  token stream, before the system executes anything. The existing detector condition
  (`"Glint:" in content and "Action:" not in content`) missed all of these because Action WAS present.
- Added `elif "Glint:" in raw_content and "Action:" in raw_content` branch:
  strips `raw_content` from the first `Glint:` onward, sets `inline_hallucination = True`,
  falls through to real `parse_actions()` on the truncated (clean) response.
- After appending the truncated assistant message, injects a corrective user message:
  "fabricated Glint was discarded — your Action is being executed now, wait for the real Glint."
- Model then receives the real system-generated Glint and continues correctly.
- Does not increment turn count — hallucination costs 0 extra turns on this path.

[UPDATE] DELIBERATE loop quality improvements (agent.py, system_prompt.py)
- Rule 4 replaced with structured error taxonomy: Recoverable / Tool-not-found /
  Genuinely-impossible. Prevents model from treating recoverable errors as dead ends.
- Thought description rewritten in both prompts: explicit guidance for post-Glint reasoning,
  post-failure classification, and pre-Final-Answer completion check.
- Rule 11 tightened: "Once all sub-tasks are resolved — write Final Answer immediately."
- Rule 16 added to both prompts: COMPLETION CHECK before Final Answer.
- `_turn_message()` helper added in agent.py: prefixes every Glint message with `[Turn N/8]`.
  On final turn, appends `[FINAL TURN]` wrap-up instruction — forces conclusion instead of
  burning the last turn on another tool call.
- Turn budget initializer: `llm.append(user("[SYSTEM MODE: TASK] [Turn 1/8] ..."))`
- `last_response_text` tracked per-turn: fallback return gives the user the last model output
  instead of a canned "I ran out of steps" message.

[FIX] Interpreter routing — write_file with generated content (interpreter.py)
- Q18 retest: Interpreter routed "Draft a new class in core_logic/proactive_commit.py" as
  CHAT (tool=None, requires_planning=False, confidence=0.95). Root cause: routing guidance
  treated write_file as single-step regardless of whether content exists in the query or must
  be composed. "Draft" + file path → Interpreter assigned write_file but content was a placeholder
  → FAST path couldn't generate code → fell through to CHAT.
- Added rule to routing guidance: write_file where content must be GENERATED (code, structured
  text, class drafts, analysis) → requires_planning=true even if the path is clear. Generating
  content is always multi-step: compose first, then write. write_file where content IS the query
  (e.g. "write 'hello' to file.txt") → requires_planning=false as before.

[UPDATE] list_directory depth guidance in Rule 13 (system_prompt.py)
- Added explicit depth guidance to Rule 13: omit depth or use 0 by default — immediate contents
  only, no chunk risk. Only use depth > 0 when subdirectory structure is explicitly needed AND
  directory is known to be sparse. Dense directories (__pycache__, model weights, indexes) will
  overflow at depth > 0. Rule 4 chunk-limit handles recovery when it happens.
- Addresses scale concern: as project grows, more directories become dense. Model now has
  explicit in-context guidance rather than relying on training priors for depth selection.

---

## Statistics

- **Commits:** 25+ (from 2025-12-24 to 2026-04-24)
- **Briefs:** 24 (Brief 0 through Brief 23, numbered consecutively)
---

## 2026-04-29

[FIX] list_directory chunk-limit — root cause in system prompt examples (system_prompt.py, tool_executor.py)
- Stress test Q18 root cause traced: `depth: 1` was not in DC's list_directory schema at all
  (schema only exposes `path`). CLARA learned it from the Rule 14 example in SYSTEM_PROMPT:
  `Action: [{"tool": "list_directory", "path": "...", "depth": 1}]` — model pattern-matched
  from its own prompt. DC accepts unknown params silently; depth=1 descends into __pycache__,
  models/, knowledge_base/, moondream_brain/ and overflows the stdio buffer on dense directories.
  The read_file chunk error in the same batch was collateral — DC's framing corrupted by the
  oversized list_directory response, not by the file being absent.
- Fix 1 (system_prompt.py): Removed `depth: 1` from both list_directory examples in
  SYSTEM_PROMPT (Rule 14 correct example + Examples section). Model no longer learns this pattern.
- Fix 2 (system_prompt.py): Added chunk-limit as a fourth error class in Rule 4 ERROR CLASSIFICATION
  in both SYSTEM_PROMPT and TEMP_SYSTEM_PROMPT. When "chunk exceed the limit" appears, CLARA
  retries the same tool on the same path with reduced scope (omit depth, narrower subpath, or
  specific filename) — not by changing paths or treating it as a missing file.
- Fix 3 (tool_executor.py): `TOOL_ARG_DEFAULTS` had `list_directory: {depth: 2}` — worse than
  the prompt example, silently injecting depth=2 whenever the model omitted depth entirely.
  Changed to `depth: 0`. Model can still pass an explicit depth when genuinely needed.
- Capability preserved: CLARA can still use depth when genuinely needed. The fix is behavioral
  (learn from the right example, recover correctly from chunk-limit) not a hard gate.

[REFACTOR] TEMP_SYSTEM_PROMPT promoted to SYSTEM_PROMPT (system_prompt.py, agent.py)
- Stress test (20 queries, 2026-04-28) ran entirely on TEMP_SYSTEM_PROMPT and validated it.
- TEMP_SYSTEM_PROMPT is structurally better: no hardcoded tool list (can't go stale), no
  project-specific path examples, tool_search schema injected inline, cleaner rules throughout.
- Old SYSTEM_PROMPT deleted. TEMP_SYSTEM_PROMPT renamed to SYSTEM_PROMPT in system_prompt.py.
- agent.py import and self.system_prompt reference updated accordingly.
- TEMP_SYSTEM_PROMPT no longer exists as a separate variable.

[UPDATE] Rule 13 search-first pattern — filesystem resolution (system_prompt.py)
- Old Rule 13: always list_directory first to confirm a path exists.
- New Rule 13: when given a filename, use start_search first — confirms existence and returns
  exact path in one call, no chunk-limit risk. Only fall back to list_directory (no depth) if
  search returns nothing, to check for typos or casing in the parent directory.
- list_directory is no longer the first move for named file resolution.

- **Native tools:** 6 (web_search, python_repl, date_time, vision_tool, consult_archive, query_task_status)
- **MCP tools (Desktop Commander):** 26
- **Total registry tools:** 32 (tool_search is injected to DELIBERATE, not registered)
- **Lines of code:** \~12K Python, \~4K JavaScript/React
- **Project duration:** \~5 months (Dec 2025 — Apr 2026)
