# CLARA Project Timeline

**Purpose:** Track all features, updates, fixes, refactors, and enhancements chronologically. 
For tracing what changed, when, and why — not for motivation.

**Markers:**
- `[FEATURE]` — New capability added
- `[FIX]` — Bug fixed or reliability improved
- `[UPDATE]` — Existing feature modified or improved
- `[REFACTOR]` — Internal restructuring, no functional change
- `[ENHANCEMENT]` — Performance or efficiency improvement

---

## 2025-12-24

[FEATURE] Project initialization
Initial commit. Clean slate with CLI-only implementation (no Claude Code usage yet).

---

## 2026-02-04

[FEATURE] First functional agent + memory system
Core agent architecture with episodic logging and long-term memory vault working.
Added context retrieval for last 10 interactions. Memory persisted to JSON.

[FEATURE] Basic interface foundation
Initial React UI connected to FastAPI backend via WebSocket. Message sending/receiving working.

---

## 2026-02-05

[FEATURE] Fully functional agent + interface
Agent responding to user messages through UI. Core conversation loop complete.
Basic prompt routing (CHAT vs TASK mode) operational.

---

## 2026-02-08

[UPDATE] UI improvements
Added better visual design with improved transitions. Image functionality integrated for vision tasks.
First iteration of image analysis support.

---

## 2026-02-10

[UPDATE] Documentation and requirements
Updated README and requirements.txt to reflect current project state and dependencies.

---

## 2026-03-06

[FEATURE] Streaming responses
Implemented response streaming from backend to frontend. Changed interface dynamics to display
tokens as they arrive instead of waiting for full completion. Rewrote consolidation logic.

---

## 2026-03-08

[FIX] Consolidation logic
Fixed bug where system prompts were being included in memory consolidation, causing context pollution.

---

## 2026-03-10

[FEATURE] Gatekeeper with MiniLM + Phi-3 Mini
Replaced simple gatekeeper with semantic routing. MiniLM encodes queries, Phi-3 Mini makes 
routing decisions (CHAT vs TASK). First structured classifier added to system.

---

## 2026-03-11

[FIX] Gatekeeper reliability
Phi-3 Mini output parsing was failing (0% pass rate on XML output). Fixed structured output
reliability to achieve 100% pass rate on test cases.

---

## 2026-03-13

[UPDATE] Gatekeeper redesign
Complete rewrite of gatekeeper routing logic. Clara architecture documentation created as PNG diagram.
Shows major components and execution flow (now outdated).

---

## 2026-03-29

[FEATURE] Parallel tool batching
Implemented asyncio.gather() for parallel execution of multiple tools in single ReAct turn.
Tools can now be batched via JSON action format: `[{"tool": "X", ...}, {"tool": "Y", ...}]`

[FEATURE] Interface redesign
Major redesign of React UI. New layout with sidebar (identity), center (chat), right panel (neural stream).
Added visual indicators for execution mode, task board, thought stream.

---

## 2026-04-09

[FIX] MiniLM embedding issues
Fixed PyTorch/HuggingFace version incompatibility in embedding model. Model now loads without errors
on CUDA. Enabled episodic semantic retrieval to work reliably.

[UPDATE] Persistent browser memory
Added localStorage persistence for chat messages on frontend. Messages now survive page refresh.
Browser state no longer lost on reload.

[FEATURE] Quote feature
Added ability to highlight text in chat and quote it with `> [Clara]:` or `> [Alkama]:` prefix.
Improves conversation clarity when referencing previous messages.

---

## 2026-04-11 - 2026-04-12

[FEATURE] Autonomy foundation architecture (Briefs 0-12)
Multi-brief implementation week establishing the autonomous system foundation:

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

[FEATURE] Interpreter + Router (Brief 13)
Replaced old Gatekeeper. New architecture:
- Interpreter: Grok non-reasoning → structured intent JSON (tool, args, confidence, uncertainty, requires_planning)
- Router: Deterministic rules (confidence ≥ 0.75, uncertainty ≤ 0.30) → FAST or DELIBERATE
- FAST: Direct tool execution with Interpreter args, no LLM reasoning
- DELIBERATE: ReAct loop with reasoning for complex tasks
- FAST escalation: On failure, context injected into DELIBERATE for adaptation

[FEATURE] Grok Vision API integration (Brief 14)
Replaced Moondream2 with Grok Vision. Auto-detail-selection based on query intent.
Image compression (JPEG 85%, ≤1280px width) reduces payload 5-10×. Works with single or multi-image.

[FEATURE] Voice Phase 1 (Brief 15)
Foundation for voice I/O. Thin wrappers for STT (ears.py) and TTS (kokoro_mouth.py) added.
VoiceCoordinator not yet active. Infrastructure in place for future voice support.

[UPDATE] Repository cleanup
Removed ignored folders and refined directory structure. Updated core logic modules.

---

## 2026-04-15

[UPDATE] Vault synchronization (Brief 16.1)
Implemented vault write protection using threading.Lock. Prevents duplicate facts from concurrent
requests. Exact-match fast-path + cosine dedup (0.85 threshold) inside lock.

[UPDATE] Voice prerequisites (Brief 16.2)
Prepared groundwork for voice phase. System still using text, but infrastructure ready
for voice CoW-time integration.

[FIX] Chat latency optimization (Brief 16.3)
Switched CHAT path from grok-4-1-fast-reasoning to grok-4-1-fast-non-reasoning.
TTFT dropped from 3-8s to ~0.5s. Streaming now more responsive. Persona guardrails added
to prevent self-description, fabrication, and technical claims.

[UPDATE] Environment noise reduction (Brief 16.4)
Memory growth trigger threshold raised from 5 → 20 user-facing episodic entries.
Filters out [AUTONOMOUS], [TASK FAILED], [TASK RETRY] prefixed entries from memory threshold.

---

## 2026-04-16

[FEATURE] RAG knowledge base rebuild (Brief 17)
Implemented FAISS vector index for knowledge base. Indexes: CLAUDE.md, ROADMAP.md, core_logic/docs/
Auto-rebuild on file change via rag_rebuild event. Hot-reload via reload_rag_engine() without restart.
Chunk size 800 with 80 overlap, markdown-aware separators.

---

## 2026-04-17

[FEATURE] Archive context injection (Brief 18)
Passive retrieval: Before Interpreter, query is embedded with MiniLM. If cosine similarity ≥ 0.35
against FAISS chunks, top 3 results injected as [ARCHIVE CONTEXT]. Zero overhead if below threshold.
Complements active tool `consult_archive` for deeper searches.

---

## 2026-04-18

[UPDATE] Tool resolution strategy (Brief 19)
Defined routing for tool naming conflicts. fs_* tools remapped to Desktop Commander native names.
Tool discovery workflow: old name returns "not found" → FAST escalates to DELIBERATE → 
DELIBERATE calls tool_search → finds correct tool.

---

## 2026-04-22

[FEATURE] Tool Registry (Brief 21-A)
Central schema store for all tools. Native tools + MCP tools registered at startup.
ToolRegistry.search(q_emb, top_k=5) uses cosine similarity for semantic discovery.
MiniLM encodes all tool descriptions → (N, 384) tensor stored CPU-side.

[FEATURE] MCP Client (Brief 21-A)
Manages MCP server subprocesses via JSON-RPC over stdio. MCPClient.connect() performs handshake.
Serializes all calls with asyncio.Lock. Works with Desktop Commander + future servers.
Absolute paths required for Windows stdio stability (npx.cmd breaks pipe transport).

[FEATURE] Tool Registry integration (Brief 21-B)
Wired registry into request pipeline. Pre-Interpreter: `tool_registry.search(q_emb, top_k=5)` 
returns most relevant schemas. Appended as [DISCOVERED_TOOLS] in context. 
Interpreter sees top 5 tools for query, not all 33.

[FEATURE] Tool executor (Brief 21-B)
Unified dispatcher: execute_fast() and execute_deliberate() route to native Python or MCP.
Reads tool._server tag to decide dispatch target. Handles arg mapping from flat query string.

[FEATURE] tool_search native tool (Brief 21-B)
New tool in DELIBERATE ReAct loop. Query returns matching schemas via registry.search().
Enables dynamic tool discovery mid-task. Returns formatted schemas for subsequent calls.

---

## 2026-04-23

[FEATURE] Desktop Commander setup and testing (Brief 22)
Integrated Desktop Commander MCP server. Connected at startup via configured DC_NODE_PATH + DC_CLI_PATH.
24 DC tools registered. Full test suite passing: registry (7 native), MCP (26 DC), search, format, live.

[FIX] Unicode emoji encoding
Removed emojis from print statements in tools.py and crud.py. Windows console encoding (cp1252)
cannot render Unicode emojis — caused silent encoding failures and exception handling issues.

---

## 2026-04-24

[FIX] Tool Registry surgical fixes (Brief 23)
Fixed three bugs in tool discovery and validation:
1. Removed tool_search from NATIVE_TOOL_SCHEMAS — prevents it from appearing in [DISCOVERED_TOOLS] via semantic search
2. Made VALID_TOOLS dynamic in parse_action — built from registry.keys() at runtime, always includes tool_search, handles all MCP tools
3. Updated [SYSTEM MODE: TASK] injection — accurate description of 6 core tools + tool_search + [DISCOVERED_TOOLS]
4. Updated Rule 13 in system_prompt — corrected tool names (read_file, list_directory) with tool_search fallback guidance
Result: Filesystem queries no longer route to tool_search in Interpreter; DELIBERATE can still use it for dynamic discovery.

---

## 2026-04-24 (continued)

[FIX] Tool discovery quality + runtime bugs (Brief 24)
Seven bugs from session log analysis. Four fix groups:

Group A — Tool discovery quality (root cause of ranking failures):
- Added _clean_description() to ToolRegistry.register_server_tools() — strips DC boilerplate
  (\nIMPORTANT:, \nThis command can be referenced, etc.) so each tool embeds its actual function
- Increased top_k from 5 → 8 in both process_request (agent.py) and tool_search handler
  (tool_executor.py) — correct tool was frequently ranking 6-8 under top_k=5
- format_tool_schemas_for_context() truncates descriptions to 150 chars to keep token cost low

Group B — Multi-arg MCP tools (start_process timeout_ms missing):
- Added TOOL_ARG_DEFAULTS in _build_args_from_query() — fills timeout_ms for start_process
  (10000ms), read_process_output (5000ms), interact_with_process (8000ms) when not explicitly set

Group C — vision_tool None client crash:
- Added None guard at top of analyze_image_grok() in tools.py
- Added _xai_client_ref None guards in execute_fast() and execute_deliberate() in tool_executor.py

Group D — Orchestrator background task re-activation warning noise:
- system_trigger handler now silently skips tasks in completed/failed/invalidated state
  (normal for background tasks that complete and re-fire their scheduler)
- Only warns for tasks in unexpected non-pending states

Also added full [DISCOVERED_TOOLS] debug log to session logs (agent.py) — untruncated schema
dump after every pre-Interpreter search, enabling tool ranking diagnosis.

---

## 2026-04-24 (continued)

[FIX] ReAct integrity, discovery reliability, and runtime fixes (Brief 25)
Seven bugs from session log analysis. Five files changed:

Fix A — Hallucinated tool observations (Critical):
- DELIBERATE loop now detects model-fabricated Observations (model generates
  "Observation:" without calling a tool). Strips content, appends truncated assistant
  message, injects corrective system message, increments turn counter, and continues.
  Forces a real tool call on the next turn instead of reasoning from invented data.

Fix B — list_directory missing for enumeration queries:
- Added ENUMERATION_KEYWORDS check in process_request after cosine search.
  If query contains find/list/all/search/directory/folder/files etc., list_directory
  and start_search are guaranteed to appear in [DISCOVERED_TOOLS] regardless of cosine rank.

Fix C — FAST vision contaminated with episodic memory:
- format_llm in _run_fast now uses a vision-specific system prompt when tool=vision_tool.
  Instructs model to describe ONLY visual content from the result — no session history,
  no memory context. intent string (which carries memory context) not passed for vision calls.

Fix D — consult_archive misused for personal memory queries:
- Added personal memory routing rules to INTERPRETER_SYSTEM_PROMPT.
  Queries about remembered people/conversations → tool=null, answer from MEMORY_CONTEXT_BLOCK.
  consult_archive explicitly excluded from personal memory lookups.

Fix E — list_directory depth arg via comma format crashes:
- Added list_directory special-case in _build_args_from_query. Detects "path,depth" format
  before JSON parse, splits correctly. Added "list_directory: {depth: 2}" to TOOL_ARG_DEFAULTS.

Fix F — Concurrent user tasks run out of conversational order:
- _handle_user_input checks for running user tasks. If one exists, new task priority set
  to 0.95 (vs 1.0) — queues behind the running task. Background tasks unaffected.

Fix G — Orchestrator system_trigger log spam:
- Changed residual slog.warning to slog.debug for already-completed background task
  re-activation events. Message updated to "already completed (normal for background tasks)".

---

## 2026-04-25

[FIX] No-arg tool validation in DELIBERATE parser
- agent.py `_validate_actions()` now checks tool registry schema to determine if a tool
  requires arguments instead of hardcoding `date_time` as the only exception.
- Allows model to call no-arg tools like `list_searches`, `list_sessions`, `list_processes`,
  `get_usage_stats`, `give_feedback_to_desktop_commander` without providing empty query errors.
- Uses schema.inputSchema.required length: if empty, tool is no-arg and allows empty query.

[FIX] RAG knowledge base and Archive tool session logging
- Replaced all print() calls in rag_db_builder.py with slog calls (info/warning/debug/error).
- Replaced all print() calls in tools.py Archive context injection and RAG operations with slog.
- Added threading.Lock to RAG rebuilds to prevent duplicate loads at startup
  (concurrent calls from startup thread + EnvironmentWatcher race now serialized).

---

## Known Issues

- **RAG build incompatibility:** PyTorch/HuggingFace version mismatch causes "Cannot copy out of meta tensor" error
  at startup. Affects archive injection initialization but does not crash core functionality.
- **Voice system:** STT (ears.py) and TTS (kokoro_mouth.py) are thin wrappers. VoiceCoordinator not implemented.
- **Architecture diagram:** PNG diagram from Mar 13 is outdated. Current system includes Tool Registry, MCP Client, Desktop Commander.

---

## Statistics

- **Commits:** 25+ (from 2025-12-24 to 2026-04-24)
- **Briefs:** 24 (Brief 0 through Brief 23, numbered consecutively)
- **Native tools:** 6 (web_search, python_repl, date_time, vision_tool, consult_archive, query_task_status)
- **MCP tools (Desktop Commander):** 26
- **Total registry tools:** 32 (tool_search is injected to DELIBERATE, not registered)
- **Lines of code:** ~12K Python, ~4K JavaScript/React
- **Project duration:** ~5 months (Dec 2025 — Apr 2026)
