# Research Brief: MCP Tool Ecosystem for CLARA

---

## Topic Name

MCP Server Ecosystem Assessment and Integration Roadmap for CLARA —
Auditing Existing Tool Robustness and Identifying Capability Gaps
Against the Ambient Autonomous Agent Vision.

---

## Research Objective

Determine which of CLARA's current tools have meaningful quality ceilings
that MCP servers could raise, identify capability domains entirely absent
from CLARA's current stack that are required for the ambient agent vision,
and produce a ranked, hardware-constrained shortlist of MCP servers —
with RAM footprint per server — that are worth integrating into
CLARA's Orchestrator/tools layer on an RTX 3050 mobile system with
~1.5–2.0 GB RAM headroom during active inference.

---

## Current Implementation

CLARA's tool layer is a custom Python implementation in `core_logic/tools.py`.
Tools are registered in `core_logic/tool_descriptions.json` and dispatched
through three paths: FAST (Interpreter-directed, single tool, no loop),
DELIBERATE (ReAct loop, multi-turn, parallel batching via asyncio.gather),
and background (Orchestrator system tasks).

Tools currently available to CLARA:

### 1. `web_search` — Tavily API
- `TavilyClient.search(query, include_answer="advanced", search_depth="advanced", max_results=2)`
- Returns a Tavily-synthesized `answer` string only; raw results array discarded
- Single query per call; no multi-query, no freshness filter, no news routing
- Fails silently if `answer` key absent (returns "No results found" even with results)
- New TavilyClient instantiated on every call (no connection reuse)

### 2. `python_repl` — bare exec()
- Captures stdout via StringIO redirect
- No timeout, no memory limit, no package isolation
- Stateless by design — all imports and variables reset each turn
- No persistent REPL session, no structured output format

### 3. `fs_read_file` — pathlib.read_text()
- UTF-8 with error replacement, 10,000 char hard truncation
- No encoding detection, no binary support, no partial read with offset

### 4. `fs_list_directory` — pathlib.iterdir()
- Single-level listing only, no recursive option
- No filtering by extension, no size/date sort, no glob support

### 5. `fs_write_file` — pathlib.write_text()
- Full overwrite only; no append mode, no atomic write
- No backup before overwrite

### 6. `fs_run_command` — subprocess.run() via PowerShell
- 30s timeout, combined stdout+stderr, 5,000 char truncation
- No structured output, no environment variable injection, no working dir control

### 7. `consult_archive` — FAISS + MiniLM (LangChain)
- Indexed sources: core_logic/docs/, CLAUDE.md, ROADMAP.md
- Chunk size 800, overlap 80, k=3 results
- Passive injection also active (threshold 0.35) before every Interpreter call
- Cannot index arbitrary URLs or live documents

### 8. `vision_tool` — Grok Vision API (analyze_image_grok)
- Local file path input, JPEG compression to <=1280px at 85% quality
- Auto-selects detail level (high for text/OCR queries, low otherwise)
- No webcam, no screen capture, no PDF-page rendering

### 9. `date_time` — datetime.now()
- Returns raw datetime string, no timezone handling, no scheduling

### 10. `query_task_status` — TaskGraph + SQLite read
- In-memory tasks + last 50 from SQLite, keyword-match filter
- Read-only; cannot modify task state

---

## Current Metrics

| Tool | Latency (observed) | Reliability | Robustness Notes |
|---|---|---|---|
| web_search | 800ms–2s per call | Good | answer key absence causes silent failure |
| python_repl | <100ms (simple) | Good | No timeout; hangs on blocking code |
| fs_read_file | <10ms | Good | 10K char limit cuts large files |
| fs_list_directory | <10ms | Good | Single level only |
| fs_write_file | <10ms | Good | No atomicity |
| fs_run_command | Variable, 30s max | Acceptable | No structured output |
| consult_archive | 50–150ms | Good | Limited to indexed static docs |
| vision_tool | 1–3s (API) | Good | File path only; no live capture |
| date_time | <1ms | Perfect | No scheduling awareness |
| query_task_status | &lt;20ms | Good | Read-only |

Overall RAM footprint of current tool layer at idle: \~200–300 MB (FAISS index + MiniLM CPU instance in [tools.py](http://tools.py) + LangChain embeddings).

---

## Gap Analysis: What CLARA Cannot Currently Do

The following capability domains are entirely absent from CLARA's current tool stack. Each is required to meaningfully progress toward the ambient autonomous agent vision described in [ROADMAP.md](http://ROADMAP.md) and OPUS_EVAL_REPORT.md Section 8.

### GAP-1: Browser / Web Interaction

CLARA can search via Tavily but cannot navigate URLs, click, scroll, fill forms, authenticate into web services, or scrape structured data from live pages. This blocks any task that requires interacting with a web app rather than just retrieving a fact.

### GAP-2: Communication Layer
No email read/write, no calendar read/write, no messaging integration.
The ROADMAP explicitly lists WhatsApp/external event integration as a
long-term goal. Currently zero infrastructure exists for this.

### GAP-3: Structured Data / Database
No SQL query capability beyond the TaskGraph internal SQLite.
No CSV/tabular data processing beyond what python_repl can do ad-hoc.
No persistent key-value store accessible as a tool.

### GAP-4: Development / Codebase Awareness
No Git integration — CLARA cannot check branch state, read diffs,
stage commits, or run tests programmatically. Given CLARA assists with
her own development, this is a notable self-referential gap.
No code search (grep-level is available via fs_run_command but not
semantic/structured), no test runner, no linter access.

### GAP-5: Search Quality Ceiling
Tavily is a good general-purpose search tool but has a hard ceiling: no raw web access, no news-specific routing, no semantic search over the open web, no financial/data API access, no academic paper search. For the ambient vision, CLARA needs higher-fidelity information retrieval.

### GAP-6: Screen / Environment Observation

CLARA is blind to what is happening on the screen unless an image is explicitly sent. No screenshot-and-analyze loop, no clipboard access, no notification observation. Required for true ambient awareness. Partial coverage exists: `user_profile.environment.known_locations` in memory.json holds key path mappings injected into every context as `[KNOWN LOCATIONS]` — but this is manually maintained and does not observe live screen or system state.

### GAP-7: Memory / Knowledge Mutation as a Tool

CLARA can read her memory passively but has no tool-callable interface to deliberately update the vault, update user profile fields, mark episodes as important, or prune stale entries. Memory can only be mutated through the background consolidation pipeline. Note: `known_locations` can be manually edited in memory.json but is not callable as a tool from within task execution.

---

## Constraints

### Hardware

- CPU: AMD Ryzen 7 4800H (8 cores / 16 threads), \~2.9 GHz base
- RAM: 15.4 GB total. At full working load (VS Code + CLARA + Brave + Chrome + Spotify): \~13.7 GB consumed. Headroom: \~1.5–2.0 GB.
- GPU: NVIDIA RTX 3050 Laptop, 4 GB GDDR6 VRAM (128-bit bus)
  - At inference: MiniLM ~90 MB + Grok API (cloud, no local VRAM cost)
  - Voice phase will add: Faster-Whisper ~800 MB + Kokoro ~300 MB
  - Post-voice VRAM headroom: ~2.8 GB for vision/other
- Storage: E: drive comfortable (309 GB free); C: tight (67 GB free)
- OS: Windows 11, PowerShell default shell

### MCP Server RAM Budget
Each MCP server is a Node.js or Python sidecar process.
Typical idle footprint: 50–150 MB. Active (tool call in flight): 200–400 MB.
With 1.5–2.0 GB RAM headroom:
  Safe ceiling: 8–10 servers at idle simultaneously.
  Practical rule: count by weight, not headcount.
  Lightweight (fs, git, time, sqlite): ~50 MB each.
  Heavy (browser automation, DB connection pools): ~200–400 MB each.
  A browser automation server (Playwright) alone consumes 300–500 MB.

### Integration Constraints
- All tools must be callable from CLARA's Orchestrator via the existing
  tool dispatch pattern in agent.py (_execute_fast_tool and execute_tool).
- MCP tools must be expressible as a tool_name + args dict in the
  Interpreter's TOOL_ARG_SCHEMAS.
- Dual dispatch interface: FAST path uses structured args dict (e.g.
  {"path": "...", "question": "..."}); DELIBERATE loop uses a flat
  "query" string per Action format. Every new tool must work in both.
  This is a non-trivial integration constraint — any MCP server
  evaluation must address how its inputs map to both arg shapes.
- Hidden capability: `analyze_images_grok` (multi-image, paths: list)
  exists in tools.py but is not registered in TOOL_ARG_SCHEMAS and is
  not Interpreter-callable. Research on vision/screen MCP servers should
  note whether this gap is better filled by registering it or replacing
  it with an MCP server.
- No Docker. Windows-native or WSL-compatible only.
- Must not load models at import time (thin wrapper discipline).
- Must not conflict with existing FAISS/MiniLM/LangChain imports.

### Non-Goals for This Research
- MCP servers for Claude Code / development workflow (separate concern).
- Cloud-hosted MCP servers that require ongoing subscription beyond
  what is already paid (Grok API, Tavily).
- Any server requiring >400 MB RAM at idle.

---

## Research Scope: Sub-Domains to Investigate

The research must cover the following five domains. Each domain maps
to one or more capability gaps above and must produce a candidate
shortlist with RAM footprint, maturity rating, and integration verdict.

### Domain A: Web Interaction (GAP-1)
Candidate MCP servers enabling browser control, page navigation,
form interaction, and structured web scraping.
Key candidates to investigate: Playwright MCP, Puppeteer MCP,
Browserbase MCP, Stagehand MCP.
Primary question: which server provides the best reliability/RAM
tradeoff for task-driven (non-interactive) browser automation on
Windows, and how does it integrate into the async CLARA pipeline?

### Domain B: Search Quality Upgrade (GAP-5)
Candidate MCP servers that supplement or replace Tavily for
higher-fidelity information retrieval.
Key candidates: Brave Search MCP, Exa MCP, Perplexity MCP,
DuckDuckGo MCP, academic search (Semantic Scholar MCP).
Primary question: what does each server offer that Tavily's
`include_answer="advanced"` path does not, and is the quality
delta worth an additional RAM footprint?

### Domain C: Communication and External Events (GAP-2)
Candidate MCP servers for email, calendar, and messaging.
Key candidates: Gmail MCP, Google Calendar MCP, Outlook MCP,
Ntfy/Gotify for push notification reception.
Primary question: which server has the lightest footprint and
most robust read/send capability for an agent that needs to
observe and act on communication, not just read it?

### Domain D: Development and Codebase Tools (GAP-4)
Candidate MCP servers for Git awareness, code search, and
test execution.
Key candidates: Git MCP (official), GitHub MCP (official),
Sourcegraph MCP, Ripgrep/AST-grep MCP.
Primary question: what is the minimum viable Git integration
for CLARA to be aware of her own codebase state without
requiring human to narrate changes to her?

### Domain E: Memory and Knowledge Mutation (GAP-7)
Candidate MCP servers for persistent memory that goes beyond
CLARA's current JSON-based vault.
Key candidates: Memory MCP (Anthropic official), Mem0 MCP,
basic SQLite MCP, Obsidian MCP.
Primary question: does any server provide a meaningful upgrade
over CLARA's custom memory system, or is the custom system
already more capable than what off-the-shelf MCP memory servers
offer for this specific use case?

---

## What the Research Must Determine Per Server

For every candidate server evaluated across all five domains:

1. **What it does** — concrete capabilities, not marketing description
2. **RAM footprint** — idle MB and active MB (from benchmarks or issues)
3. **Windows compatibility** — native, WSL-only, or broken
4. **Maturity** — version, last commit date, open issues count, stars
5. **Integration pattern** — how tool calls map to CLARA's arg schema
6. **Known failure modes** — from GitHub issues and practitioner reports
7. **Verdict** — integrate / defer / skip + one-sentence rationale

---

## Output

Produce: `Research_MCP_Tool_Ecosystem.md` saved to
`E:\ML PROJECTS\AGENT_ZERO\research\`

The Implementation Objective section (Section 6) must produce:
- A prioritized integration order (which servers to add first,
  sequenced by impact-to-RAM-cost ratio)
- For each recommended server: exact install command, the specific
  tool_name entries to add to TOOL_ARG_SCHEMAS in interpreter.py,
  and the dispatch block to add to _execute_fast_tool in agent.py
- An updated RAM budget table showing total headroom after each
  addition, so the integration sequence can be halted at the
  safe ceiling without rework
