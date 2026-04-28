# Research: MCP Tool Ecosystem for CLARA

**Date:** 2026-04-19
**Objective:** Determine which of CLARA's current tools have meaningful quality ceilings that
MCP servers could raise, identify capability domains absent from the current stack required
for the ambient agent vision, and produce a ranked, hardware-constrained shortlist of MCP
servers with RAM footprints for integration on an RTX 3050 mobile system with ~1.5–2.0 GB
RAM headroom during active inference.
**Target Context:** AMD Ryzen 7 4800H, 15.4 GB RAM (~1.5–2.0 GB headroom at load),
RTX 3050 Laptop 4 GB VRAM, Windows 11, Python 3.11 venv.

---

## 1. Overview

The Model Context Protocol (MCP) is an open standard introduced by Anthropic in November 2024,
now governed by the Agentic AI Foundation (AAIF) under the Linux Foundation (since December 2025).
It standardizes how AI agents connect to external tools and data sources through a uniform
JSON-RPC interface over stdio or HTTP transports. By April 2026, over 20,000 servers exist
across public registries, with OpenAI, Google, Microsoft, and GitHub all shipping official
first-party servers.

For CLARA, MCP is not a replacement for the custom tool layer — it is a selective upgrade
mechanism. The question is not "which MCP servers exist" but "which servers fill real gaps
in CLARA's current capability stack or raise the quality ceiling of tools already present,
within the hard RAM constraint of this hardware."

CLARA's current tool layer is a custom Python implementation with 10 tools spanning web search,
code execution, filesystem access, vision, memory, and task management. The architecture
dispatches tools through two paths: FAST (Interpreter-directed, structured args dict) and
DELIBERATE (ReAct loop, flat "query" string). Any MCP integration must work in both.

---

## 2. How MCP Servers Work (Integration Architecture)

### 2.1 Process Model
Each MCP server runs as a separate OS process, communicating with the host via stdio pipes
or HTTP. In CLARA's context, the "host" would be a thin Python client layer in tools.py that
spawns the server process on startup and maintains a persistent stdio connection.

```
CLARA agent.py
    → process_request()
        → _execute_fast_tool(tool_name, args)      [FAST path]
        → execute_tool(action)                      [DELIBERATE path]
            → MCPClientWrapper.call(server, tool, args)
                → stdio pipe → MCP server process
                    → returns result string
```

The critical integration constraint: FAST path uses a structured args dict
`{"path": "...", "question": "..."}` while DELIBERATE uses a flat `"query"` string.
Every new MCP tool must have arg schemas registered in `TOOL_ARG_SCHEMAS` in interpreter.py
and a dispatch block in both `_execute_fast_tool` and `execute_tool` in agent.py.

### 2.2 RAM Model
MCP servers are sidecar processes. RAM cost is:

| Runtime | Idle RAM (measured) | Notes |
|---|---|---|
| Node.js (npx) | 50–150 MB | Includes V8 heap + process overhead |
| Python (uvx) | 30–80 MB | Lower baseline than Node.js |
| Go binary | 15–25 MB | Minimal; Go GC is efficient |
| Chromium (browser) | 300–600 MB | Per-context; not per-server |

Source: [TMDevLab MCP benchmark, Feb 2026]; [Google Antigravity IDE bug report, Mar 2026,
4 workspaces × 9 Node.js servers = ~4.5GB total]; practical Node.js baseline.

### 2.3 Windows Considerations
Windows adds friction specific to the stdio transport:
- `npx` on Windows resolves to `npx.cmd` (a batch wrapper). When spawned programmatically,
  the stdin/stdout pipes used by MCP's stdio transport may not connect correctly to the
  underlying Node.js process.
- Workaround: use absolute `node` binary path + absolute `cli.js` path instead of `npx`.
  This bypasses the .cmd wrapper and connects stdio correctly.
  Source: [Playwright MCP issue #1540, github.com/microsoft/playwright-mcp, Apr 2026]
- Python-based servers (via `uvx`) do not have this issue — uvx launches Python directly.
- Go binaries also unaffected — single native executable, no wrapper layer.

---

## 3. Benchmarks & Performance

### 3.1 Current Tool Audit — Quality Ceiling Analysis

| Tool | Current Quality | Ceiling Issue | MCP Upgrade Opportunity |
|---|---|---|---|
| web_search (Tavily) | Good | Answer-only, no raw results, acquisition uncertainty | HIGH — Brave Search MCP |
| python_repl | Acceptable | No timeout, no isolation | LOW — exec() acceptable for local agent |
| fs_read/write/list | Good | Single-level list, 10K char limit, no recursive | LOW — custom is fine for agent use |
| fs_run_command | Acceptable | No structured output, PowerShell only | LOW — already powerful enough |
| consult_archive | Good | Static docs only, no live URL indexing | MEDIUM — Firecrawl fills live web gap |
| vision_tool | Good | File-path only, no screen capture | LOW — Grok Vision is strong |
| date_time | Perfect | None | SKIP |
| query_task_status | Good | Read-only | LOW |
| (missing) browser | — | Entire domain absent | HIGH — Playwright MCP |
| (missing) git | — | Entire domain absent | HIGH — mcp-server-git |
| (missing) memory mutation | — | No tool-callable memory update | MEDIUM — custom remains superior |

### 3.2 Search API Benchmark Comparison

| API | Agent Score | Latency (median) | Index | Full Content | Source |
|---|---|---|---|---|---|
| Brave Search | 14.89 | 669 ms | Independent | Snippets only | AIMultiple 2026 |
| Firecrawl | ~14.8 | ~1,335 ms | Web crawl | Full page | AIMultiple 2026 |
| Exa | ~14.7 | ~1,335 ms | Neural/semantic | Full page | AIMultiple 2026 |
| Tavily (pre-acquisition) | ~13.8 | ~998 ms | Web | Optional full | AIMultiple 2026 |
| Perplexity | ~13.0 | 2–5s | Synthesis | Synthesized answer | AIMultiple 2026 |

**[HIGH]** Brave Search leads on latency (669ms vs Tavily ~1000ms) and overall agent score,
outperforming Tavily by ~1 full point in a statistically significant benchmark
[AIMultiple agentic search benchmark, April 2026, concurrent agent queries].

**[HIGH]** Tavily was acquired by Nebius in February 2026, creating supply-chain and
pricing uncertainty for long-term reliance [multiple practitioner sources, Feb-Apr 2026].

**[MEDIUM]** Exa wins on semantic/research queries (highest mean relevance score 4.30);
Brave wins on speed; Tavily had best agent ergonomics (synthesized `answer` key).
Task-type dependency means no single winner for all CLARA use cases.

### 3.3 Playwright MCP Performance & RAM

| Metric | Value | Conditions | Source |
|---|---|---|---|
| Playwright MCP server idle (Node.js only) | ~80–120 MB | No browser launched | Practical Node.js baseline |
| Chromium headless (per context) | 300–500 MB | Single tab, idle | Playwright production post-mortems |
| Chromium + active page | 400–800 MB | JS-heavy SPA | GitHub issue #15400 |
| Total (server + browser) | 400–650 MB | 1 headless Chromium tab | Estimated from above |

**[HIGH]** Playwright MCP on Windows with `npx` fails via stdio in multiple documented cases
(Claude Code, Codex, VS Code). Root cause: `npx.cmd` batch wrapper doesn't pipe stdio
correctly. Workaround: use absolute `node cli.js` path. Not a hard blocker but requires
manual config per machine [github.com/microsoft/playwright-mcp/issues/1540, Apr 2026].

**[HIGH]** Playwright MCP fails when Chrome is already running on Windows — Playwright
cannot launch a second Chrome with a different user-data-dir. The running Brave browser
(CLARA's interface) uses Chromium-based rendering that does not conflict, but any open
Chrome instance does [github.com/anthropics/claude-code/issues/24144, Feb 2026].

**[HIGH]** Chromium headless instances do not free memory between pages without explicit
`context.close()` calls. Memory grows progressively if sessions are not managed.
Post-voice headroom (~400-900MB) is insufficient for Playwright + Chromium simultaneously.

### 3.4 mcp-server-git Performance

| Metric | Value | Notes |
|---|---|---|
| Runtime | Python (uvx) | ~30–60 MB idle |
| Install | `uvx mcp-server-git` | No npm, no Node.js process |
| Tools exposed | git_status, git_diff, git_log, git_commit, git_add, git_search_code, git_read_file, git_list_branches, git_create_branch | Full read + write |
| Windows compatibility | Native | Python + git binary, no wrappers |
| Version | 2026.1.14 | Actively maintained |

Source: [pypi.org/project/mcp-server-git, 2026]; [mcpservers.org/servers/modelcontextprotocol/git]

### 3.5 Anthropic Memory MCP Server vs CLARA's Custom System

| Dimension | Official Memory MCP | CLARA Custom |
|---|---|---|
| Storage format | JSONL knowledge graph | JSON (episodic log + vault) |
| Retrieval | In-memory entity/relation graph | MiniLM cosine similarity (semantic) + recency |
| Deduplication | None built-in | Exact string + 0.85 cosine threshold (vault) |
| Vault/episodic separation | No — single flat graph | Yes — explicit two-tier |
| Tool-callable mutation | Yes — create_entities, add_relations | No |
| RAM | ~60–80 MB (Node.js) | Already loaded (crud.py in-process) |
| Verdict | Inferior retrieval, no dedup, but callable | Superior system, but no tool interface |

**[HIGH]** The official Memory MCP server is architecturally simpler than CLARA's custom
system and would be a downgrade for retrieval quality. It does offer one thing CLARA
currently lacks: **tool-callable memory mutation** (create entity, add relation, delete entity).
The gap to address is not the memory system itself but the absence of a mutation interface.

---

## 4. Known Limitations & Failure Modes

### 4.1 Playwright MCP — Windows

**Failure 1: npx stdio pipe disconnect [HIGH confidence, pattern-level]**
When Playwright MCP is launched via `npx @playwright/mcp@latest` on Windows, the `npx`
command resolves to `npx.cmd` (a batch file wrapper). When this batch wrapper is spawned
programmatically, stdin/stdout pipes used by MCP's stdio transport do not connect correctly
to the underlying Node.js process. The MCP server process starts but hangs silently.
Confirmed across: Claude Code (github.com/anthropics/claude-code), OpenAI Codex
(github.com/openai/codex/issues/3310), and multiple VS Code users (2025-2026).
**Workaround:** Use absolute path to `node.exe` + absolute path to `@playwright/mcp/cli.js`.
`"command": "node", "args": ["C:/path/node_modules/@playwright/mcp/cli.js"]`
Source: [github.com/microsoft/playwright-mcp/issues/1540, Apr 2026]

**Failure 2: Chrome conflict [HIGH confidence, Windows-specific]**
Playwright MCP configured to use Chrome (`--browser chrome`) cannot launch when Chrome
is already running on Windows. Chrome does not support multiple instances with different
user-data-dirs — the second launch attaches to the existing session and exits immediately.
Since Brave (CLARA's interface) is open during normal operation and is also Chromium-based,
this creates a hard conflict if Playwright is configured to use Chrome.
**Fix:** Configure Playwright MCP to use Chromium (bundled): `--browser chromium`.
Bundled Chromium is separate from system Chrome and does not conflict.
Source: [github.com/anthropics/claude-code/issues/24144, Feb 2026]

**Failure 3: Memory growth without session management [HIGH confidence]**
Chromium contexts accumulate memory if `context.close()` is not explicitly called after
each task. In a long-running autonomous agent, each navigation sequence adds residual memory.
Without active session lifecycle management, RAM usage grows unbounded over hours.
In production post-mortems, servers with open contexts consumed 1GB+ in under 20 minutes
under moderate load. For CLARA, each invocation must explicitly close the browser context.
Source: [medium.com/@onurmaciit, "8GB Was a Lie", Dec 2025]

**Failure 4: Post-voice RAM infeasibility**
Current RAM headroom: ~1.5-2.0 GB. Playwright MCP + Chromium requires ~400-650 MB.
Post-voice phase adds Faster-Whisper (~800MB) + Kokoro (~300MB) = ~1.1GB additional load.
Post-voice headroom shrinks to ~400-900MB. A single Chromium context (~300-500MB) would
consume all remaining headroom, leaving nothing for concurrent DELIBERATE inference.
**Conclusion:** Playwright MCP is viable pre-voice but must be treated as a conditional
tool — launched on-demand, session closed immediately after each task, not running idle.

### 4.2 Brave Search MCP — Limitations

**Snippets-only response schema [HIGH confidence]**
Brave Search returns structured result objects with title, URL, and description snippets.
It does NOT return a synthesized `answer` string like Tavily. CLARA's current web_search
implementation uses `res.get("answer", "No results found.")` — switching to Brave requires
a response processing change: extract the top result snippets and join them, or pass the
raw results array to the Interpreter as context.
Source: [brave.com/search/api/guides, 2025]

**Niche query coverage gap [MEDIUM confidence]**
Brave's independent index is smaller than Google's. Queries for niche or highly specialized
topics may return fewer results or miss sources that appear on Google/Bing.
Source: [oreateai.com comparison, Jan 2026]; [brave.com self-assessment, 2025]

**Free tier: 2,000 queries/month [LOW risk for CLARA's use]**
Brave API free tier allows 2,000 queries/month. For a personal agent with intermittent use
this is sufficient. Paid tier starts at $3/1000 queries. Tavily (pre-acquisition) was
$8/1000 queries, making Brave significantly cheaper at scale.

### 4.3 mcp-server-git — Limitations

**"Early development" label [MEDIUM concern]**
The official documentation on mcpservers.org states the server is "currently in early
development." However, the PyPI package (mcp-server-git, version 2026.1.14) shows active
maintenance with recent releases. The concern is primarily about API stability and
potential breaking changes rather than functional gaps.

**Repository path must be pre-specified [LOW concern for CLARA]**
The server requires `--repository` flag pointing to a specific repo path at startup.
For CLARA, the path is always fixed: `E:\ML PROJECTS\AGENT_ZERO`. This is not a limitation
in practice, but it means the server cannot dynamically switch repositories without restart.

### 4.4 Firecrawl MCP — Limitations

**Free tier is non-renewable [MEDIUM concern]**
Firecrawl free tier is 500 one-time credits, not monthly. Once exhausted, requires a paid
plan (~$16/month starter). The MCP server itself is a thin wrapper (~80-100MB idle) — the
heavy work is cloud-side. For an ambient agent doing frequent web extraction, costs could
accumulate quickly.

**MCP server not updated since September 2025 [MEDIUM concern]**
Firecrawl's parent platform shipped major updates (FIRE-1, Browser Sandbox, Extract v2) in
late 2025, but the MCP server package (v3.2.1) has not been updated since September 2025.
Platform features are not accessible via MCP. The gap may close, but it's a current risk.
Source: [dev.to/grove_chatforest, Mar 2026]

**Over-engineering for simple web reading [LOW concern]**
For basic URL-to-content extraction (what CLARA currently does with consult_archive on live
pages), the free Fetch MCP server from the official modelcontextprotocol/servers repo handles
80% of use cases at zero API cost. Firecrawl's value is in JS-heavy pages, structured
extraction, and crawling — capabilities CLARA does not yet need for core operation.

### 4.5 Official Memory MCP Server — Limitations

**In-memory only — data lost on restart [HIGH concern]**
The official Memory MCP server stores its knowledge graph in memory backed by a JSONL file.
The graph itself is held in RAM; the JSONL is a persistence layer. There is no built-in
compaction, deduplication, or similarity-gated dedup. If the server crashes, state is lost
until next restart.

**Flat entity-relation model lacks CLARA's vault/episodic separation [HIGH concern]**
The official server treats all memory as entities with relations (person → has_trait → value).
CLARA's system distinguishes permanent facts (vault) from temporal summaries (episodic log)
and retrieves them differently. The flat model would collapse this distinction.

---

## 5. Comparison: Current CLARA Tools vs MCP Candidates

### 5.1 web_search (Tavily) vs Brave Search MCP

**Current approach strengths:**
- Synthesized `answer` key — single string directly usable in FAST path with zero processing
- `search_depth="advanced"` + `include_answer="advanced"` — high-quality synthesis
- Battle-tested in CLARA's current pipeline; established response schema

**Where Brave Search MCP is stronger:**
- Latency: 669ms vs ~1000ms median [AIMultiple 2026 benchmark, concurrent agent queries]
- Quality: 14.89 vs ~13.8 agent score in the same benchmark — consistent ~1 point advantage
- No acquisition uncertainty (Brave is stable; Tavily acquired by Nebius Feb 2026)
- News-specific routing, image search, local search — capabilities Tavily basic doesn't expose
- Lower cost: ~$3/1000 queries vs Tavily ~$8/1000 queries

**Where current approach holds or wins:**
- Synthesized `answer` string is ergonomically perfect for CLARA's FAST path. Brave returns
  raw result objects — requires a new response formatter in tools.py
- Tavily's `include_raw_content` flag gives full page text in one call when needed;
  Brave is snippets-only (requires Firecrawl for full content)

**Unresolved:**
- Brave's independent index has documented niche coverage gaps vs Bing/Google-backed sources.
  How frequently CLARA's queries hit this gap is unknown without empirical testing.

**Verdict: Upgrade.** Replace Tavily with Brave Search MCP. Latency and quality both improve,
acquisition risk eliminated, cost drops. Response formatter needs one change in tools.py.

---

### 5.2 (Gap) Browser/Web Interaction vs Playwright MCP

**Current state:** Entirely absent. CLARA can retrieve text via Tavily but cannot navigate,
click, fill forms, or interact with any web application. The ReAct loop cannot self-correct
by observing a live page.

**What Playwright MCP adds:**
- Navigate to any URL, click elements, fill forms, take screenshots, read page content
- Uses Playwright's accessibility tree (not pixels) — structured, deterministic, no vision model
- Persistent browser profile — stays logged in across CLARA tasks within a session
- Full JS rendering — sees dynamic content that Tavily and Firecrawl snippet-mode miss

**Integration constraints:**
- Windows setup requires absolute node + cli.js path workaround
- Chromium (~300-500MB) must be launched on demand, not kept idle
- Cannot be reliably used with Chrome open on Windows — must use bundled Chromium
- Post-voice, RAM budget makes this a conditional tool (on-demand only)

**Verdict: Integrate (pre-voice only; conditional architecture).** High capability gap filled.
RAM constraint means it should not be an always-running server — load on first use, close
context after task, unload server when not needed for extended periods.

---

### 5.3 (Gap) Codebase Awareness vs mcp-server-git

**Current state:** CLARA can read files via fs_read_file and run shell commands via
fs_run_command (which could run `git log` etc.), but has no structured git awareness.
She cannot read diffs, check branch state, or understand what changed without explicit
instruction from Alkama. She is narratively dependent on the developer to explain the
codebase state.

**What mcp-server-git adds:**
- `git_status` — current working tree state
- `git_diff` — changes between commits or HEAD vs working tree
- `git_log` — commit history with filtering
- `git_read_file` — file content at any ref/commit
- `git_search_code` — semantic code search across repo
- `git_create_branch`, `git_commit`, `git_add` — full write access (with Alkama approval gating)

**What it doesn't add:**
- GitHub API (PRs, issues, Actions) — that's github-mcp-server, a separate concern
- For CLARA's self-awareness, local git operations are sufficient; GitHub API is not needed

**Verdict: Integrate.** 30-60MB Python process, no Windows friction, fills a genuine
self-referential gap — CLARA aware of her own codebase state without narrative dependency.

---

### 5.4 (Gap) Memory Mutation vs Official Memory MCP

**Current state:** CLARA's memory system (vault + episodic) is more capable than the official
MCP Memory server for retrieval. The gap is the absence of a **tool-callable mutation
interface** — CLARA cannot deliberately update her own vault mid-task.

**What the official Memory MCP offers:**
- `create_entities`, `create_relations`, `add_observations`, `delete_entities`
- Tool-callable from within DELIBERATE loop
- ~60-80MB Node.js idle footprint

**Why replacing CLARA's system with it is a downgrade:**
- No cosine similarity dedup — vault would accumulate duplicates again
- No episodic/vault tier separation — all facts treated as equal
- In-memory graph loses state faster than JSON file persistence

**Recommended approach:** Build a thin `memory_tool` wrapper over CLARA's existing
`crud.py` — expose `vault_add`, `vault_search`, `vault_delete`, `style_update` as
callable tools registered in TOOL_ARG_SCHEMAS. This requires no MCP server at all —
a native Python tool dispatch in `_execute_fast_tool`. ~0MB additional RAM.

**Verdict: Skip the official Memory MCP.** Build a native CLARA memory mutation tool
instead. The custom system is architecturally superior; the gap is a missing callable
interface, not a missing system.

---

### 5.5 (Gap) Search + Full Content vs Firecrawl MCP

**Current state:** CLARA's `consult_archive` covers indexed static docs. For live URLs or
web pages not in the FAISS index, she has no way to retrieve full page content. Tavily
returns snippets for non-subscribed pages. The Orchestrator's EnvironmentWatcher can trigger
a rag_rebuild when docs change but cannot pull live content on demand.

**What Firecrawl adds:**
- `firecrawl_scrape` — full page content from any URL, JS-rendered, cleaned markdown
- `firecrawl_search` — web search + scrape combined in one call
- `firecrawl_extract` — structured JSON extraction from a URL via schema
- Production-grade anti-bot handling, proxies, dynamic content rendering

**Adoption considerations:**
- Free tier is 500 credits one-time. For occasional research, sufficient. For continuous
  ambient agent operations, will exhaust quickly.
- MCP server itself is lightweight (~80-100MB idle, all heavy work is cloud-side)
- The free `fetch` MCP server (official modelcontextprotocol/servers) handles simple
  URL-to-markdown extraction at zero cost — covers 80% of CLARA's web reading needs
  without Firecrawl's credit consumption

**Recommended approach:** Start with the free official Fetch MCP server. Integrate
Firecrawl MCP only if Fetch proves insufficient for JS-heavy or auth-gated pages.

**Verdict: Defer Firecrawl; integrate Fetch MCP first.** Lower risk, zero ongoing cost,
covers the majority use case. Firecrawl is the upgrade path when Fetch hits walls.

---

## 6. Implementation Objective

This section is the brief for Claude Code. Written for a developer executing the changes,
not a strategist evaluating them. Execute in order — each stage is self-contained.

---

### Stage 1: Replace Tavily with Brave Search MCP (Highest ROI, Lowest Risk)
**Why first:** Biggest latency + quality improvement, no new capability gaps, minimal code change.

**Install:**
```bash
npm install -g @brave/brave-search-mcp-server
# Get API key at: https://api.search.brave.com/ — free tier: 2,000 queries/month
```

**Windows config** (avoids npx.cmd stdio issue):
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "node",
      "args": ["C:\\Users\\alkam\\AppData\\Roaming\\npm\\node_modules\\@brave\\brave-search-mcp-server\\dist\\index.js"],
      "env": { "BRAVE_API_KEY": "YOUR_KEY_HERE" }
    }
  }
}
```

**Key parameter:** The Brave MCP server exposes `brave_web_search` tool. Arguments:
`{"query": "string", "count": 5, "freshness": "pd"}` (pd = past day for time-sensitive queries)

**Response schema change required in `core_logic/tools.py`:**
```python
# OLD (Tavily):
def web_search(query: str) -> dict:
    response = client.search(query=query, include_answer="advanced", ...)
    return response  # agent uses res.get("answer")

# NEW (Brave MCP client wrapper):
def web_search(query: str) -> dict:
    result = mcp_client.call("brave-search", "brave_web_search", {"query": query, "count": 5})
    # Brave returns: {"web": {"results": [{"title": ..., "description": ..., "url": ...}]}}
    snippets = [f"{r['title']}: {r['description']}" for r in result.get("web", {}).get("results", [])[:3]]
    answer = "\n".join(snippets) if snippets else "No results found."
    return {"answer": answer, "results": result.get("web", {}).get("results", [])}
```

**What stays the same:** The `"answer"` key interface is preserved. FAST path
`res.get("answer", "No results found.")` continues to work unchanged.

**Files changed:** `core_logic/tools.py` — replace TavilyClient with Brave MCP client.
`core_logic/interpreter.py` — TOOL_ARG_SCHEMAS for web_search is already correct.

**Do NOT:** Install via Docker (adds ~200MB overhead for a lightweight Node.js server).
Do not use `npx` as the command — use absolute `node` binary path.

**Expected post-implementation metrics:**
- Latency: ~669ms [HIGH confidence, AIMultiple 2026] vs current ~1000ms
- Quality: +1 agent score point average [HIGH confidence]
- RAM: ~80-100MB idle Node.js process [HIGH confidence]

---

### Stage 2: Integrate mcp-server-git (Zero Windows Friction)
**Why second:** Python-based (no Windows stdio issue), low RAM, high value for self-awareness.

**Install:**
```bash
# In CLARA's venv (jarvis_v2):
pip install uv --break-system-packages
# Then use uvx to run — no global install needed
```

**Config** (add to CLARA's startup or api.py lifespan):
```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "E:\\ML PROJECTS\\AGENT_ZERO"]
    }
  }
}
```

**Tool schema additions in `core_logic/interpreter.py` TOOL_ARG_SCHEMAS:**
```python
"git_status":      {},
"git_diff":        {"ref": "string — optional commit ref or 'HEAD' (default: working tree)"},
"git_log":         {"max_count": "int — number of commits to return (default: 10)"},
"git_read_file":   {"path": "string — file path relative to repo root",
                    "ref": "string — optional git ref (default: HEAD)"},
"git_search_code": {"query": "string — code search query"},
```

**Dispatch in `_execute_fast_tool` and `execute_tool`:**
```python
elif tool_name in ("git_status", "git_diff", "git_log", "git_read_file", "git_search_code"):
    return await asyncio.to_thread(
        mcp_client.call, "git", tool_name, args
    )
```

**Key parameters for CLARA's DELIBERATE use:**
- `git_diff` with `ref="HEAD"` — see what changed since last commit
- `git_log` with `max_count=5` — brief history before autonomous code tasks
- `git_read_file` with `path` + `ref` — read any file at any point in history

**Do NOT:** Use this for git commit/push without explicit user confirmation gate.
Write operations (git_commit, git_add) should be in a separate `git_write` tool group
gated behind user approval in the Orchestrator's conflict model.

**Files changed:** `core_logic/tools.py`, `core_logic/interpreter.py`
**RAM:** ~30-60 MB Python process [HIGH confidence, Python process baseline]

---

### Stage 3: Integrate Fetch MCP (Free URL Content Retrieval)
**Why third:** Zero cost, fills the live-URL content gap before considering Firecrawl.

**Install:**
```bash
npm install -g @modelcontextprotocol/server-fetch
```

**Config:**
```json
{
  "mcpServers": {
    "fetch": {
      "command": "node",
      "args": ["C:\\Users\\alkam\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-fetch\\dist\\index.js"]
    }
  }
}
```

**Tool schema addition in interpreter.py TOOL_ARG_SCHEMAS:**
```python
"fetch_url": {"url": "string — full URL to fetch", "max_length": "int — optional char limit"},
```

**Key parameters:**
- Returns clean markdown from URL
- `max_length=5000` recommended to protect CLARA's context window from oversized responses
- Does NOT handle JS-heavy pages — static HTML only. For JS pages, Firecrawl is the upgrade.

**Do NOT:** Use for authentication-gated pages (cookies are not maintained).
Do not remove `consult_archive` — RAG over indexed docs remains faster and more relevant
for known-document lookups. `fetch_url` is for live, on-demand URL retrieval only.

**Files changed:** `core_logic/tools.py`, `core_logic/interpreter.py`
**RAM:** ~60-80 MB Node.js process [HIGH confidence]

---

### Stage 4: Playwright MCP (Pre-Voice Only, Conditional Load)
**Why fourth:** Highest RAM cost, most Windows friction — defer until Stages 1-3 are stable.

**Install:**
```bash
# Install locally in project, not globally
cd "E:\ML PROJECTS\AGENT_ZERO"
npm install @playwright/mcp
npx playwright install chromium   # Download bundled Chromium — NOT system Chrome
```

**Config (absolute path — avoids npx.cmd Windows bug):**
```json
{
  "mcpServers": {
    "playwright": {
      "command": "node",
      "args": ["E:\\ML PROJECTS\\AGENT_ZERO\\node_modules\\@playwright\\mcp\\cli.js",
               "--browser", "chromium",
               "--headless"]
    }
  }
}
```

**Key parameters:**
- `--browser chromium` MANDATORY — do not use `chrome`. Bundled Chromium does not
  conflict with running Brave or Chrome instances on Windows.
- `--headless` for background agent tasks. Remove for debugging.
- Do NOT use `--persistent` — ephemeral profile is safer for autonomous tasks.

**Architecture constraint:** This server must NOT be started at CLARA's boot time.
Start it on-demand when a browser task is requested. After the task completes, send
`browser_close` tool call. This keeps RAM idle usage at ~0 vs ~400-650MB.

**RAM budget reality check:**
- Pre-voice: 1.5-2.0 GB headroom. Playwright + Chromium = ~400-650 MB. Feasible.
- Post-voice: ~400-900 MB headroom. Playwright + Chromium = ~400-650 MB. Infeasible without
  unloading voice models first. Do not run concurrently with active voice sessions.

**Files changed:** `api.py` (on-demand server lifecycle management), `core_logic/tools.py`,
`core_logic/interpreter.py`

---

### Updated RAM Budget Table (Cumulative)

| Stage | Server Added | Idle RAM | Cumulative Idle | Headroom Remaining |
|---|---|---|---|---|
| Baseline | CLARA current stack | ~200-300 MB | ~200-300 MB | ~1.5-1.8 GB |
| Stage 1 | Brave Search MCP (Node.js) | ~80-100 MB | ~300-400 MB | ~1.4-1.7 GB |
| Stage 2 | mcp-server-git (Python) | ~30-60 MB | ~330-460 MB | ~1.3-1.7 GB |
| Stage 3 | Fetch MCP (Node.js) | ~60-80 MB | ~390-540 MB | ~1.2-1.6 GB |
| Stage 4 | Playwright MCP (on-demand) | ~0 idle / ~400-650 MB active | ~390-540 MB idle | Feasible pre-voice |
| Post-voice | Voice stack added | ~1.1 GB | ~1.5 GB + active | Playwright infeasible |

**Safe ceiling confirmed:** Stages 1-3 are comfortably within the ~1.5-2.0 GB headroom.
Stage 4 is feasible pre-voice with on-demand loading. Halt integration at Stage 3 before
Voice Phase 1 begins; defer Playwright to a post-voice headroom reassessment.

---

## Key Findings

- **[HIGH]** Brave Search API consistently outperforms Tavily by ~1 agent score point with
  669ms median latency vs ~1000ms [AIMultiple agentic search benchmark, 2026, concurrent
  agent queries]. Tavily was acquired by Nebius in February 2026, adding long-term risk.
  Brave replaces Tavily with a single response formatter change in tools.py.

- **[HIGH]** Playwright MCP on Windows requires absolute `node cli.js` path instead of
  `npx` — the npx.cmd batch wrapper breaks MCP stdio transport [github.com/microsoft/
  playwright-mcp/issues/1540, 2026]. Must use `--browser chromium` (bundled) not `chrome`
  to avoid conflicts with running Chrome/Brave instances [claude-code issue #24144, 2026].
  Post-voice phase (~400-900MB headroom) makes Playwright + Chromium (~400-650MB) infeasible
  as an idle server — on-demand loading only.

- **[HIGH]** mcp-server-git (Python/uvx, ~30-60MB, PyPI version 2026.1.14) has no Windows
  friction, low RAM cost, and fills CLARA's entire codebase self-awareness gap. This is the
  highest ROI server per MB of RAM consumed.

- **[HIGH]** The official Anthropic Memory MCP server is architecturally inferior to CLARA's
  custom system — no cosine similarity dedup, no vault/episodic separation, in-memory only.
  The real gap is the absence of a **tool-callable mutation interface**, best solved by a
  native `memory_tool` wrapper over existing crud.py (~0MB overhead, no MCP server needed).

- **[MEDIUM]** The free official Fetch MCP server fills ~80% of the live-URL content gap
  at zero ongoing cost before considering Firecrawl. Firecrawl's value is JS-heavy pages
  and structured extraction — capabilities CLARA doesn't yet require for core operation.
  Firecrawl free tier is 500 non-renewable credits, making it a deferred upgrade.

**Verdict:** Execute four stages sequentially: (1) Replace Tavily with Brave Search MCP
for immediate latency + quality improvement, (2) Add mcp-server-git for codebase
self-awareness, (3) Add Fetch MCP for free live-URL content, (4) Add Playwright MCP
pre-voice for browser interaction with on-demand lifecycle management. Skip the official
Memory MCP — build a native memory_tool wrapper instead. Halt Stage 4 before Voice Phase 1.
