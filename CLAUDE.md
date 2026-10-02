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

## Work Backlog & Busy-Day Mode (Global Instruction)

`BACKLOG.md` (repo root) is the **single canonical queue** of pending work, grouped by autonomy tier
(🟢 do-unattended / 🟡 build-on-`autonomous`-dormant-uncommitted-Alkama-reviews-diff-and-commits / 🔴 queue-only-needs-Alkama-or-arming-risk). Keep it
**fed** and **comprehensive**: whenever a feature/idea is discussed and confirmed, add it to **BOTH**
`BACKLOG.md` **and** `briefs/ROADMAP.md` (the backlog is the operational queue; the roadmap is the
sequenced plan + dependencies — they must stay in sync). Don't let pending state scatter across TIMELINE
notes.

**The decision chain (how to choose what to work on): `BACKLOG.md` → `briefs/ROADMAP.md` → the brief →
decide.** Start at the backlog (the queue). For a candidate, trace to the ROADMAP to see where it sits and
**its dependencies** — is it the right time, are its prerequisites done? Then check for a **brief** (design
doc / proposed code): if briefed-and-confirmed, implement; if briefed-awaiting-confirmation, don't; if it's
complex and unbriefed, write the brief instead of editing. Only then decide whether to start it **or not** —
an item whose dependencies aren't met should not be started; do the prerequisite first or pick another.
Each backlog item should therefore carry its dependencies + roadmap/brief references.

**Busy-day mode** — when Alkama says *"use busy-mode" / "busy day / work the backlog"*, it's an explicit
**permission-bypass** (same live session, not `claude -p`): do 🟢/🟡 work **without asking** (typical and
complex), self-solve forks with the most reversible option + log it, **queue/brief 🔴 instead of
interrupting** (there is no one to answer — briefs are async questions), over-consume the window freely
(under-consume never), self-check every action. Log everything **as you go** to a per-lifetime report in
`busy-mode-reports/` (status Ongoing→Ended; a 150–200-word "which & why" before each task — the
skill-evaluation signal), **in addition to** TIMELINE/BACKLOG. Full operating contract + the loop is the
`busy-mode` skill (`.claude/skills/busy-mode/SKILL.md`); queue/contract in `BACKLOG.md`.

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

### Daily Test Harness — The Drill (analysis + rotation protocol)

The harness fires 20 questions per session from `tests/questions_morning.json` and
`tests/questions_evening.json`, then verifies the answers (Layer 1) and writes a report.
Each question carries `fail_count` (consecutive fails, resets to 0 on pass), `last_result`
("pass" | "fail" | "pending"), and an optional `verification` block (see below).
**Trigger:** Alkama says "same drill" / "analyze the report" after a run — manual only, never automatic.

**Delivery hardening (Brief 41, 2026-06-13) — after three cron casualties in three days, two from
no-internet.** The drill is meaningless without DeepSeek (every question needs it), so the harness now: (A)
**connectivity pre-flight** — `wait_for_internet()` checks the DeepSeek host BEFORE the lock or any spawn;
offline → capped retry (2× / 15 min) then a CLEAN `sys.exit(0)` (Telegram heads-up, nothing spawned, no
orphan, no lock held). (B) **HTTP-poll readiness + reuse** — `backend_is_up()` (`/soul`) is the SOLE
readiness gate (dropped the old `"Voice system loaded"` log-scrape); a live/leftover-healthy backend is
REUSED and left running (`_HARNESS_OWNS_BACKEND` ownership flag); a half-started zombie holding :8001 is
killed before respawn (`_kill_stale_backend`). (C) **guaranteed teardown** — `stop_backend()` is
ownership-gated + idempotent, and `atexit.register(stop_backend)` reaps a harness-spawned backend on any
exception/sys.exit so a mid-run crash no longer orphans it. The lock was already robust (steals a dead-pid or
>3h-old lock). If a cron is missed, the manual rerun is `python tests/test_harness.py --session {morning|evening}`
— it reuses a live backend or starts one, and is safe to run after killing any orphan.

**Coherence Drill (Phase 4) — wired into the MORNING run only (2026-06-05).** After the morning
scorecard/report is written and BEFORE the backend is stopped, `test_harness.py` Phase 3 runs
`coherence_drill.run_drill()` and APPENDS a "## Coherence Drill (multi-turn — Phase 4)" section to the
morning report. This is a SEPARATE axis from the single-turn scorecard: scripted multi-turn dialogues
(`tests/coherence_dialogues.json`) measure conversational memory — entity-recall, didn't-need-to-ask
(infer when the referent is clear), and appropriately-asked (the ambiguity CONTROLS — ask when genuinely
ambiguous). Scorer + self-test in `tests/coherence_drill.py` / `tests/test_coherence_drill.py`; docs in
`tests/COHERENCE_DRILL.md`. Morning only and once/day by design (coherence is a slow-moving capability
metric, not a per-change regression guard like the L1-L5 scorecard — twice-daily would just double cost).
Evening runs the L1-L5 scorecard alone. Wrapped so a coherence hiccup never fails the main harness.

**The drill, step by step:**
1. **Read the report** `reports/YYYY-MM-DD-{session}.md`. It now contains a **Verification Scorecard**
   (Layer 1, `tests/verification.py`, Brief 31) — deterministic PASS / FAIL / UNVERIFIABLE per question,
   re-derived from current source.
2. **Determine pass/fail per question, anchored to the scorecard:**
   - Scorecard **PASS / FAIL is strong but NOT infallible** — for every **FAIL**, confirm against
     independent ground truth (your own `grep`/read) BEFORE accepting it. A verifier bug looks exactly
     like a Clara failure. **(2026-06-01 evening: `search_set` FALSE-failed two correct answers — it counted
     `core_logic/memory.json`'s episodic mentions of `os.replace`/`asyncio.Lock` as code occurrences, so
     truth "7 across 2" became "18 across 3". Now fixed (`search_set` is code-only), but the lesson stands.)**
     A confirmed FAIL = genuinely wrong (search undercount, wrong count, wrong computation).
   - Scorecard **UNVERIFIABLE** (knowledge, semantic, file_op, multi-line quotes) = judge manually.
   - Known Layer 1 v1 gaps to catch by hand: it can MISS a real PASS when a quote spans multiple source
     lines (string-concat / split f-strings), and it does not yet verify list-counts ("said 12, there are
     14") or whether a verbatim quote is the *relevant* line. Verbatim PASS confirms the quote is real,
     not that the answer is correct — spot-check the substance.
3. **Cross-reference the session log** `logs/session_*.log` for every FAIL and anything suspicious:
   `>> [Router] Mode:`, the actual `>> [FAST] tool=` / `-> Tool:` calls + their raw results, `Off-format`,
   `Status: COMPLETED / Total results found`, `Malformed JSON`. This reveals the *mechanism* (e.g. format_llm
   corrupting a number, a search returning a spurious 0, a Rule-19 violation) — not just the symptom.
   Note: one log can hold multiple runs if the backend stayed up; `build_session_digest` matches by question.
4. **For each FAILED question:** increment `fail_count`, set `last_result: "fail"`, **keep verbatim** — UNLESS
   the question itself is flawed (e.g. a project-wide search whose ground truth is dominated by accumulating
   doc/report mentions). In that case fix the question's *scope* and note it (carry the fail_count).
5. **For each PASSED question:** set `last_result: "pass"`, then replace it with a NEW question that
   **climbs the difficulty ladder** (below) rather than shuffling sideways at the same depth. Default:
   keep the **same `expected_mode`** and target the *same capability area one rung higher*; once that area
   reaches L5/L6, open a fresh area at L1. Reset `fail_count: 0`, `last_result: "pending"`, and **add a
   `verification` block** (below) — if the new rung isn't mechanically checkable yet (L3/L4/L6), mark it
   `{"type":"knowledge"}` so it routes to manual judgment, and **flag it as a Layer-1 extension candidate**
   in the TIMELINE note (the questions lead; her self-verification follows — see the ladder note).
6. **Write the JSON**, then update **TIMELINE.md** with an `[UPDATE]` entry (pass rate, what each FAIL was +
   its mechanism, Layer 1's performance, what rotated).
7. **WRITE THE ANALYSIS INTO THE REPORT FILE — MANDATORY, NON-SKIPPABLE.** Overwrite the report's
   `## Claude's Analysis` placeholder (`*Pending — open this report…*`) with the actual verdict for THIS run:
   per-question pass/fail anchored to the scorecard, every confirmed FAIL elaborated per the
   elaborate-on-real-problems rule (what/why/root-cause/blast-radius/disposition; clean passes stay terse),
   Layer-1 performance, and what rotated. **The report file is the durable record — chat and TIMELINE are
   not enough** (2026-06-19: 7 reports, 06-15e..06-19m, had real analysis in chat+TIMELINE but their report
   sections silently stayed "Pending", so they were untrackable). The drill is NOT complete until this
   section is written. Verify with `python tests/report_analysis_status.py` — it must report that day's
   report as ANALYZED (it greps the section for the "Pending" placeholder and exits non-zero on any gap).
   **That SAME gate now also enforces climb-due (2026-07-20):** it reads the live `pass_streak` fields
   from both question sets and exits non-zero while ANY climbable question sits at/over `CLIMB_AT` (5)
   consecutive passes owed a climb. Detection was always automatic (harness Phase 1.7), but actioning it
   was skippable — so streaks silently piled to 12-13 (the "climb backlog"). **Actioning every CLIMB DUE
   (promote one rung, reset the streak) OR recording an explicit deferral in the analysis is now a
   mandatory, gated part of the drill: the backlog is not clear until this check exits 0.** Because it
   reads the live question JSONs (not the report snapshot), a climb clears the gate the instant it's
   actioned, and it catches climbs in the OTHER session's set too (e.g. an evening climb the morning
   report never showed).

**`verification` block (add to every new question — the reliable path, Brief 31):**
- `{"type":"compute","code":"<one-liner that prints the answer>"}` — FAST math. Self-check it runs.
- `{"type":"count","target_file":"core_logic/memory.json","json_path":"long_term"}` — counts.
- `{"type":"search_set","pattern":"<regex>","scope":"core_logic"}` — searches. ALWAYS scope to `core_logic/`,
  never "the project" (doc mentions in reports/briefs/TIMELINE grow over time → unwinnable enumeration).
  The question text MUST demand an explicit file+line list — the verifier's recall currency is LINE
  NUMBERS, so a counts-only answer scores recall≈0 and can FAIL despite being correct (Brief 36 E-1).
- `{"type":"verbatim_quote","target_file":"core_logic/x.py"}` — "quote the line verbatim". Confirm the target
  string actually exists in the file so the question is answerable AND verifiable.
- `{"type":"key_facts","must_include":["term",["syn1","syn2"],…]}` — L3/L4 chains: the answer must CONTAIN the
  terminal facts (the function/tool/class names a chain resolves to). Necessary-condition check (method=key_facts,
  conf 0.75) — PASS means the facts are present, NOT full correctness, so spot-check substance. A plain string must
  appear; a nested list is an any-of synonym group. FAIL only when the majority are missing.
- `{"type":"absence_honesty","pattern":"<regex>","scope":"core_logic"}` — L5 Rule-19: pick a string GENUINELY
  absent from code → PASS if Clara reports absence, FAIL if she fabricates a file:line. Confirm it is actually absent.
- `{"type":"file_op","path":"tests/probe_X.txt"}` — probe write/read/delete.
- `{"type":"knowledge"}` — CHAT training-knowledge questions, AND genuinely-semantic L4-synthesis / L6
  self-diagnosis (advisory; no source oracle) — judged manually by Claude during the drill.
  Validate new questions before saving: compute blocks run, verbatim/absence targets exist (or are absent) in
  source, key_facts terms appear in a correct answer.

**Verifier self-test (the meta-guardrail, Brief 31 hardening):** the scorecard engine itself is regression-tested
by `tests/test_verification.py` (fixture-based, 14 cases — `python tests/test_verification.py`, exit 1 on any
deviation; also a pytest `test_self_test`). **It runs automatically every harness run** as a Phase 1.4 pre-flight:
if the engine fails its own fixtures, the report's scorecard section is stamped "⚠️ VERIFIER SELF-TEST FAILED —
scorecard suspect this run" so a buggy verifier is surfaced loudly instead of silently false-failing Clara. No
manual step needed; also run it by hand whenever you edit `verification.py`. It exists because **a
verifier bug looks exactly like a Clara failure** (2026-06-01 evening: `search_set` counted `core_logic/memory.json`
episodic mentions as code occurrences and FALSE-failed two correct answers — true "7 across 2" became "18 across 3").
It locks in the fixes (search_set is code-only via `CODE_EXT`; verbatim candidates strip surrounding `"`/`*`
decoration) so they cannot silently regress. **Anchor every confirmed FAIL to independent ground truth before
accepting it** (step 2 above) — the self-test guards the engine, your grep guards the run.

**Self-Assessment Layer 3 — Fix Proposals (Brief 38, harness Phase 1.8, 2026-06-14):** for a CONFIRMED,
persistent (`fail_count >= 2`), real-axis (Layer-2 classified `real`) FAIL, Clara reads the responsible
code and proposes a specific fix — a review-only ARTIFACT in `reports/proposals/`, NEVER applied (apply is
the far-future Brief 34). The design center is the **fabrication gate** (`tests/fix_proposals.py`,
deterministic, reuses Layer-1's `_read`/`_norm_ws`): every proposal's `current_code_quote` must verify
VERBATIM against the named in-scope file — a quote of code that doesn't exist is auto-rejected
`fabricated_quote`, no LLM judge. Scope guard: `core_logic/`+`tests/` only, EXCEPT the assessment stack
itself (verifier/harness/this module — no grading-the-grader). Trigger is rare by design → ~0 proposals on
a clean run; the gate-rejection rate is the live fabrication-rate metric. Trust ledger
(`reports/proposals/ledger.json`) tracks gate-passed / Claude-endorsed / Alkama-accepted / proven — Layer 4
(auto-apply) stays undiscussed until ≥5 consecutive gate-passed+accepted+proven. Gate gold-seed self-test:
`tests/test_fix_proposals.py` (19 cases, both directions, no backend). Wrapped — never fails a run.

**Difficulty ladder (rotation climbs this — Topic 2, 2026-06-01):** the suite must grow *deeper* as
CLARA's capacity grows, not circle at one altitude. ~85% of the suite was single-hop retrieval (which she
has mastered at 19/20); rotation now promotes up these rungs:
- **L1 — single-hop retrieval/recall:** quote one line, recall a definition, one compute. *(mastered — keep
  ~5 as fixed regression anchors so a higher-rung fix can't silently break L1.)*
- **L2 — completeness enumeration:** list every occurrence / all matches across files (Q06's class).
- **L3 — multi-hop chains:** output of read N feeds read N+1 (e.g. "find the guard `_run_fast` calls, then
  open the module that defines it and give its threshold"). Tests planning, not just retrieval.
- **L4 — cross-source synthesis:** doc-vs-code agreement (e.g. "does CLAUDE.md's atomic-search description
  match `tool_executor.py`?"). The Q5-stress-test class.
- **L5 — adversarial / guardrail:** malformed Windows path (Rule 14), a string that genuinely is absent
  (Rule 19 honesty, not fabricated absence). Exercises the guardrails nothing currently tests.
- **L6 — self-diagnosis / meta:** "read your last failed task in the session log, explain why it failed."

**Verifier note (Topic 2 resolution):** the question ladder is **not** gated by CLARA's *own* Layer-1
verifier in the near term — **Claude (me) is the verifier during the drill** and can hand-check L3/L4/L6
answers regardless of what Layer 1 can mechanically grade. So climb now; for each rung Layer 1 cannot yet
verify, flag it as a Layer-1 extension target (Brief 32+) so her self-verification grows toward where the
questions already are. The CLARA-self-grades-unsupervised constraint only applies to the autonomous future.

**Rules:** rotation only on explicit trigger; no rewording of failed questions (verbatim until they pass,
scope-fix excepted); promote on PASS per the ladder; hold ~5 L1 regression anchors fixed; new questions
cover untested modules, recent code changes, routing edge cases, memory integrity, multi-step file ops,
persona guardrails, error recovery — avoid the remaining set's areas.

### Environment Variables (core_logic/.env)
- `DEEPSEEK_API_KEY` — DeepSeek API key (all LLM calls via OpenAI-compatible API)
- `GEMINI_API_KEY` — Google Gemini API key for the vision tool (`gemini-2.5-flash`). **SET as of 2026-06-11 — vision is LIVE** (validated: test image correctly described). Free tier throws transient 503s; the tool retries 3× with backoff.
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
Interpreter (DeepSeek non-reasoning) → structured intent JSON
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

**Per-turn stream watchdog (BRIEF_59 / G17, 2026-08-02).** Each DELIBERATE turn opens a streaming
DeepSeek call. The old bare `async for` had no inner bound, so an upstream freeze hung the whole request
(07-08e Q11 hung ~22 min). `run_task` now bounds the two real stall points: `create()` is wrapped in
`asyncio.wait_for(REACT_STREAM_CONNECT_TIMEOUT_S=30)`, and the stream is iterated by hand with each
`__anext__()` wrapped in `asyncio.wait_for(REACT_STREAM_IDLE_TIMEOUT_S=30)` — an **idle** timer that resets
on every chunk, so a slow-but-progressing (thinking-mode) turn never trips and only a true no-progress
freeze does. On a stall: best-effort `_stream.close()`, an honest `user()` retry note appended to `llm`
(Clara re-runs the turn), and a **consecutive**-stall cap (`REACT_STREAM_MAX_CONSEC_STALLS=2`) that returns
an honest "upstream outage" message on the 2nd consecutive stall (fail-fast) and resets on any completed
turn. This is the *inner* per-turn budget; Brief 37's `asyncio.wait_for(…, 600s)` remains the *outer*
request backstop. Knobs are env-overridable (defined after `load_dotenv`). Test:
`tests/test_react_stream_timeout.py` (synthetic-hang, no backend). `_run_chat`'s single stream has the same
`async for` shape and is a noted follow-up (outer 600s wrapper only for now).

### Key Modules

| Module | Path | Role |
|--------|------|------|
| API server | `api.py` | FastAPI + concurrent WebSocket (fire-and-forget, message_id) |
| Agent | `core_logic/agent.py` | process_request, route(), _run_fast, _run_chat, run_task |
| Interpreter | `core_logic/interpreter.py` | DeepSeek non-reasoning → structured intent JSON |
| Orchestrator | `core_logic/orchestrator.py` | OrchestratorLoop, task dispatch, retry architecture |
| TaskGraph | `core_logic/task_graph.py` | SQLite-backed task state machine with crash recovery |
| EventQueue | `core_logic/event_queue.py` | Async priority queue; `drain_blocking` default timeout 1.0s, orchestrator drives it at 0.1s |
| Memory CRUD | `core_logic/crud.py` | get_smart_context, episodic log, vault |
| System prompt | `core_logic/system_prompt.py` | PERSONA + CHAT_SYSTEM_PROMPT + SYSTEM_PROMPT |
| Tools | `core_logic/tools.py` | All tool implementations + vision tool (Gemini 2.5 Flash, live 2026-06-11) |
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

### Computation always routes to a tool, never CHAT (2026-08-01)
Any question that **computes** a value — arithmetic, a statistic, a date offset, a time offset, a count —
routes to a tool (`python_repl` for math; `date_time` with `offset_days`/`offset_minutes` for date/time),
`requires_planning=false` → FAST; a multi-STEP computation → DELIBERATE. `tool=null` → CHAT is for
conversational/knowledge answers ONLY, and `CHAT_SYSTEM_PROMPT` explicitly forbids emitting a tool call
there (CHAT has no tool execution). **Root cause of the 07-31e `<tool_call>` blob:** a time-delta was routed
to CHAT, where the model hallucinated a tool-call it cannot run. `get_time_date` gained `offset_minutes`
(deterministic target clock time, wraps across midnight) so time-deltas are FAST + deterministic, mirroring
Brief-50's `offset_days`. `_run_fast` dispatch forwards both offsets; a `_run_chat` backstop suppresses any
stray native tool-call block.

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

**Crash-safe atomic persistence (`crud._save_memory`, added 2026-05-29):** memory.json is written
via a UNIQUE `tempfile.mkstemp(prefix=".memory.json.", suffix=".tmp")` → fsync → `os.replace`, so a
crash or hard-kill mid-write can never truncate the live file (the 2026-05-29 truncation that lost
~4000 episodes came from the old `open('w')` truncate-then-stream). The temp name is unique per call
because background consolidation + autonomous episodes can write concurrently (a shared temp name
would let them interleave). On Windows `os.replace` raises PermissionError when a reader (e.g.
`/soul`, a harness `python_repl`) has the file open, so it is retried with backoff. Because every
save writes the FULL in-RAM dict, a rare dropped replace self-heals on the next save. On
`JSONDecodeError` at load, `_load_memory` backs up the corrupt file (timestamped) before falling
back to default memory — it never silently overwrites recoverable data. `EnvironmentWatcher` ignores
`.memory.json.` temp files so saves don't spam `file_change` events.

### Temporal Grounding (`[NOW]` line, 2026-06-12)
`crud._now_line()` opens EVERY memory context with one compact line: weekday, date, 24h+12h time,
part-of-day, yesterday/tomorrow anchors. Placed inside the (per-request-varying) context block —
NEVER the system prompt — so the per-second timestamp costs nothing against the DeepSeek prefix
cache. Reaches both the Interpreter (relative-time math: ambient_recall windows, "yesterday"
resolution) and all answer paths. The `date_time` tool was also upgraded from a bare datetime repr
to a rich block (formats, timezone, week/day-of-year, adjacent days). Watch-item: probes that
MANDATE the date_time tool now test instruction-following, not necessity.

### Ambient Awareness A0 + A1 (BRIEF_39, live 2026-06-12)
**A0 (silent perception):** `ambient_watch.py` (repo root) is a STANDALONE 24/7 process — no GPU, no models,
NO API keys — that samples consent-gated sensors (`active_window`, `system_state`, `session_rhythm`; gated by
`AMBIENT_SENSORS` in `.env`, empty = all off) and is the SOLE writer of `core_logic/ambient.json` (ring cap
2000, atomic flushes). Runs via Task Scheduler at logon (`setup_ambient_task.ps1`); kill switch =
`schtasks /end /tn CLARA_AmbientWatch` or empty the env var. The backend only READS the store. Separate file
on purpose (telemetry, not memories — avoids the B-20 bloat class); `ambient.json`/`ambient_patterns.json` are
in `IGNORED_PATTERNS` + `.gitignore`. Sensors + store + `recall()` live in `core_logic/ambient.py`.
**A1 (grounded recall):** native tool `ambient_recall(date|window, query)` — "what was I doing yesterday /
on June 11 / in the last 2h". Prefer the **`date` anchor** ('yesterday'/'June 11'/ISO) over hours-back. PERSONA
+ recall enforce Rule-19 parity: an unobserved window says "I wasn't watching" — never a reconstruction.
**Q1-fix hardening (2026-06-13, BRIEF_39 addendum):** the date anchor + an hour-by-hour rollup for busy
windows (the old 40-record tail-cap HID in-window records, which caused the first real ambient drill FAIL).
Ambient drill questions use a DYNAMIC verifier (`verification.v_ambient_recall`) that reads ambient.json at
grade time — both question anchor and oracle resolve at grade time, so they never rot (live-truth pattern).
A2 (salience-gated proactivity, BRIEF_40) waits on ~a week of A0 baseline.

### Smart Context Retrieval (`get_smart_context`)
- Filters out `[AUTONOMOUS]`, `[TASK FAILED]`, `[TASK RETRY]` prefixed entries entirely
- Returns: last 3 **user-facing** episodic entries (recency) + top 2 semantic hits (MiniLM)
- Vault always included
- Deduplicates via set union — max ~5 episodic entries total
- Injected as assistant message with `[MEMORY_CONTEXT_BLOCK]` tags

### Conversation Hold (Topic 4 — human-like coherence)
Two tiers sit *on top* of the summary-based retrieval above so implicit references resolve from what was
actually said, not a lossy summary:
- **Phase 1 — Verbatim recent window (`recent_exchanges`).** `crud.append_recent_exchange(user, clara)` stores
  the raw last-10 exchanges — **user query + final answer ONLY, never the ReAct loop** (each side length-bounded
  600/900 chars). Written as a background task in `process_request` for `source=="user"`, decoupled from
  consolidation so a parse-failure never costs a turn. `get_smart_context` injects the last 6 as
  `[RECENT CONVERSATION — verbatim]`.
- **Phase 2 — Active-discourse state (`discourse_state`).** `memorize_episode` consolidation extracts a
  `discourse` field (1-5 concrete subject tags) — **user turns only** (`memorize_episode(..., source)` gates the
  discourse update to `source=="user"` so system/autonomous tasks don't pollute "what WE are discussing";
  episodic/facts/self-learning still extract from all sources). `crud.update_discourse_state()` keeps a rolling,
  deduped, most-recent-first, **cap-8** list (stale topics fall off as the conversation moves). Injected as
  `[CURRENTLY DISCUSSING: …]`. PERSONA has a paired directive: *resolve implicit references from the recent
  window + these tags; infer when the referent is clear, ask only when genuinely ambiguous* (preserves good
  pushback, doesn't train it away). Phases 3 (stronger semantic retrieval) + 4 (multi-turn Coherence Drill) are
  on the roadmap.

### Test Memory Isolation (`memory_mode`, 2026-06-07)
The daily harness and the Coherence Drill run against the **live backend** through `POST /query`, so without
isolation every test turn wrote to memory via `memorize_episode` / `append_recent_exchange` /
`update_discourse_state`. That actually happened: scripted drill **fixtures** (a fake "two job offers" dialogue,
"manager Priya", "brother in Lisbon", Kleppmann/DDIA, a PostgreSQL analytics service, Go auth/billing
microservices) became **real episodic memories**, and Clara surfaced them in a genuine conversation ("what's
occupying your head — the job decision?"). The 08:02-cron "monolithic vs microservices" L1-L5 knowledge question
likewise logged a false "Alkama asked about…" episode every morning. Fix: a **`memory_mode`** field on
`QueryRequest` threaded `api.py → orchestrator.submit_user_event → _handle_user_input` (task context) →
`process_request`. Tri-state, because the two test types have **different** needs:
- **`"full"`** (default; real users via WS + Telegram) — normal persistence.
- **`"ephemeral"`** (Coherence Drill) — `recent_exchanges` DOES write (the drill's recall test needs turn K to see
  turns 1..K-1 in the verbatim window), but `memorize_episode` (permanent episodic + vault + discourse) is SKIPPED.
  The drill resets the transient window between dialogues **and once after the run** (trailing `reset_fn`) so the
  last dialogue can't leak into the next real chat.
- **`"none"`** (L1-L5 harness) — single-turn questions write NOTHING (full isolation).

In `process_request`: `write_recent = memory_mode != "none"`, `write_episodic = memory_mode == "full"`. The
permanent-pollution source is `memorize_episode` only — `recent_exchanges`/`discourse_state` are transient,
reset-clearable working memory, not "real memories." **Anything routed through `/query` for testing MUST set
`memory_mode`** (`none` for the L1-L5 harness, `ephemeral` for multi-turn coherence) — never let test traffic
default into persistence.

### Memory Consolidation
Runs in `asyncio.to_thread` after every response (never blocks main path):
- Disposable non-reasoning DeepSeek instance extracts `summary` + `facts`
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

### Self-Knowledge (memory.json → `self_knowledge`)
Three categories stored under `self_knowledge`:
- `architecture_facts` — things discovered through use that aren't in CLAUDE.md
- `failure_patterns` — recurring failure modes CLARA has encountered
- `recovery_methods` — successful recovery strategies for known failure classes

Seeded with 8 entries from stress test analysis (bench_logger lock, two-phase start_search,
api.py root location, CHAT no-tool limitation, RAG stale-doc pattern, list_directory empty
recovery, chunk-limit recovery). Injected as `[SELF KNOWLEDGE]` block into the **LLM paths only**
(CHAT/FAST/DELIBERATE) — **NOT the Interpreter** (2026-06-07): the interpreter only routes and
doesn't need operational learnings, so `get_smart_context(..., include_self_knowledge=False)` feeds it
a context without the block (the block lives in `crud._self_knowledge_block()` and is appended to
`llm_context` in `process_request`). SK was previously injected into BOTH calls → counted twice → a
material slice of the CHAT token bloat. Keep the entry count under the 20-cap (deduped 30→18 on
2026-06-07) so the block stays small.

**Auto-population (Phase B):** `memorize_episode()` consolidation prompt includes a
`self_learning` extraction field. Only fires when CLARA made a mistake and corrected it,
or discovered a new architectural fact. Explicitly excluded: routine successes, Alkama facts,
things already in CLAUDE.md.

`crud.py` function: `add_self_knowledge(category, key, detail)` — dedup-guarded write.

### Filesystem Map (memory.json → `filesystem_map`)
Hierarchical path tree: `drive → dir object → file null`. Injected as compact `[FILE SYSTEM MAP]`
block on every query — CLARA sees the filesystem structure she has already explored.

Seeded with all paths from `known_locations` plus actual `core_logic/` and project root files.

**Auto-population:** After any successful MCP filesystem tool call (`read_file`, `write_file`,
`list_directory`, `create_directory`, `get_more_search_results`), `_update_filesystem_map()`
in `tool_executor.py` runs. Extracts paths from args (reliable) and attempts to parse
`list_directory` results for children (JSON first, text fallback). Search results scanned by
regex for Windows path patterns. All parsing is defensive — never raises, never disrupts execution.

`crud.py` functions: `merge_filesystem_path(path)` (additive tree merge),
`remove_filesystem_path(path)` (stale entry removal), `_serialize_filesystem_map()` (context
injection serializer).

---

## Persona System

Defined in `core_logic/system_prompt.py`:

```python
PERSONA            # Shared identity — injected into ALL three paths
CHAT_SYSTEM_PROMPT # PERSONA + minimal chat operational line
SYSTEM_PROMPT      # PERSONA + full DELIBERATE operational block
```

**FAST path:** `format_llm` gets `PERSONA + "Format the tool result into a natural response."`
plus a strict fidelity constraint (added 2026-05-30 after the Q12 fabrication): present the tool
output faithfully — do NOT interpret, analyze, or assert behavior, correctness, existence, or
runtime effects beyond what the output literally states. A search result is presented as file:line,
never as a verdict on what the code does. This is Rule-19 parity for the FAST formatter, which has
no ReAct loop and therefore no Rule 19. The same principle is also a shared PERSONA guardrail.
The prompt also requires reproducing numbers/values/identifiers digit-for-digit. As a structural
backstop (format_llm is a non-reasoning relay and once transposed `print(2**16)=65536`→`65636`),
`_run_fast` has a **numeric-fidelity guard**: for `python_repl`, if any number the tool printed is
not preserved in the formatted response, it returns the RAW tool output. Targeted to numbers, so it
never over-triggers on legitimate reframing ("True"→"97 is prime") or comma-formatting.
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

**Disconnect safety (current mechanism):** all sends go through `_broadcast()`, which try/excepts per socket and prunes dead connections from `active_connections` on failure — a client disconnect mid-stream can never error-spam. (The older `send_update` `client_state` guard this section used to describe was replaced by broadcast-with-prune; the WS handler also discards its socket in a `finally`.) Additionally (Brief 37): both `submit_user_event` call sites (WS `handle_message` + `/query`) are wrapped in `asyncio.wait_for(…, 600s)` — any bug that drops a response future now produces an honest timeout message instead of permanent silence (the cancelled future makes a late worker result drop harmlessly via the `future.done()` guard).

---

## Autonomous System

### Background Scheduler
- `health_check` — every 2 minutes
- `memory_maintenance` — every 5 minutes  
- `context_warmup` — every 10 minutes

**Heartbeat hygiene (Brief 37, 2026-06-10):** routine results from these three triggers are NO
LONGER written to episodic memory (they had accumulated 468/1028 episodes of pure noise) — an
`[AUTONOMOUS]` episode is logged only when something notable happened (repair/prune/failure, gated
in `_run_worker`). `memory_maintenance` now runs a real **janitor sweep** every ~6h: deletes
traces/logs >14d and benchmarks >30d, removes upload temp files >1d (age-gated so a Brief-35
detached retry can still use a fresh one), prunes terminal task rows >7d
(`TaskGraph.prune_terminal`), and rotates memory.json backups (keep 3). The orchestrator
tick-trace is also gated (emit on change or 60s heartbeat) — the unconditional ~10/sec idle tick
had accumulated 653 MB of traces. Episodic retrieval + growth counting use ONE shared filter:
`crud.SYSTEM_PREFIXES = ("[AUTONOMOUS]", "[TASK")` — prefix-matched so every `[TASK …]` variant is
excluded (the old 3-prefix literal let `[TASK SOFT-RETRY]` episodes leak into user-facing context).
Scheduler + EnvironmentWatcher also DEDUPE triggers: a same-trigger (same-path) task still
pending/running suppresses a new one (no rebuild pile-ups, no backlog stacking).

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

`IGNORED_PATTERNS` (in `environment.py`) filters high-churn noise before it ever reaches the queue —
`__pycache__`, `.pyc`, `tasks.db*`, `.log`, `session_`, `.faiss`/`.pkl`, atomic-write temps, editor
swaps, **and `node_modules` / `.git`**. The `node_modules` entry is load-bearing: on 2026-06-04 an
`npm install` under `core_logic/interface/` emitted **12,640** `file_change` events in a single harness
run, each spawning an autonomous task, saturating the orchestrator until user requests timed out at 180s.
Anything matching a pattern (substring) is dropped by `_should_ignore`. **Root cause fixed (2026-06-05):**
the frontend had been living *inside* `core_logic/` (`core_logic/interface/`), which is what put `node_modules`
— and the interface source files — in the watched tree (they kept generating `[AUTONOMOUS]` file_change
episodes on every run). It was **moved back to repo-root `interface/`**, removing the frontend from the
watched tree entirely. The `node_modules`/`.git` ignore patterns remain as defense-in-depth.

### SIMPLE_TRIGGERS
Known lightweight system tasks bypass the Interpreter and go directly to `run_background_task`.
Unknown/complex system tasks go through the full Interpreter → Router → Execution pipeline.

### Concurrent Task Resource Safety (Layers 2+3 + ResourceLedger)

Three layers protect concurrent user tasks from filesystem conflicts:

**Layer 2 — Active Task Awareness:** When a user task starts, `_run_worker` checks what other tasks are currently `running`/`active` and injects an `[ACTIVE TASKS]` block into `full_context`. The LLM sees what else is in-flight before it starts executing. Soft signal only — no enforcement.

**Layer 3 — Live Resource Ledger (dispatch-time):** `_task_resources` dict on the orchestrator (`task_id → {"reads": set, "writes": set}`) is populated via `resource_callback` as tools execute. `ConflictDetector.check()` receives this at dispatch time so pending tasks see what running tasks are actually touching. Cleans up in `_run_worker` finally.

**ResourceLedger — Mid-execution conflict detection** (`core_logic/resource_ledger.py`):
Module-level singleton `resource_ledger` shared across all tasks. Two mechanisms:

1. **Read-modify-write protection:** `record_read(task_id, path, content)` hashes file content after every successful `read_file`. Before every `write_file`, `check_write(task_id, path)` re-hashes the file on disk and compares. If changed → returns `"Error: Write blocked..."` to the ReAct loop so Clara re-reads first. Hash mismatch is caught before the lock is even attempted.

2. **Pure-write exclusivity:** `acquire_write(path, task_id)` acquires an `asyncio.Lock` per path, held only for the duration of the `mcp_client.call`. A coroutine suspends cooperatively at `await lock.acquire()` if another task holds the lock — resumes automatically when released. Covers the edge case where two tasks write a file neither has read.

`release_task(task_id)` cleans up all read hashes in `_run_worker` finally alongside `_task_resources`.

**Threading:** `task.id` injected into `task.context["task_id"]` in `_handle_user_input` → extracted in `process_request` → passed to `_run_fast` / `run_task` → `execute_fast` / `execute_deliberate` in `tool_executor.py` (both accept `task_id=None`, ops are no-ops when None so background tasks are unaffected).

**Latency impact:** Negligible. Hash check is one extra disk read at write time (<1ms for typical files). Uncontested lock acquire is nanoseconds.

### Retry Architecture
`MAX_ATTEMPTS = 3`. On failure: summarize failure context, create new task with
`failure_summary` in context. At max attempts: resolve future with failure message, log to episodic.

---

## Observability

- **Session logs:** `logs/session_YYYY-MM-DD_HH-MM-SS.log` — every request, full response text
  - `>> [FAST] Response:` — full FAST response
  - `>> [CHAT] Response:` — full CHAT response
  - `>> [DELIBERATE] Final Answer:` — full DELIBERATE final answer
  - `>> [MEMORY_CONTEXT] Injecting into LLM:` — full memory context (file only, not console). **Currently commented out** in `crud.get_smart_context` (the old line referenced "Grok"; Grok is gone — the LLM is DeepSeek).
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

Usage captured from OpenAI SDK `response.usage` / final streaming chunk `chunk.usage`:

- `prompt_tokens` — input tokens
- `completion_tokens` — output tokens
- `total_tokens` — sum
- `prompt_cache_hit_tokens` — tokens served from DeepSeek disk cache (billed at 1/10 rate)

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

**STATUS: LIVE — ground truth as of 2026-06-11.** Alkama provisioned `GEMINI_API_KEY` and it was wired
up + validated (probe image correctly described on the first non-503 attempt). The vision tool
`analyze_image_grok` in `tools.py` calls `model="gemini-2.5-flash"` via the `google-genai` SDK, reading
`GEMINI_API_KEY` from `core_logic/.env`. The free tier throws transient **503 UNAVAILABLE** under load —
the tool retries 3× with backoff (8s/16s) before reporting failure. The registry's "[CURRENTLY
UNAVAILABLE]" description prefix is applied only when the key is missing, so it self-clears at startup.
This also unblocks the `markitdown-ocr` follow-up (scanned PDFs) and the future Ambient Awareness
screenshot sensor (BRIEF_36 F.6/F.7).

History: vision originally used **Grok Vision** (`grok-4-1-fast-non-reasoning` via `xai_sdk`, injected with
`set_xai_client`). It was rewritten to Gemini 2.5 Flash in commit `edf81a8` (2026-05-30) but the Gemini key
was never provisioned, and the Grok/xAI path was removed. The function and wrapper are still *named*
`analyze_image_grok` / `analyze_images_grok` (legacy naming) and the module comment still says "Grok Vision
API" — those names are vestigial; **Grok is gone** (see "LLM Models in Use" — the whole stack moved to
DeepSeek in Brief 28). `core_logic/sight.py` (Moondream2) is kept but not imported.

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

## Telegram Integration (Brief 27)

`core_logic/telegram_bot.py` — `TelegramBot` + `TelegramNotifier`.

Messages from Alkama's personal Telegram bot are processed as `user_input` events through
the full orchestrator pipeline (`submit_user_event`), identical to the web UI.
`source="user"`, full memory and persona active, identical routing.

Uses **long-polling** (not webhooks) — no public URL or tunnel required. Works behind VPN,
dynamic IP, or any network. Poll interval: 1s, timeout: 10s.

**Security:** `TELEGRAM_ALLOWED_CHAT_ID` env var gates all incoming messages. Any sender
not matching the allowed chat ID is silently rejected before any processing occurs.

**`TelegramNotifier` singleton** (`notifier`) available for proactive outbound messaging
from any component:
```python
from .telegram_bot import notifier
await notifier.send("Memory maintenance complete.")
await notifier.send("**Bold alert**", parse_markdown=True)
```
No-ops gracefully if Telegram is not configured (env vars missing).

**MarkdownV2 escaping** (`_to_telegram_md`): all Telegram special chars escaped, then
bold/italic/code re-applied. Responses split at paragraph boundaries if over 4096 chars.

**Config:** `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_CHAT_ID` in `core_logic/.env`.
Telegram bot is optional — server starts normally if env vars are not set.

**Bot startup/shutdown** wired into `api.py` lifespan after tool registry injection.
Requires: `pip install "python-telegram-bot>=21.0"`

---

## Conventions

### LLM Models in Use

**Inference:** DeepSeek V4 Flash (`deepseek-v4-flash`, resolved from `core_logic/llm_config.py`
`DEEPSEEK_MODEL`, env-overridable — G24, 2026-08-01) via OpenAI-compatible API
(`base_url: https://api.deepseek.com`) for all LLM calls — Interpreter, FAST format_llm,
CHAT stream, DELIBERATE ReAct loop, memory consolidation.

**Vision:** Gemini 2.5 Flash via `google-genai` SDK (`google.genai.Client`) — **LIVE as of 2026-06-11** (`GEMINI_API_KEY` set; 503-retry built in).

**Embeddings:** MiniLM (`all-MiniLM-L6-v2`) locally on CUDA — episodic embeddings and tool registry only.

Disk cache enabled by default on DeepSeek — PERSONA + system prompt form a stable prefix
cache hit after the first request. Cache hit tokens tracked in `TokenUsage.cached_tokens`
via `prompt_cache_hit_tokens` field on usage object.

Migration from xAI Grok SDK → DeepSeek OpenAI-compatible SDK (Brief 28, May 2026).
All three paths (CHAT/FAST/DELIBERATE) use `DEEPSEEK_MODEL` (default `deepseek-v4-flash`; the
`deepseek-chat` alias was retired 2026-07-25 and the name is now centralized in one source,
`core_logic/llm_config.py` — G24, so the next rename is one line). Async calls via `AsyncOpenAI`,
memory consolidation uses sync `OpenAI` client (runs in `asyncio.to_thread`).

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

**MarkItDown:** Microsoft's `markitdown-mcp` STDIO server, connected at startup in the `api.py` lifespan immediately after DC, via `mcp_client.connect("markitdown", sys.executable, ["-m", "markitdown_mcp"])` (runs in the backend's own venv — no separate path config). Registered under server name "markitdown" with one tool: `convert_to_markdown(uri)` — converts PDF / DOCX / XLSX / PPTX / EPUB and 20+ formats to clean Markdown. Fills the gap DC `read_file` cannot: binary office formats (DC returns gibberish on them). `uri` is a `file:///`, `http(s):`, or `data:` URI. Install: `pip install markitdown-mcp markitdown[pdf]`. **Dependency hazard:** markitdown's `magika` dep installs CPU `onnxruntime`, which shadows `onnxruntime-gpu` and removes `CUDAExecutionProvider` (would silently drop Kokoro TTS to CPU). Fix: uninstall plain `onnxruntime`, force-reinstall `onnxruntime-gpu` (it satisfies `import onnxruntime` for magika AND provides CUDA). Verify with `onnxruntime.get_available_providers()` after any markitdown reinstall. OCR for scanned/complex PDFs (`markitdown-ocr` → Gemini) is a deferred follow-up — plain-text PDFs and all office formats work without it.

**Document upload pipeline (mirrors the image path, 2026-06-01):** a document can now be attached in the UI end-to-end, not just read by an on-disk path. Frontend: the file picker (`<input accept="image/*,.pdf,.docx,…">`) branches in `handleImageUpload` — `image/*` → `selectedImage` (vision path), everything else → `selectedFile = {name, data}` (a base64 data-URL). `sendMessage` sends it under a `file` field. Backend: `api.py` reads `payload.get("file")` and threads `file_data` through `submit_user_event` → `_handle_user_input` (task context) → `process_request(file_data=…)` — exactly parallel to `image_data`. In `process_request`, `file_data` is base64-decoded to `temp_doc_<uuid><ext>` (original extension preserved), and the prompt is augmented with `[SYSTEM: A document named '…' has been uploaded and saved at 'PATH'. Use convert_to_markdown with uri 'file:///PATH' …]`. The interpreter/ReAct then calls `convert_to_markdown`; `_build_args_from_query` maps the flat URI to the tool's single required `uri` arg. Image + document in the same turn both keep their notes (the document block appends to `final_prompt` rather than rebuilding it).

**Pre-Interpreter injection:** Before every `interpret()` call, `tool_registry.search(q_emb_cpu, top_k=8)` runs and top-8 schemas are appended to context under `[DISCOVERED_TOOLS]` tag. Interpreter uses these for accurate tool names and args.

**Mandatory injection (Brief 25):** After cosine top-8 search, `process_request` checks `ENUMERATION_KEYWORDS` (find, list, all, search, what files, directory, folder, etc.) against the query. If matched, `list_directory` and `start_search` schemas are appended to `discovered` if not already present. Max discovered set grows to top_k + 2. Prevents cosine similarity from missing enumeration tools when queries describe the target (image files) rather than the operation (list directory).

**DC description cleaning (Brief 24):** MCP tool descriptions are cleaned at registration time in `register_server_tools()`. Boilerplate (`\nIMPORTANT:`, `\nThis command can be referenced`, etc.) is stripped so each DC tool's embedding reflects its actual function rather than shared boilerplate. Raw descriptions from the MCP handshake are not retained. `format_tool_schemas_for_context()` truncates to 150 chars in the injected context to keep token cost low.

**TOOL_ARG_DEFAULTS (Brief 24):** `_build_args_from_query()` in `tool_executor.py` applies default values for known multi-required-arg DC tools after mapping the primary arg: `start_process → timeout_ms: 10000`, `read_process_output → timeout_ms: 5000`, `interact_with_process → timeout_ms: 8000`, `list_directory → depth: 0`, `write_file → mode: "rewrite"` (five entries — the `write_file` default was previously undocumented here; corrected 2026-08-07 after the 08-07 evening drill answer correctly said five against a doc that said four). A companion `TOOL_ARG_NORMALIZERS` runs immediately AFTER the defaults and repairs known-wrong VALUES (currently `write_file`: `mode` `w`→`rewrite`, `a`→`append`) — defaults fill absent args, normalizers fix present ones. Only fills args not already present — never overwrites explicit values. `list_directory` default is 0 (immediate contents only) — prevents silent chunk-limit overflow when the model omits the depth arg on dense directories.

**tool_search architectural note (Brief 23):** `tool_search` is NOT in the tool registry and is NOT embedded or returned by `registry.search()`. It is injected directly into the DELIBERATE \[SYSTEM MODE: TASK\] prompt. This prevents it from appearing in \[DISCOVERED_TOOLS\] and being mistakenly selected by the Interpreter as an action tool for arbitrary queries. `VALID_TOOLS` in `parse_action` is built dynamically from the registry at call time (| {"tool_search"}) so all MCP tools are always valid without manual maintenance. DELIBERATE can always call `tool_search` to discover filesystem/process/MCP capabilities by semantic query.

**Tool executor routing:**

- `execute_fast(tool_name, args_dict, ...)` — FAST path, structured args from Interpreter
- `execute_deliberate(tool_name, query_str, ...)` — DELIBERATE path, flat string from ReAct Action
- Both routes to native Python functions or `await mcp_client.call(server, tool, args)` based on `registry.get_server(tool_name)`
- `_build_args_from_query` maps flat DELIBERATE query string to MCP tool's required args (transitional until Pattern B streaming migration)

**Atomic search (`_atomic_search`, added 2026-05-29):** `start_search` is a two-phase DC tool — the
first call returns a session handle with `Status: RUNNING / Total results: 0`, and real results only
arrive from `get_more_search_results`. A model reading "0 results" while RUNNING could not tell it
from "no matches" — the cause of confident false negatives (e.g. MAX_ATTEMPTS / resource_callback
reported as absent), and FAST (single-shot) could never make the second call at all. The executor now
intercepts `start_search` in both `execute_fast` and `execute_deliberate`, polls
`get_more_search_results` until `Status: COMPLETED` (up to MAX_SEARCH_POLLS=20, ~7s), and returns
only the terminal result — so the caller never sees the ambiguous mid-state and FAST searches work.
On timeout it appends an explicit "PARTIAL/INCONCLUSIVE — not confirmation that no matches exist"
note so a slow search is never misread as absence. It also (a) drops a meaningless `filePattern`
(`*`/`**`/empty — that means "all files" = the default and once produced a spurious 0 that became a
false "os.replace does not exist"), and (b) appends a Rule-19 reminder to any COMPLETED-with-0-results
search ("0 results... NOT proof the string is absent... read a known file to confirm") — putting the
reminder in the tool output where it is seen in the moment. `python_repl` (`run_python_code`) injects
a UTF-8-defaulting `open()` into its exec namespace so model code that omits `encoding=` does not hit
Windows cp1252 charmap errors on UTF-8 files.

*fs\_ tool names:*\* Changed from `fs_read_file`/`fs_write_file`/`fs_list_directory`/`fs_run_command` to DC's native names (`read_file`, `write_file`, `list_directory`, `start_process`). Old names return "not found" → FAST escalates → DELIBERATE calls tool_search → finds correct DC tool.

**Scaling:** Every new MCP server: `mcp_client.connect()` → `tool_registry.register_server_tools()` → `tool_registry.rebuild_embeddings()`. Zero changes to TOOL_ARG_SCHEMAS or system prompt.

**Config:** `core_logic/.env` requires `DC_NODE_PATH` and `DC_CLI_PATH`. Registry + MCP init in `api.py` lifespan after `orchestrator._broadcast_fn` injection. `clara.tool_registry` and `clara.mcp_client` injected after `rebuild_embeddings()` completes.

### ReAct Loop Format Enforcement

Rules 11-19 in SYSTEM_PROMPT:

- Rule 11: After a Glint that answers the question, next output MUST be Thought → Final Answer. No prose dumps, no markdown headers before Final Answer.
- Rule 12: Never simulate or fabricate metrics/statistics/telemetry. python_repl must not be used to generate random numbers presented as real data.
- Rule 13: FILESYSTEM RESOLUTION — when given a filename, use `start_search` first (confirms existence + returns exact path in one call, no chunk-limit risk). Only fall back to `list_directory` (no depth) if search returns nothing. Never use `list_directory` as the first move for a named file. `list_directory` depth: omit or use 0 by default — immediate contents only. Only use depth > 0 when subdirectory structure is explicitly needed AND the directory is known to be sparse. Dense directories (`__pycache__`, model weights, indexes) overflow at depth > 0.
- Rule 14: ACTION FORMAT IS MANDATORY — every Action must be a valid JSON array. No markdown, no prose, no code fences. A malformatted Action cannot be parsed and wastes the turn.
- Rule 15: TOOL DISCOVERY — for filesystem, process, or MCP-backed operations not in the core tools list, call tool_search first with a semantic query. Use returned schemas exactly. One retry with refined query allowed; do not repeat the same query.
- Rule 16: COMPLETION CHECK — before writing Final Answer, Thought must confirm every sub-task is complete or genuinely impossible. Partial results do not constitute a complete answer.
- Rule 17: TOOL SELECTION (enumeration vs parsing) — DC tools for directory listing/discovery (NEVER python_repl for filesystem enumeration); python_repl ONLY for single-line parsing/computation on a known path, and ALWAYS with `encoding='utf-8'` when opening files. Multi-line python embedded in a JSON Action is escape-fragile and frequently fails to parse — prefer `read_file` for multi-step inspection.
- Rule 18: ARCHITECTURE SELF-KNOWLEDGE — questions about CLARA's own modules, file locations, paths, modes, or class behavior must be verified against CLAUDE.md + the source file before answering. Never answer architecture questions from parametric memory alone — it drifts and fabricates.
- Rule 19: NEGATIVE CLAIMS & VERIFICATION HONESTY — a tool returning no results, an error, or a "search/index" problem is a TOOL FAILURE, not evidence of absence. Never conclude "X does not exist / is not defined / not in this file" without confirming via an independent reliable method (e.g. a known-path python_repl read). And never claim to have read, searched, or verified more than you actually did.

**Chunk-limit error class (Rule 4):** When a tool returns "chunk exceed the limit" or "Separator is not found", the response is too large for the stdio transport. Recovery: retry the SAME tool on the SAME path with reduced scope — omit depth, use a narrower subpath, or read a specific file by name instead of listing a directory. Do NOT change the path or assume the error means the file/directory doesn't exist.

Safety net in run_task: a turn with no Thought/Action/Final Answer markers is off-format. On LATE
turns (5+) its content is returned as an implicit Final Answer. On EARLY turns (≤4) the loop injects
a corrective and continues — and that corrective (fixed 2026-05-30) tells the model its last turn was
NOT shown to the user, so if it already answered it must re-send the answer IN FULL prefixed with
`Final Answer:` (never "as above" / "already delivered"). This fixes the Q11/Q17 bug where a complete
answer written off-format on an early turn was discarded and the model only referenced it as "above",
leaving the user with no actual content.

**Hallucination detection (two forms):**

1. **Bare Glint** — model emits a `Glint:` line without a preceding Action. Loop detects it, strips fabricated content, appends truncated assistant message, injects corrective system message ("Glints can ONLY come from actual tool execution"), increments turn counter, and `continue`s — forcing a real tool call on the next turn.

2. **Inline fabrication** — model writes `Action: [...]` then immediately writes a fabricated `Glint:` in the same turn (before the system executes anything). Loop splits on `"Glint:"`, keeps only the `pre_glint` portion (the real Action), appends it as the assistant turn, injects a corrective message, and `continue`s — the system then executes the Action normally on the next turn.

`pre_glint` is computed once before the if/elif/else branch to avoid duplication. Turn budget applies to both cases. The custom `Glint:` token (replacing "Observation") reduces hallucination pressure from training bigrams on the word "Observation".

### Vision Tool Improvements

`analyze_image_grok` in [tools.py](http://tools.py) **(LIVE as of 2026-06-11 — see Vision Tool section; name is legacy)**:

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
`user_profile.environment.known_locations` in `memory.json` holds labeled shortcut path mappings (human-readable bookmarks).
`get_smart_context` injects a `[KNOWN LOCATIONS]` block into every context string.
Add entries manually to `memory.json` when new paths need to be known. Format:
```json
"environment": {
  "known_locations": {
    "Screenshots": "C:\\Users\\alkam\\OneDrive\\Pictures\\Screenshots",
    "AGENT_ZERO (Clara)": "E:\\ML_PROJECTS\\AGENT_ZERO"
  }
}
```

### Self-Knowledge (CLARA's persistent self-model)
`self_knowledge` is a top-level key in `memory.json`. It stores CLARA's operational learnings about her own architecture, failure patterns, and recovery methods — things discovered through use that are not in CLAUDE.md.

Three categories:
- `architecture_facts` — definitive facts about how the system works (e.g., which file contains which handler, two-phase tool behavior)
- `failure_patterns` — specific past mistakes with the correct approach documented alongside (status: active | resolved)
- `recovery_methods` — specific procedures that worked when a tool or path failed

Each entry has: `id`, `summary`/`trigger`/`problem`, `detail`/`correct_approach`/`method`, `confidence`, `learned_at`, `status`.

Injection: `get_smart_context()` injects all `status: active` entries as `[SELF KNOWLEDGE]` block into every request. CLAUDE.md takes precedence — self_knowledge complements but never overrides architecture documentation.

Write path: `crud.add_self_knowledge(category, entry)` with exact-string dedup on the primary key field. Called from `memorize_episode()` when consolidation extracts a `self_learning` entry.

Cap: keep under 20 total entries. Mark resolved entries `"status": "resolved"` when the underlying code fix is applied — they are excluded from injection automatically and can be pruned manually.

### Filesystem Map (progressively discovered path tree)
`filesystem_map` is a top-level key in `memory.json`. It holds a hierarchical tree of the filesystem that CLARA builds up as she explores via tool calls.

Schema:
- Drive letter = top-level key (`"C"`, `"E"`)
- Directory = key → object `{}` (empty = known but unexplored; populated = partially known)
- File = key → `null`

```json
"filesystem_map": {
  "E": {
    "ML_PROJECTS": {
      "AGENT_ZERO": {
        "api.py": null,
        "core_logic": { "agent.py": null, "tools.py": null }
      }
    }
  }
}
```

Injection: `get_smart_context()` serializes the tree as compact indented text under `[FILE SYSTEM MAP]`. Files shown inline per directory (up to 8, then `[+N more]`); unexplored dirs labeled `[unexplored]`.

Write paths:
- `crud.merge_filesystem_path(path_str, is_file)` — additive merge, never overwrites existing nodes
- `crud.remove_filesystem_path(path_str)` — removes a stale entry on confirmed not-found

Population is manual/Phase-A only. Phase B will wire `tool_executor.py` to auto-populate after successful filesystem tool calls.

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
- Runs at startup via `lifespan` in `api.py` — the build runs in a thread but IS awaited, so startup blocks for the few seconds the rebuild takes
- Auto-rebuild triggered by `rag_rebuild` event when any source file changes
- Hot-reload via `reload_rag_engine()` in `tools.py` — updates the global `RAG_ENGINE` in place
  without restarting the server

**Chunk settings:** `chunk_size=800`, `chunk_overlap=80`, markdown-aware separators
(`\n## `, `\n### `, `\n\n`, `\n`).

To add a new permanent document: drop it into `core_logic/docs/` and restart (or wait for
auto-rebuild if the file lands in a watched path).

### Files That Are Dead / Legacy
- `core_logic/sight.py`: Moondream2 vision, no longer imported. Vision later moved to Grok, then to Gemini 2.5 Flash, which has been **LIVE since 2026-06-11**; the older "keyless/non-functional stub" wording here contradicted the Vision Tool section above and was corrected 2026-09-18. **Its model weights, `core_logic/moondream_brain/` (3.68 GB, 74 files), were DELETED 2026-09-18**, reachable only through this file, which nothing imports. `sight.py` itself is kept deliberately: it is the subject of a live evening drill question, so deleting it would break a Rule-19 absence check to save 4 KB.
- `core_logic/tool_descriptions.json` — was MiniLM embedding source, Interpreter replaced this role
- `core_logic/ears.py` — superseded by `core_logic/voice.py`, no longer imported
- `core_logic/kokoro_mouth.py` — superseded by `core_logic/voice.py`, no longer imported
- Architecture PNG (`Clara_Architecture_Fixed_And_Updated.png`) — outdated, does not reflect current system

**REMOVED 2026-09-18 (BRIEF_65 cleanup). Listed so the record says what left and why. All three are
recoverable from git history:**
- `cleanup_tasks.py` and `cleanup_orphaned_tasks.py`: one-off SQLite surgery that marked pending
  tasks invalidated. **Superseded by live automation:** `TaskGraph.prune_terminal(days=7)`
  ([task_graph.py:364](core_logic/task_graph.py#L364)) is called by the janitor sweep in
  [background_tasks.py:196](core_logic/background_tasks.py#L196). Recover: `git show 4d2d200:cleanup_tasks.py`
- `update_diagram.py`: a one-off editor for the architecture diagram JSON, carrying a hardcoded
  userId and timestamp, targeting the PNG listed as outdated directly above it.
  Recover: `git show 22cfb91:update_diagram.py`

⚠️ **Some load-bearing tooling lives outside the tracked tree.** It runs from `.git/hooks/pre-commit`
or from standing instructions outside this repo, so nothing in the tree imports it and a static
reachability pass will read it as dead. Never nominate it for deletion from such a pass.

### Branch
All work on `autonomous`. (`features/stream-and-functionality` was the old dev branch — closed long ago; do not reference it.) Never merge to `main` until the full system is validated.
