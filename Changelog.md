# C.L.A.R.A. System State

## Current Objective
- Migrating Gatekeeper from Grok API to local Phi-3 (50ms latency target).

## Architecture Decisions (The "Why")
- [2026-03-06] Severed memory consolidation from `self.llm` to prevent 30k TPM quadratic bloat. Memory now uses a stateless, disposable ghost-instance(chat_snapshot).
- [2026-03-06] Replaced synchronous sampling with a state-machine stream. React frontend UI utilizes `turn_id` to stack thoughts instead of overwriting.

## Recent Kills/Bugs (Changelog)
- [2026-03-06] Fixed React visual cascade glitch by overwriting the active stream block.
- [2026-03-06] Added regex guillotine `\n?(?:Action|Final Answer|Observation):?` to prevent syntax bleed into the UI stream.
- [2026-03-08] Fixed `m.role` integer/string mismatch that was leaking the system prompt into the memory summarizer (memorize_episode, chat_snapshot).

## [2026-03-09] Gatekeeper: MiniLM Embedding-Based Tool Routing

### What Changed
Replaced the 3-6s Grok API gatekeeper call with a local MiniLM (`all-MiniLM-L6-v2`) embedding classifier that routes user queries to the correct tool in <50ms on GPU.

### Why It Works — Max-Similarity vs Mean-Pooling
The original approach **mean-pooled** all tool descriptions into a single embedding per tool. This fundamentally caps cosine similarity at ~0.50-0.60 for tools that serve diverse query types (e.g., `web_search` must match "gold price", "cricket match", AND "OpenAI news" — semantically distant queries whose average embedding matches none of them well).

**Max-similarity** stores all sub_description embeddings individually and takes the **highest** cosine score across them. Each sub_description is a short phrase that mirrors how a user would ask for that tool (e.g., `"what is the price of gold"` for `web_search`, `"calculate the square root of 1445"` for `python_repl`). The query only needs to closely match *one* prototype to score highly. The highest-scoring tool wins.

### Score Improvements (Before → After)

| Query | Mean-Pool Score | Max-Sim Score |
|-------|:-:|:-:|
| "price of gold" | 0.52 | **0.93** |
| "square root of 144" | 0.29 | **0.97** |
| "what time is it" | 0.49 | **0.95** |
| "phone number in resume" | 0.53 | **0.80** |
| "analyze this image" | 0.50 | **0.63** |
| "hello how are you" (no tool) | 0.06 | 0.21 (correct reject) |

**100% routing accuracy** across all 34 test queries (20 baseline + 14 adversarial). Zero misroutes.

### Routing Logic
1. Compute max-similarity for each tool. **Highest-scoring tool wins.**
2. If top score `>0.40` — route to that tool (high confidence).
3. If top score `0.15–0.40` — low confidence zone; route to tool but flag for review.
4. If **all** tools `<0.15` — no tool needed, fall through to direct chat.

### Files
- `core_logic/tool_descriptions.json` — Optimized tool descriptions (short, query-mirroring phrases)
- `core_logic/scripts/eval_embeddings.py` — Eval harness with 20-query test bank + adversarial support
- `core_logic/test.py` — Updated with max-similarity method and new descriptions

## Active Bugs / Debt
- ~~6-second API wall on Gatekeeper intent classification (Pending local Phi-3 swap).~~ Resolved by MiniLM embedding router.
- Nvidia Riva ASR integration pending.
- MiniLM gatekeeper not yet wired into `agent.py` — currently validated in `test.py` only.

## [2026-03-10] Gatekeeper: MiniLM Embedding-Based Tool Routing(Final_Update): Fixes #1
- Sticking to Max Pooling in Cosine Similarity
- Observed Drastic improvements in comparison to mean pooling


Tier	Threshold	Queries	Examples
T1: AUTO	> 0.70	15/21	square root (1.00), phone number (1.00), gold price (0.92), news (0.97), cricket (0.84), bitcoin (0.87)
T2: PHI3	0.40–0.70	3/21	Samsung S25 (0.55), iPhone+% (0.58), 30% of 5000 (0.51)
T3: GROK	< 0.40	3/21	"hello" (0.25), "tell me a joke" (0.35), "climate change" (0.11)

## [2026-03-10] Gatekeeper: GateKeeper Refractoring(Isolation & Leveraging Phi3:Mini) Fixes #2
- Dropped The prior Grok gatekeepr's architecture entirely
- Replaced with MiniLM+Phi3:Mini for faster inference(90% faster inference 700-900ms Over 4-6 Seconds)
- Boost Tool Calling method: Prior tool call with Phi3 mini inference before The main React Loop

## [2026-03-10] Snall Bugs:
- Removed Local Asyncio imports causing local variable assumptions conflicts.
- Fallback tightened.

## [2026-03-11] Gatekeeper: Phi-3 Mini XML Reliability Fix (Fixes #3)
- Replaced LangChain `Ollama` wrapper with native `ollama.chat()` — Phi-3 Mini is a chat-tuned model; the wrapper was calling `/api/generate` (text completion) which bypasses the chat template and causes hallucination
- Added stop sequence `["</analysis>"]` — model was running past the XML block into irrelevant text; stop forces it to halt at the right point
- Added system message constraining output to XML only
- Strengthened Low Confidence routing rule (explicit: "You MUST select NONE")
- Added TOOL QUERY RULES section so `tool_query` is always populated when a tool is selected
- Fixed `context_needed` XML parser: `(.*?)</context_needed>` → `(TRUE|FALSE)` — closing tag was being mangled by Phi-3 into echoed prompt text
- **Result:** 15% → 100% pass rate across 20 eval queries (8 outer iterations via `gatekeeper_test.py`)
