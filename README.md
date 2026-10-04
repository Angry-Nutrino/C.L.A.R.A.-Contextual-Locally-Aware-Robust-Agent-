# C.L.A.R.A.

**Contextual Locally Aware Robust Agent**: an autonomous AI system that runs its orchestration, memory and voice on consumer hardware, with an optional governance gate on its file and process actions and a harness that grades it against its own source code.

> Built on an RTX 3050 laptop (4GB VRAM). The constraint is the interesting part.

---

## What this actually is

CLARA is built around two questions that matter once an agent runs unsupervised:

1. **How do you stop it doing something it shouldn't?** → a governance gate that adjudicates its file and process actions and writes a receipt for each one to a local ledger.
2. **How do you know it still works?** → a deterministic evaluation harness that grades her against the source, with no model in the grading path.

Everything else (the orchestrator, the router, the memory, the tooling) exists to make those two things possible on hardware that cannot hold a large model in VRAM.

---

## The execution pipeline

Every input enters through the same queue and loop. A user message, a background trigger and an environment event are all the same kind of thing. Known lightweight system tasks skip the Interpreter and run directly.

```
INPUT (user / system / background / environment)
        ↓
   EventQueue  (async priority queue)
        ↓
  OrchestratorLoop
        ↓
   Interpreter  →  structured intent JSON
        ↓
     Router  →  FAST | CHAT | DELIBERATE
        ↓
  Governance gate  →  ALLOW / REVIEW / DENY   (file and process tool calls, when the gate is on)
        ↓
    Execution  →  response
        ↓
  memorize_episode  (background)
```

**Three execution modes**, so compute is spent in proportion to how hard the task is:

| Mode | When | Latency |
|---|---|---|
| `FAST` | tool is known, high confidence, no planning needed | ~2-4s |
| `CHAT` | no tool needed, conversational | ~1.5-2.5s |
| `DELIBERATE` | planning required, low confidence, or FAST failed | ~5-30s (ReAct, up to 8 turns) |

FAST escalates to DELIBERATE on failure, injecting what was tried and why it failed, so the retry adapts.

---

## The two things worth looking at

### 1. Pre-execution governance

When the gate is on, each mutating file or process tool call is abstracted into a **privacy-preserving envelope** before it runs: an operation class, a coarse target class, a hash of the target, and coarse risk and reversibility labels. Raw content never leaves the machine.

That envelope is adjudicated by a pluggable policy adapter (a local policy, or an external governance engine), which returns `ALLOW`, `REVIEW`, or `DENY`. The envelope is signed when a signing key is configured, and the verdict and a receipt go to a local ledger. With a local policy, or in enforce mode, that happens **before** the action is allowed to proceed. In shadow mode an external engine is consulted off the hot path, so its verdict can land after the action runs.

The design principle: *a record written after the fact, by the process that acted, is not evidence.* Authorization and its evidence have to be causally upstream of execution.

The gate is off by default. Switched on, it runs in shadow mode (verdicts recorded, nothing blocked) unless set to enforce, with an explicit fail-open/closed posture.

### 2. Self-verification (the Drill)

A harness fires 23 questions per run at the live system, from a morning set and an evening set, and grades every checkable answer **deterministically**, with no model doing the grading: code questions against the source, computations by running them. The rest are marked unverifiable.

- The checkable question classes carry their own oracle: exact counts, set coverage, verbatim quotes, executable acceptance tests, and **absence-honesty probes** where the correct answer is "this does not exist" and any fabricated file:line citation auto-fails.
- A question that passes five runs in a row is flagged and moved one rung up a **six-level difficulty ladder**, so the benchmark gets harder as the system improves.
- **The grader is itself under test.** A fixture suite regression-tests the scoring engine on every run and stamps the report if the engine fails its own fixtures. This exists because a scoring bug once quietly failed a set of answers that were correct, and a broken evaluation looks exactly like a broken model until something checks.

```bash
python tests/test_harness.py --session morning   # or: evening
```

---

## Architecture map

| Module | Path | Role |
|---|---|---|
| API server | `api.py` | FastAPI + concurrent WebSocket |
| Agent | `core_logic/agent.py` | routing, FAST / CHAT / DELIBERATE execution |
| Interpreter | `core_logic/interpreter.py` | intent + routing decision |
| Orchestrator | `core_logic/orchestrator.py` | the persistent loop, dispatch, retry |
| TaskGraph | `core_logic/task_graph.py` | SQLite task state machine + crash recovery |
| EventQueue | `core_logic/event_queue.py` | async priority queue |
| Governance | `core_logic/admissibility.py` | envelopes, risk classification, verdicts |
| Memory | `core_logic/crud.py` | episodic log, fact vault, semantic retrieval |
| Tool registry | `core_logic/tool_registry.py` | native + MCP tool schemas, semantic search |
| Tool executor | `core_logic/tool_executor.py` | unified dispatch |
| Voice | `core_logic/voice.py` | Whisper STT + Kokoro TTS on CUDA |

**Memory** is a three-tier store: an episodic log with vector retrieval (recency + cosine similarity), a deduplicated long-term fact vault, and a verbatim recent-conversation window. Persistence is crash-safe (temp file → fsync → atomic replace), because a hard kill mid-write once truncated the store.

**Tooling** is 30+ tools across native Python functions and MCP servers, retrieved semantically per query, so each prompt carries a small matched subset.

---

## Running it

**Prerequisites:** Python 3.11, Node.js, NVIDIA CUDA 12.x, [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) on PATH, and FFmpeg.

```bash
git clone https://github.com/Angry-Nutrino/C.L.A.R.A.-Contextual-Locally-Aware-Robust-Agent-.git
cd C.L.A.R.A.-Contextual-Locally-Aware-Robust-Agent-

python -m venv jarvis_v2
jarvis_v2\Scripts\activate                 # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd interface && npm install && cd ..
```

**Configuration:** create `core_logic/.env`:

```env
DEEPSEEK_API_KEY=...      # cloud reasoning, via an OpenAI-compatible API
GEMINI_API_KEY=...        # vision
tavily_api=...            # web search
DC_NODE_PATH=...          # optional: Desktop Commander MCP
DC_CLI_PATH=...
```

**Start** (two terminals):

```bash
python api.py             # backend on :8001
cd interface && npm run dev   # dashboard on :5173
```

Or start the whole stack with `bash start_clara.sh` (and `bash stop_clara.sh` to stop it).

---

## Status and honesty

This is a personal system.

- The governance gate ships **off**. Switched on, it defaults to **shadow mode**; enforce mode exists, and the policy is still maturing.
- Code execution through `python_repl` is classified as mutating, but its dispatch doesn't call the gate yet, so it produces no envelope and no receipt.
- Layer 4 of the self-assessment ladder (the agent applying its own fixes) is **deliberately not built**. She writes fix proposals for persistent failures; every one is a review-only artifact and nothing is auto-applied.
- Some modules are legacy and nothing imports them (`sight.py`, `ears.py`, `kokoro_mouth.py`).
- It runs on 4GB of VRAM. That shapes almost every architectural decision here.

---

## License

No license granted. All rights reserved. Read it, learn from it, but please do not redistribute.
