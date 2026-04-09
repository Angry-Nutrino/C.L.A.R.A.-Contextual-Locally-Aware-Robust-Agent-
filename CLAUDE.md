# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (FastAPI + WebSocket)
```bash
# Activate venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies (CUDA 12.x required for GPU models)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run API server (port 8001)
python api.py
```

### Frontend (React + Vite)
```bash
cd interface
npm install
npm run dev      # Dev server on port 5173
npm run build    # Production build
npm run lint     # ESLint
```

### Environment Variables (.env in core_logic/)
- `XAI_API_KEY` — xAI Grok API key (primary LLM)
- `tavily_api` — Tavily API key (web search tool)

## Architecture

**C.L.A.R.A.** (Custom Local Aware Robust Agent) — a hybrid edge-cloud AI agent optimized for consumer GPUs (RTX 3050, 4GB VRAM).

### Core Data Flow
```
React UI (5173) ←WebSocket→ FastAPI (8001) → Clara_Agent → Tools/LLM
```

1. User sends `{text, image}` JSON over WebSocket (`/ws`)
2. `api.py` passes to `Clara_Agent.process_request()`
3. Agent classifies intent via **Gatekeeper** (TASK vs CHAT)
4. TASK → `run_task()` ReAct loop (max 8 turns): Thought → Action → Observation → Final Answer
5. CHAT → `run_chat()` direct streaming response
6. After completion, `memorize_episode()` consolidates memory in a background thread via `asyncio.to_thread`

### Key Modules

| Module | Path | Role |
|--------|------|------|
| API server | `api.py` | FastAPI + WebSocket endpoint + `/soul` vitals endpoint |
| Agent orchestrator | `core_logic/agent.py` | ReAct loop, intent gating, streaming, memory consolidation |
| Memory CRUD | `core_logic/crud.py` | JSON persistence: episodic log, long-term facts, user profile |
| Tools | `core_logic/tools.py` | python_repl, web_search, date_time, consult_archive |
| Tool descriptions | `core_logic/tool_descriptions.json` | MiniLM embedding source for gatekeeper scoring |
| Vision | `core_logic/sight.py` | Moondream2 image analysis (lazy-loaded, VRAM-managed) |
| Speech-to-text | `core_logic/ears.py` | Faster-Whisper transcription |
| Text-to-speech | `core_logic/kokoro_mouth.py` | Kokoro ONNX TTS |
| GPU cleanup | `core_logic/memory_manager.py` | VRAM deallocation utility |
| RAG builder | `core_logic/rag_db_builder.py` | FAISS vector DB from PDFs in `./docs` |
| Session logger | `core_logic/session_logger.py` | Per-session timestamped `.log` files in `logs/` |
| React UI | `interface/src/Layout.jsx` | Three-panel layout (sidebar, chat, neural stream) |
| WS hook | `interface/src/hooks/useClara.js` | WebSocket state management + reconnection |

### Memory System (3-tier, JSON-based)
Stored in `core_logic/memory.json`:
- **Episodic log** — last N interaction summaries with timestamps
- **Long-term vault** — permanent extracted facts (deduped via cosine similarity)
- **User profile** — name, role, preferences, interests

### WebSocket Message Protocol
Backend sends messages with `type` field:
- `"thought"` — internal reasoning (displayed in neural stream panel, keyed by `turn_id`)
- `"stream"` — final answer tokens (accumulated in streaming buffer)
- `"status"` — system status updates
- `"final_answer"` — complete response, ends the interaction

## Gatekeeper (Intent Routing)

The gatekeeper runs on every request before anything else. It does two things in sequence:

### 1. MiniLM Semantic Scoring
`self.miniLM` (SentenceTransformer `all-MiniLM-L6-v2`, loaded on CUDA) encodes the user query and computes cosine similarity against pre-built embeddings for each tool's `description` + `sub_descriptions` from `tool_descriptions.json`. Returns `top_tool`, `top_score`, and `tool_margin` (gap between #1 and #2 tool scores).

**Device strategy:** MiniLM encodes on CUDA for speed. All stored embeddings (tool embeddings, episodic embeddings) are immediately moved to CPU via `.to('cpu')` after encoding. This keeps the tensor list device-consistent and avoids CUDA/CPU mixing errors in `torch.stack`.

### 2. Phi3-mini Routing Decision
Ollama runs `phi3:mini` locally to classify intent as `TASK` or `CHAT` and select the best tool. Output is XML-parsed.

### 3. Boost Gate
If `top_tool != NONE` AND `top_score >= 0.35` AND `tool_margin >= 0.10`, the gatekeeper pre-executes the top tool and injects the result into Grok's context as a `[GATEKEEPER_BOOST]` user message before the main LLM call. This gives Grok pre-fetched data to reason from immediately.

The margin gate (`>= 0.10`) prevents boosting on ambiguous queries where two tools score similarly — only fires when there's a clear winner.

### 4. Smart Context Retrieval (`get_smart_context`)
Before every LLM call, memory context is retrieved via `crud.get_smart_context()`:
- **Last 3** episodic entries by recency
- **Top 2** semantic hits via MiniLM cosine similarity against `episodic_embeddings`
- Deduped, max 5 entries total
- Vault facts and user profile always included

Injected as an assistant message with `[MEMORY_CONTEXT_BLOCK]` tags so Grok recognizes it as its own prior context.

## Memory Consolidation

After every response, `memorize_episode(chat_snapshot)` runs in a **background thread** (`asyncio.to_thread`) so it never blocks the main response path.

- Uses a **disposable ghost LLM** (`grok-4-1-fast-non-reasoning`) — separate instance to avoid TPM bloat on the main Grok instance
- Extracts: `summary` (1-2 sentence description of the interaction) and `facts` (permanent vault entries)
- New episodic embedding is encoded with `self.miniLM` and stored CPU-side, appended to `episodic_embeddings` incrementally
- Vault dedup: new facts are cosine-similarity checked against existing vault entries (threshold 0.85) before saving

**JSON parse safety:** `parse_json_safely()` uses a 3-layer parser — direct parse, markdown fence strip, then regex extraction — to handle Grok wrapping output in code fences or adding escape sequences.

## Conventions

### LLM Integration
- Primary LLM: **xAI Grok** via `xai_sdk.Client` (model: `grok-4-1-fast-reasoning`)
- Messages use xai_sdk chat objects with numeric roles: `1` (user), `2` (assistant), `3` (system)
- Streaming via `self.llm.stream()` generator
- `self.llm` is re-initialized after every request (`self.client.chat.create(...)`) to clear context
- Memory consolidation uses a **separate temporary LLM instance** (`grok-4-1-fast-non-reasoning`)

### Action Format
Tools are invoked via JSON array format in LLM output:
```
Action: [{"tool": "tool_name", "query": "input"}]
```
Multiple tools can be batched in one action array when their inputs are independent (parallel execution via `asyncio.gather`). Parser in `parse_actions()` uses a 3-layer approach: direct JSON parse, bracket-counting extraction, then old-format `tool[input]` fallback.

### Session Logging
Every `api.py` startup creates a new timestamped log file in `logs/`. Use `slog` (from `core_logic/session_logger.py`) for all logging — never `print()` for anything that should appear in session logs.

### Vision Model Pattern
Moondream2 (in `core_logic/moondream_brain/`) is **lazy-loaded** — only instantiated when `vision_tool` is called, then VRAM is freed via `free_gpu_memory()`. CPU inference is completely unusable (>40 min). Always keep on GPU, always lazy-load.

### Frontend
- React 19 + Vite + Tailwind CSS 4 (no StrictMode — causes double WebSocket connections)
- Dark theme with emerald (`#10b981`) accent
- Three-panel layout: left sidebar (identity/vitals), center (chat), right (neural stream/thoughts)
- `turn_id` in thought messages controls stacking vs overwriting in the neural stream panel
- **No `className` on `<ReactMarkdown>`** — react-markdown v10 removed this prop. Wrap in a `<div>` instead.
- Messages persisted to `localStorage` (`clara_messages` key) — survive page refreshes and backend restarts
- WebSocket reconnects with exponential backoff (1s → 2s → 4s → ... → 30s cap) on disconnect

### Quote Feature
Users can highlight any text in any message bubble → floating QUOTE button appears → injects `> [Clara]: text` or `> [Alkama]: text` into the textarea. The system prompt instructs Grok to treat `[Clara]` quotes as references to its own prior output and `[Alkama]` quotes as user-provided context anchors.
