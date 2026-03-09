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

### Environment Variables (.env in project root)
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
6. After completion, `memorize_episode()` consolidates memory via a disposable "ghost" LLM instance

### Key Modules

| Module | Path | Role |
|--------|------|------|
| API server | `api.py` | FastAPI + WebSocket endpoint + `/soul` vitals endpoint |
| Agent orchestrator | `core_logic/agent.py` | ReAct loop, intent gating, streaming, memory consolidation |
| Memory CRUD | `core_logic/crud.py` | JSON persistence: episodic log, long-term facts, user profile |
| Tools | `core_logic/tools.py` | python_repl, web_search, date_time, consult_archive |
| Vision | `core_logic/sight.py` | Moondream2 image analysis (lazy-loaded, VRAM-managed) |
| Speech-to-text | `core_logic/ears.py` | Faster-Whisper transcription |
| Text-to-speech | `core_logic/kokoro_mouth.py` | Kokoro ONNX TTS |
| GPU cleanup | `core_logic/memory_manager.py` | VRAM deallocation utility |
| RAG builder | `core_logic/rag_db_builder.py` | FAISS vector DB from PDFs in `./docs` |
| React UI | `interface/src/Layout.jsx` | Three-panel layout (sidebar, chat, neural stream) |
| WS hook | `interface/src/hooks/useClara.js` | WebSocket state management |

### Memory System (3-tier, JSON-based)
Stored in `core_logic/memory.json`:
- **Episodic log** — last 10 interaction summaries with timestamps
- **Long-term vault** — permanent extracted facts
- **User profile** — name, role, preferences, interests

### WebSocket Message Protocol
Backend sends messages with `type` field:
- `"thought"` — internal reasoning (displayed in neural stream panel, keyed by `turn_id`)
- `"stream"` — final answer tokens (accumulated in streaming buffer)
- `"status"` — system status updates
- `"final_answer"` — complete response, ends the interaction

## Conventions

### LLM Integration
- Primary LLM: **xAI Grok** via `xai_sdk.Client` (model: `grok-4-1-fast-reasoning`)
- Messages use xai_sdk chat objects with numeric roles: `1` (user), `2` (assistant), `3` (system)
- Streaming via `self.llm.stream()` generator
- Memory consolidation uses a **separate temporary LLM instance** to avoid TPM bloat on the main instance

### Tool Parsing
Tools are invoked via regex-parsed format in LLM output:
```
Action: tool_name[input]
```
Parser: `re.compile(r"Action:\s*\{?\s*(\w+)\[(.*)\]", re.DOTALL)` in `agent.py`

### Vision Model Pattern
Moondream2 (in `core_logic/moondream_brain/`) is **lazy-loaded** — only instantiated when `vision_tool` is called, then VRAM is freed via `free_gpu_memory()`. This pattern exists because all models share 4GB VRAM.

### Frontend
- React 19 + Vite + Tailwind CSS 4
- Dark theme with emerald (#10b981) accent
- Three-panel layout: left sidebar (identity/vitals), center (chat), right (neural stream/thoughts)
- `turn_id` in thought messages controls stacking vs overwriting in the neural stream panel
