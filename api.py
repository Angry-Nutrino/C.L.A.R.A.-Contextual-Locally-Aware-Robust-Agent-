# api.py
import sys
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import psutil
import platform
import asyncio
import threading
import uuid as _uuid

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), "core_logic"))

# --- HF CACHE REDIRECT (MUST precede any HuggingFace import below) ---
# Point the HuggingFace cache into the repo (.hf_cache) instead of ~/.cache/huggingface.
# The user-profile cache became permission-walled to non-interactive / sandboxed contexts
# (2026-06-21 — collision + ACL state that survived icacls /reset and a reboot), which
# blocked every backend boot (MiniLM/Whisper would not load). A repo-local cache is writable
# by ALL contexts (cron, tool, interactive), so the backend now boots uniformly everywhere.
# setdefault so an explicit HF_HOME in the environment still wins.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"))

from core_logic.agent import Clara_Agent
from core_logic.task_graph import TaskGraph
from core_logic.event_queue import EventQueue, make_event
from core_logic.orchestrator import Orchestrator
from core_logic.background_tasks import BackgroundScheduler
from core_logic.environment import EnvironmentWatcher
from core_logic.tracer import Tracer
from core_logic.tools import set_task_graph, set_agent_ref
from core_logic.session_logger import init_session_log, slog
from core_logic.bench_logger import init_bench_log, close_bench_log
from core_logic.tool_registry import ToolRegistry
from core_logic.mcp_client import MCPClient, MCPError
from core_logic.voice import VoiceCoordinator, set_voice
from core_logic.telegram_bot import TelegramBot
from starlette.websockets import WebSocketState

# Start session log before anything else
init_session_log()

# --- Module-level singletons (set during lifespan startup) ---
clara: Clara_Agent | None = None
task_graph: TaskGraph | None = None
event_queue: EventQueue | None = None
orchestrator: Orchestrator | None = None
scheduler: BackgroundScheduler | None = None
env_watcher: EnvironmentWatcher | None = None
tracer: Tracer | None = None
tool_registry: ToolRegistry | None = None
mcp_client: MCPClient | None = None
voice: VoiceCoordinator | None = None
telegram_bot: TelegramBot | None = None
_whatsapp_task = None   # Brief 45 P1 — the read-only WhatsApp poller task (dormant unless WHATSAPP_ENABLED)
_ambient_task = None    # Brief 40 Y1c — the A2 ambient shadow loop task (dormant unless A2_MODE=shadow|live)
_a3_task = None         # Brief 36 F.7 — the A3 screen sensor task (dormant unless A3_SCREEN_SENSOR=on)
active_connections: set = set()  # live WebSocket connections — used for speaking_start/stop broadcast

async def _broadcast(payload: dict) -> None:
    """Send a JSON payload to all connected WS clients, pruning dead connections."""
    if not active_connections:
        return
    dead = set()
    for ws in list(active_connections):  # snapshot — a disconnect mid-await mutates the set and raises RuntimeError on live iteration
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    if dead:
        active_connections.difference_update(dead)


async def broadcast_task_event(task_id: str, goal: str, state: str, priority: float = 0.5, source: str = "system", message_id: str = ""):
    await _broadcast({
        "type": "task_event",
        "task_id": task_id,
        "goal": goal,
        "state": state,
        "priority": priority,
        "source": source,
        "message_id": message_id,
    })


async def _broadcast_speaking(is_speaking: bool):
    await _broadcast({"type": "speaking_start" if is_speaking else "speaking_stop"})


async def _broadcast_console(role: str, text: str, source: str, message_id: str = None):
    """Live cross-channel mirror — pushes a message into the master console the instant it
    happens, so telegram/whatsapp/voice exchanges appear WITHOUT a /history refresh. role is
    'user' | 'clara'; source is 'telegram' | 'whatsapp' | 'voice'. No-op if no UI is connected."""
    await _broadcast({"type": "console_message", "role": role, "content": text,
                      "source": source, "message_id": message_id})


async def _whatsapp_poll_loop():
    """Brief 45 P1 — every ~2s, compile the due 15s batches and route them. SURFACE (Shobha) →
    broadcast a whatsapp_alert + notifier + log; HOLD (everyone else) → archived to the console
    (source='whatsapp'), NO interrupt. Read-only — nothing is ever sent back to WhatsApp."""
    from core_logic.whatsapp_gate import poll
    from core_logic.conversations import record_message, record_whatsapp_held
    from core_logic.telegram_bot import notifier
    while True:
        try:
            for d in poll():
                sender, text, decision = d["sender"], d["text"], d["decision"]
                if decision == "surface":
                    # Priority sender (Shobha): into the chat feed + a live incoming alert + Telegram.
                    # source='whatsapp' makes the UI render it as an INCOMING bubble (left, badged), never
                    # as Alkama's own; the [sender] prefix carries the name on both live + /history reload.
                    record_message("whatsapp", "user", f"[{sender}] {text}")
                    await _broadcast({"type": "whatsapp_alert", "sender": sender,
                                      "content": text, "count": d["count"]})
                    try:
                        await notifier.send(f"WhatsApp — {sender}:\n{text}")
                    except Exception:
                        pass
                    slog.info(f"[WhatsApp] SURFACE {sender}: {text[:80]}")
                else:
                    # Everyone else (incl. spam): HELD QUIETLY — separate archive, NOT the chat feed,
                    # NOT broadcast. Reviewable on demand ('what did I miss on WhatsApp?').
                    record_whatsapp_held(sender, text)
                    slog.info(f"[WhatsApp] HOLD {sender} ({d['count']} msg) — archived quietly, no chat, no interrupt")
        except Exception as e:
            slog.warning(f"[WhatsApp] poll loop error: {e}")
        await asyncio.sleep(2)


async def _a3_screen_loop():
    """A3 screen sensor (Brief 36 F.7) — DORMANT unless A3_SCREEN_SENSOR=on. Each cycle: capture the screen,
    store ONLY a one-line Gemini description (the raw image is captured in-memory, used for the Gemini call,
    then deleted — never persisted). Self-gates per cycle, so disarming mid-run takes effect next tick.
    Read-only; non-fatal. Privacy contract lives in core_logic/screen_sensor.py."""
    from core_logic.screen_sensor import run_once, interval_seconds, a3_enabled
    while True:
        try:
            if a3_enabled():
                desc = await asyncio.to_thread(run_once)
                if desc:
                    slog.info(f"[A3] screen: {desc[:80]}")
        except Exception as e:
            slog.warning(f"[A3] loop error: {e}")
        await asyncio.sleep(interval_seconds())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clara, task_graph, event_queue, orchestrator, scheduler, env_watcher, tracer, voice

    # Startup
    slog.info("[API] Starting CLARA system...")

    # ── BRIEF 62 — POLICY GATE ON BOOT. This must run BEFORE anything authority-bearing starts. ──
    # Prints the effective security policy and the capability-manifest hash, then REFUSES TO START if
    # any security-relevant field is absent or holds a value we do not recognise AND falling back to
    # its default would give the system more authority.
    #
    # Why fail rather than warn: arming PYTHON_REPL_COMPUTE_ONLY on 2026-09-07 was a silent no-op
    # because the module reading it never loaded the env file, and on 2026-09-09 an invalid value was
    # measured to degrade silently toward permissive (ADMISSIBILITY_MODE=enforced -> "shadow";
    # ADMISSIBILITY_FAIL=close -> fail-OPEN). Both leave the operator believing a control is armed when
    # it is not. A backend that will not boot is loud; a control that is quietly off is not.
    #
    # 🔴 CONSEQUENCE, DELIBERATE AND ACCEPTED: a typo in one of those keys stops the backend, which
    # means the unattended 08:02 drill cron produces no run and no report. That is the trade, and it is
    # the one that was ratified. The exception message names the offending field.
    from core_logic import policy_config as _policy_config
    _policy_config.startup_check(strict=True)
    slog.info("[API] Policy config validated. manifest=%s" % _policy_config.manifest_hash())
    _pid_file = os.path.join(os.path.dirname(__file__), "clara_backend.pid")
    try:
        with open(_pid_file, "w") as _f:
            _f.write(str(os.getpid()))
    except Exception:
        pass
    init_bench_log("benchmarks")
    slog.info("[API] Benchmark logger initialized.")
    tracer = Tracer(enabled=True, traces_dir="traces")
    slog.info("[API] Tracer initialized.")
    clara = Clara_Agent()
    task_graph = TaskGraph()
    set_task_graph(task_graph)
    set_agent_ref(clara)
    slog.info("[API] TaskGraph + agent references injected into tools.")
    # Inject the agent's LIVE crud instance into the tool executor (Brief 36 C-35 —
    # it previously imported the crud CLASS, so fsmap auto-population never ran).
    from core_logic.tool_executor import set_db
    set_db(clara.db)
    slog.info("[API] Memory db reference injected into tool executor.")
    event_queue = EventQueue()
    orchestrator = Orchestrator(clara, event_queue, task_graph, tracer=tracer)
    await orchestrator.start()
    orchestrator._broadcast_fn = broadcast_task_event  # inject callback — avoids circular import
    orchestrator._send_message_fn = _broadcast  # general WS push for Brief 35 proactive retry delivery
    slog.info("[API] Orchestrator running.")
    scheduler = BackgroundScheduler(task_graph, event_queue, clara)
    await scheduler.start()
    slog.info("[API] BackgroundScheduler running.")
    env_watcher = EnvironmentWatcher(
        task_graph=task_graph,
        event_queue=event_queue,
        agent=clara,
        event_loop=asyncio.get_event_loop(),
        watch_paths=[
            "core_logic/",
            "CLAUDE.md",
            "briefs/ROADMAP.md",
        ],
    )
    await env_watcher.start()
    slog.info("[API] EnvironmentWatcher running.")

    # Build/verify RAG knowledge base at startup
    try:
        from core_logic.rag_db_builder import build_knowledge_base
        from core_logic.tools import reload_rag_engine
        slog.info("[API] Building RAG knowledge base...")
        await asyncio.to_thread(build_knowledge_base)
        reload_rag_engine()
        slog.info("[API] RAG knowledge base ready.")
    except Exception as e:
        slog.error(f"[API] RAG build failed at startup: {e}")

    # ── Tool Registry + MCP Client ────────────────────────────────────────────
    global tool_registry, mcp_client

    tool_registry = ToolRegistry()
    tool_registry.register_native_tools()

    mcp_client = MCPClient()

    dc_node = os.getenv("DC_NODE_PATH", "")
    dc_cli  = os.getenv("DC_CLI_PATH", "")

    if dc_node and dc_cli and os.path.exists(dc_node) and os.path.exists(dc_cli):
        try:
            dc_tools = await mcp_client.connect("desktop_commander", dc_node, [dc_cli])
            tool_registry.register_server_tools("desktop_commander", dc_tools)
            slog.info(f"[API] Desktop Commander connected: {len(dc_tools)} tools registered.")
        except MCPError as e:
            slog.warning(f"[API] Desktop Commander connection failed: {e}. Continuing without DC tools.")
    else:
        slog.warning("[API] DC_NODE_PATH or DC_CLI_PATH not set. DC tools unavailable.")

    # MarkItDown — STDIO MCP server (Microsoft). One tool: convert_to_markdown(uri).
    # Fills the gap DC read_file cannot: PDF / DOCX / XLSX / PPTX / EPUB and other
    # binary office formats → clean Markdown. Runs in the same venv as the backend.
    try:
        md_tools = await mcp_client.connect("markitdown", sys.executable, ["-m", "markitdown_mcp"])
        tool_registry.register_server_tools("markitdown", md_tools)
        slog.info(f"[API] MarkItDown connected: {len(md_tools)} tools registered.")
    except MCPError as e:
        slog.warning(f"[API] MarkItDown connection failed: {e}. Continuing without document-conversion tools.")

    await tool_registry.rebuild_embeddings(clara._encode)
    slog.info(f"[API] Tool registry ready: {tool_registry.tool_count} tools indexed.")

    clara.tool_registry = tool_registry
    clara.mcp_client = mcp_client
    slog.info("[API] Tool registry injected into agent.")

    # ── Telegram Bot ──────────────────────────────────────────────────────────
    global telegram_bot
    tg_token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat_id = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "")

    if tg_token and tg_chat_id:
        # Telegram is a PERIPHERAL notifier — a connectivity/auth failure (e.g.
        # api.telegram.org unreachable, VPN/region block) must NEVER take down the core
        # backend. Before the fix, get_me() raising TimedOut here crashed the whole lifespan
        # ("Application startup failed. Exiting.") and /soul never served — a Telegram outage
        # took the entire AI system offline. Degrade gracefully, exactly like the Voice block.
        try:
            telegram_bot = TelegramBot(orchestrator, tg_token, tg_chat_id)
            telegram_bot.on_console = _broadcast_console   # live console mirror (no refresh)
            await telegram_bot.start()
            slog.info("[API] Telegram bot active.")
        except Exception as e:
            telegram_bot = None
            slog.warning(f"[API] Telegram bot failed to start ({e}) — continuing without it.")
    else:
        slog.info("[API] Telegram not configured — TELEGRAM_BOT_TOKEN or CHAT_ID missing.")

    # Voice system — load after everything else; failure is non-fatal
    try:
        voice = VoiceCoordinator()
        voice.load()
        set_voice(voice)
        loop = asyncio.get_event_loop()
        voice.set_speaking_callback(
            lambda is_spk: asyncio.run_coroutine_threadsafe(
                _broadcast_speaking(is_spk), loop
            )
        )
        slog.info("[API] Voice system loaded.")
    except Exception as e:
        slog.warning(f"[API] Voice system unavailable: {e}. Continuing without voice.")
        voice = None

    # WhatsApp read-only poller (Brief 45 P1) — DORMANT unless WHATSAPP_ENABLED is set in .env.
    # Off by default so the Node service can be stood up + QR-logged before this ever runs; when on,
    # it compiles the 15s batches and routes them (Shobha → surface/notify, others → hold/store).
    # Non-fatal, like Voice/Telegram — a poller hiccup never affects startup.
    global _whatsapp_task
    _whatsapp_task = None
    if os.getenv("WHATSAPP_ENABLED", "").strip():
        try:
            _whatsapp_task = asyncio.create_task(_whatsapp_poll_loop())
            slog.info("[API] WhatsApp read-only poller started (WHATSAPP_ENABLED).")
        except Exception as e:
            slog.warning(f"[API] WhatsApp poller failed to start ({e}) — continuing without it.")
    else:
        slog.info("[API] WhatsApp poller dormant (set WHATSAPP_ENABLED in .env to activate).")

    # A2 ambient shadow loop (Brief 40 Y1c) — DORMANT unless A2_MODE is shadow|live in .env. In shadow it
    # logs candidate proactive remarks to ambient_shadow.jsonl (sends nothing); off = not even started.
    # Non-fatal like the others. Live delivery needs an injected notifier sink, so shadow can never spam.
    global _ambient_task
    _ambient_task = None
    try:
        from core_logic.ambient_loop import ambient_shadow_loop, a2_mode, set_broadcast
        set_broadcast(_broadcast)   # live sink pushes ambient nudges to the UI via this primitive
        if a2_mode() != "off":
            _ambient_task = asyncio.create_task(ambient_shadow_loop())
            slog.info(f"[API] A2 ambient loop started (A2_MODE={a2_mode()}).")
        else:
            slog.info("[API] A2 ambient loop dormant (set A2_MODE=shadow|live in .env).")
    except Exception as e:
        slog.warning(f"[API] A2 ambient loop failed to start ({e}) — continuing without it.")

    # A3 screen sensor (Brief 36 F.7) — DORMANT unless A3_SCREEN_SENSOR=on. Captures the screen on a slow
    # cadence and stores ONLY a one-line Gemini description (never the raw image). Off by default; non-fatal.
    global _a3_task
    _a3_task = None
    try:
        from core_logic.screen_sensor import a3_enabled, interval_seconds
        if a3_enabled():
            _a3_task = asyncio.create_task(_a3_screen_loop())
            slog.info(f"[API] A3 screen sensor ARMED (interval {interval_seconds() / 60:.0f} min).")
        else:
            slog.info("[API] A3 screen sensor dormant (set A3_SCREEN_SENSOR=on in .env to arm).")
    except Exception as e:
        slog.warning(f"[API] A3 screen sensor failed to start ({e}) — continuing without it.")

    yield  # server is live here

    # Shutdown
    slog.info("[API] Shutting down CLARA system...")
    try:
        if os.path.exists(_pid_file):
            os.remove(_pid_file)
    except Exception:
        pass
    close_bench_log()
    if _whatsapp_task:
        _whatsapp_task.cancel()
    if _ambient_task:
        _ambient_task.cancel()
    if _a3_task:
        _a3_task.cancel()
    if voice:
        voice.unload()
        slog.info("[API] Voice system unloaded.")
    if telegram_bot:
        await telegram_bot.stop()
    if mcp_client:
        await mcp_client.disconnect_all()
    await env_watcher.stop()
    slog.info("[API] EnvironmentWatcher stopped.")
    await scheduler.stop()
    slog.info("[API] BackgroundScheduler stopped.")
    await orchestrator.stop()
    slog.info("[API] Clean shutdown complete.")


app = FastAPI(lifespan=lifespan)

# Enable React to talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    async def send_update(content: str, type="thought",
                          turn_id=None, message_id=None, extra=None):
        if not active_connections:
            return
        payload = {
            "type": type,
            "content": content,
            "turn_id": turn_id,
            "message_id": message_id,
        }
        if extra:
            payload["extra"] = extra
        await _broadcast(payload)

    def _voice_ready() -> bool:
        return bool(voice and voice.is_enabled())

    def _speak_ack(interpreted: dict, mode: str):
        if _voice_ready():
            ack = voice.get_acknowledgment(interpreted, mode)
            if ack:
                voice.speak(ack, block=False)

    async def handle_message(user_text: str, image_data, file_data, message_id: str, via_voice: bool = False):
        try:
            # Upload ceiling (2026-07-03, UI-audit deferred item): base64 beyond ~11MB (≈8MB raw)
            # risks the WS transport cap and stalls the pipeline decoding it. The client enforces
            # 8MB with a friendly notice; this is the server-side belt.
            _MAX_B64 = 11 * 1024 * 1024
            _file_blob = (file_data or {}).get("data") if isinstance(file_data, dict) else file_data
            for _blob, _label in ((image_data, "image"), (_file_blob, "file")):
                if isinstance(_blob, str) and len(_blob) > _MAX_B64:
                    slog.warning(f"[WS] Rejected oversized {_label} upload ({len(_blob) // 1048576}MB base64) for {message_id}.")
                    await _broadcast({
                        "type": "final_answer",
                        "content": f"That {_label} is too large for me to take — the upload limit is 8MB.",
                        "message_id": message_id,
                        "source": "interface",
                    })
                    return

            # Brief 43.4 — voice cancel-filter: a spoken request ending in "leave it / never mind"
            # is REJECTED before process_request (never hits the LLM); Clara just acks. Text queries
            # are not filtered (this is the omnipresent-voice "false request" case). Deterministic.
            if via_voice and not image_data and not file_data:
                from core_logic.intent_filters import is_false_request
                if is_false_request(user_text):
                    slog.info(f"[Voice] False-request (cancel) detected — skipping process: {user_text!r}")
                    if _voice_ready():
                        threading.Thread(target=voice.speak, args=("Got it.",), kwargs={"block": False}, daemon=True).start()
                    await _broadcast({"type": "final_answer", "content": "Got it.", "message_id": message_id, "source": "voice"})
                    return

            async def on_step(content, type="thought", turn_id=None, extra=None):
                await send_update(
                    content, type=type,
                    turn_id=turn_id, message_id=message_id, extra=extra
                )
            # 600s ceiling (Brief 36 A-13): without it, ANY bug that drops a response
            # future means permanent silence for this message. On timeout the awaited
            # future is cancelled, so a late worker result is dropped harmlessly
            # (set_result is guarded by future.done()).
            response = await asyncio.wait_for(
                orchestrator.submit_user_event(
                    text=user_text,
                    image_data=image_data,
                    file_data=file_data,
                    message_id=message_id,
                    on_step_update=on_step,
                    on_interpreted=_speak_ack if via_voice else None,
                    channel=("voice" if via_voice else "interface"),
                ),
                timeout=600,
            )
            env_watcher.notify_interaction()
            if via_voice and _voice_ready():
                slog.info(f"[Voice] Speaking response ({len(response)} chars), speaking={voice.is_speaking()}")
                # Start synthesis in background thread before sending WS — hides synthesis latency
                threading.Thread(target=voice.speak, args=(response,), kwargs={"block": True}, daemon=True).start()
            await _broadcast({
                "type": "final_answer",
                "content": response,
                "message_id": message_id,
                "source": ("voice" if via_voice else "interface"),  # Brief 43.3 — live source badge
            })
        except asyncio.TimeoutError:
            slog.error(f"[WS] handle_message timed out after 600s (message {message_id}).")
            await _broadcast({
                "type": "final_answer",
                "content": ("This is taking far longer than it should — I've let it go for now. "
                            "The task may still finish in the background; ask me again in a moment."),
                "message_id": message_id,
            })
        except Exception as e:
            slog.error(f"[WS] handle_message failed: {e}")
            await _broadcast({
                "type": "final_answer",
                "content": f"Error: {e}",
                "message_id": message_id,
            })

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload    = json.loads(raw_data)
                msg_type   = payload.get("type", "")
                message_id = payload.get("message_id", str(_uuid.uuid4()))

                # The F4 in-interface voice path (voice_start/voice_stop WS handlers) was RETIRED 2026-06-24:
                # the standalone F10 hotkey (own-mic → POST /voice_query) replaced it, and the frontend no
                # longer sends these. The backend's persistent-mic capture (start_recording/stop_recording_async
                # in voice.py) is now orphaned — a follow-up cleanup. voice_interrupt stays (TTS-stop is still
                # a valid capability via interrupt_speech).
                if msg_type == "voice_interrupt":
                    if voice:
                        voice.interrupt_speech()
                    continue

                if msg_type == "cancel_task":
                    target_id = payload.get("task_id", "")
                    if target_id and orchestrator:
                        cancelled = await orchestrator.cancel_task(target_id)
                        await websocket.send_json({
                            "type": "task_cancelled",
                            "task_id": target_id,
                            "success": cancelled,
                        })
                    continue

                user_text  = payload.get("text", "")
                image_data = payload.get("image", None)
                file_data  = payload.get("file", None)
            except json.JSONDecodeError:
                user_text  = raw_data
                image_data = None
                file_data  = None
                message_id = str(_uuid.uuid4())

            # Fire and forget — do NOT await
            asyncio.create_task(
                handle_message(user_text, image_data, file_data, message_id)
            )

    except WebSocketDisconnect:
        slog.info("[API] Client disconnected.")
    finally:
        # Always remove — a non-Disconnect exception (e.g. send_json on a socket that
        # closed mid-reply) previously left the dead socket in active_connections.
        active_connections.discard(websocket)


@app.get("/soul")
async def get_soul():
    """
    Returns the Agent's perception of the User + Real Hardware Vitals.
    """
    profile = {
        "identity": {"name": "Alkama", "role": "Unknown", "location": "India", "clearance": "Lvl 1"},
        "skills": ["System Offline"],
        "mission": {"current": "Initializing...", "status": "WAIT", "phase": "Init"},
        "vitals": {"cpu": "Unknown", "gpu": "Offline", "memory_usage": "0%", "status": "OFFLINE"}
    }

    try:
        # Serve from the agent's LIVE in-RAM memory (Brief 36 D-5). The old disk read
        # was the documented READER causing os.replace PermissionError contention in
        # crud._save_memory — and the RAM dict is always fresher anyway.
        if clara is not None:
            memory = clara.db.memory

            user = memory.get("user_profile", {})
            state = memory.get("project_state", {})

            profile["identity"] = {
                "name": user.get("name", "Alkama"),
                "role": user.get("role", "Architect"),
                "location": "India",
                "clearance": "Lvl 5 (Admin)"
            }

            tools = user.get("preferences", {}).get("tools", [])
            interests = user.get("interests", [])
            profile["skills"] = (tools + interests)[:8] or [
                "Python (AsyncIO)", "React + Vite", "FastAPI",
                "Grok API", "Docker", "Generative AI"
            ]

            profile["mission"] = {
                "current": state.get("current_phase", "Unknown"),
                "status": "IN PROGRESS",
                "phase": "V2.0"
            }
    except Exception as e:
        slog.error(f"Memory Load Error: {e}")

    try:
        ram_percent = psutil.virtual_memory().percent
        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, 0.1)  # off-loop: 100ms blocking sample must not stall the event loop
        cpu_name = platform.processor()
        if "Intel" in cpu_name: cpu_name = "Intel Core i5"
        if "AMD" in cpu_name: cpu_name = "AMD Ryzen 4800H"

        # VRAM via torch if available
        vram_used_gb, vram_total_gb = 0.0, 4.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_used_gb  = round(torch.cuda.memory_allocated(0) / 1e9, 2)
                vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:
            pass

        profile["vitals"] = {
            "cpu":          f"{cpu_percent}%",
            "cpu_name":     cpu_name,
            "gpu":          f"{vram_used_gb}GB / {vram_total_gb}GB",
            "memory_usage": f"{ram_percent}%",
            "status":       "ONLINE"
        }
    except Exception as e:
        slog.error(f"Vitals Error: {e}")

    return {**profile, "version": "v2.6"}


class QueryRequest(BaseModel):
    text: str
    # Test-only endpoint. memory_mode isolates test traffic from Clara's real memory —
    # without it, scripted drill fixtures (fake job offers, manager "Priya", brother in
    # "Lisbon") leaked into episodic memory and Clara later surfaced them as real facts in
    # genuine conversations (2026-06-07 confabulation). Values:
    #   "none"      — write nothing (L1-L5 harness; single-turn, needs full isolation)
    #   "ephemeral" — transient recent_exchanges only, NO permanent episodic/vault
    #                 (coherence drill; needs within-dialogue recall, resets between dialogues)
    #   "full"      — normal persistence (never used by the harness; real users only)
    memory_mode: str = "none"
    # Brief 32: when True, the response includes the raw post-routing ReAct loop
    # (react_trace) so the harness can feed a FAILED query's actual turns to Clara for
    # Self-Assessment Layer 2 root-cause diagnosis. Off by default (real callers never need it).
    return_trace: bool = False

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    Simple HTTP endpoint for the daily test harness.
    Fires a query through the full orchestrator pipeline and returns the final answer.
    Local-only, unauthenticated — do not expose publicly.
    memory_mode defaults "none" so test traffic never persists to Clara's memory.
    """
    if not orchestrator:
        return {"response": "Error: orchestrator not ready."}
    try:
        response = await asyncio.wait_for(
            orchestrator.submit_user_event(
                text=req.text, memory_mode=req.memory_mode, return_trace=req.return_trace,
                channel="harness",   # Brief 43.3 — /query is the test harness; excluded from the console
            ),
            timeout=600,  # Brief 36 A-13 — a dropped future becomes an honest error, not a hang
        )
        # With return_trace, the worker resolves with {"response", "react_trace"}; pass it through.
        if isinstance(response, dict):
            return response
        return {"response": response}
    except Exception as e:
        slog.error(f"[/query] Error: {e}")
        return {"response": f"Error: {e}"}


@app.get("/history")
async def history_endpoint(limit: int = 200, include_harness: bool = False):
    """Brief 43.3 — the persistent cross-channel conversation archive that feeds the unified master
    console (one thread, source-badged). Harness/drill traffic excluded by default. Read-only, local."""
    try:
        from core_logic.conversations import load_recent
        return {"messages": load_recent(limit=limit, include_harness=include_harness)}
    except Exception as e:
        slog.error(f"[/history] Error: {e}")
        return {"messages": []}


class VoiceQueryRequest(BaseModel):
    audio_b64: str            # base64 WAV captured by the F10 hotkey listener (mic on-press only)
    speak: bool = True        # speak the reply via Kokoro (set False if simultaneous play/record distorts)


@app.post("/voice_query")
async def voice_query_endpoint(req: VoiceQueryRequest):
    """Brief 44.1 — the global F10 hotkey path. The standalone listener records its OWN audio (mic
    only while F10 is held) and POSTs the WAV here; the backend transcribes with the loaded Whisper,
    applies the cancel-filter, runs the full pipeline (channel='voice'), and optionally speaks. The
    backend's persistent mic is NOT used for capture — honoring 'mic only on press'. Local-only."""
    import base64, tempfile, os as _os
    v = voice            # module global, set in lifespan (api.py imports set_voice, not get_voice)
    if v is None or not orchestrator:
        return {"transcript": "", "response": "Voice/orchestrator unavailable."}
    tmp = None
    try:
        try:
            data = base64.b64decode(req.audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
                f.write(data)
        except Exception as e:
            return {"transcript": "", "response": f"bad audio: {e}"}
        transcript = await asyncio.to_thread(v.transcribe_file, tmp)
    finally:
        if tmp:
            try: _os.unlink(tmp)
            except OSError: pass
    if not transcript:
        return {"transcript": "", "response": ""}
    import uuid as _uuid
    mid = str(_uuid.uuid4())
    # Live console: surface what the hotkey heard immediately (reuses the user_transcript
    # handler → User bubble + query card + "thinking"), source-badged 'voice'.
    await _broadcast({"type": "user_transcript", "content": transcript, "message_id": mid, "source": "voice"})
    # Cancel-filter (Brief 43.4): "leave it / never mind" → reject before the LLM.
    from core_logic.intent_filters import is_false_request
    if is_false_request(transcript):
        if req.speak and v:
            threading.Thread(target=v.speak, args=("Got it.",), kwargs={"block": False}, daemon=True).start()
        await _broadcast({"type": "final_answer", "content": "Got it.", "message_id": mid, "source": "voice"})
        return {"transcript": transcript, "response": "Got it.", "cancelled": True}
    try:
        response = await asyncio.wait_for(
            orchestrator.submit_user_event(text=transcript, channel="voice"), timeout=600)
    except Exception as e:
        await _broadcast({"type": "final_answer", "content": f"Error: {e}", "message_id": mid, "source": "voice"})
        return {"transcript": transcript, "response": f"Error: {e}"}
    if isinstance(response, dict):
        response = response.get("response", "")
    if req.speak and v:
        threading.Thread(target=v.speak, args=(response,), kwargs={"block": True}, daemon=True).start()
    await _broadcast({"type": "final_answer", "content": response, "message_id": mid, "source": "voice"})
    return {"transcript": transcript, "response": response}


@app.get("/ambient_feed")
async def ambient_feed_endpoint(limit: int = 50):
    """Recent ambient nudges for the interface feed (Brief 40 Y1e — passive, novelty-gated). The UI loads
    this on connect so the feed is populated even for nudges surfaced while it was closed. Local-only.
    TTL (2026-07-08, Alkama's 07-04 rule "a nudge from yesterday should not load at all"): only entries
    from the last AMBIENT_FEED_TTL_H hours (default 12) are served — a nudge is context-sensitive to its
    moment; the full history stays in the ledger for calibration, it just doesn't LOAD."""
    from datetime import datetime, timedelta
    from core_logic.ambient_loop import read_ledger
    rows = read_ledger(limit=max(1, min(int(limit or 50), 200)))
    ttl_h = float(os.getenv("AMBIENT_FEED_TTL_H", "12"))
    cutoff = (datetime.now() - timedelta(hours=ttl_h)).isoformat(timespec="seconds")
    return {"feed": [r for r in rows if str(r.get("ts", "")) >= cutoff]}


class AmbientFeedback(BaseModel):
    id: str
    vote: str            # "up" | "down" — the 👍/👎 calibration tap


@app.post("/ambient_feedback")
async def ambient_feedback_endpoint(fb: AmbientFeedback):
    """Record a 👍/👎 on an ambient nudge (Brief 40 §4 — calibrate the novelty picks). Local-only."""
    from core_logic.ambient_loop import set_feedback
    return {"ok": set_feedback(fb.id, fb.vote)}


class WhatsAppIncoming(BaseModel):
    sender: str
    text: str


@app.post("/whatsapp_incoming")
async def whatsapp_incoming_endpoint(msg: WhatsAppIncoming):
    """Brief 45 P1 (read-only) — the external whatsapp-web.js service POSTs each incoming message here.
    It enters the 15s per-sender debounce; the backend poller compiles + routes it (Shobha → surface,
    others → hold). NEVER sends anything back to WhatsApp. Local-only."""
    try:
        from core_logic.whatsapp_gate import record_incoming
        record_incoming(msg.sender, msg.text)
        return {"ok": True}
    except Exception as e:
        slog.error(f"[/whatsapp_incoming] Error: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/reset_conversation")
async def reset_conversation_endpoint():
    """Clear ONLY the short-term conversational substrate (recent_exchanges + discourse_state)
    — episodic memory, vault and self_knowledge are untouched. Used by the Coherence Drill to
    isolate scripted dialogues. Local-only, unauthenticated — do not expose publicly."""
    if not clara:
        return {"ok": False, "error": "agent not ready"}
    try:
        cleared = clara.db.reset_conversation_state()
        return {"ok": True, "cleared": cleared}
    except Exception as e:
        slog.error(f"[/reset_conversation] Error: {e}")
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
