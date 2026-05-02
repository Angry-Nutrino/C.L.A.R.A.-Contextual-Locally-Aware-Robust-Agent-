# api.py
import sys
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import psutil
import platform
import asyncio
import uuid as _uuid

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), "core_logic"))

from core_logic.agent import Clara_Agent
from core_logic.task_graph import TaskGraph
from core_logic.event_queue import EventQueue, make_event
from core_logic.orchestrator import Orchestrator
from core_logic.background_tasks import BackgroundScheduler
from core_logic.environment import EnvironmentWatcher
from core_logic.tracer import Tracer
from core_logic.tools import set_task_graph, set_xai_client
from core_logic.session_logger import init_session_log, slog
from core_logic.bench_logger import init_bench_log, close_bench_log
from core_logic.tool_registry import ToolRegistry
from core_logic.mcp_client import MCPClient, MCPError
from core_logic.voice import VoiceCoordinator, set_voice
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
active_connections: set = set()  # live WebSocket connections — used for speaking_start/stop broadcast

async def broadcast_task_event(task_id: str, goal: str, state: str, priority: float = 0.5, source: str = "system"):
    """Broadcast task state changes to all connected clients for the task board."""
    global active_connections
    if not active_connections:
        return
    payload = {
        "type": "task_event",
        "task_id": task_id,
        "goal": goal,
        "state": state,
        "priority": priority,
        "source": source,
    }
    dead = set()
    for ws in active_connections:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    if dead:
        active_connections -= dead


async def _broadcast_speaking(is_speaking: bool):
    """Send speaking_start/speaking_stop to all connected WS clients."""
    msg_type = "speaking_start" if is_speaking else "speaking_stop"
    dead = set()
    for ws in active_connections:
        try:
            await ws.send_json({"type": msg_type})
        except Exception:
            dead.add(ws)
    if dead:
        active_connections.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clara, task_graph, event_queue, orchestrator, scheduler, env_watcher, tracer, voice

    # Startup
    slog.info("[API] Starting CLARA system...")
    init_bench_log("benchmarks")
    slog.info("[API] Benchmark logger initialized.")
    tracer = Tracer(enabled=True, traces_dir="traces")
    slog.info("[API] Tracer initialized.")
    clara = Clara_Agent()
    set_xai_client(clara.client)
    slog.info("[API] xAI client reference injected into tools.")
    task_graph = TaskGraph()
    set_task_graph(task_graph)
    slog.info("[API] TaskGraph reference injected into tools.")
    event_queue = EventQueue()
    orchestrator = Orchestrator(clara, event_queue, task_graph, tracer=tracer)
    await orchestrator.start()
    orchestrator._broadcast_fn = broadcast_task_event  # inject callback — avoids circular import
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

    await tool_registry.rebuild_embeddings(clara._encode)
    slog.info(f"[API] Tool registry ready: {tool_registry.tool_count} tools indexed.")

    clara.tool_registry = tool_registry
    clara.mcp_client = mcp_client
    slog.info("[API] Tool registry injected into agent.")

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

    yield  # server is live here

    # Shutdown
    slog.info("[API] Shutting down CLARA system...")
    close_bench_log()
    if voice:
        voice.unload()
        slog.info("[API] Voice system unloaded.")
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
        if websocket.client_state != WebSocketState.CONNECTED:
            return
        try:
            payload = {
                "type": type,
                "content": content,
                "turn_id": turn_id,
                "message_id": message_id,
            }
            if extra:
                payload["extra"] = extra
            await websocket.send_json(payload)
        except Exception as e:
            slog.debug(f"send_update skipped — connection closed: {e}")

    def _speak_ack(interpreted: dict, mode: str):
        if voice and voice.is_enabled():
            ack = voice.get_acknowledgment(interpreted, mode)
            if ack:
                voice.speak(ack, block=False)

    async def handle_message(user_text: str, image_data, message_id: str):
        try:
            async def on_step(content, type="thought", turn_id=None, extra=None):
                await send_update(
                    content, type=type,
                    turn_id=turn_id, message_id=message_id, extra=extra
                )
            response = await orchestrator.submit_user_event(
                text=user_text,
                image_data=image_data,
                on_step_update=on_step,
                on_interpreted=_speak_ack,
            )
            env_watcher.notify_interaction()
            if voice and voice.is_enabled():
                voice.speak(response, block=False)
            await websocket.send_json({
                "type": "final_answer",
                "content": response,
                "message_id": message_id,
            })
        except Exception as e:
            slog.error(f"[WS] handle_message failed: {e}")
            try:
                await websocket.send_json({
                    "type": "final_answer",
                    "content": f"Error: {e}",
                    "message_id": message_id,
                })
            except Exception:
                pass

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload    = json.loads(raw_data)
                msg_type   = payload.get("type", "")
                message_id = payload.get("message_id", str(_uuid.uuid4()))

                if msg_type == "voice_start":
                    if voice and voice.is_enabled():
                        voice.start_recording()
                    continue

                if msg_type == "voice_stop":
                    if voice and voice.is_enabled():
                        text = await voice.stop_recording_async()
                        if text:
                            asyncio.create_task(
                                handle_message(text, None, message_id)
                            )
                    continue

                if msg_type == "voice_interrupt":
                    if voice:
                        voice.interrupt_speech()
                    continue

                user_text  = payload.get("text", "")
                image_data = payload.get("image", None)
            except json.JSONDecodeError:
                user_text  = raw_data
                image_data = None
                message_id = str(_uuid.uuid4())

            # Fire and forget — do NOT await
            asyncio.create_task(
                handle_message(user_text, image_data, message_id)
            )

    except WebSocketDisconnect:
        active_connections.discard(websocket)
        slog.info("[API] Client disconnected.")


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
        if os.path.exists("core_logic/memory.json"):
            with open("core_logic/memory.json", "r") as f:
                memory = json.load(f)

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
        cpu_percent = psutil.cpu_percent(interval=0.1)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
