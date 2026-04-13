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
from core_logic.tools import set_task_graph
from core_logic.session_logger import init_session_log, slog

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clara, task_graph, event_queue, orchestrator, scheduler, env_watcher, tracer

    # Startup
    slog.info("[API] Starting CLARA system...")
    tracer = Tracer(enabled=True, traces_dir="traces")
    slog.info("[API] Tracer initialized.")
    clara = Clara_Agent()
    task_graph = TaskGraph()
    set_task_graph(task_graph)
    slog.info("[API] TaskGraph reference injected into tools.")
    event_queue = EventQueue()
    orchestrator = Orchestrator(clara, event_queue, task_graph, tracer=tracer)
    await orchestrator.start()
    slog.info("[API] Orchestrator running.")
    scheduler = BackgroundScheduler(task_graph, event_queue, clara)
    await scheduler.start()
    slog.info("[API] BackgroundScheduler running.")
    env_watcher = EnvironmentWatcher(
        task_graph=task_graph,
        event_queue=event_queue,
        agent=clara,
        event_loop=asyncio.get_event_loop(),
        watch_paths=["core_logic/"],
    )
    await env_watcher.start()
    slog.info("[API] EnvironmentWatcher running.")

    yield  # server is live here

    # Shutdown
    slog.info("[API] Shutting down CLARA system...")
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

    async def send_update(content: str, type="thought",
                          turn_id=None, message_id=None):
        try:
            await websocket.send_json({
                "type": type,
                "content": content,
                "turn_id": turn_id,
                "message_id": message_id,
            })
        except Exception as e:
            slog.error(f"Error sending update: {e}")

    async def handle_message(user_text: str, image_data, message_id: str):
        try:
            async def on_step(content, type="thought", turn_id=None):
                await send_update(
                    content, type=type,
                    turn_id=turn_id, message_id=message_id
                )
            response = await orchestrator.submit_user_event(
                text=user_text,
                image_data=image_data,
                on_step_update=on_step,
            )
            env_watcher.notify_interaction()
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
                user_text  = payload.get("text", "")
                image_data = payload.get("image", None)
                message_id = payload.get("message_id", str(_uuid.uuid4()))
            except json.JSONDecodeError:
                user_text  = raw_data
                image_data = None
                message_id = str(_uuid.uuid4())

            # Fire and forget — do NOT await
            asyncio.create_task(
                handle_message(user_text, image_data, message_id)
            )

    except WebSocketDisconnect:
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
            profile["skills"] = (tools + interests)[:8]

            profile["mission"] = {
                "current": state.get("current_phase", "Unknown"),
                "status": "IN PROGRESS",
                "phase": "V2.0"
            }
    except Exception as e:
        slog.error(f"Memory Load Error: {e}")

    try:
        ram_percent = psutil.virtual_memory().percent
        cpu_name = platform.processor()
        if "Intel" in cpu_name: cpu_name = "Intel Core i5"
        if "AMD" in cpu_name: cpu_name = "AMD Ryzen 4800H"

        profile["vitals"] = {
            "cpu": cpu_name,
            "gpu": "RTX 3050",
            "memory_usage": f"{ram_percent}%",
            "status": "ONLINE"
        }
    except Exception as e:
        slog.error(f"Vitals Error: {e}")

    return profile


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
