# api.py
import sys
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import psutil
import platform

# --- PATH SETUP ---
# This tells Python: "Look for agent.py inside the core_logic folder"
sys.path.append(os.path.join(os.path.dirname(__file__), "core_logic"))

# Import the Class
from core_logic.agent import Clara_Agent
from core_logic.session_logger import init_session_log

# Start session log before anything else
init_session_log()

app = FastAPI()

# Enable React to talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INSTANTIATION (The part we missed) ---
print("⚙️ Initializing System...")
# We create the 'Object' here. This loads the models into RAM immediately.
agent = Clara_Agent()
print("✅ Clara is Alive and Ready.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    async def send_update(content: str, type="thought", turn_id=None):
        try:
            await websocket.send_json({ "type": type, "content": content, "turn_id": turn_id })
        except Exception as e:
            print(f"Error sending update: {e}")

    try:
        while True:
            # 1. Receive the JSON payload from React
            raw_data = await websocket.receive_text()
            
            try:
                # 2. Parse it
                payload = json.loads(raw_data)
                user_text = payload.get("text", "")
                image_data = payload.get("image", None) # Might be None
            except json.JSONDecodeError:
                # Fallback if it's somehow just a string
                user_text = raw_data
                image_data = None
            
            # 3. Pass BOTH to the Agent
            # Notice we pass 'image_data' as a separate argument now
            response = await agent.process_request(
                query=user_text, 
                image_data=image_data, 
                on_step_update=send_update
            )
            
            await websocket.send_json({ "type": "final_answer", "content": response })
            
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/soul")
async def get_soul():
    """
    Returns the Agent's perception of the User + Real Hardware Vitals.
    """
    # 1. Default / Fallback Profile
    profile = {
        "identity": {"name": "Alkama", "role": "Unknown", "location": "India", "clearance": "Lvl 1"},
        "skills": ["System Offline"],
        "mission": {"current": "Initializing...", "status": "WAIT", "phase": "Init"},
        "vitals": {"cpu": "Unknown", "gpu": "Offline", "memory_usage": "0%", "status": "OFFLINE"}
    }

    # 2. Load Real Memory (User Identity)
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
            
            # Combine 'Interests' and 'Tools'
            tools = user.get("preferences", {}).get("tools", [])
            interests = user.get("interests", [])
            profile["skills"] = (tools + interests)[:8] 
            
            profile["mission"] = {
                "current": state.get("current_phase", "Unknown"),
                "status": "IN PROGRESS",
                "phase": "V2.0"
            }
    except Exception as e:
        print(f"⚠️ Memory Load Error: {e}")

    # 3. Load Real Hardware Vitals (Separate Try-Block to prevent crashes)
    try:
        # Get RAM usage as percentage
        ram_percent = psutil.virtual_memory().percent
        
        # Get CPU Name (Windows friendly)
        cpu_name = platform.processor()
        # Clean up the long CPU name if needed
        if "Intel" in cpu_name: cpu_name = "Intel Core i5" # Simplified for UI
        if "AMD" in cpu_name: cpu_name = "AMD Ryzen 4800H" # Simplified for UI

        profile["vitals"] = {
            "cpu": cpu_name,
            "gpu": "RTX 3050", # Still hardcoded (Python needs torch to see GPU)
            "memory_usage": f"{ram_percent}%", # <--- REAL LIVE DATA
            "status": "ONLINE"
        }
    except Exception as e:
        print(f"⚠️ Vitals Error: {e}")

    return profile

if __name__ == "__main__":
    # We use port 8001 since 8000 was blocked earlier
    uvicorn.run(app, host="0.0.0.0", port=8001)