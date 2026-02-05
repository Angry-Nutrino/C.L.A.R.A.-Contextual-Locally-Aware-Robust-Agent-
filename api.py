# api.py
import sys
import os
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- PATH SETUP ---
# This tells Python: "Look for agent.py inside the core_logic folder"
sys.path.append(os.path.join(os.path.dirname(__file__), "core_logic"))

# Import the Class
from core_logic.agent import Clara_Agent

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
    print("✅ React Client Connected")
    
    try:
        while True:
            # 1. Receive Raw Data
            raw_data = await websocket.receive_text()
            
            user_text = ""
            image_b64 = None
            
            # 2. Parse (Just extraction, no logic)
            try:
                payload = json.loads(raw_data)
                user_text = payload.get("text", "")
                image_b64 = payload.get("image", None)
            except json.JSONDecodeError:
                user_text = raw_data

            # 3. Log
            log_msg = f"🧠 Brain: Processing '{user_text}'"
            if image_b64: log_msg += " [with Image]"
            
            await websocket.send_text(json.dumps({
                "type": "log", 
                "content": log_msg + "..."
            }))
            
            # 4. PASS EVERYTHING TO AGENT
            try:
                # We pass the raw base64. The Agent decides what to do with it.
                response_text = agent.run(direct_input=user_text, image_data=image_b64)
                
                await websocket.send_text(json.dumps({
                    "type": "response", 
                    "content": response_text
                }))
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await websocket.send_text(json.dumps({"type": "log", "content": f"Error: {str(e)}"}))
                
    except Exception as e:
        print(f"❌ Connection Closed: {e}")

if __name__ == "__main__":
    # We use port 8001 since 8000 was blocked earlier
    uvicorn.run(app, host="0.0.0.0", port=8001)