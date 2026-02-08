# api.py
import sys
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    
    async def send_update(content: str, type="thought"):
        try:
            await websocket.send_json({ "type": type, "content": content })
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

if __name__ == "__main__":
    # We use port 8001 since 8000 was blocked earlier
    uvicorn.run(app, host="0.0.0.0", port=8001)