# CLARA (Contextual Locally Aware Robust Agent)

> **"The Brain Gets a Body."**

A fully local, multi-modal AI orchestration engine designed to run on **consumer hardware (RTX 3050 - 4GB VRAM)**.

CLARA is not just a chatbot; it is a **Hybrid Agentic System** that combines local sensory processing (Vision/Voice) with high-level reasoning (Cloud/Local Hybrid) to execute complex tasks with stateful memory.

---

## 🧠 The Architecture (Update 3.0)

CLARA runs on a strict memory budget by using a **"Split-Brain" Architecture**. Sensors are local (Privacy/Speed), while reasoning is routed based on complexity.

| Module | Engine | VRAM Cost | Description |
| :--- | :--- | :--- | :--- |
| **The Soul (UI)** | React + Vite | N/A | Real-time dashboard visualizing memory streams, state, and thought processes. |
| **Gatekeeper** | Hybrid (Phi-3 / Grok) | Variable | The "Pre-frontal Cortex." Routes tasks: **Simple** (Local) vs **Complex** (Cloud). |
| **Hippocampus** | JSON Stream / Vector | Low | **Long-Term Memory.** Retains context across turns (e.g., "What was that watch I showed you?"). |
| **Ears** | Faster-Whisper | ~1.2 GB | Real-time transcription (Medium.en model). |
| **Mouth** | Kokoro v0.19 | ~0.8 GB | Neural TTS (ONNX/CUDA). Tuned for "Witty/Seductive" tone. |
| **Eyes** | Moondream2 | ~1.6 GB | Lightweight Vision Transformer for instant image analysis. |

---

## 🛠️ Prerequisites

Before installing, ensure your Windows environment is ready for GPU acceleration.

1.  **NVIDIA CUDA Toolkit:** Required for GPU acceleration (CUDA 12.x).
2.  **Node.js & npm:** Required to run the React Dashboard.
3.  **eSpeak NG (Critical):**
    * Download from [GitHub Releases](https://github.com/espeak-ng/espeak-ng/releases).
    * **Action:** Add `C:\Program Files\eSpeak NG` to your System PATH.
4.  **FFmpeg:**
    * Run `winget install ffmpeg` in PowerShell.

---

## 🚀 Installation Guide

### 1. Clone the Repo
```bash
git clone [https://github.com/Angry-Nutrino/AGENT_ZERO.git](https://github.com/Angry-Nutrino/AGENT_ZERO.git)
cd AGENT_ZERO
```


Backend Setup (The Brain)

```bash
# Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate
```

# Install PyTorch with CUDA support (CRITICAL STEP)
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

# Install the rest of the stack
```bash
pip install -r requirements.txt
```

# Frontend Setup (The Soul)
```bash
cd frontend
npm install
# Keep this terminal open later to run "npm run dev"
```
### Configuration (API Keys)
CLARA uses a Hybrid Brain.

Local Mode: Uses Phi-3 Mini (No keys required, higher VRAM usage).

Cloud Mode (Recommended for Logic): Uses Grok/xAI API.

Create a .env file in the root directory:

Code snippet
XAI_API_KEY=your_key_here
(Note: The system gracefully falls back if no key is found, but complex reasoning may degrade).

### 🎮 Usage
### Start the System
You need two terminals running (One for Brain, One for Body).

### Terminal 1: The Backend (FastAPI)

```bash
python api.py
# Server starts at http://localhost:8000
```
### Terminal 2: The Frontend (React)

```bash
cd frontend
npm run dev
# Dashboard opens at http://localhost:5173
```
 
### Module Testing
Debug individual senses without launching the full UI:

Hearing: python core_logic/ears.py

Speech: python core_logic/mouth.py

Vision: python core_logic/eyes.py

### ⚠️ Troubleshooting
"DLL load failed" (Kokoro): You forgot to install eSpeak NG or add it to your PATH. Restart your terminal.

Voice sounds robotic/slow: Check if onnxruntime-gpu is installed. If it's running on CPU, latency will be high.

Git Push Failures: Large model weights (Bin files) are ignored by .gitignore. Do not try to force push them.