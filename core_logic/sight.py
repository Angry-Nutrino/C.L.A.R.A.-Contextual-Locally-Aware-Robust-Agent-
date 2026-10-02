# ⚠️ THE MODEL THIS FILE LOADS NO LONGER EXISTS. `core_logic/moondream_brain/` (3.68 GB) was deleted
# on 2026-09-18. Vision moved Moondream2 -> Grok -> Gemini 2.5 Flash and has run on Gemini since
# 2026-06-11; the live tool is `analyze_image_grok` in tools.py. Nothing imports this module, which
# is why the weights were removable at all, and `analyze_image` already returns a clean "Model
# folder not found" instead of raising (see the os.path.exists guard below).
#
# The file is KEPT ON PURPOSE. It is the subject of a live evening drill question (a cross-source
# absence check that asks Clara to verify CLAUDE.md's claim that nothing imports sight), so deleting
# it would silently break a Rule-19 honesty test to save 4 KB. Retire the question first, then this.

import sys
import os

# --- 1. GLOBAL PATH SETUP (The Universal Fix) ---
# We calculate these once so they work for both imports and standalone execution.
current_file_path = os.path.abspath(__file__)
VISION_DIR = os.path.dirname(current_file_path)     # Path to .../core_logic
PROJECT_ROOT = os.path.dirname(VISION_DIR)          # Path to .../AGENT_Zero

# If we are running standalone (not via main.py), tell Python where the root is.
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ------------------------------------------------

import torch
from PIL import Image
from transformers import AutoTokenizer

# Now these imports work safely in all scenarios
from core_logic.memory_manager import free_gpu_memory
from core_logic.moondream_brain.hf_moondream import HfMoondream as Moondream

def analyze_image(image_path, prompt="Describe this image."):
    """
    Lazy Loader:
    1. Finds the model path using the pre-calculated global var.
    2. Loads the model (ONLY when this function is called).
    3. Runs inference.
    4. Cleans up VRAM immediately.
    """
    print(f"\n👁️ Vision Tool Triggered for: {image_path}")
    
    # Use the Global Variable we defined at the top
    model_path = os.path.join(VISION_DIR, "moondream_brain")
    
    if not os.path.exists(model_path):
        return f"❌ Error: Model folder not found at: {model_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    tokenizer = None
    answer = "Error during vision processing"

    try:
        print(f"   -> Loading Moondream from: {model_path}")
        
        # --- 2. LOAD (The Heavy Lift) ---
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # We use the custom class (Moondream) directly
        model = Moondream.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True,
            dtype=dtype,
            local_files_only=True # Safety: Force local use
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # --- 3. INFERENCE ---
        print(f"   -> Analyzing...")
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        print(f"   -> Result: {answer}")

    except Exception as e:
        answer = f"❌ Vision Error: {e}"
        print(answer)
        
    finally:
        # --- 4. CLEANUP (The Broom) ---
        # This runs whether the analysis succeeded or failed
        print("   -> Cleaning up vision resources...")
        free_gpu_memory(model, tokenizer)
        
    return answer

# --- TEST BLOCK ---
# Runs only if you execute: python core_logic/vision.py
if __name__ == "__main__":
    # Ensure you have a 'test.jpg' in your AGENT_Zero folder, or update this path
    test_image = os.path.join(PROJECT_ROOT, "test.png")
    if os.path.exists(test_image):
        print(analyze_image(test_image, "What is in this image?"))
    else:
        print(f"⚠️ Create a 'test.jpg' in {PROJECT_ROOT} to test this.")