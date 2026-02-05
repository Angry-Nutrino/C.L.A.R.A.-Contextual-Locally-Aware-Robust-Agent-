import torch
import gc

def free_gpu_memory(model=None, tokenizer=None):
    """
    Aggressively clears GPU memory by deleting Python references
    and forcing PyTorch to release cached VRAM.
    """
    print(" Cleaning GPU memory...")
    if model:
        del model
        print("   Model deleted.")
    if tokenizer:
        del tokenizer
        print("   Tokenizer deleted.")
    
    gc.collect()
    print("   Python garbage collector invoked.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   PyTorch GPU cache cleared.")

    
    print("✨ GPU Memory flushed.")