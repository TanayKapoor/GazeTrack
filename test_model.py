import torch
from transformers import AutoModelForCausalLM

print("Loading model...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=device
    )
    print("Model loaded successfully!")
    model.to(device)
    print("Model moved to device!")
    print("Ready to run demo.py")
except Exception as e:
    print(f"Error: {e}")