# test_llm.py
from src.llm_interface import LLMInterface
import torch

# Initialize the LLM (local model path assumed)
llm = LLMInterface("Model")

# Check device
print(f"Model device: {llm.model.device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test generation
prompt = "Draft a short legal opinion on land acquisition disputes in Telangana."
print("\n=== Generating text ===")
output = llm.generate(prompt, max_new_tokens=150)
print("\n=== Output ===")
print(output)
