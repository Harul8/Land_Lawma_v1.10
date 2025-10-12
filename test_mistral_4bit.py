from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

# Local model path
local_model_path = "C:/Users/rahul/Landa_Lawma/models/Mistral-7B-Instruct-v0.3"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Define 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model
print("\n=== Loading model... ===")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Device & GPU Check
print("\n=== Device Check ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print("GPU Device:", device_name)
    print("Model device map:")
    print(model.hf_device_map)

# Prepare prompt
prompt = "Summarize Indian land law in two sentences."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Clear GPU cache before timing
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Measure inference time
print("\n=== Running inference... ===")
start_time = time.time()

outputs = model.generate(**inputs, max_new_tokens=100)
end_time = time.time()

# Decode output
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Calculate timing
elapsed_time = end_time - start_time
num_tokens = outputs.shape[-1]
tokens_per_sec = num_tokens / elapsed_time

print("\n=== Model Output ===")
print(result)

print("\n=== Performance Stats ===")
print(f"Total time taken: {elapsed_time:.2f} seconds")
print(f"Tokens generated: {num_tokens}")
print(f"Speed: {tokens_per_sec:.2f} tokens/sec")

# VRAM usage (if GPU)
if torch.cuda.is_available():
    vram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    vram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    print(f"VRAM Allocated: {vram_allocated:.2f} MB")
    print(f"VRAM Reserved: {vram_reserved:.2f} MB")
