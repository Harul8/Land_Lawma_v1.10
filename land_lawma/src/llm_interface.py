# src/llm_interface.py - simple local LLM wrapper (Mistral-7B)
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class LLMInterface:
    def __init__(self, model_name_or_path: str = None):
        self.model_name = model_name_or_path or os.getenv('LOCAL_LLM_PATH', 'mistral-7b-instruct-v0.3')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                load_in_4bit=True,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f'Warning: could not load 4-bit model: {e}. Trying normal load. May OOM on 8GB VRAM.')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto')

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.95) -> str:
        cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=(cfg.temperature > 0),
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
