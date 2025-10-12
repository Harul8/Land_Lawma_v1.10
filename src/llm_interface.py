# src/llm_interface.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMInterface:
    def __init__(self, model_path: str = "Model", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device.upper()} from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print("✓ Model loaded successfully")

    def generate(
        self,
        facts: str,
        retrieved_context: str,
        max_new_tokens: int = 300,
        **kwargs
    ) -> str:
        """
        Generate a legal draft using the LLM.

        For greedy decoding, we enforce do_sample=False instead of temperature=0.0.
        """
        prompt = self._compose_prompt(facts, retrieved_context)
        return self.llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # <-- important for greedy decoding
            **kwargs
        )
