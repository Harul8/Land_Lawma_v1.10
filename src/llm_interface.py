# src/llm_interface.py
# ------------------------------------------------------------
# Optimized for: Mistral 7B Instruct v0.3 on RTX 4060 (8 GB)
# ------------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

class LLMInterface:
    def __init__(self, model_path: str = "Model", device: str = None, load_in_4bit: bool = True):
        """
        Initialize the LLM interface for text generation.

        Args:
            model_path (str): Path to local Mistral model directory.
            device (str): 'cuda' or 'cpu'. Auto-detect if None.
            load_in_4bit (bool): If True, loads model in 4-bit precision for faster inference.
        """
        start = time.time()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🧠 Loading Mistral 7B model from: {model_path}")
        print(f"💻 Device: {self.device.upper()} | 4-bit quantization: {load_in_4bit}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # ✅ Load model with quantization if GPU supports it
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto" if self.device == "cuda" else None
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model_kwargs["quantization_config"] = quant_config
                print("⚡ Using 4-bit quantization to save VRAM.")
            except Exception as e:
                print(f"❌ 4-bit quantization unavailable: {e}")
                print("→ Falling back to 16-bit model.")

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        print(f"✅ Model loaded successfully in {time.time() - start:.2f}s\n")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from a composed prompt (optimized for low VRAM).

        Args:
            prompt (str): Input text prompt.
            max_new_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature (low = more deterministic).
            top_p (float): Nucleus sampling (controls diversity).
            repetition_penalty (float): Penalize repeated text.
            do_sample (bool): Use sampling or greedy decoding.

        Returns:
            str: Generated text output.
        """
        # Tokenize efficiently
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Free unused VRAM
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return generated_text
