# src/draft_generator.py
# ---------------------------------------------
from typing import List
from .llm_interface import LLMInterface

class DraftGenerator:
    def __init__(self, model_path: str = "Model"):
        self.llm = LLMInterface(model_path=model_path)

    def _compose_prompt(self, facts: str, retrieved_context: str) -> str:
        prompt = (
            f"You are an expert legal draftsman specialising in Indian land law.\n"
            f"Using ONLY the facts and evidence provided below, draft a legal memorandum/opinion.\n"
            f"Structure the output with:\n"
            f"1. Short facts summary (max 150 words)\n"
            f"2. Issues\n"
            f"3. Applicable law (Acts & Sections)\n"
            f"4. Analysis (cite Evidence items like [EVIDENCE 1])\n"
            f"5. Conclusion & relief sought\n"
            f"If evidence does not support a conclusion, mark it as 'Not supported by provided evidence'.\n\n"
            f"FACTS:\n{facts}\n\n"
            f"RELEVANT ACTS / EVIDENCE:\n{retrieved_context}\n\n"
            f"Draft the legal opinion now. Use numbered sections and inline citations where needed."
        )
        return prompt

    def generate(
        self,
        facts: str,
        retrieved_context: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        Generate a legal draft using the LLM.

        For deterministic (greedy) output, set temperature=None and do_sample=False.
        """
        prompt = self._compose_prompt(facts, retrieved_context)

        # Determine if we should do greedy decoding
        if temperature is None or temperature <= 0.0:
            return self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs
            )
        else:
            return self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
