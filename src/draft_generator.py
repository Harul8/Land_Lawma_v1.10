# src/draft_generator.py - prompt composition and opinion generation
from typing import List, Dict
from .llm_interface import LLMInterface

class DraftGenerator:
    def __init__(self, llm: LLMInterface, retriever=None):
        self.llm = llm
        self.retriever = retriever

    def _compose_prompt(self, facts: str, evidence: List[Dict]) -> str:
        header = (

            "You are an expert legal draftsman specialising in Indian land law.\n"
            "Using ONLY the facts and evidence provided below, draft a legal memorandum/opinion.\n"
            "Structure the output with: (1) Short facts summary (max 150 words); (2) Issues; (3) Applicable law (Acts & Sections); (4) Analysis (cite Evidence items like [EVIDENCE 1]); (5) Conclusion & relief sought.\n"
            "If evidence does not support a conclusion, mark it as 'Not supported by provided evidence'.\n\n"
        )
        evidence_text = "\n\n".join([f"[EVIDENCE {i+1}] Source: {e['meta'].get('source','unknown')}\n{e['text']}" for i, e in enumerate(evidence)])
        prompt = f"{header}\nFACTS:\n{facts}\n\nEVIDENCE:\n{evidence_text}\n\nDraft the legal opinion now. Use clear numbered sections and include inline evidence citations."
        return prompt

    def generate(self, facts: str, retrieved: List[Dict], max_new_tokens: int = 512) -> str:
        prompt = self._compose_prompt(facts, retrieved)
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=0.95)
