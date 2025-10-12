# src/legal_agent.py
# ---------------------------------------------
# Legal Agent orchestrator: GPU-ready embeddings + retrieval + LLM draft generation
# ---------------------------------------------

from .embeddings_manager import EmbeddingsManagerGPU
from .retriever import Retriever
from .llm_interface import LLMInterface
from .draft_generator import DraftGenerator

class LegalAgent:
    def __init__(self, model_name: str = None):
        # Initialize GPU embeddings manager
        self.emb_mgr = EmbeddingsManagerGPU()
        self.retriever = Retriever(self.emb_mgr)

        # Initialize LLM interface (local path or HF model)
        self.llm = LLMInterface(model_name)
        self.draft_gen = DraftGenerator(self.llm, self.retriever)

    def run(self, facts: str, top_k: int = 8):
        # Step 1: retrieve relevant chunks
        retrieved = self.retriever.retrieve(facts, top_k=top_k)

        # Step 2: generate draft opinion using LLM
        opinion = self.draft_gen.generate(facts, retrieved)

        # Return both the opinion and retrieved evidence
        return {'opinion': opinion, 'retrieved': retrieved}
