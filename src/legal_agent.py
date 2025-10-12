# src/legal_agent.py - simple orchestrator for pipeline
from .embeddings_manager import EmbeddingsManager
from .retriever import Retriever
from .llm_interface import LLMInterface
from .draft_generator import DraftGenerator

class LegalAgent:
    def __init__(self, model_name: str = None):
        self.emb_mgr = EmbeddingsManager()
        self.retriever = Retriever(self.emb_mgr)
        self.llm = LLMInterface(model_name)
        self.draft_gen = DraftGenerator(self.llm, self.retriever)

    def run(self, facts: str, top_k: int = 8):
        retrieved = self.retriever.retrieve(facts, top_k=top_k)
        opinion = self.draft_gen.generate(facts, retrieved)
        return {'opinion': opinion, 'retrieved': retrieved}
