# src/retriever.py - query FAISS index using sentence-transformers embedding model
from typing import List, Dict
import faiss
import numpy as np
from .embeddings_manager import EmbeddingsManager

class Retriever:
    def __init__(self, emb_mgr: EmbeddingsManager = None):
        self.emb_mgr = emb_mgr if emb_mgr is not None else EmbeddingsManager()
        self.index, self.metas = self.emb_mgr.load_index()

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict]:
        if self.index is None or self.metas is None:
            print('Index not loaded — build the index first.')
            return []
        q_emb = self.emb_mgr.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            m = self.metas[idx]
            results.append({'score': float(dist), 'text': m['text'], 'meta': m.get('meta', {})})
        return results
