# src/retriever.py
# ---------------------------------------------
# Query FAISS index using SentenceTransformers embeddings on GPU
# ---------------------------------------------

from typing import List, Dict
from .embeddings_manager import EmbeddingsManagerGPU  # Updated import

class Retriever:
    def __init__(self, emb_mgr: EmbeddingsManagerGPU = None):
        """
        Initialize Retriever with an embeddings manager.

        Args:
            emb_mgr: Instance of EmbeddingsManagerGPU. If None, a new one is created.
        """
        self.emb_mgr = emb_mgr if emb_mgr is not None else EmbeddingsManagerGPU()
        self.index, self.metas = self.emb_mgr.load_index()

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict]:
        """
        Retrieve top-k relevant chunks from the FAISS index for a query.

        Args:
            query: Query string.
            top_k: Number of top chunks to retrieve.

        Returns:
            List[Dict]: Each dict contains 'score', 'text', and 'meta' keys.
        """
        if self.index is None or self.metas is None:
            print('Index not loaded — build the index first.')
            return []

        # Encode query and perform FAISS search on GPU
        distances, indices = self.emb_mgr.search(query, k=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            m = self.metas[idx]
            results.append({
                'score': float(dist),
                'text': m.get('meta', {}).get('text', m.get('meta', {}).get('act_name', '')),
                'meta': m.get('meta', {})
            })
        return results
