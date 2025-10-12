# src/embeddings_manager_gpu.py
# ---------------------------------------------
# Memory-efficient FAISS embeddings manager using SentenceTransformers on GPU
# ---------------------------------------------

import os
import gc
import json
import torch
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from .text_preprocessor import chunk_text
from .data_loader import stream_bare_acts  # generator yielding (act_name, text)
import numpy as np

# Paths for FAISS index and metadata
INDEX_PATH = 'data/vector_store/faiss_index.idx'
METADATA_PATH = 'data/vector_store/faiss_metadata.json'
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingsManagerGPU:
    def __init__(self, model_name=MODEL_NAME):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("="*60)
        print(f"Device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print("="*60)

        # Load SentenceTransformer model on GPU
        self.model = SentenceTransformer(model_name, device=self.device)
        print("✓ SentenceTransformer loaded")

        self.metas: List[Dict] = []

        # Initialize FAISS index
        self.dim = self.model.get_sentence_embedding_dimension()
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            cpu_index = faiss.IndexFlatIP(self.dim)
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index) if self.device=='cuda' else cpu_index

        # Load metadata if exists
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                self.metas = json.load(f)
                print(f"✓ Loaded metadata: {len(self.metas)} items")

    def embed_batch(self, texts: List[str]):
        """Embed a batch of texts on GPU, returns normalized numpy array"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_from_folder(self, folder='BareActs', chunk_size=1000, overlap=200, batch_size=128):
        """Build FAISS index incrementally from PDFs in folder"""
        vector_id = len(self.metas)
        total_chunks = 0

        for act_name, text in stream_bare_acts(folder):
            batch_chunks, batch_metas = [], []
            for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                batch_chunks.append(chunk)
                batch_metas.append({'id': vector_id, 'meta': {'act_name': act_name, 'source': act_name}})
                vector_id += 1

                if len(batch_chunks) >= batch_size:
                    self._embed_and_add(batch_chunks, batch_metas)
                    total_chunks += len(batch_chunks)
                    batch_chunks, batch_metas = [], []
                    print(f"  Processed {total_chunks} chunks...", end='\r')

            if batch_chunks:
                self._embed_and_add(batch_chunks, batch_metas)
                total_chunks += len(batch_chunks)

            self._save_metadata()
            print(f"\n  ✓ Completed {act_name}: total chunks {total_chunks}")
            gc.collect()
            if self.device=='cuda':
                torch.cuda.empty_cache()

        self._save_index()
        print(f"\n✓ INDEX BUILD COMPLETE | Total vectors: {len(self.metas)}")

    def _embed_and_add(self, batch_chunks: List[str], batch_metas: List[Dict]):
        embeddings = self.embed_batch(batch_chunks)
        self.index.add(embeddings)
        self.metas.extend(batch_metas)
        del embeddings
        gc.collect()
        if self.device=='cuda':
            torch.cuda.empty_cache()

    def _save_index(self):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.device=='cuda' else self.index, INDEX_PATH)

    def _save_metadata(self):
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metas, f, ensure_ascii=False, indent=2)

    def search(self, query: str, k=5) -> Tuple[List[np.ndarray], List[int]]:
        """Search top-k similar vectors for query"""
        emb = self.embed_batch([query])
        distances, indices = self.index.search(emb, k)
        return distances, indices

    def load_index(self) -> Tuple[faiss.Index, List[Dict]]:
        """Return FAISS index and metadata"""
        return self.index, self.metas
