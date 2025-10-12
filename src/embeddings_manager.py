# src/embeddings_manager.py
# ---------------------------------------------
# Embeddings Manager for BareActs PDFs
# Memory-efficient + GPU-friendly
# Processes one PDF at a time
# Embeds chunks in small batches to utilize RTX 4060
# Incrementally updates FAISS index and metadata
# ---------------------------------------------

import os
import json
import torch
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from .text_preprocessor import chunk_text
from .data_loader import load_bare_acts

# Paths for storing FAISS index and metadata
INDEX_PATH = 'data/vector_store/faiss_index.idx'
METADATA_PATH = 'data/vector_store/faiss_metadata.json'

# Default embedding model
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

class EmbeddingsManager:
    def __init__(self, model_name: str = EMBED_MODEL_NAME, device: str = 'cuda'):
        """
        Initialize embeddings manager.
        Forces GPU usage (CUDA) if available.
        Loads existing FAISS index and metadata if present.
        """
        # Set device to CUDA if available, else CPU
        self.device = device if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("CUDA not available, falling back to CPU.")
        else:
            print(f"Using device: {self.device}")

        # Load SentenceTransformer model on chosen device
        self.model = SentenceTransformer(model_name, device=self.device)

        # Ensure storage directory exists
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

        # Load existing FAISS index & metadata if available
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    self.metas = json.load(f)
            else:
                self.metas = []
            print(f"Loaded existing FAISS index with {len(self.metas)} vectors")
        else:
            self.index = None
            self.metas = []

    def build_from_folder(self, data_folder: str = 'BareActs', 
                          chunk_size: int = 500, overlap: int = 100,
                          batch_size: int = 8) -> None:
        """
        Build FAISS index from PDFs in folder, memory-efficient.

        Args:
            data_folder: Path to BareActs PDFs
            chunk_size: Characters per chunk
            overlap: Overlap characters between chunks
            batch_size: Number of chunks to embed at once
        """
        print("=== Building FAISS index from BareActs folder ===")
        docs = load_bare_acts(data_folder)  # dict {act_name: text}

        # Embedding dimension for all-MiniLM-L6-v2
        dim = 384
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)

        # Start vector ID after existing vectors
        vector_id = len(self.metas)

        for act_name, text in docs.items():
            print(f"\nProcessing PDF: {act_name}, length: {len(text)} chars")
            # Prepare batches
            batch = []
            batch_metas = []

            # Process chunks
            for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                batch.append(chunk)
                batch_metas.append({'id': vector_id, 'text': chunk, 'meta': {'act_name': act_name}})
                vector_id += 1

                # When batch is full, embed and add to FAISS
                if len(batch) == batch_size:
                    self._add_batch_to_index(batch, batch_metas)
                    batch = []
                    batch_metas = []

            # Flush remaining chunks in batch
            if batch:
                self._add_batch_to_index(batch, batch_metas)

            # Save FAISS index and metadata after each PDF
            self._save_index_and_metadata()
            print(f"Completed PDF: {act_name}, total vectors so far: {vector_id}")

        print(f"\n=== FAISS index build complete: total vectors = {len(self.metas)} ===")

    def _add_batch_to_index(self, batch: List[str], batch_metas: List[Dict]) -> None:
        """
        Embed a batch of chunks and add to FAISS index.
        """
        # Encode batch on GPU
        embeddings = self.model.encode(batch, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metas.extend(batch_metas)

    def _save_index_and_metadata(self) -> None:
        """
        Save FAISS index and metadata to disk.
        """
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metas, f, ensure_ascii=False, indent=2)

    def load_index(self) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
        """
        Load FAISS index and metadata from disk.

        Returns:
            index: FAISS index object
            metas: List of metadata dictionaries
        """
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            print("FAISS index or metadata not found. Build index first.")
            return None, None

        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metas = json.load(f)

        print(f"Loaded FAISS index with {len(metas)} vectors")
        return index, metas
