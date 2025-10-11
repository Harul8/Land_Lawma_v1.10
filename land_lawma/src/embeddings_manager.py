# src/embeddings_manager.py - embeddings + FAISS index management
import os, json
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer
from .text_preprocessor import chunk_text
from .data_loader import load_pdfs_from_folder

INDEX_PATH = 'data/vector_store/faiss_index.idx'
METADATA_PATH = 'data/vector_store/faiss_metadata.json'
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

class EmbeddingsManager:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    def build_from_folder(self, data_folder: str = 'data') -> None:
        docs = load_pdfs_from_folder(data_folder)
        chunks = []
        for d in docs:
            text = d.get('text','')
            for c in chunk_text(text):
                chunks.append({'text': c, 'meta': d.get('meta', {})})

        texts = [c['text'] for c in chunks]
        if not texts:
            print('No texts found to index.')
            return

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(index, INDEX_PATH)

        metas = [{'id': i, 'text': texts[i], 'meta': chunks[i]['meta']} for i in range(len(texts))]
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)
        print(f'Built FAISS index with {len(texts)} vectors')

    def load_index(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            return None, None
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metas = json.load(f)
        return index, metas
