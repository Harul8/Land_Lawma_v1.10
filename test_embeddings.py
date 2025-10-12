# test_embeddings.py
"""
Test script to verify:
1. BareActs PDFs are loaded correctly.
2. Text chunks are generated.
3. Embeddings are created.
4. FAISS index is built and can be queried.
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Correct import from src package
from src.embeddings_manager import EmbeddingsManager

# Initialize embeddings manager
em = EmbeddingsManager()  # Automatically uses GPU if available

# Step 1: Build FAISS index from BareActs folder
bareacts_folder = 'BareActs'
if not os.path.exists(bareacts_folder):
    print(f"Folder '{bareacts_folder}' not found. Please place PDFs inside it.")
else:
    print("=== Building FAISS index from BareActs folder ===")
    em.build_from_folder(bareacts_folder)

# Step 2: Load FAISS index and metadata
print("\n=== Loading FAISS index and metadata ===")
index, metas = em.load_index()

if index is None or metas is None:
    print("Index not found. Please check previous step.")
    exit(1)
else:
    print(f"FAISS index loaded with {index.ntotal} vectors")
    print(f"Metadata loaded for {len(metas)} chunks")

# Step 3: Test a sample query
query = "What is the definition of land revenue?"
print(f"\n=== Running test query: '{query}' ===")

# Use the same embedding model to encode query
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if faiss.get_num_gpus() > 0 else 'cpu')
query_embedding = embed_model.encode([query], convert_to_numpy=True)
faiss.normalize_L2(query_embedding)

# Search top 3 relevant chunks
k = 3
D, I = index.search(query_embedding, k)

print("\nTop results:")
for rank, idx in enumerate(I[0], start=1):
    chunk_text = metas[idx]['text'][:500] + ("..." if len(metas[idx]['text']) > 500 else "")
    act_name = metas[idx]['meta']['act_name']
    print(f"Rank {rank}:")
    print(f"Act: {act_name}")
    print(f"Text: {chunk_text}\n")
