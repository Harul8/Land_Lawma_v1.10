import os
import time
import torch
from src.embeddings_manager import EmbeddingsManagerGPU

def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU STATISTICS")
        print(f"{'='*60}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
        print(f"Free: {free/1e9:.2f} GB")
        print(f"{'='*60}\n")

def main():
    folder = 'BareActs'
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found!")
        return

    em = EmbeddingsManagerGPU()
    print_gpu_stats()

    rebuild = True
    if os.path.exists('data/vector_store/faiss_index.idx'):
        rebuild = input("Index exists. Rebuild? (y/n): ").lower() == 'y'

    if rebuild:
        start = time.time()
        em.build_from_folder(folder, chunk_size=500, overlap=100, batch_size=32)
        print(f"\n✓ Index built in {time.time()-start:.2f}s")
        print_gpu_stats()
    else:
        index, metas = em.load_index()
        print(f"Loaded {len(metas)} vectors")

    queries = [
        "Definition of land revenue",
        "Penalties for violation",
        "Property tax calculation",
        "Powers of the collector"
    ]
    index, metas = em.load_index()
    for q in queries:
        start = time.time()
        dists, idxs = em.search(q, k=3)
        print(f"\nQuery: {q} | Search time: {(time.time()-start)*1000:.2f} ms")
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < len(metas):
                print(f"  Similarity: {dist:.4f} | Act: {metas[idx]['meta']['act_name']}")

    print_gpu_stats()
    print("✓ TEST COMPLETED")

if __name__ == "__main__":
    main()
