# app.py - Streamlit App for Land Law Draft Generation

import os
import streamlit as st
from src.embeddings_manager import EmbeddingsManagerGPU
from src.retriever import Retriever
from src.draft_generator import DraftGenerator
import torch

# ----------------------------
# Utility: GPU Stats
# ----------------------------
def print_gpu_stats():
    if torch.cuda.is_available():
        st.text(f"GPU: {torch.cuda.get_device_name(0)}")
        st.text(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        st.text(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
        st.text(f"Free: {free/1e9:.2f} GB")

# ----------------------------
# App Title
# ----------------------------
st.title("Land Law Legal Draft Generator")

# ----------------------------
# Initialize Components
# ----------------------------
st.sidebar.header("Setup")
bareacts_folder = st.sidebar.text_input("BareActs Folder Path:", "BareActs")
model_path = st.sidebar.text_input("LLM Model Folder:", "Model")

# GPU stats
if torch.cuda.is_available():
    st.sidebar.subheader("GPU Info")
    print_gpu_stats()
else:
    st.sidebar.warning("CUDA GPU not detected. Using CPU.")

# Embeddings manager
em = EmbeddingsManagerGPU()
# Rebuild FAISS index if needed
rebuild_index = st.sidebar.checkbox("Rebuild FAISS Index?", value=False)
if rebuild_index:
    if not os.path.exists(bareacts_folder):
        st.error(f"Folder '{bareacts_folder}' not found!")
    else:
        with st.spinner("Building FAISS index..."):
            em.build_from_folder(
                bareacts_folder,
                chunk_size=500,   # adjust for GPU
                overlap=100,
                batch_size=32     # adjust for RTX 4060 + FP16
            )
            st.success("FAISS index built successfully!")
            print_gpu_stats()

# Initialize Retriever
retriever = Retriever(emb_mgr=em)

# Initialize Draft Generator
draft_gen = DraftGenerator(model_path=model_path)

# ----------------------------
# User Input Section
# ----------------------------
st.subheader("Enter Case Facts")
facts = st.text_area("Facts of the case:", height=200)

top_k = st.number_input("Number of retrieved evidence chunks:", min_value=1, max_value=10, value=5)

if st.button("Generate Legal Draft"):
    if not facts.strip():
        st.warning("Please enter case facts to generate a draft.")
    else:
        # Retrieve top-k relevant chunks
        with st.spinner("Retrieving relevant legal evidence..."):
            retrieved_docs = retriever.retrieve(facts, top_k=top_k)
            retrieved_text = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Generate draft
        with st.spinner("Generating legal draft..."):
            try:
                # Use temperature=None for greedy deterministic output
                draft = draft_gen.generate(
                    facts=facts,
                    retrieved_context=retrieved_text,
                    max_new_tokens=300,
                    temperature=None
                )
                st.subheader("Generated Legal Draft")
                st.text_area("Draft Opinion", draft, height=400)
            except Exception as e:
                st.error(f"Error during draft generation: {e}")

# ----------------------------
# Optional: Display FAISS stats
# ----------------------------
if st.sidebar.checkbox("Show FAISS Index Stats"):
    index, metas = em.load_index()
    if metas:
        st.sidebar.write(f"Total vectors in FAISS index: {len(metas)}")
    else:
        st.sidebar.info("FAISS index not found or empty. Build it first.")
