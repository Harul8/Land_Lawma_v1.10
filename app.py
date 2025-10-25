# app.py - Streamlit front-end for Land_Lawma
# -------------------------------------------------------
# ✅ Fully compatible with your local Mistral 7B model
# ✅ GPU-optimized with FAISS + SentenceTransformers
# ✅ Streamlit UI kept exactly as before
# -------------------------------------------------------

import os
import streamlit as st
from src.embeddings_manager import EmbeddingsManagerGPU
from src.retriever import Retriever
from src.llm_interface import LLMInterface
from src.draft_generator import DraftGenerator
from src.data_loader import load_bare_acts as load_pdfs_from_folder


# -------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Land_Lawma", layout="wide")
st.title("🏛️ Land Lawma – Legal Drafting Assistant")

# -------------------------------------------------------
# INITIALIZATION FUNCTION (CACHED)
# -------------------------------------------------------
@st.cache_resource(show_spinner="🚀 Initializing models and pipelines...")
def initialize_model_and_pipeline():
    """
    Initializes:
      - SentenceTransformer embeddings manager (GPU optimized)
      - Retriever (FAISS)
      - Local Mistral LLM interface (quantized)
      - DraftGenerator (drives final output)
    """
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_MISTRAL_PATH = "Model"  # Folder where Mistral-7B is stored

    # 1️⃣ Embeddings + Retriever
    emb_manager = EmbeddingsManagerGPU(model_name=MODEL_NAME)
    retriever = Retriever(emb_manager)

    # 2️⃣ LLM Interface (Mistral)
    llm_interface = LLMInterface(model_path=LOCAL_MISTRAL_PATH, load_in_4bit=True)

    # 3️⃣ Draft Generator (connects to LLM)
    draft_gen = DraftGenerator(llm_interface)

    return draft_gen, retriever, emb_manager


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
draft_gen, retriever, emb_manager = initialize_model_and_pipeline()


# -------------------------------------------------------
# STREAMLIT UI SECTION
# -------------------------------------------------------
st.markdown("### 📂 Upload or Select Legal Documents")

# Sidebar for document selection
with st.sidebar:
    st.header("⚖️ Legal Document Loader")
    folder = st.text_input("Enter folder name containing Bare Acts:", "BareActs")

    if st.button("📘 Load Documents"):
        st.session_state["pdf_data"] = load_pdfs_from_folder(folder)
        st.success(f"Loaded documents from: {folder}")

# Input area for facts and query
st.markdown("### 🧾 Enter Case Facts")
facts = st.text_area("Provide the facts of the case here...", height=150)

st.markdown("### ❓ Ask a Legal Question")
query = st.text_input("What do you want to draft or find?")

# -------------------------------------------------------
# DRAFT GENERATION LOGIC
# -------------------------------------------------------
if st.button("🪶 Generate Legal Draft"):
    if not facts or not query:
        st.warning("Please provide both case facts and a question.")
    else:
        with st.spinner("Retrieving relevant legal sections..."):
            retrieved = retriever.retrieve(query)

        with st.spinner("Generating draft using local Mistral model..."):
            opinion = draft_gen.generate(facts, retrieved)

        st.success("✅ Draft generated successfully!")
        st.markdown("### 🧾 Generated Draft")
        st.write(opinion)

        # Save the output
        output_path = "data/output/draft_output.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(opinion)

        st.download_button(
            label="⬇️ Download Draft",
            data=opinion,
            file_name="draft_output.txt",
            mime="text/plain",
        )
