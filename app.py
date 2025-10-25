# app.py - Streamlit front-end for Land_Lawma
# -------------------------------------------------------
# ✅ Fully compatible with your local Mistral 7B model
# ✅ GPU-optimized with FAISS + SentenceTransformers
# ✅ Streamlit UI kept exactly as before (with requested layout changes)
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
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_MISTRAL_PATH = "Model"  # Folder where Mistral-7B is stored

    emb_manager = EmbeddingsManagerGPU(model_name=MODEL_NAME)
    retriever = Retriever(emb_manager)
    llm_interface = LLMInterface(model_path=LOCAL_MISTRAL_PATH, load_in_4bit=True)
    draft_gen = DraftGenerator(llm_interface)

    return draft_gen, retriever, emb_manager


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
draft_gen, retriever, emb_manager = initialize_model_and_pipeline()


# -------------------------------------------------------
# STREAMLIT UI SECTION
# -------------------------------------------------------
st.markdown("### 🧾 Enter Case Facts and Query")

facts = st.text_area("Provide the facts of the case here...", height=150)
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

        # --- Single line heading before the two columns
        st.success("✅ Draft generated successfully!")

        # --- Split page into two columns
        col_left, col_right = st.columns([3, 1])  # Left column wider

        # --- LEFT COLUMN: Generated Draft
        with col_left:
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

        # --- RIGHT COLUMN: Referenced Documents
        with col_right:
            st.markdown("### 📄 Referenced Documents")
            if retrieved:
                for doc_entry in retrieved:
                    # Assume each item is a dict with keys like {'act_name': ..., 'source': ...}
                    doc_name = doc_entry.get("act_name") or str(doc_entry)
                    doc_path = doc_entry.get("source") or doc_name

                    # Make clickable link (open in new tab)
                    if os.path.exists(doc_path):
                        st.markdown(f'- <a href="{doc_path}" target="_blank">{doc_name}</a>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"- {doc_name} (not found)")
            else:
                st.info("No documents retrieved for this query.")
