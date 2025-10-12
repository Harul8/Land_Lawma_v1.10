# app.py - Streamlit front-end for land_lawma
import os
import streamlit as st
from src.data_loader import load_pdfs_from_folder
from src.embeddings_manager import EmbeddingsManager
from src.retriever import Retriever
from src.llm_interface import LLMInterface
from src.draft_generator import DraftGenerator
from dotenv import load_dotenv

load_dotenv()  # load .env if present

st.set_page_config(page_title='land_lawma - Legal RAG', layout='wide')
st.title('land_lawma — Legal RAG (Mistral 7B + FAISS)')

# Sidebar: configuration & ingestion
st.sidebar.header('Configuration')
local_model = st.sidebar.text_input('Local LLM path or HF id', value=os.getenv('LOCAL_LLM_PATH', 'mistral-7b-instruct-v0.3'))

st.sidebar.header('Data ingestion')
if st.sidebar.button('Load PDFs from data folder'):
    st.sidebar.info('Loading PDFs from data/acts and data/judgments...')
    pdfs = load_pdfs_from_folder('data')
    st.sidebar.success(f'Loaded {len(pdfs)} PDFs (use Build Index to index them)')

# Main UI for facts input
st.header('Facts of the case')
facts = st.text_area('Paste facts here', height=260)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button('Build / Rebuild Index'):
        st.info('Building embeddings and FAISS index — this may take a few minutes...')
        emb_mgr = EmbeddingsManager()
        emb_mgr.build_from_folder('data')
        st.success('Index built and saved to disk.')
with col2:
    top_k = st.number_input('Top-K retrieval', min_value=1, max_value=20, value=8)

if st.button('Generate Draft Opinion'):
    if not facts.strip():
        st.warning('Please provide facts first.')
    else:
        emb_mgr = EmbeddingsManager()
        retriever = Retriever(emb_mgr)
        llm = LLMInterface(local_model)
        draft_gen = DraftGenerator(llm, retriever)

        with st.spinner('Retrieving evidence...'):
            retrieved = retriever.retrieve(facts, top_k=top_k)

        st.subheader('Top evidence snippets (short)')
        for i, r in enumerate(retrieved):
            st.markdown(f"**Evidence {i+1}** — source: {r['meta'].get('source','unknown')} — score: {r['score']:.3f}")
            st.write(r['text'][:500] + ('...' if len(r['text'])>500 else ''))

        with st.spinner('Generating draft opinion (this may take time)...'):
            opinion = draft_gen.generate(facts, retrieved)
        st.success('Draft generated — review carefully')
        st.download_button('Download opinion', opinion, file_name='draft_opinion.txt')
        st.markdown('### Draft Opinion')
        st.write(opinion)
