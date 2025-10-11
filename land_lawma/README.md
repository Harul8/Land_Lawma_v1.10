# land_lawma - Legal RAG Pipeline (Mistral-7B-Instruct-v0.3 + FAISS)

This repository is a starter legal Retrieval-Augmented Generation (RAG) pipeline
optimized for local execution on a laptop with an NVIDIA RTX 4060 (8GB VRAM).

**Core features**:
- Ingest PDFs (acts & judgments)
- OCR-friendly text extraction and chunking
- Embeddings using SentenceTransformers
- FAISS-based retrieval
- Local generation using Mistral-7B-Instruct (quantized)
- Streamlit interface (simple chatbot)

## Quickstart

1. Create conda env & install dependencies:

```bash
conda create -n land_lawma python=3.10 -y
conda activate land_lawma
pip install -r requirements.txt
```

2. Place your PDFs into `data/acts/` and `data/judgments/` (or upload through app).
3. Set `LOCAL_LLM_PATH` in `.env` if you have a local model path; otherwise the app will attempt to load from Hugging Face id.
4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Notes & cautions
- This is a prototype starter. Do NOT use to provide unaudited legal advice without a human reviewer.
- Respect licensing for paid legal databases. Do not scrape paywalled content.
- Quantized model loading requires compatible CUDA, bitsandbytes and driver versions.
