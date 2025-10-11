# src/data_loader.py - PDF ingestion utilities
import os
from typing import List, Dict
import pdfplumber

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file. For scanned PDFs, consider running OCR separately before ingest."""
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        print(f"Error extracting {path}: {e}")
    return "\n".join(texts)

def load_pdfs_from_folder(base_folder: str = 'data') -> List[Dict]:
    """Search `base_folder` recursively for PDFs and return a list of dicts: {path, text, meta}"""
    results = []
    for root, dirs, files in os.walk(base_folder):
        for name in files:
            if name.lower().endswith('.pdf'):
                path = os.path.join(root, name)
                text = extract_text_from_pdf(path)
                meta = {'source': os.path.relpath(path, base_folder)}
                results.append({'path': path, 'text': text, 'meta': meta})
    return results
