# src/text_preprocessor.py
# ---------------------------
# Memory-efficient text cleaning and chunking
# ---------------------------

import re
from typing import List

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ''
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Chunk text safely with controlled overlap (no infinite loops).

    Args:
        text: Input string
        chunk_size: Characters per chunk
        overlap: Characters overlap between chunks

    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []

    text = clean_text(text)
    length = len(text)
    chunks = []

    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()

        if len(chunk) > 50:
            chunks.append(chunk)

        # ✅ Stop safely when at end of text
        if end >= length:
            break

        # Move start forward safely for overlap
        start = end - overlap

    print(f"✅ Created {len(chunks)} chunks for text of length {length}")
    return chunks
