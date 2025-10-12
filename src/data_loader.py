import os
import gc
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from typing import Iterator, Tuple

# Optional: Poppler path if not in system PATH
POPPLER_PATH = r"C:\poppler\Library\bin"

def load_bare_acts(folder_path):
    """
    Load all PDFs in the folder (both digital and scanned),
    return dict {Act name: text}.
    
    WARNING: This loads ALL PDFs into memory at once.
    For large collections, use stream_bare_acts() instead.
    """
    bareacts_data = {}
    failed_files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            act_name = os.path.splitext(file_name)[0]

            # Try pdfplumber first
            text = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                if not text.strip():  # empty, fallback to OCR
                    raise ValueError("No text extracted")
            except Exception:
                # Fallback: OCR
                try:
                    pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
                    for page_image in pages:
                        text += pytesseract.image_to_string(page_image)
                except Exception as e:
                    print(f"OCR failed for {file_name}: {e}")
                    failed_files.append(file_name)
                    continue

            bareacts_data[act_name] = text
            print(f"Loaded: {act_name}, Characters extracted: {len(text)}")

    if failed_files:
        print(f"Failed files: {failed_files}")

    return bareacts_data


def stream_bare_acts(folder_path: str) -> Iterator[Tuple[str, str]]:
    """
    Stream PDFs one at a time (memory efficient).
    Yields (act_name, text) tuples.
    
    Usage:
        for act_name, text in stream_bare_acts('BareActs'):
            process(act_name, text)
    """
    failed_files = []
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files to process")
    
    for idx, file_name in enumerate(pdf_files, 1):
        file_path = os.path.join(folder_path, file_name)
        act_name = os.path.splitext(file_name)[0]
        
        print(f"\n[{idx}/{total_files}] Loading: {file_name}...")
        
        # Try pdfplumber first
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                    
            if not text.strip():  # empty, fallback to OCR
                raise ValueError("No text extracted")
                
        except Exception as e:
            print(f"  Digital extraction failed, trying OCR...")
            # Fallback: OCR
            try:
                pages = convert_from_path(
                    file_path, 
                    dpi=300, 
                    poppler_path=POPPLER_PATH
                )
                for page_image in pages:
                    text += pytesseract.image_to_string(page_image)
                    
            except Exception as ocr_error:
                print(f"  ✗ OCR failed: {ocr_error}")
                failed_files.append(file_name)
                continue
        
        if text.strip():
            print(f"  ✓ Extracted {len(text):,} characters")
            yield act_name, text
            
            # Clear memory after yielding
            del text
            gc.collect()
        else:
            print(f"  ✗ No text extracted")
            failed_files.append(file_name)
    
    if failed_files:
        print(f"\n⚠ Failed to process {len(failed_files)} files: {failed_files}")


def load_single_pdf(file_path: str) -> str:
    """
    Load a single PDF file and return its text.
    Memory efficient for processing one file at a time.
    """
    text = ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    
        if not text.strip():
            raise ValueError("No text extracted")
            
    except Exception:
        # Fallback: OCR
        try:
            pages = convert_from_path(
                file_path,
                dpi=300,
                poppler_path=POPPLER_PATH
            )
            for page_image in pages:
                text += pytesseract.image_to_string(page_image)
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    return text