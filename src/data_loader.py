import os
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

# Optional: Poppler path if not in system PATH
POPPLER_PATH = r"C:\poppler\Library\bin"

def load_bare_acts(folder_path):
    """
    Load all PDFs in the folder (both digital and scanned),
    return dict {Act name: text}.
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
