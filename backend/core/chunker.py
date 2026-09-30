from io import BytesIO
from typing import List, Dict, Any
from pypdf import PdfReader


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """
    Split text into chunks of `size` characters with `overlap` character overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_pages_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text per page from a PDF file buffer.
    """
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page_number": page_num, "text": text})
    return pages
