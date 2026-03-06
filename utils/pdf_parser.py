import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of dicts with page_number and text
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    pages = []

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            if text:  # skip empty pages
                pages.append({
                    "page_number": page_num,
                    "text": text
                })

    return pages