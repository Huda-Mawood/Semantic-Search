from utils.pdf_parser import extract_text_from_pdf
from config import settings


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The full text to split
        chunk_size: Max characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def ingest_pdf(file_path: str) -> list[dict]:
    """
    Full pipeline: extract text from PDF → chunk it.

    Args:
        file_path: Path to the uploaded PDF

    Returns:
        List of dicts with chunk text + metadata
    """
    # Step 1: Extract text page by page
    pages = extract_text_from_pdf(file_path)

    # Step 2: Chunk each page
    all_chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        for chunk in page_chunks:
            if chunk.strip():  # skip empty chunks
                all_chunks.append({
                    "chunk_id": str(chunk_id),
                    "text": chunk.strip(),
                    "page_number": page["page_number"],
                    "source": file_path
                })
                chunk_id += 1

    return all_chunks