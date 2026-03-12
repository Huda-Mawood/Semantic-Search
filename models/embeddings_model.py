from sentence_transformers import SentenceTransformer
from config import settings

# Load model once at startup
model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of document chunks.
    Used during ingestion (indexing).

    Args:
        texts: List of text chunks

    Returns:
        List of embedding vectors
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a search query.
    Used during search.

    Args:
        query: The user's search question

    Returns:
        Single embedding vector
    """
    embedding = model.encode(query)
    return embedding.tolist()