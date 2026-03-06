import cohere
from config import settings

client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of document chunks.
    Used during ingestion (indexing).

    Args:
        texts: List of text chunks

    Returns:
        List of embedding vectors
    """
    response = client.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )

    return response.embeddings.float


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a search query.
    Used during search.

    Args:
        query: The user's search question

    Returns:
        Single embedding vector
    """
    response = client.embed(
        texts=[query],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"]
    )

    return response.embeddings.float[0]