from models.embeddings_model import embed_query
from models.vector_store import query_collection
from config import settings


def search(query: str) -> list[dict]:
    """
    Full search pipeline: query → embed → search ChromaDB → return results.

    Args:
        query: User's search question

    Returns:
        List of matched chunks with text, metadata, and score
    """
    # Step 1: Embed the query
    query_embedding = embed_query(query)

    # Step 2: Search ChromaDB
    results = query_collection(
        query_embedding=query_embedding,
        top_k=settings.TOP_K_RESULTS
    )

    # Step 3: Format results
    formatted = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        formatted.append({
            "text": doc,
            "page_number": meta.get("page_number"),
            "source": meta.get("source"),
            "score": round(1 - distance, 4)  # cosine similarity
        })

    return formatted