import chromadb
from config import settings


def get_client():
    """Create ChromaDB client lazily."""
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def get_collection():
    """Get or create ChromaDB collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def store_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Store text chunks and their embeddings in ChromaDB.
    """
    collection = get_collection()

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "page_number": chunk["page_number"],
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def query_collection(query_embedding: list[float], top_k: int) -> dict:
    """
    Search ChromaDB for most similar chunks.
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results