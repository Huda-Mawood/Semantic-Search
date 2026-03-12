from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Ollama
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ChromaDB
    CHROMA_COLLECTION_NAME: str = "llm_book"
    CHROMA_DB_PATH: str = "./data/chroma_db"
    TOP_K_RESULTS: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()