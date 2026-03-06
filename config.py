from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    COHERE_API_KEY: str
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    CHROMA_COLLECTION_NAME: str = "llm_book"
    CHROMA_DB_PATH: str = "./data/chroma_db"
    TOP_K_RESULTS: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()