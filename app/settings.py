from pydantic import BaseSettings

class Settings(BaseSettings):
    # Generation backends: "openai" or "ollama"
    GEN_BACKEND: str = "openai"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "llama3:8b"

    # Vector store / embeddings
    CHROMA_DIR: str = "/app/storage"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
