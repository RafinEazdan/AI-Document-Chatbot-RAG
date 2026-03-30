"""Centralized configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    LLM_GUARD_MODEL: str = os.getenv("LLM_GUARD_MODEL", "gemini-2.5-flash-lite")

    # RAG settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths
    DOCUMENT_PATH: str = os.getenv("DOCUMENT_PATH", "documents/")
    INDEX_PATH: str = os.getenv("INDEX_PATH", "vector_store/")

    # Hallucination control
    NOT_FOUND_RESPONSE: str = (
        "This information is not present in the provided document."
    )
