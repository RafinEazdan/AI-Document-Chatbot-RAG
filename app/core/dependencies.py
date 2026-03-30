"""FastAPI dependency providers — wire app.state and config into route handlers."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Config
from app.core.interfaces import IEmbeddingManager, ILLMProvider, IGuard, IDocumentLoader
from app.memory.memory import ConversationMemory


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a process-wide cached Config instance."""
    return Config()


def get_embedding_manager(request: Request) -> IEmbeddingManager:
    """Pull the EmbeddingManager from app.state (populated during lifespan startup)."""
    return request.app.state.embedding_manager


def get_memory(request: Request) -> ConversationMemory:
    """Pull the single ConversationMemory from app.state."""
    return request.app.state.memory


def get_llm_provider(
    config: Annotated[Config, Depends(get_config)],
) -> ILLMProvider:
    from app.rag.llm import GeminiProvider
    return GeminiProvider(config)


def get_guard(
    config: Annotated[Config, Depends(get_config)],
) -> IGuard:
    from app.rag.guard import TwoTierGuard
    return TwoTierGuard(config)


def get_document_loader(
    config: Annotated[Config, Depends(get_config)],
) -> IDocumentLoader:
    from app.document_loader import DocumentLoader
    return DocumentLoader(config)
