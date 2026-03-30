"""Abstract base classes for dependency injection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class IEmbeddingManager(ABC):
    """Base class for embedding model and FAISS index management."""

    @abstractmethod
    def build_index(self, chunks: list) -> None: ...

    @abstractmethod
    def save_index(self, path: str = None) -> None: ...

    @abstractmethod
    def load_index(self, path: str = None) -> bool: ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...


class ILLMProvider(ABC):
    """Base class for LLM chat-completion backends."""

    @abstractmethod
    def complete(self, messages: List[dict]) -> str: ...


class IGuard(ABC):
    """Base class for prompt-injection detection."""

    @abstractmethod
    def check(self, text: str) -> Tuple[bool, str]: ...


class IDocumentLoader(ABC):
    """Base class for loading documents from disk."""

    @abstractmethod
    def load(self, path: str) -> str: ...

    @abstractmethod
    def load_all(self, directory: str) -> str: ...
