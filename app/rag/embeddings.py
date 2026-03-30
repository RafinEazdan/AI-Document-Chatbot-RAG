"""Embedding generation and FAISS index management."""

import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from app.core.config import Config
from app.core.interfaces import IEmbeddingManager


class EmbeddingManager(IEmbeddingManager):
    """Manages embedding model, FAISS index, and chunk storage."""

    def __init__(self, config: Config) -> None:
        self._config = config
        print(f"  Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Document] = []

    def build_index(self, chunks: List[Document]) -> None:
        """Create a FAISS index from document chunks."""
        self.chunks = chunks
        texts = [chunk.page_content for chunk in chunks]

        print(f"  Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity (using inner product on normalized vectors)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def save_index(self, path: str = None) -> None:
        """Persist FAISS index and chunks to disk."""
        path = path or self._config.INDEX_PATH
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        import json
        chunk_data = [c.page_content for c in self.chunks]
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(chunk_data, f)
        print(f"  Index saved to {path}/")

    def load_index(self, path: str = None) -> bool:
        """Load a previously saved index. Returns True if successful."""
        path = path or self._config.INDEX_PATH
        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.json")

        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            return False

        import json
        self.index = faiss.read_index(index_file)
        with open(chunks_file, "r") as f:
            chunk_data = json.load(f)
        self.chunks = [Document(page_content=text) for text in chunk_data]
        print(f"  Loaded existing index: {self.index.ntotal} vectors")
        return True

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        vec = self.model.encode([query])
        vec = np.array(vec, dtype="float32")
        faiss.normalize_L2(vec)
        return vec
