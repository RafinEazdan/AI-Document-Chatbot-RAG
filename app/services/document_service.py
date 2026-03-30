"""Document service — handles upload, indexing, and status logic."""

import os
import shutil

from fastapi import UploadFile

from app.core.config import Config
from app.core.interfaces import IEmbeddingManager, IDocumentLoader
from app.document_loader import chunk_text


class DocumentService:
    """Stateless document upload and indexing logic.

    All mutable state (the embedding manager) is injected per-request via
    FastAPI's Depends(), so this class holds no module-level globals.
    """

    def __init__(self, config: Config, loader: IDocumentLoader) -> None:
        self._config = config
        self._loader = loader

    async def upload_and_index(
        self,
        file: UploadFile,
        embedding_manager: IEmbeddingManager,
    ) -> dict:
        """Save an uploaded file and rebuild the vector index."""
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".pdf", ".docx", ".doc"):
            raise ValueError(
                f"Unsupported file type: {ext}. Only PDF and DOCX are supported."
            )

        os.makedirs(self._config.DOCUMENT_PATH, exist_ok=True)
        file_path = os.path.join(self._config.DOCUMENT_PATH, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        raw_text = self._loader.load_all(self._config.DOCUMENT_PATH)
        chunks = chunk_text(raw_text)

        embedding_manager.build_index(chunks)
        embedding_manager.save_index()

        return {"filename": file.filename, "total_chunks": len(chunks)}

    async def reindex_all(self, embedding_manager: IEmbeddingManager) -> int:
        """Re-index all documents in the documents directory."""
        raw_text = self._loader.load_all(self._config.DOCUMENT_PATH)
        chunks = chunk_text(raw_text)

        embedding_manager.build_index(chunks)
        embedding_manager.save_index()

        return len(chunks)

    def get_index_status(self, embedding_manager: IEmbeddingManager | None) -> dict:
        """Return current index status."""
        if embedding_manager is None or embedding_manager.index is None:
            return {
                "indexed": False,
                "total_vectors": 0,
                "document_path": self._config.DOCUMENT_PATH,
            }
        return {
            "indexed": True,
            "total_vectors": embedding_manager.index.ntotal,
            "document_path": self._config.DOCUMENT_PATH,
        }
