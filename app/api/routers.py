"""All API routes — imported into main.py via APIRouter."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.schemas.schemas import (
    UploadResponse,
    ReindexResponse,
    IndexStatusResponse,
    ChatRequest,
    ChatResponse,
    SourceChunk,
    ClearMemoryResponse,
)
from app.services import chat_service
from app.services.document_service import DocumentService
from app.core.config import Config
from app.core.interfaces import IEmbeddingManager, ILLMProvider, IGuard, IDocumentLoader
from app.core.dependencies import (
    get_config,
    get_embedding_manager,
    get_memory,
    get_llm_provider,
    get_guard,
    get_document_loader,
)
from app.memory.memory import ConversationMemory


# ── Document Router ──

document_router = APIRouter(prefix="/documents", tags=["Documents"])


@document_router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    config: Annotated[Config, Depends(get_config)] = None,
    embedding_manager: Annotated[IEmbeddingManager, Depends(get_embedding_manager)] = None,
    loader: Annotated[IDocumentLoader, Depends(get_document_loader)] = None,
):
    """Upload a PDF or DOCX file and build the vector index."""
    svc = DocumentService(config, loader)
    try:
        result = await svc.upload_and_index(file, embedding_manager)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        filename=result["filename"],
        total_chunks=result["total_chunks"],
    )



@document_router.get("/status", response_model=IndexStatusResponse)
async def index_status(
    request: Request,
    config: Annotated[Config, Depends(get_config)] = None,
    loader: Annotated[IDocumentLoader, Depends(get_document_loader)] = None,
):
    """Check whether a vector index is loaded and how many vectors it has."""
    em = getattr(request.app.state, "embedding_manager", None)
    svc = DocumentService(config, loader)
    status = svc.get_index_status(em)
    return IndexStatusResponse(**status)


# ── Chat Router ──

chat_router = APIRouter(prefix="/chat", tags=["Chat"])


@chat_router.post("/ask", response_model=ChatResponse)
async def ask_question(
    body: ChatRequest,
    embedding_manager: Annotated[IEmbeddingManager, Depends(get_embedding_manager)],
    memory: Annotated[ConversationMemory, Depends(get_memory)],
    llm_provider: Annotated[ILLMProvider, Depends(get_llm_provider)],
    guard: Annotated[IGuard, Depends(get_guard)],
):
    """Ask a question about the uploaded document."""
    try:
        result = chat_service.ask_question(
            body.question,
            embedding_manager,
            memory,
            llm_provider=llm_provider,
            guard=guard,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@chat_router.post("/clear", response_model=ClearMemoryResponse)
async def clear_memory_route(
    memory: Annotated[ConversationMemory, Depends(get_memory)] = None,
):
    """Clear the conversation memory."""
    message = chat_service.clear_memory(memory)
    return ClearMemoryResponse(message=message)
