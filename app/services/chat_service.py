"""Chat service — handles question answering and conversation memory."""

from app.memory.chain import ask
from app.rag.retriever import retrieve
from app.core.interfaces import IEmbeddingManager, ILLMProvider, IGuard
from app.memory.memory import ConversationMemory


def ask_question(
    question: str,
    embedding_manager: IEmbeddingManager,
    memory: ConversationMemory,
    llm_provider: ILLMProvider,
    guard: IGuard,
) -> dict:
    """Process a question through the RAG pipeline."""
    full_answer = ask(
        question,
        embedding_manager,
        memory,
        llm_provider=llm_provider,
        guard=guard,
    )

    answer_text = full_answer.split("\n📎 Sources:")[0].strip()

    results = retrieve(question, embedding_manager)
    sources = [
        {
            "chunk_index": r.chunk_index,
            "score": round(r.score, 4),
            "preview": r.content[:120].replace("\n", " "),
        }
        for r in results
    ]

    return {"answer": answer_text, "sources": sources}


def clear_memory(memory: ConversationMemory) -> str:
    """Clear conversation memory."""
    memory.clear()
    return "Conversation memory cleared."
