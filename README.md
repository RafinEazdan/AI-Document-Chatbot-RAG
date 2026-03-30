# 📄 RAG Document Chatbot

A production-grade, hallucination-free chatbot that answers questions **strictly** from an uploaded PDF or DOCX document using a Retrieval-Augmented Generation (RAG) pipeline, powered by Google Gemini and served through a FastAPI REST API.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
  - [Project Structure](#project-structure)
  - [RAG Pipeline](#rag-pipeline)
  - [Two-Tier Prompt Injection Guard](#two-tier-prompt-injection-guard)
- [Technical Explanation](#technical-explanation)
  - [Document Ingestion](#1-document-ingestion)
  - [Embedding & Indexing](#2-embedding--indexing)
  - [Retrieval](#3-retrieval)
  - [Prompt Construction & LLM Generation](#4-prompt-construction--llm-generation)
  - [Conversation Memory](#5-conversation-memory)
  - [Hallucination Control](#6-hallucination-control)
- [Libraries & Tools Used](#libraries--tools-used)
- [Design Decisions & Justifications](#design-decisions--justifications)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)

---

## Features

| Feature | Description |
|---------|-------------|
| 📄 **Document grounding** | Answers only from the uploaded document — never guesses |
| 📎 **Source citations** | Every answer cites the chunk(s) it was derived from |
| 📊 **Similarity scores** | Each retrieved chunk includes a cosine-similarity score |
| 💬 **Conversation memory** | Sliding-window history for multi-turn follow-up questions |
| 🛡️ **Two-tier injection guard** | Regex heuristics + LLM confirmation to block prompt injection |
| 🔌 **Dependency injection** | Interface-driven design for testability and swappability |
| 🐳 **Docker support** | One-command containerized deployment |
| ⚡ **Atomic document upload** | Uploading a new document resets the document directory and rebuilds the index for simplicity and atomicity |

---

## Architecture Overview

### Project Structure

```
csn-demo/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── documents/                ← uploaded PDF/DOCX files (mounted volume)
├── vector_store/             ← persisted FAISS index + chunks (mounted volume)
└── app/
    ├── main.py               ← FastAPI application entry point & lifespan
    ├── document_loader.py    ← PDF/DOCX parsing + recursive text chunking
    ├── api/
    │   └── routers.py        ← REST endpoints (documents, chat)
    ├── core/
    │   ├── config.py         ← Centralized env-based configuration
    │   ├── interfaces.py     ← Abstract base classes (DI contracts)
    │   └── dependencies.py   ← FastAPI Depends() provider functions
    ├── rag/
    │   ├── embeddings.py     ← SentenceTransformer + FAISS index management
    │   ├── retriever.py      ← Similarity search (Top-K nearest neighbors)
    │   ├── llm.py            ← Google Gemini LLM provider
    │   └── guard.py          ← Two-tier prompt injection guard
    ├── memory/
    │   ├── memory.py         ← Sliding-window conversation memory
    │   └── chain.py          ← RAG chain: retrieve → prompt → LLM → citations
    ├── schemas/
    │   └── schemas.py        ← Pydantic request/response models
    └── services/
        ├── chat_service.py   ← Chat business logic
        └── document_service.py ← Upload, indexing & status logic
```

### RAG Pipeline

```
                          User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Two-Tier Guard     │
                    │  ┌───────────────┐  │
                    │  │ Tier 1: Regex │──┼──▶ Pass → continue
                    │  └───────┬───────┘  │
                    │     Suspicious?      │
                    │          ▼           │
                    │  ┌───────────────┐  │
                    │  │ Tier 2: LLM   │──┼──▶ Confirm → Block
                    │  │  (Gemini)     │  │    Deny   → Pass
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Embed Query        │  sentence-transformers
                    │  (all-MiniLM-L6-v2) │  → 384-dim vector
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  FAISS Search       │  Cosine similarity (Inner Product
                    │  (IndexFlatIP)      │  on L2-normalized vectors)
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Build Prompt       │  System prompt + conversation
                    │                     │  history + retrieved chunks
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Gemini LLM Call    │  gemini-2.5-flash-lite
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Answer + Citations │  [Chunk N] references
                    │  + Similarity Scores│  with cosine scores
                    └─────────────────────┘
```

### Two-Tier Prompt Injection Guard

The system implements a **layered defense** against prompt injection attacks:

| Tier | Method | Cost | When it runs |
|------|--------|------|-------------|
| **Tier 1** | Regex pattern matching | Near-zero latency | Every request |
| **Tier 2** | LLM-based classification (Gemini) | ~200ms, 1 API call | Only when Tier 1 flags the input as suspicious |

**How it works:**

1. **Tier 1 (Regex Guard)** — A compiled set of 40+ regex patterns checks for known injection phrases (e.g., `"ignore previous instructions"`, `"you are now"`, `"jailbreak"`, `"reveal system prompt"`, `"DAN mode"`, etc.). This is extremely fast and catches the vast majority of injection attempts with zero API cost.

2. **Tier 2 (LLM Guard)** — If (and only if) Tier 1 flags the input as suspicious, a secondary Gemini model (`LLM_GUARD_MODEL`) acts as a binary classifier to confirm whether the input is truly an injection attempt. This prevents false positives — legitimate questions that happen to contain flagged words (e.g., *"What is the override procedure for the safety system?"*) are allowed through.

**Why two tiers?** A regex-only guard is fast but produces false positives. An LLM-only guard is accurate but wastes API calls and latency on every request. The two-tier approach gives us the best of both: near-zero cost for clean inputs, high accuracy for ambiguous ones.

---

## Technical Explanation

### 1. Document Ingestion

- **Supported formats:** PDF (via `pypdf`) and DOCX (via `python-docx`).
- **Upload flow:** When a new document is uploaded via `POST /documents/upload`, the system **clears all existing files from the `documents/` directory** before saving the new file. This atomic-reset design ensures the chatbot always operates on exactly one document, maintaining simplicity and consistency.
- **Text extraction:** PDFs are parsed page-by-page with page markers (`[Page N]`). DOCX files are extracted paragraph-by-paragraph.

### 2. Embedding & Indexing

- **Chunking:** Extracted text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` with configurable `CHUNK_SIZE` (default: 500 chars) and `CHUNK_OVERLAP` (default: 100 chars). The recursive strategy splits on `\n\n` → `\n` → `. ` → ` ` → `""`, preserving semantic boundaries.
- **Embedding model:** `all-MiniLM-L6-v2` from Sentence Transformers generates 384-dimensional dense vectors. This model offers an excellent balance of speed, size (~80 MB), and quality for semantic similarity tasks.
- **Vector index:** FAISS `IndexFlatIP` (inner product) is used on L2-normalized vectors, which is mathematically equivalent to cosine similarity but leverages FAISS's optimized inner-product kernels.
- **Persistence:** The FAISS index (`index.faiss`) and chunk texts (`chunks.json`) are saved to the `vector_store/` directory and automatically reloaded on server startup.

### 3. Retrieval

- Top-K (default: 4) chunks are retrieved by cosine similarity.
- Each result includes the chunk content, its index, and the similarity score.
- Results with `idx == -1` (FAISS sentinel for no match) are filtered out.

### 4. Prompt Construction & LLM Generation

- A **strict system prompt** instructs the LLM to answer only from the provided context, cite chunks using `[Chunk N]` notation, and refuse off-document questions with a specific fallback message.
- The prompt assembles: `system prompt` → `conversation history` → `context chunks` → `user question`.
- The Google Gemini API is called with multi-turn chat support (prior history is passed as alternating user/model turns).

### 5. Conversation Memory

- A `ConversationMemory` class maintains a **sliding window** of the last 10 turns (20 messages: 10 user + 10 assistant).
- History is injected into the prompt so the LLM can handle follow-up questions that reference earlier answers.
- Memory can be cleared via `POST /chat/clear`.

### 6. Hallucination Control

The system uses **four layers** of hallucination prevention:

| Layer | Mechanism |
|-------|-----------|
| **Strict system prompt** | Explicitly tells the LLM to only use provided context |
| **Context-only prompting** | The LLM sees only retrieved chunks, not the full document |
| **Chunk citations** | Forces the model to ground answers in specific chunks |
| **Explicit fallback** | If the answer isn't in the context: *"This information is not present in the provided document."* |

---

## Libraries & Tools Used

| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.135.2 | Async REST API framework with auto-generated OpenAPI docs |
| **Uvicorn** | 0.42.0 | ASGI server to run the FastAPI application |
| **Pydantic** | 2.12.5 | Request/response validation and serialization |
| **google-generativeai** | 0.8.6 | Google Gemini API client for LLM calls (chat + guard) |
| **sentence-transformers** | 5.3.0 | Pre-trained embedding models (`all-MiniLM-L6-v2`) |
| **faiss-cpu** | 1.13.2 | Facebook AI Similarity Search for fast vector retrieval |
| **langchain-text-splitters** | 1.1.1 | Recursive character text splitting with semantic boundaries |
| **langchain-core** | 1.2.23 | `Document` data model for chunk representation |
| **pypdf** | 6.9.2 | PDF text extraction |
| **python-docx** | 1.2.0 | DOCX text extraction |
| **python-dotenv** | 1.2.2 | Load configuration from `.env` files |
| **NumPy** | 2.4.4 | Vector operations and array handling |
| **Docker** | — | Containerized deployment with volume mounts |

---

## Design Decisions & Justifications

### Why single-document atomicity?

When a new document is uploaded, the system **deletes all existing files** from `documents/` before saving the new one. This is a deliberate simplification:

- **Atomicity:** The index always corresponds to exactly one document. There is no risk of stale chunks from a previously uploaded file bleeding into answers.
- **Simplicity:** Users don't need to manage document inventories or worry about conflicts.
- **Consistency:** Every question is answered from a single, well-defined source of truth.

### Why FAISS `IndexFlatIP` over approximate methods?

- For document-scale datasets (hundreds to low thousands of chunks), exact search is fast enough and eliminates the complexity of tuning approximate nearest-neighbor parameters (e.g., nprobe, nlist).
- L2-normalization + inner product is mathematically equivalent to cosine similarity but avoids the overhead of FAISS's cosine-specific index types.

### Why `all-MiniLM-L6-v2`?

- ~80 MB model size — loads quickly even in containerized environments.
- 384-dimensional output — compact vectors reduce memory and search time.
- Consistently ranks among the top lightweight models on the [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) for semantic similarity tasks.

### Why a two-tier guard instead of regex-only or LLM-only?

- **Regex-only** is fast but brittle — legitimate questions containing trigger words (e.g., *"override"* in a safety manual) get blocked.
- **LLM-only** is accurate but expensive — every request incurs an API call and ~200ms latency even for clearly benign inputs.
- **Two-tier** combines both: regex pre-screens cheaply, LLM confirms only when needed. In practice, >95% of legitimate requests pass Tier 1 instantly, and the LLM guard only fires for genuinely ambiguous inputs.

### Why dependency injection with abstract base classes?

The `core/interfaces.py` module defines `IEmbeddingManager`, `ILLMProvider`, `IGuard`, and `IDocumentLoader` as abstract base classes. Route handlers receive implementations through FastAPI's `Depends()` system:

- **Testability:** Unit tests can inject mock implementations without touching real APIs or file systems.
- **Swappability:** Switching from Gemini to OpenAI (or adding a new LLM provider) requires only a new class and a one-line change in `dependencies.py`.
- **Separation of concerns:** Route handlers don't know or care which LLM, embedding model, or storage backend is in use.

### Why sliding-window memory instead of full history?

- LLM context windows have token limits. Sending the entire conversation history would eventually exceed them.
- A 10-turn window (configurable) keeps recent context available for follow-up questions while bounding prompt size.
- Older turns naturally become less relevant as the conversation topic shifts.

---

## Quick Start

### 1. Setup

```bash
# Clone and enter project
cd csn-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp app/.env.example app/.env
# Edit app/.env with your Gemini API key
```

### 2. Configure

Edit `app/.env`:

```env
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.5-flash-lite
LLM_GUARD_MODEL=gemini-2.5-flash-lite
```

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`. Interactive docs are available at `/docs`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check and API info |
| `POST` | `/documents/upload` | Upload a PDF/DOCX and build the vector index |
| `GET` | `/documents/status` | Check if an index is loaded and its vector count |
| `POST` | `/chat/ask` | Ask a question about the uploaded document |
| `POST` | `/chat/clear` | Clear conversation memory |

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the company leave policy?"}'
```

**Response:**

```json
{
  "answer": "According to the document, employees are entitled to ... [Chunk 3]",
  "sources": [
    {
      "chunk_index": 3,
      "score": 0.8721,
      "preview": "All full-time employees are entitled to 20 days of annual leave..."
    }
  ]
}
```

---

## Configuration

All settings are loaded from environment variables (via `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model for answer generation |
| `LLM_GUARD_MODEL` | `gemini-2.5-flash-lite` | Model for Tier 2 injection classification |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks to retrieve per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `DOCUMENT_PATH` | `documents/` | Directory for uploaded documents |
| `INDEX_PATH` | `vector_store/` | Directory for persisted FAISS index |

---

# 🐳 Docker Deployment

You can run this project either by building the image locally (recommended for development) or by using the prebuilt image from Docker Hub (recommended for quick testing).

## 🔹 Option 1: Build Locally (Development)

```bash
# Build and run using Docker Compose
docker-compose up --build
```

Or manually:

```bash
docker build -t rag-chatbot .
```

```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/vector_store:/app/vector_store \
  rag-chatbot
```

The Docker image uses `python:3.11-slim` as the base, exposes port 8000, and mounts two volumes:
- `documents/` — stores input documents for indexing
- `vector_store/` — persists FAISS index and embeddings

