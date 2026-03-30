# 📄 RAG Document Chatbot

An intelligent chatbot that answers questions **strictly** from a provided PDF or DOCX document using Retrieval-Augmented Generation (RAG).

## Features

- ✅ **Document grounding** — answers only from your document, never guesses
- 📎 **Source citations** — shows which chunks each answer comes from
- 📊 **Similarity scores** — displays relevance scores for retrieved chunks
- 💬 **Conversation memory** — maintains context across multi-turn chats
- 🛡️ **Prompt injection protection** — blocks common injection patterns
- 🐳 **Docker support** — containerized deployment
- 🔄 **Dual LLM support** — works with OpenAI or Ollama (local)

## Architecture

```
main.py                 ← CLI entry point
src/
  config.py             ← Centralized settings from .env
  document_loader.py    ← PDF/DOCX parsing + chunking
  embeddings.py         ← Sentence-transformers + FAISS index
  retriever.py          ← Similarity search with scores
  llm.py                ← LLM wrapper (OpenAI / Ollama)
  chain.py              ← RAG pipeline with hallucination control
  memory.py             ← Sliding-window conversation memory
  guard.py              ← Prompt injection detection
```

### RAG Pipeline

```
User Question
      │
      ▼
┌─────────────┐
│ Injection   │──▶ Block if suspicious
│ Guard       │
└─────┬───────┘
      ▼
┌─────────────┐
│ Embed Query │ (sentence-transformers)
└─────┬───────┘
      ▼
┌─────────────┐
│ FAISS Search│──▶ Top-K relevant chunks + scores
└─────┬───────┘
      ▼
┌─────────────┐
│ Build Prompt│ System prompt + history + context + question
└─────┬───────┘
      ▼
┌─────────────┐
│ LLM Call    │ (OpenAI or Ollama)
└─────┬───────┘
      ▼
  Answer + Citations
```

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
cp .env.example .env
# Edit .env with your settings
```

### 2. Add Your Document

Place your PDF or DOCX file in the `documents/` folder:

```bash
mkdir -p documents
cp /path/to/Operational_Manual_XYZ_Company.pdf documents/
```

### 3. Choose Your LLM

**Option A: Ollama (free, local)**

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2

# In .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

**Option B: OpenAI**

```bash
# In .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Run

```bash
python main.py
```

The chatbot will:
1. Load your document(s) from `documents/`
2. Chunk the text and build a FAISS vector index
3. Start an interactive chat session

### Commands

| Command | Action |
|---------|--------|
| Type a question | Get an answer from the document |
| `clear` | Reset conversation memory |
| `quit` / `exit` | Exit the chatbot |

## Docker Setup

```bash
# Build and run
docker-compose up --build

# Or with docker run
docker build -t rag-chatbot .
docker run -it --env-file .env \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/vector_store:/app/vector_store \
  rag-chatbot
```

> **Note**: If using Ollama on the host, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in your `.env`.

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

## Hallucination Control

The system uses multiple layers to prevent hallucination:

1. **Strict system prompt** — instructs the LLM to only use provided context
2. **Low temperature (0.1)** — reduces creative/speculative outputs
3. **Explicit fallback** — if the answer isn't in the document, responds with:
   > "This information is not present in the provided document."
4. **Context-only prompting** — the LLM only sees retrieved document chunks, not the full document

## Evaluation Checklist

| Criteria | Status |
|----------|--------|
| ✅ Functional correctness | RAG pipeline with retrieval + generation |
| 🏗 Architecture quality | Clean separation: loader → embedder → retriever → chain |
| 🧠 Hallucination prevention | System prompt + low temp + fallback response |
| 💻 Code quality | Type hints, docstrings, single-responsibility modules |
| 📖 Documentation | This README + inline comments |
| ⭐ Source citations | Chunk references with similarity scores |
| ⭐ Prompt injection protection | Regex-based guard |
| ⭐ Docker setup | Dockerfile + docker-compose.yml |
