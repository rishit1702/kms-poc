# KMS POC — VVDN Knowledge Management System

A multimodal Knowledge Management System backend POC for VVDN's internal portal.
Employees upload videos, PDFs, and images; the system processes them into a
searchable knowledge base, and a single chat endpoint answers questions with
context-aware responses powered by RAG.

## What this POC demonstrates

- **Single `/chat` inference endpoint** — handles all multimedia, docs, PDFs.
- **Multimodal ingestion** — PDF text extraction, image vision-captioning, video
  transcription via Whisper.
- **Knowledge base** — ChromaDB vector store with local embeddings (no API key
  needed for embeddings).
- **Context engineering** — carefully crafted system prompt + structured RAG
  context blocks with source attribution.
- **Swappable LLM providers** — Gemini (default), Anthropic Claude, or Ollama
  (local). Change `LLM_PROVIDER` in `.env`, no code changes.
- **MCP server** — exposes the knowledge base as MCP tools any compatible LLM
  client can call (`python -m app.mcp.server`).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Single Endpoint:  POST /chat                           │
│  - text query  +  optional file (PDF/image/video)       │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌────────────────┐            ┌──────────────────┐
│  Ingestion     │            │  Retrieval       │
│  - PDF -> text │            │  - vector search │
│  - Img -> vis. │   ──────►  │  - top-k chunks  │
│  - Vid -> ASR  │            │                  │
│  - chunks      │            └────────┬─────────┘
└────────┬───────┘                     │
         ▼                             ▼
   ┌─────────────────┐         ┌──────────────────┐
   │  ChromaDB       │         │  LLM (swappable) │
   │  Knowledge Base │         │  Gemini/Claude/  │
   │  (embeddings)   │         │  Ollama          │
   └─────────────────┘         └────────┬─────────┘
                                        ▼
                               ┌──────────────────┐
                               │  Answer + sources│
                               └──────────────────┘
```

## Setup (5 minutes)

### 1. Install Python dependencies

```bash
cd kms-poc
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install ffmpeg (required for video transcription)

- **Ubuntu / WSL:** `sudo apt install ffmpeg`
- **Mac:** `brew install ffmpeg`
- **Windows:** download from https://ffmpeg.org/download.html and add to PATH

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set ONE of these:
- `GEMINI_API_KEY=...`  (get free key at https://aistudio.google.com/apikey)
- `ANTHROPIC_API_KEY=...`
- For Ollama: install from https://ollama.com, run `ollama pull llama3.2`

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## Usage

### Upload content to knowledge base

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"

curl -X POST http://localhost:8000/ingest \
  -F "file=@meeting_recording.mp4"

curl -X POST http://localhost:8000/ingest \
  -F "file=@diagram.png"
```

### Ask a question (single endpoint, all media types)

```bash
curl -X POST http://localhost:8000/chat \
  -F "query=What was decided in the Q3 planning meeting?"
```

### Ask + upload in one call

```bash
curl -X POST http://localhost:8000/chat \
  -F "query=Summarize this video" \
  -F "file=@new_video.mp4"
```

## Swap LLM providers

Just change `LLM_PROVIDER` in `.env`:

```
LLM_PROVIDER=gemini      # or anthropic, or ollama
```

Restart the server. Zero code changes needed.

## Run the MCP server

```bash
python -m app.mcp.server
```

This exposes three MCP tools to any compatible client (Claude Desktop, etc.):
- `search_knowledge_base`
- `ingest_document`
- `kb_stats`

## Project structure

```
kms-poc/
├── app/
│   ├── main.py                    FastAPI entrypoint
│   ├── config.py                  Settings (.env loader)
│   ├── routes/
│   │   └── api.py                 /chat, /ingest, /kb/stats, /health
│   ├── services/
│   │   ├── llm.py                 Swappable LLM providers
│   │   ├── knowledge_base.py      ChromaDB vector store
│   │   ├── ingestion.py           PDF/image/video processors
│   │   └── chat.py                RAG + context engineering
│   └── mcp/
│       └── server.py              MCP server
├── data/                          ChromaDB storage (auto-created)
├── uploads/                       Uploaded files (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```
