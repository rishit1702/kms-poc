"""KMS POC — FastAPI app entry."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routes.api import router

app = FastAPI(
    title="KMS POC — VVDN Knowledge Management System",
    description=(
        "Multimodal Knowledge Management System. Single chat endpoint that handles "
        "text, PDFs, images, and videos. Built with RAG, context engineering, and "
        "swappable LLM providers (Gemini / Anthropic / Ollama / OpenRouter). "
        "MCP server included."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the frontend at /
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "service": "KMS POC",
            "docs": "/docs",
            "endpoints": ["/chat", "/ingest", "/kb/stats", "/health"],
        }
