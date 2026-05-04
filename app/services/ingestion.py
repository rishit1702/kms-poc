"""
Multimodal ingestion pipeline.
Handles PDF, images, video, and text. Each becomes searchable text in the KB.

Image handling has two modes (controlled by ENABLE_IMAGE_VISION in .env):
- True  -> Ollama LLaVA generates a detailed caption (high RAM, can hang on 8GB)
- False -> Metadata-only ingestion, safe on low-RAM machines (default)
"""
from pathlib import Path
from typing import List
from pypdf import PdfReader
import base64
import os
import httpx

from app.services.knowledge_base import kb
from app.config import settings


# ---------- chunking helper ----------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ---------- PDF / text ----------
def ingest_pdf(file_path: str, source_name: str) -> int:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    chunks = chunk_text(full_text)
    return kb.add_chunks(chunks, {"source": source_name, "type": "pdf"})


def ingest_text(text: str, source_name: str) -> int:
    chunks = chunk_text(text)
    return kb.add_chunks(chunks, {"source": source_name, "type": "text"})


# ---------- Image (vision optional) ----------
VISION_MODEL = "llava:7b"
ENABLE_VISION = os.getenv("ENABLE_IMAGE_VISION", "false").lower() == "true"


def ingest_image(file_path: str, source_name: str) -> int:
    if ENABLE_VISION:
        description = _describe_image_ollama(file_path)
    else:
        description = (
            f"Image file uploaded: {source_name}. "
            f"This image is stored in the knowledge base. "
            f"Vision-based content extraction is currently disabled "
            f"on this deployment to conserve memory. "
            f"To enable detailed image content search, set "
            f"ENABLE_IMAGE_VISION=true in the environment configuration."
        )
    chunks = chunk_text(description)
    return kb.add_chunks(
        chunks,
        {"source": source_name, "type": "image", "vision_used": ENABLE_VISION},
    )


def _describe_image_ollama(file_path: str) -> str:
    with open(file_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    prompt = (
        "Describe this image in detail. Include all visible text, "
        "objects, diagrams, charts, and technical content. This will be "
        "stored in a searchable knowledge base, so be thorough and specific."
    )
    with httpx.Client(timeout=300.0) as client:
        r = client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            },
        )
        r.raise_for_status()
        return r.json().get("response", "(no description)")


# ---------- Video (Whisper transcription) ----------
def ingest_video(file_path: str, source_name: str) -> int:
    transcript = _transcribe_video(file_path)
    chunks = chunk_text(transcript)
    return kb.add_chunks(
        chunks,
        {"source": source_name, "type": "video", "transcript_preview": transcript[:200]},
    )


def _transcribe_video(file_path: str) -> str:
    import whisper
    model = whisper.load_model("small")
    result = model.transcribe(file_path)
    return result["text"]


# ---------- main router ----------
def ingest_file(file_path: str, source_name: str) -> dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        n = ingest_pdf(file_path, source_name)
        return {"type": "pdf", "chunks_added": n}
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        n = ingest_image(file_path, source_name)
        return {"type": "image", "chunks_added": n, "vision_used": ENABLE_VISION}
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}:
        n = ingest_video(file_path, source_name)
        return {"type": "video", "chunks_added": n}
    if ext in {".txt", ".md"}:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        n = ingest_text(text, source_name)
        return {"type": "text", "chunks_added": n}
    raise ValueError(f"Unsupported file type: {ext}")
