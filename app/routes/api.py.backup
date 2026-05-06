"""API routes."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import shutil

from app.config import settings
from app.services.chat import answer_question
from app.services.ingestion import ingest_file
from app.services.knowledge_base import kb

router = APIRouter()


class ChatResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int
    query_variants: Optional[List[str]] = None


@router.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER}


@router.get("/kb/stats")
def kb_stats():
    return kb.stats()


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    upload_dir = Path(settings.UPLOAD_PATH)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = ingest_file(str(file_path), file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": file.filename, **result}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    query: str = Form(...),
    top_k: int = Form(5),
    file: Optional[UploadFile] = File(None),
):
    """Single chat inference endpoint. Accepts text query + optional file."""
    if file is not None and getattr(file, "filename", "") and file.filename.strip():
        upload_dir = Path(settings.UPLOAD_PATH)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            ingest_file(str(file_path), file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    result = answer_question(query, k=top_k)
    return ChatResponse(**result)
