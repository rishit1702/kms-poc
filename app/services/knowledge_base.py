"""
Knowledge Base service — handles embedding & retrieval via ChromaDB.
Videos, PDFs, and images all get processed into text chunks and stored here.
"""
import chromadb
from chromadb.utils import embedding_functions
from app.config import settings
from typing import List, Dict
import uuid


class KnowledgeBase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        # Local embedding model — no API key needed, runs on CPU
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="kms_kb",
            embedding_function=self.embedder,
        )

    def add_chunks(self, chunks: List[str], metadata: Dict):
        """Store text chunks with metadata (source file, type, etc.)."""
        if not chunks:
            return 0
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata.copy() for _ in chunks]
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        results = self.collection.query(query_texts=[query], n_results=k)
        if not results["documents"] or not results["documents"][0]:
            return []
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def stats(self) -> Dict:
        return {"total_chunks": self.collection.count()}


# Singleton
kb = KnowledgeBase()
