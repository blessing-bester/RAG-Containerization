import os
from pathlib import Path
from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import tiktoken  # lightweight tokenizer (optional for chunking)

from .settings import settings

def _read_text_file(p: Path) -> str:
    text = p.read_text(encoding="utf-8", errors="ignore")
    return text.strip()

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # token-aware chunking is better; this is a simple char-based fallback:
    if len(text) <= chunk_size: 
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks

class RAGPipeline:
    def __init__(self):
        os.makedirs(settings.CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space":"cosine"}
        )
        # Local embedding model
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def ingest_folder(self, folder: str = "/app/data") -> dict:
        folder_path = Path(folder)
        assert folder_path.exists(), f"{folder} not found"
        files = [p for p in folder_path.glob("**/*") if p.suffix.lower() in {".txt", ".md"}]
        added = 0
        for p in files:
            doc_text = _read_text_file(p)
            if not doc_text: 
                continue
            chunks = _chunk_text(doc_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            ids = [f"{p.name}-{i}" for i in range(len(chunks))]
            embeddings = self._embed(chunks)
            metadatas = [{"source": str(p), "chunk": i} for i in range(len(chunks))]
            self.collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
            added += len(chunks)
        return {"files": len(files), "chunks_added": added}

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[str, dict, float]]:
        k = top_k or settings.TOP_K
        q_emb = self._embed([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances"])
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return list(zip(docs, metas, dists))
