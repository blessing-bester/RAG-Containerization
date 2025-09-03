from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os

from .rag import RAGPipeline
from .settings import settings

app = FastAPI(title="RAG Starter")
rag = RAGPipeline()

SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer.
If the answer is not in the context, say you don't know. Cite sources.
"""

def build_prompt(question: str, contexts):
    joined = ""
    for i, (doc, meta, dist) in enumerate(contexts, start=1):
        joined += f"\n[Chunk {i} | {meta.get('source')}]\n{doc}\n"
    return f"""{SYSTEM_PROMPT}

Context:
{joined}

Question: {question}
Answer with citations like [source: filename]."""

class IngestResp(BaseModel):
    files: int
    chunks_added: int

class QueryReq(BaseModel):
    question: str
    top_k: int | None = None

class QueryResp(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ingest", response_model=IngestResp)
def ingest():
    return rag.ingest_folder("/app/data")

async def call_openai(prompt: str) -> str:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set and GEN_BACKEND is 'openai'")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [{"role":"user","content": prompt}],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

async def call_ollama(prompt: str) -> str:
    url = f"{settings.OLLAMA_HOST}/api/generate"
    payload = {"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False, "options":{"temperature":0.1}}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response","").strip()

@app.post("/query", response_model=QueryResp)
async def query(req: QueryReq):
    ctx = rag.retrieve(req.question, req.top_k)
    prompt = build_prompt(req.question, ctx)
    if settings.GEN_BACKEND.lower() == "ollama":
        answer = await call_ollama(prompt)
    else:
        answer = await call_openai(prompt)
    sources = []
    for _, meta, _ in ctx:
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)
    return {"answer": answer, "sources": sources}
