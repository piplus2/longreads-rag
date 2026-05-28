"""
main.py
-------
FastAPI app exposing the RAG pipeline as a REST API.

Run:
    uvicorn app.main:app --reload

Endpoints:
    POST /ask          — ask a question, get answer + sources
    GET  /health       — health check
    GET  /stats        — index stats
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import LongReadRAG, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Long-Read Sequencing Literature RAG",
    description="Ask questions over ~3000 long-read sequencing papers from PubMed/PMC.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MLFLOW_EXPERIMENT = "longread_rag_queries"
request_queue = asyncio.Queue()
active_queries = 0
MAX_CONCURRENT = 2

executor = ThreadPoolExecutor(max_workers=4)


# ── Lazy-load the RAG pipeline (expensive to initialise) ──────────────────────


@lru_cache(maxsize=1)
def get_rag() -> LongReadRAG:
    logger.info("Initialising RAG pipeline...")
    return LongReadRAG(top_k=5, llm="ollama")


# ── Request / Response schemas ────────────────────────────────────────────────


class AskRequest(BaseModel):
    query: str = Field(
        ..., min_length=5, max_length=500, example="What are the main error profiles of Oxford Nanopore reads?"
    )
    top_k: int = Field(5, ge=1, le=20)


class SourceItem(BaseModel):
    pmid: str
    title: str
    year: str
    authors: str
    score: float
    has_full: bool


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


class StatsResponse(BaseModel):
    n_vectors: int
    n_chunks: int
    index_path: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/queue")
async def queue_status():
    return {
        "queued": request_queue.qsize(),
        "active": active_queries,
        "message": f"{request_queue.qsize()} queries waiting",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    index_ready = Path("data/chromadb").exists()
    return HealthResponse(status="ok", index_ready=index_ready)


@app.get("/stats", response_model=StatsResponse)
def stats():
    rag = get_rag()
    n = rag.collection.count()
    return StatsResponse(
        n_vectors=n,
        n_chunks=n,
        index_path="data/chromadb/",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    rag = get_rag()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_param("query", request.query)
        mlflow.log_param("top_k", request.top_k)

        t0 = time.perf_counter()
        try:
            # Override top_k per request if specified
            rag.top_k = request.top_k

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(executor, lambda: rag.ask(request.query))
        except Exception as e:
            logger.error(f"RAG error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        latency_ms = (time.perf_counter() - t0) * 1000
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("n_sources", len(response.sources))

    return AskResponse(
        query=response.query,
        answer=response.answer,
        latency_ms=round(latency_ms, 1),
        sources=[
            SourceItem(
                pmid=s.pmid,
                title=s.title,
                year=s.year,
                authors=s.authors,
                score=round(s.score, 4),
                has_full=s.has_full,
            )
            for s in response.sources
        ],
    )
