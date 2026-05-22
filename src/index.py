"""
index.py
--------
Load raw papers, chunk text, embed with sentence-transformers,
store in FAISS. Tracks the experiment with MLflow.

Usage:
    python -m src.index
    python -m src.index --model BAAI/bge-small-en-v1.5 --chunk_size 400
"""

import json
import logging
import argparse
from pathlib import Path

import mlflow
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chromadb")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "longread_papers"
MLFLOW_EXPERIMENT = "longread_rag_indexing"


# ── Load ──────────────────────────────────────────────────────────────────────


def load_papers(path: Path) -> list[dict]:
    with open(path) as f:
        papers = json.load(f)
    logger.info(f"Loaded {len(papers)} papers")
    return papers


# ── Chunk ─────────────────────────────────────────────────────────────────────


def build_chunks(
    papers: list[dict],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> tuple[list[str], list[dict]]:
    """
    Split each paper's text into overlapping chunks.
    Returns:
        chunks    — list of text strings
        metadatas — list of dicts with pmid, title, year, source per chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks, metadatas = [], []

    for paper in papers:
        # Prefer full text; fall back to abstract
        text_body = paper.get("full_text") or paper.get("abstract", "")
        if not text_body:
            continue

        # Prepend title so every chunk has some context
        full_text = f"Title: {paper['title']}\n\n{text_body}"
        splits = splitter.split_text(full_text)

        for split in splits:
            chunks.append(split)
            metadatas.append(
                {
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "authors": ", ".join(paper.get("authors", [])[:3]),
                    "source": paper["source"],
                    "has_full": paper.get("has_full_text", False),
                }
            )

    logger.info(f"Created {len(chunks)} chunks from {len(papers)} papers")
    return chunks, metadatas


# ── ChromaDB ──────────────────────────────────────────────────────────────────


def get_collection(model_name: str, reset: bool = False, device: str = "cpu"):
    """
    Get or create a ChromaDB collection.
    If reset=True, deletes and recreates the collection.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Existing collection deleted")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def upsert_chunks(
    collection,
    texts: list[str],
    metadatas: list[dict],
    batch_size: int = 256,
) -> int:
    """
    Upsert chunks into ChromaDB in batches.
    Upsert = insert if new, update if exists — safe to re-run.
    Returns number of new chunks added.
    """
    existing_count = collection.count()

    ids = [f"{meta['pmid']}_{idx}" for idx, meta in enumerate(metadatas)]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        collection.upsert(
            documents=batch_ids and batch_texts,
            metadatas=batch_metas,
            ids=batch_ids,
        )

        if i % (batch_size * 5) == 0:
            logger.info(f"  Upserted {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    new_count = collection.count() - existing_count
    logger.info(f"Index now contains {collection.count()} chunks ({new_count} new)")
    return new_count


# ── Main ──────────────────────────────────────────────────────────────────────


def main(model_name: str, chunk_size: int, chunk_overlap: int, batch_size: int, reset: bool, device: str) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        # Log params
        mlflow.log_params(
            {
                "model_name": model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "batch_size": batch_size,
                "reset": reset,
                "backend": "chromadb",
            }
        )

        papers = load_papers(RAW_DIR / "papers.json")
        mlflow.log_metric("n_papers", len(papers))
        mlflow.log_metric("n_full_text", sum(1 for p in papers if p.get("has_full_text")))

        chunks, metadatas = build_chunks(papers, chunk_size, chunk_overlap)
        mlflow.log_metric("n_chunks", len(chunks))

        collection = get_collection(model_name, reset, device)
        new_chunks = upsert_chunks(collection, chunks, metadatas, batch_size=batch_size)

        mlflow.log_metric("new_chunks", new_chunks)
        mlflow.log_metric("total_in_index", collection.count())

        logger.info("Indexing run complete — check MLflow UI: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection before indexing")
    parser.add_argument("--device", default="cpu", help="Device to use for embeddings", choices=["cpu", "cuda"])
    args = parser.parse_args()
    main(args.model, args.chunk_size, args.chunk_overlap, args.batch_size, args.reset, args.device)
