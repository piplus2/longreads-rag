"""
rag.py
------
RAG pipeline: retrieve relevant chunks from FAISS,
assemble a prompt, call an LLM for the answer.

Can be used as a library or called directly:
    python -m src.rag --query "What are the main errors in nanopore sequencing?"
"""

import json
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/index")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    text: str
    score: float
    pmid: str
    title: str
    year: str
    authors: str
    has_full: bool


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    query: str


# ── RAG Pipeline ──────────────────────────────────────────────────────────────


class LongReadRAG:
    """
    Retrieval-Augmented Generation over long-read sequencing literature.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model — must match what was used at index time.
    top_k : int
        Number of chunks to retrieve per query.
    llm : str
        Which LLM to use for generation.
        Options: "anthropic" (Claude via API) | "openai" | "ollama" (local)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 5,
        llm: str = "ollama",
    ):
        self.top_k = top_k
        self.llm = llm
        self.model_name = model_name

        logger.info("Loading ChromaDB index...")
        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name, device="cpu")

        self.collection = client.get_collection(name="longread_papers", embedding_function=embedding_fn)

        logger.info(f"Loading embedding model: {model_name}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed query and find top-k nearest chunks."""
        results = self.collection.query(
            query_texts=[query], n_results=self.top_k, include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            score = 1 - dist  # convert distance to similarity
            
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    score=score,
                    pmid=meta["pmid"],
                    title=meta["title"],
                    year=meta["year"],
                    authors=meta["authors"],
                    has_full=meta["has_full"] == "True",
                )
            )
        return chunks

    # ── Prompt assembly ───────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {c.authors} ({c.year}). {c.title}\n{c.text}")
        context = "\n\n---\n\n".join(context_parts)

        return f"""You are an expert assistant in genomics and long-read sequencing technologies.
            Answer the question below using ONLY the provided context.
            Be specific and technical. Extract concrete facts, numbers, and mechanisms from the sources.
            Do not hedge with phrases like "further research is needed".
            If the context is insufficient say so briefly, then summarise what IS known.
            Cite sources by number [1], [2] etc.

            CONTEXT:
            {context}

            QUESTION:
            {query}

            ANSWER:"""

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _generate_ollama(self, prompt: str, model: str = "llama3.1:8b") -> str:
        """Local LLM via Ollama — free, no API key needed. Run: ollama pull mistral"""
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        return response.json()["response"]

    def _generate(self, prompt: str) -> str:
        if self.llm == "anthropic":
            return self._generate_anthropic(prompt)
        elif self.llm == "openai":
            return self._generate_openai(prompt)
        elif self.llm == "ollama":
            return self._generate_ollama(prompt)
        else:
            raise ValueError(f"Unknown LLM: {self.llm}")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def ask(self, query: str) -> RAGResponse:
        logger.info(f"Query: {query}")
        chunks = self.retrieve(query)
        prompt = self._build_prompt(query, chunks)
        answer = self._generate(prompt)
        return RAGResponse(answer=answer, sources=chunks, query=query)

    def format_response(self, response: RAGResponse) -> str:
        """Pretty-print a RAGResponse for CLI use."""
        lines = [
            f"\nQUESTION: {response.query}",
            f"\nANSWER:\n{response.answer}",
            "\nSOURCES:",
        ]
        for i, s in enumerate(response.sources, 1):
            lines.append(
                f"  [{i}] PMID:{s.pmid} | {s.authors} ({s.year}) | "
                f"score={s.score:.3f} | {'full text' if s.has_full else 'abstract'}"
            )
            lines.append(f"      {s.title}")
        return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--llm", default="anthropic", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    rag = LongReadRAG(model_name=args.model, top_k=args.top_k, llm=args.llm)
    response = rag.ask(args.query)
    print(rag.format_response(response))
