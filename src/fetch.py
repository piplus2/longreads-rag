"""
fetch.py
--------
Fetch long-read sequencing papers from PubMed and PubMed Central.
Saves raw results to data/raw/ as JSON.

Usage:
    python -m src.fetch --query "long read sequencing" --max_results 3000
"""

import os
import re

import time
import json
import logging
import argparse
from pathlib import Path

from Bio import Entrez

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

Entrez.email = os.getenv("ENTREZ_EMAIL", "your@email.com")  # set via env var

PUBMED_QUERY = (
    '("long-read sequencing" OR "long read sequencing" OR '
    '"Oxford Nanopore" OR "PacBio" OR "SMRT sequencing" OR '
    '"nanopore sequencing") AND '
    '("High-Throughput Nucleotide Sequencing"[MeSH])'
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── PubMed: fetch abstracts ───────────────────────────────────────────────────


def search_pubmed(query: str, max_results: int = 3000) -> list[str] | None:
    """Return list of PubMed IDs matching query."""
    logger.info(f"Searching PubMed for: {query[:80]}...")
    with Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y") as handle:
        record = Entrez.read(handle)
    if not record:
        logger.warning("No records found in PubMed search.")
        return None
    pmids: list[str] = record["IdList"]  # type: ignore
    logger.info(f"Found {len(pmids)} PubMed IDs")
    return pmids


def fetch_europe_pmc(
    query: str = "long read sequencing",
    max_results: int = 1000,
) -> list[dict]:
    """
    Fetch papers from Europe PMC — broader coverage than PubMed,
    includes Wellcome Trust and European funder mandated papers.
    """
    import requests

    logger.info(f"Fetching Europe PMC papers for: {query}")
    papers = []
    cursor_mark = "*"
    page_size = 100

    while len(papers) < max_results:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
            "resultType": "core",  # includes abstracts
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Europe PMC request failed: {e}")
            break

        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        for item in results:
            abstract = item.get("abstractText", "").strip()
            if not abstract:
                continue

            # Avoid duplicates with PubMed — skip if PMID already fetched
            pmid = str(item.get("pmid", ""))
            epmc_id = f"epmc_{item.get('id', '')}"

            papers.append(
                {
                    "pmid": pmid if pmid else epmc_id,
                    "pmc_id": item.get("pmcid"),
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "full_text": None,
                    "authors": [a.get("fullName", "") for a in item.get("authorList", {}).get("author", [])[:5]],
                    "year": str(item.get("pubYear", "")),
                    "source": "europe_pmc",
                    "has_full_text": False,
                }
            )

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor
        time.sleep(0.3)

    logger.info(f"Retrieved {len(papers)} Europe PMC papers")
    return papers


def fetch_biorxiv(
    query: str = "long-read sequencing",
    start_date: str = "2020-01-01",
    end_date: str = "2025-04-01",
    max_results: int = 1000,
) -> list[dict]:
    import requests

    logger.info(f"Fetching bioRxiv preprints for: {query}")
    papers = []
    cursor = 0

    while len(papers) < max_results:
        url = f"https://api.biorxiv.org/details/biorxiv/{query}/{start_date}/{end_date}"
        try:
            resp = requests.get(url, timeout=10).json()
            resp.raise_for_statue()
            data = resp.json()
        except Exception as e:
            logger.warning(f"bioRxiv request failed: {e}")
            break

        collection = data.get("collection", [])
        if not collection:
            break

        for item in collection:
            abstract = item.get("abstract", "").strip()
            if not abstract:
                continue

            papers.append(
                {
                    "pmid": f"biorxiv_{item['doi'].replace('/', '_')}",
                    "pmc_id": None,
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "full_text": None,
                    "authors": [a.strip() for a in item.get("authors", "").split(";")],
                    "year": item.get("date", "")[:4],
                    "source": "biorxiv",
                    "has_full_text": False,
                }
            )

        total = data.get("messages", [{}])[0].get("count", 0)
        logger.info(f"  cursor={cursor}, collected so far={len(papers)}, total in range={total}")

        cursor += 100
        if cursor >= total:
            break

        time.sleep(0.5)

    logger.info(f"Retrieved {len(papers)} bioRxiv preprints")
    return papers


def fetch_abstracts(pmids: list[str], batch_size: int = 200) -> list[dict]:
    """
    Fetch title + abstract for a list of PubMed IDs.
    Returns list of dicts: {pmid, title, abstract, authors, year, pmc_id}
    """
    papers = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        logger.info(f"Fetching abstracts {i}–{i + len(batch)}")
        with Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml") as handle:
            records = Entrez.read(handle)

        if records is None or "PubmedArticle" not in records:  # type: ignore
            logger.warning("No records found in PubMed fetch.")
            continue

        for article in records.get("PubmedArticle", []):  # type: ignore
            try:
                medline = article["MedlineCitation"]
                art = medline["Article"]

                title = str(art.get("ArticleTitle", ""))
                abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(p) for p in abstract_parts)

                authors = []
                for a in art.get("AuthorList", []):
                    last = a.get("LastName", "")
                    fore = a.get("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}".strip())

                year = ""
                pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")))

                pmid = str(medline["PMID"])

                # Grab PMC ID if available (used later for full-text fetch)
                pmc_id = None
                for id_obj in article.get("PubmedData", {}).get("ArticleIdList", []):
                    if str(id_obj.attributes.get("IdType")) == "pmc":
                        pmc_id = str(id_obj)

                if abstract:  # skip papers with no abstract
                    papers.append(
                        {
                            "pmid": pmid,
                            "pmc_id": pmc_id,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": year,
                            "source": "pubmed",
                            "has_full_text": False,
                        }
                    )
            except Exception as e:
                logger.warning(f"Skipping article: {e}")

        time.sleep(0.4)  # NCBI rate limit: max 3 requests/sec without API key

    logger.info(f"Retrieved {len(papers)} papers with abstracts")
    return papers


# ── PMC: fetch full text ──────────────────────────────────────────────────────


def fetch_full_text(papers: list[dict], batch_size: int = 50) -> list[dict]:
    """
    For papers that have a PMC ID, fetch full text and replace abstract.
    Updates papers in-place and sets has_full_text=True.
    """
    pmc_papers = [p for p in papers if p["pmc_id"]]
    logger.info(f"Fetching full text for {len(pmc_papers)} PMC papers")

    pmc_index = {p["pmid"]: i for i, p in enumerate(papers)}

    for i in range(0, len(pmc_papers), batch_size):
        batch = pmc_papers[i : i + batch_size]
        pmc_ids = [p["pmc_id"] for p in batch]
        logger.info(f"  Full text batch {i}–{i + len(batch)}")

        try:
            with Entrez.efetch(db="pmc", id=",".join(pmc_ids), rettype="xml", retmode="xml") as handle:
                raw = handle.read()

            if not raw:
                logger.warning("No data returned for PMC full text fetch.")
                continue

            # Simple extraction: pull all <p> text from XML
            # For production, use a proper PMC XML parser
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", raw.decode("utf-8"), re.DOTALL)  # type: ignore
            clean = [re.sub(r"<[^>]+>", "", p).strip() for p in paragraphs]
            full_text = " ".join(t for t in clean if len(t) > 50)

            # Distribute text back — rough heuristic per paper
            chunk_size = len(full_text) // max(len(batch), 1)
            for j, paper in enumerate(batch):
                start = j * chunk_size
                excerpt = full_text[start : start + chunk_size]
                if excerpt:
                    idx = pmc_index[paper["pmid"]]
                    papers[idx]["full_text"] = excerpt
                    papers[idx]["has_full_text"] = True

        except Exception as e:
            logger.warning(f"Full text batch failed: {e}")

        time.sleep(0.4)

    full_text_count = sum(1 for p in papers if p["has_full_text"])
    logger.info(f"Full text retrieved for {full_text_count} papers")
    return papers


# ── Save ──────────────────────────────────────────────────────────────────────


def save(papers: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(papers, f, indent=2)
    logger.info(f"Saved {len(papers)} papers to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(query: str, max_results: int, fetch_full: bool, include_europe_pmc: bool) -> None:
    pmids = search_pubmed(query, max_results)
    if not pmids:
        logger.error("No PubMed IDs found, exiting.")
        return
    papers = fetch_abstracts(pmids)

    if fetch_full:
        papers = fetch_full_text(papers)

    if include_europe_pmc:
        epmc_papers = fetch_europe_pmc(
            query="long read sequencing OR nanopore OR PacBio HiFi",
            max_results=1000,
        )
        existing_ids = {p["pmid"] for p in papers}
        new = [p for p in epmc_papers if p["pmid"] not in existing_ids]
        papers.extend(new)
        logger.info(f"Total after Europe PMC merge: {len(papers)} papers")

    out_path = RAW_DIR / "papers.json"
    save(papers, out_path)
    logger.info("Fetch complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=PUBMED_QUERY)
    parser.add_argument("--max_results", type=int, default=3000)
    parser.add_argument("--fetch_full", action="store_true", help="Also fetch full text from PMC where available")
    parser.add_argument("--include_europe_pmc", action="store_true", help="Also fetch preprints from Europe PMC")
    args = parser.parse_args()
    main(args.query, args.max_results, args.fetch_full, args.include_europe_pmc)
