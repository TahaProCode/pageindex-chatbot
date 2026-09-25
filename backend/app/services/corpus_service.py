"""
Corpus-wide candidate narrowing, scoped to a single upload batch.

When a user asks a question we cannot afford full tree-search on every
document. This module does a cheap pass first using document-level
summary + keywords, then (if the LLM finds nothing) falls back to a
simple BM25-style keyword ranking so we never silently return "most
recent documents" that have nothing to do with the query.
"""
import asyncio
import json
import logging
import math
import re
from collections import Counter

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Document, DocumentStatus
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.corpus")

MAX_CANDIDATE_DOCUMENTS = 15
DOCUMENTS_PER_BATCH = 40

CANDIDATE_SELECTION_SYSTEM_PROMPT = (
    "You are choosing which documents are worth searching to answer a "
    "question. You are shown a numbered list of documents, each with a "
    "short summary and keywords. Reply with ONLY the numbers of documents "
    "that might contain relevant information, separated by commas "
    "(example: 1,3,4). Include a document even if you're only somewhat "
    "sure it's relevant - it's fine to include more than the minimum. "
    "If truly none of them seem relevant, reply with exactly: NONE. "
    "Do not add any explanation."
)


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _format_document_listing(documents: list[Document]) -> str:
    lines = []
    for i, doc in enumerate(documents):
        keywords = _parse_keywords(doc.keywords)
        keywords_str = ", ".join(keywords) if keywords else "(none)"
        summary = doc.summary or "(no summary available)"
        lines.append(
            f"{i + 1}. {doc.filename}\n   Summary: {summary}\n   Keywords: {keywords_str}"
        )
    return "\n".join(lines)


def _chunk_list(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


def _bm25_rank(
    question: str,
    documents: list[Document],
    top_n: int,
) -> list[Document]:
    """
    Lightweight BM25-inspired ranking over document filename + summary + keywords.
    Used as fallback when the LLM narrowing returns nothing relevant.
    """
    query_tokens = _tokenize(question)
    if not query_tokens or not documents:
        return documents[:top_n]

    # Document "text" for scoring
    doc_texts = []
    for doc in documents:
        parts = [doc.filename or ""]
        if doc.summary:
            parts.append(doc.summary)
        kws = _parse_keywords(doc.keywords)
        parts.extend(kws)
        doc_texts.append(" ".join(parts))

    tokenized_docs = [_tokenize(t) for t in doc_texts]
    doc_lens = [len(td) or 1 for td in tokenized_docs]
    avgdl = sum(doc_lens) / len(doc_lens)

    # Document frequency
    df: Counter = Counter()
    for td in tokenized_docs:
        df.update(set(td))

    N = len(documents)
    k1 = 1.5
    b = 0.75

    scores = []
    for i, td in enumerate(tokenized_docs):
        tf = Counter(td)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            idf = math.log(1 + (N - df[qt] + 0.5) / (df[qt] + 0.5))
            freq = tf[qt]
            denom = freq + k1 * (1 - b + b * doc_lens[i] / avgdl)
            score += idf * (freq * (k1 + 1)) / denom
        scores.append((score, documents[i]))

    scores.sort(key=lambda x: x[0], reverse=True)
    ranked = [doc for sc, doc in scores if sc > 0][:top_n]

    logger.info(
        "BM25 fallback ranked %d docs (top scores): %s",
        len(ranked),
        [(round(sc, 3), d.filename) for sc, d in scores[:5]],
    )
    # If BM25 found nothing, fall back to most recent as last resort
    if not ranked:
        return documents[:top_n]
    return ranked


async def _narrow_one_batch(
    question: str, batch: list[Document], batch_label: str
) -> list[Document]:
    listing = _format_document_listing(batch)
    prompt = (
        f"Question: {question}\n\nDocuments:\n{listing}\n\n"
        "Which document numbers might contain the answer?"
    )

    logger.info(
        "[%s] Asking LLM to narrow among %d documents: %s",
        batch_label, len(batch), [d.filename for d in batch],
    )

    response = await chat_completion(prompt=prompt, system=CANDIDATE_SELECTION_SYSTEM_PROMPT)

    numbers = sorted(set(int(x) for x in re.findall(r"\b\d+\b", response)))
    selected = [batch[n - 1] for n in numbers if 1 <= n <= len(batch)]

    logger.info(
        "[%s] LLM raw reply=%r -> parsed_numbers=%s -> picked=%s",
        batch_label, response, numbers, [d.filename for d in selected],
    )
    return selected


async def narrow_candidate_documents(
    db: Session,
    question: str,
    batch_id: str,
    max_candidates: int | None = None,
    documents_per_batch: int = DOCUMENTS_PER_BATCH,
) -> list[Document]:
    """
    Returns a shortlist of Document rows worth running the full tree-search
    on for this question, scoped to a single upload-batch.
    """
    max_candidates = max_candidates or settings.max_candidate_documents

    documents = (
        db.query(Document)
        .filter(Document.batch_id == batch_id, Document.status == DocumentStatus.ready)
        .order_by(Document.created_at.desc())
        .all()
    )

    if not documents:
        logger.info("No ready documents in batch %s - nothing to narrow", batch_id)
        return []

    if len(documents) <= max_candidates:
        logger.info(
            "Batch %s has only %d ready documents (<= max_candidates=%d) - "
            "skipping LLM narrowing, using all of them",
            batch_id, len(documents), max_candidates,
        )
        return documents

    chunks = _chunk_list(documents, documents_per_batch)
    logger.info(
        "Narrowing %d documents in batch %s across %d chunk(s) of up to %d each (parallel)",
        len(documents), batch_id, len(chunks), documents_per_batch,
    )

    chunk_results = await asyncio.gather(*[
        _narrow_one_batch(
            question, chunk,
            batch_label=f"batch={batch_id} chunk={i + 1}/{len(chunks)}",
        )
        for i, chunk in enumerate(chunks)
    ])

    seen_ids: set[str] = set()
    merged: list[Document] = []
    for chunk_selected in chunk_results:
        for doc in chunk_selected:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                merged.append(doc)

    logger.info(
        "Merged candidates across all chunks for batch %s: %d documents -> %s",
        batch_id, len(merged), [d.filename for d in merged],
    )

    if not merged:
        # BM25 / keyword ranking fallback instead of blind "most recent"
        logger.info(
            "LLM found no relevant documents - applying BM25 keyword ranking fallback"
        )
        return _bm25_rank(question, documents, max_candidates)

    if len(merged) > max_candidates:
        logger.info(
            "Merged candidates (%d) exceed max_candidates (%d) - truncating",
            len(merged), max_candidates,
        )
        merged = merged[:max_candidates]

    return merged