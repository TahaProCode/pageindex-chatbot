### backend/app/services/corpus_service.py
"""
Phase 6 (Chunk 3), Step 2 (upgraded for 200-document scale, then again for
batch scoping): corpus-wide candidate narrowing.

When a user asks a question in "search across the batch" mode, we can't
afford to run the full tree-search (collect_relevant_nodes) on every
document in the batch - that's one or more Ollama calls PER DOCUMENT PER
TREE LEVEL, which doesn't scale past a handful of files.

Instead, this module does a cheap pass first: show the LLM every
candidate document's rollup summary + keywords (generated in Chunk 2 by
summary_service.py) and ask which documents are even worth searching for
this question. Only those "candidate" documents go on to the real
per-document tree-search in the next step (multi-doc orchestrator).

IMPORTANT: narrowing is now scoped to a single batch_id. A batch is the
set of documents a user uploaded together and is chatting with as one
unit - documents from OTHER batches must never leak into this search,
even if they're both "ready". Without this filter, a 10-document batch
would get its candidates diluted (or replaced) by unrelated documents
from a different 200-document batch sitting in the same table.

Only documents with status == ready are considered, since summary/
keywords/tree nodes don't exist yet for documents still indexing.
"""
import asyncio
import json
import logging
import re

from sqlalchemy.orm import Session

from app.models.models import Document, DocumentStatus
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.corpus")

# How many candidate documents to hand off to the per-document tree-search
# (Step 4 in the pipeline). 10-20 chosen so that even if a couple of true
# positives get missed by the LLM narrowing pass, there's enough headroom
# for the answer to still be found among the survivors.
MAX_CANDIDATE_DOCUMENTS = 15

# How many documents go into ONE narrowing prompt. Keeps each individual
# LLM call's listing short enough for a 4B model to reason over reliably.
# e.g. 200 documents / 40 per batch = 5 LLM calls, run in parallel.
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
        lines.append(f"{i + 1}. {doc.filename}\n   Summary: {summary}\n   Keywords: {keywords_str}")
    return "\n".join(lines)


def _chunk_list(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


async def _narrow_one_batch(question: str, batch: list[Document], batch_label: str) -> list[Document]:
    """
    Runs the narrowing prompt on a single LLM-call-sized batch of documents
    (<= DOCUMENTS_PER_BATCH) and returns the Document rows from THIS batch
    that the LLM picked.

    Note: this is an internal chunking batch for the LLM prompt, distinct
    from the user-facing upload Batch (models.Batch) - a single upload
    Batch of e.g. 200 documents gets split into several of these here.
    """
    listing = _format_document_listing(batch)
    prompt = f"Question: {question}\n\nDocuments:\n{listing}\n\nWhich document numbers might contain the answer?"

    logger.info(
        "[%s] Asking LLM to narrow among %d documents: %s",
        batch_label, len(batch), [d.filename for d in batch],
    )

    response = await chat_completion(prompt=prompt, system=CANDIDATE_SELECTION_SYSTEM_PROMPT)

    numbers = sorted(set(int(x) for x in re.findall(r"\d+", response)))
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
    max_candidates: int = MAX_CANDIDATE_DOCUMENTS,
    documents_per_batch: int = DOCUMENTS_PER_BATCH,
) -> list[Document]:
    """
    Returns a shortlist of Document rows worth running the full tree-search
    on for this question, using only their rollup summary+keywords.

    Scoped to a single upload-batch (batch_id) - documents from any other
    batch are never considered, no matter how many total documents exist
    in the system.

    For batches larger than `documents_per_batch`, documents are split
    into LLM-call-sized chunks and narrowed in parallel (one LLM call per
    chunk, not one call for the whole batch) - this keeps each prompt
    short and lets narrowing scale to hundreds of documents without the
    accuracy hit of one giant list.
    """
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
        _narrow_one_batch(question, chunk, batch_label=f"batch={batch_id} chunk={i + 1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ])

    # Merge picks from all chunks, preserving order and de-duplicating
    # (a document can only appear in one chunk anyway, but stay safe).
    seen_ids = set()
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
        logger.info(
            "LLM found no relevant documents in any chunk - falling back to "
            "most recent %d in this batch as a safety net", max_candidates,
        )
        return documents[:max_candidates]

    if len(merged) > max_candidates:
        logger.info(
            "Merged candidates (%d) exceed max_candidates (%d) - truncating "
            "to the %d most recently created among them",
            len(merged), max_candidates, max_candidates,
        )
        merged = merged[:max_candidates]

    return merged