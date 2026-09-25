"""
Multi-document orchestrator (batch-scoped) with multi-hop loop.

Pipeline:
  1. narrow_candidate_documents() - shortlist within the batch
  2. parallel tree-search on candidates
  3. embedding rerank
  4. completeness check; if incomplete, re-search for the gap (max hops)

Improvements:
- Stricter completeness prompt so local models stop early when key facts exist.
- Uses config-driven limits.
"""
import asyncio
import logging
import re

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Document, TreeNode
from app.services.corpus_service import narrow_candidate_documents
from app.services.retrieval_agent import collect_relevant_nodes
from app.services.reranker_service import rerank_nodes
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.multidoc")

CONTEXT_CHARS_PER_NODE = 3000

MULTI_DOC_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the document excerpts provided "
    "below. The excerpts come from multiple different source documents, "
    "each one labeled with its document filename and section title. When "
    "you use information from an excerpt, mention which document it came "
    "from (by filename). If the excerpts don't fully answer the question, "
    "say what you found and note what's missing. Do not use outside "
    "knowledge. Keep the answer concise and clear."
)

NO_CONTEXT_SYSTEM_PROMPT = (
    "Answer the question honestly. If you don't have enough information to "
    "answer it, say so plainly rather than guessing."
)

COMPLETENESS_CHECK_SYSTEM_PROMPT = (
    "You are checking whether a set of document excerpts contains enough "
    "information to fully answer a question.\n"
    "Rules:\n"
    "- If the key entities, dates, numbers, or facts the question asks for "
    "are present in the excerpts, reply COMPLETE.\n"
    "- Only reply INCOMPLETE if a primary fact the question explicitly "
    "requests is genuinely missing.\n"
    "- Do NOT say INCOMPLETE just because more detail would be nice.\n"
    "Reply in EXACTLY ONE of these two formats, nothing else:\n"
    "COMPLETE\n"
    "or\n"
    "INCOMPLETE: <a short, specific description of the missing information, "
    "phrased as something to search for>"
)


def _fetch_nodes_by_document(db: Session, document_ids: list[str]) -> dict[str, list[TreeNode]]:
    if not document_ids:
        return {}
    all_nodes = (
        db.query(TreeNode)
        .filter(TreeNode.document_id.in_(document_ids))
        .all()
    )
    by_document: dict[str, list[TreeNode]] = {doc_id: [] for doc_id in document_ids}
    for node in all_nodes:
        by_document.setdefault(node.document_id, []).append(node)
    return by_document


async def _collect_for_document(
    db: Session,
    document: Document,
    search_query: str,
    nodes: list[TreeNode],
) -> tuple[Document, list[TreeNode]]:
    if not nodes:
        logger.info("Document %s (%s) has no tree nodes - skipping", document.id, document.filename)
        return document, []
    collected = await collect_relevant_nodes(db, document.id, search_query, nodes=nodes)
    return document, collected


async def _gather_context(
    db: Session,
    search_query: str,
    batch_id: str,
    hop_label: str,
) -> list[tuple[Document, TreeNode]]:
    candidates = await narrow_candidate_documents(db, search_query, batch_id=batch_id)
    if not candidates:
        logger.info(
            "[%s] No candidate documents found in batch %s for query %r",
            hop_label, batch_id, search_query,
        )
        return []

    candidate_ids = [doc.id for doc in candidates]
    nodes_by_document = _fetch_nodes_by_document(db, candidate_ids)

    logger.info(
        "[%s] Running tree-search in parallel across %d candidate documents "
        "(batch=%s) for query %r: %s",
        hop_label, len(candidates), batch_id, search_query,
        [d.filename for d in candidates],
    )

    results = await asyncio.gather(*[
        _collect_for_document(db, doc, search_query, nodes_by_document.get(doc.id, []))
        for doc in candidates
    ])

    gathered: list[tuple[Document, TreeNode]] = []
    for document, collected_nodes in results:
        for node in collected_nodes:
            gathered.append((document, node))

    logger.info("[%s] Gathered %d raw nodes", hop_label, len(gathered))
    return gathered


def _merge_unique(
    existing: list[tuple[Document, TreeNode]],
    new: list[tuple[Document, TreeNode]],
) -> list[tuple[Document, TreeNode]]:
    seen_node_ids = {node.id for _, node in existing}
    merged = list(existing)
    for document, node in new:
        if node.id not in seen_node_ids:
            seen_node_ids.add(node.id)
            merged.append((document, node))
    return merged


async def _check_completeness(question: str, context_text: str) -> tuple[bool, str | None]:
    prompt = (
        f"Question: {question}\n\nGathered excerpts:\n{context_text}\n\n"
        "Is this enough to fully answer the question?"
    )
    response = await chat_completion(prompt=prompt, system=COMPLETENESS_CHECK_SYSTEM_PROMPT)

    stripped = response.strip()
    match = re.match(r"INCOMPLETE\s*:\s*(.+)", stripped, re.IGNORECASE | re.DOTALL)
    if match:
        missing_query = match.group(1).strip()
        logger.info("Completeness check: INCOMPLETE - missing=%r", missing_query)
        return False, missing_query or None

    logger.info("Completeness check: COMPLETE (raw reply=%r)", stripped)
    return True, None


def _build_context_text(
    nodes_with_documents: list[tuple[Document, TreeNode]],
) -> tuple[str, list[dict]]:
    context_blocks = []
    sources = []
    for document, node in nodes_with_documents:
        text = (node.raw_text or "").strip() or "(no direct text - see subsections)"
        context_blocks.append(
            f"### [{document.filename}] {node.title} "
            f"(pages {node.page_start}-{node.page_end})\n"
            f"{text[:CONTEXT_CHARS_PER_NODE]}"
        )
        sources.append({
            "document_id": document.id,
            "filename": document.filename,
            "node_id": node.id,
            "title": node.title,
            "page_start": node.page_start,
            "page_end": node.page_end,
        })
    return "\n\n".join(context_blocks), sources


async def answer_question_multi_doc(db: Session, question: str, batch_id: str) -> dict:
    """
    Answers a question by searching across every ready document in batch_id,
    with a multi-hop loop that re-searches for missing information.
    """
    max_hops = settings.max_hops
    max_context = settings.max_total_context_nodes

    logger.info("=== New multi-document question for batch %s: %r ===", batch_id, question)

    accumulated: list[tuple[Document, TreeNode]] = []
    search_query = question
    hop = 0

    for hop in range(1, max_hops + 1):
        hop_label = f"batch={batch_id} hop={hop}/{max_hops}"

        new_nodes = await _gather_context(
            db, search_query, batch_id=batch_id, hop_label=hop_label
        )
        accumulated = _merge_unique(accumulated, new_nodes)

        if not accumulated:
            logger.info("[%s] No relevant nodes found in any candidate document yet", hop_label)
            if hop == max_hops:
                break
            continue

        reranked = await rerank_nodes(question, accumulated, top_n=max_context)
        context_text, sources = _build_context_text(reranked)

        if hop == max_hops:
            logger.info("[%s] Reached MAX_HOPS - using accumulated context as-is", hop_label)
            break

        is_complete, missing_query = await _check_completeness(question, context_text)
        if is_complete or not missing_query:
            logger.info("[%s] Context judged complete - stopping hops", hop_label)
            break

        logger.info("[%s] Context incomplete - next hop will search for: %r", hop_label, missing_query)
        search_query = missing_query

    if not accumulated:
        logger.info("No relevant nodes found across any hop in batch %s", batch_id)
        answer = await chat_completion(
            prompt=f"Question: {question}\n\n(No relevant section of any document was found.)",
            system=NO_CONTEXT_SYSTEM_PROMPT,
        )
        return {"answer": answer, "sources": [], "hops_used": hop}

    reranked = await rerank_nodes(question, accumulated, top_n=max_context)
    context_text, sources = _build_context_text(reranked)

    logger.info(
        "Generating final multi-document answer from %d context nodes "
        "(batch=%s, hops_used=%d)",
        len(reranked), batch_id, hop,
    )
    final_prompt = f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}"
    answer = await chat_completion(prompt=final_prompt, system=MULTI_DOC_ANSWER_SYSTEM_PROMPT)

    return {"answer": answer, "sources": sources, "hops_used": hop}