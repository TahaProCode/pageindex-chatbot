# multi_doc_service.py

"""Phase 6 (Chunk 3 + Chunk 4, then batch-scoping): multi-document orchestrator.

Chunk 3 built the base pipeline:
  - corpus_service.narrow_candidate_documents(): cheap pass that shortlists
    candidate documents out of a batch using rollup summary+keywords.
  - retrieval_agent.collect_relevant_nodes(): existing single-document
    tree-search, reused unchanged, run per candidate document in parallel.
  - reranker_service.rerank_nodes(): (Chunk 4, Step 4a) embedding-based
    cutoff replacing the old naive truncation.

Chunk 4, Step 4b adds a MULTI-HOP LOOP on top of that base pipeline:
some questions need evidence from more than one "round" of searching (e.g.
"compare Company X's and Company Y's Q3 revenue" - a single narrow+search
pass might surface only one company's numbers). After gathering context,
a cheap completeness-check LLM call decides if the context is enough to
answer the question. If not, it also tells us what's missing, we gather
context for THAT, merge it with what we already have, and try again -
up to MAX_HOPS times total, so the loop always terminates.

This orchestrator is now BATCH-SCOPED: every call requires a batch_id, and
every narrowing pass only considers documents belonging to that batch, so
a 10-document batch and a 200-document batch never bleed into each other
even though they live in the same documents table.

The single-hop gathering logic (narrow -> parallel tree-search -> rerank)
is pulled into _gather_context() so the multi-hop loop can call it
repeatedly without duplicating that logic.
"""
import asyncio
import logging
import re

from sqlalchemy.orm import Session

from app.models.models import Document, TreeNode
from app.services.corpus_service import narrow_candidate_documents
from app.services.retrieval_agent import collect_relevant_nodes
from app.services.reranker_service import rerank_nodes
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.multidoc")

# How many nodes survive reranking to go into the final answer prompt.
MAX_TOTAL_CONTEXT_NODES = 8
CONTEXT_CHARS_PER_NODE = 3000

# Multi-hop loop bounds (Chunk 4, Step 4b). MAX_HOPS=3 means: 1 initial
# gather + up to 2 additional "fill the gap" gathers, matching the
# project plan's "max 2-3 hops". This is a hard cap so the loop always
# terminates even if the completeness-check keeps saying "incomplete".
MAX_HOPS = 3

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
    "information to fully answer a question. Reply in EXACTLY ONE of these "
    "two formats, nothing else:\n"
    "COMPLETE\n"
    "or\n"
    "INCOMPLETE: <a short, specific description of the missing information, "
    "phrased as something to search for>\n"
    "Only say INCOMPLETE if something the question explicitly asks for is "
    "genuinely missing from the excerpts - not just because more detail "
    "would be nice. If the excerpts already cover the question reasonably "
    "well, say COMPLETE."
)


def _fetch_nodes_by_document(db: Session, document_ids: list[str]) -> dict[str, list[TreeNode]]:
    """One bulk query for all candidate documents' tree nodes, grouped by document_id."""
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
    """Runs the existing single-document tree-search for one candidate document."""
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
    """
    One full "round" of gathering: narrow candidate documents WITHIN
    batch_id for search_query, run tree-search on each candidate in
    parallel, and return the raw (document, node) pairs collected (NOT
    yet reranked - the caller merges across hops before doing a final
    rerank).
    """
    candidates = await narrow_candidate_documents(db, search_query, batch_id=batch_id)
    if not candidates:
        logger.info("[%s] No candidate documents found in batch %s for query %r", hop_label, batch_id, search_query)
        return []

    candidate_ids = [doc.id for doc in candidates]
    nodes_by_document = _fetch_nodes_by_document(db, candidate_ids)

    logger.info(
        "[%s] Running tree-search in parallel across %d candidate documents (batch=%s) for query %r: %s",
        hop_label, len(candidates), batch_id, search_query, [d.filename for d in candidates],
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
    """Merges two (document, node) lists, de-duplicating by node id."""
    seen_node_ids = {node.id for _, node in existing}
    merged = list(existing)
    for document, node in new:
        if node.id not in seen_node_ids:
            seen_node_ids.add(node.id)
            merged.append((document, node))
    return merged


async def _check_completeness(question: str, context_text: str) -> tuple[bool, str | None]:
    """
    Asks the LLM whether the gathered context is enough to answer the
    question. Returns (is_complete, missing_info_query_or_None).
    """
    prompt = f"Question: {question}\n\nGathered excerpts:\n{context_text}\n\nIs this enough to fully answer the question?"
    response = await chat_completion(prompt=prompt, system=COMPLETENESS_CHECK_SYSTEM_PROMPT)

    stripped = response.strip()
    match = re.match(r"INCOMPLETE\s*:\s*(.+)", stripped, re.IGNORECASE | re.DOTALL)
    if match:
        missing_query = match.group(1).strip()
        logger.info("Completeness check: INCOMPLETE - missing=%r", missing_query)
        return False, missing_query or None

    logger.info("Completeness check: COMPLETE (raw reply=%r)", stripped)
    return True, None


def _build_context_text(nodes_with_documents: list[tuple[Document, TreeNode]]) -> tuple[str, list[dict]]:
    context_blocks = []
    sources = []
    for document, node in nodes_with_documents:
        text = (node.raw_text or "").strip() or "(no direct text - see subsections)"
        context_blocks.append(
            f"### [{document.filename}] {node.title} (pages {node.page_start}-{node.page_end})\n"
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
    Answers a question by searching across every ready document in
    batch_id, with a multi-hop loop (Chunk 4, Step 4b) that re-searches
    for missing information when the first pass's context looks
    incomplete.

    Returns:
        {
            "answer": str,
            "sources": [{"document_id", "filename", "node_id", "title",
                          "page_start", "page_end"}, ...],
            "hops_used": int,
        }
    """
    logger.info("=== New multi-document question for batch %s: %r ===", batch_id, question)

    accumulated: list[tuple[Document, TreeNode]] = []
    search_query = question

    for hop in range(1, MAX_HOPS + 1):
        hop_label = f"batch={batch_id} hop={hop}/{MAX_HOPS}"

        new_nodes = await _gather_context(db, search_query, batch_id=batch_id, hop_label=hop_label)
        accumulated = _merge_unique(accumulated, new_nodes)

        if not accumulated:
            logger.info("[%s] No relevant nodes found in any candidate document yet", hop_label)
            if hop == MAX_HOPS:
                break
            continue

        # Rerank the full accumulated set against the ORIGINAL question
        # (not the sub-query) so relevance is always judged against what
        # the user actually asked, even on later hops.
        reranked = await rerank_nodes(question, accumulated, top_n=MAX_TOTAL_CONTEXT_NODES)
        context_text, sources = _build_context_text(reranked)

        if hop == MAX_HOPS:
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

    reranked = await rerank_nodes(question, accumulated, top_n=MAX_TOTAL_CONTEXT_NODES)
    context_text, sources = _build_context_text(reranked)

    logger.info("Generating final multi-document answer from %d context nodes (batch=%s, hops_used=%d)", len(reranked), batch_id, hop)
    final_prompt = f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}"
    answer = await chat_completion(prompt=final_prompt, system=MULTI_DOC_ANSWER_SYSTEM_PROMPT)

    return {"answer": answer, "sources": sources, "hops_used": hop}