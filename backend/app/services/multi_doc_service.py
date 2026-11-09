"""
Multi-document orchestrator with Phase 1–4 retrieval upgrades
(same behavior as LangGraph path for the non-LangGraph fallback).
"""
from __future__ import annotations

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
from app.services.graph_retrieval_service import (
    expand_context_via_graph,
    sections_from_query_entities,
)
from app.services.graph_db import is_graph_available
from app.services.query_rewrite_service import is_weak_context, generate_two_search_queries

logger = logging.getLogger("pageindex.multidoc")

CONTEXT_CHARS_PER_NODE = 3000

MULTI_DOC_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the document excerpts provided "
    "below. Excerpts may come from multiple documents.\n"
    "Rules:\n"
    "- Focus only on what was asked; ignore unrelated sections.\n"
    "- Write clear Markdown: short summary, then ## headings and bullets.\n"
    "- Mention source filename in parentheses when citing a fact.\n"
    "- Do not invent facts. Keep the answer concise and structured."
)

FINAL_CONTEXT_NODES = 4

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
    "INCOMPLETE: <short search phrase for the missing fact, max 12 words>"
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
        return document, []
    collected = await collect_relevant_nodes(db, document.id, search_query, nodes=nodes)
    return document, collected


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


def _items_to_tuples(items: list[dict]) -> list[tuple[Document, TreeNode]]:
    return [(i["document"], i["node"]) for i in items]


def _tuples_to_items(pairs: list[tuple[Document, TreeNode]]) -> list[dict]:
    return [{"document": d, "node": n} for d, n in pairs]


async def _gather_tree(
    db: Session,
    search_query: str,
    batch_id: str,
    hop_label: str,
) -> list[tuple[Document, TreeNode]]:
    candidates = await narrow_candidate_documents(db, search_query, batch_id=batch_id)
    if not candidates:
        logger.info("[%s] No candidate documents for %r", hop_label, search_query)
        return []

    nodes_by_document = _fetch_nodes_by_document(db, [d.id for d in candidates])
    results = await asyncio.gather(*[
        _collect_for_document(db, doc, search_query, nodes_by_document.get(doc.id, []))
        for doc in candidates
    ])

    gathered: list[tuple[Document, TreeNode]] = []
    for document, collected_nodes in results:
        for node in collected_nodes:
            gathered.append((document, node))
    logger.info("[%s] Tree gathered %d nodes", hop_label, len(gathered))
    return gathered


def _shorten_missing_query(raw: str, original: str) -> str:
    text = (raw or "").strip()
    text = re.sub(
        r"^(the provided excerpts.*?needed\.?\s*)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 12:
        text = " ".join(words[:12])
    return text if len(text) >= 3 else original


async def _check_completeness(question: str, context_text: str) -> tuple[bool, str | None]:
    prompt = (
        f"Question: {question}\n\nGathered excerpts:\n{context_text}\n\n"
        "Is this enough to fully answer the question?"
    )
    response = await chat_completion(prompt=prompt, system=COMPLETENESS_CHECK_SYSTEM_PROMPT)
    stripped = response.strip()
    match = re.match(r"INCOMPLETE\s*:\s*(.+)", stripped, re.IGNORECASE | re.DOTALL)
    if match:
        missing_query = _shorten_missing_query(match.group(1).strip(), question)
        logger.info("Completeness: INCOMPLETE short=%r", missing_query)
        return False, missing_query
    logger.info("Completeness: COMPLETE")
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


async def _enrich_with_graph(
    db: Session,
    batch_id: str,
    search_query: str,
    accumulated: list[tuple[Document, TreeNode]],
) -> list[tuple[Document, TreeNode]]:
    if not is_graph_available():
        return accumulated

    try:
        entity_items = sections_from_query_entities(db, search_query, batch_id)
        accumulated = _merge_unique(accumulated, _items_to_tuples(entity_items))
        logger.info("Query-entity added=%d total=%d", len(entity_items), len(accumulated))
    except Exception:
        logger.exception("Query-entity search failed")

    if not accumulated:
        return accumulated

    try:
        extra = expand_context_via_graph(
            db,
            seed_items=_tuples_to_items(accumulated),
            batch_id=batch_id,
            limit=settings.graph_expand_limit,
        )
        before = len(accumulated)
        accumulated = _merge_unique(accumulated, _items_to_tuples(extra))
        logger.info(
            "Graph expand: graph_added=%d total=%d",
            len(accumulated) - before,
            len(accumulated),
        )
    except Exception:
        logger.exception("Graph expand failed")

    return accumulated


async def answer_question_multi_doc(db: Session, question: str, batch_id: str) -> dict:
    max_hops = settings.max_hops
    max_context = settings.max_total_context_nodes

    logger.info("=== multi-doc question batch=%s: %r ===", batch_id, question)

    accumulated: list[tuple[Document, TreeNode]] = []
    search_query = question
    hop = 0
    rewrite_attempted = False

    for hop in range(1, max_hops + 1):
        hop_label = f"batch={batch_id} hop={hop}/{max_hops}"

        tree_nodes = await _gather_tree(db, search_query, batch_id, hop_label)
        accumulated = _merge_unique(accumulated, tree_nodes)
        accumulated = await _enrich_with_graph(db, batch_id, search_query, accumulated)

        if is_weak_context(len(accumulated)) and not rewrite_attempted:
            rewrite_attempted = True
            logger.info("[%s] Weak context — rewrite 2 queries", hop_label)
            try:
                rewritten = await generate_two_search_queries(db, batch_id, question)
                for rq in rewritten:
                    more = await _gather_tree(
                        db, rq, batch_id, hop_label + f" rewrite={rq[:40]}"
                    )
                    accumulated = _merge_unique(accumulated, more)
                    accumulated = await _enrich_with_graph(db, batch_id, rq, accumulated)
            except Exception:
                logger.exception("Rewrite recovery failed")

        if not accumulated:
            if hop == max_hops:
                break
            continue

        reranked = await rerank_nodes(question, accumulated, top_n=max_context)
        context_text, sources = _build_context_text(reranked)

        if hop == max_hops:
            break

        is_complete, missing_query = await _check_completeness(question, context_text)
        if is_complete or not missing_query:
            break

        search_query = missing_query
        logger.info("[%s] Next hop search=%r", hop_label, search_query)

    if not accumulated:
        answer = await chat_completion(
            prompt=f"Question: {question}\n\n(No relevant section of any document was found.)",
            system=NO_CONTEXT_SYSTEM_PROMPT,
        )
        return {
            "answer": answer,
            "sources": [],
            "hops_used": hop,
            "graph_used": is_graph_available(),
        }

    reranked = await rerank_nodes(question, accumulated, top_n=max_context)
    focused = reranked[:FINAL_CONTEXT_NODES]
    context_text, sources = _build_context_text(focused)
    final_prompt = (
        f"Document excerpts:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Write a structured Markdown answer focused only on this question."
    )
    answer = await chat_completion(prompt=final_prompt, system=MULTI_DOC_ANSWER_SYSTEM_PROMPT)

    logger.info("Final answer from %d/%d nodes, citations=%d", len(focused), len(reranked), len(sources))
    return {
        "answer": answer,
        "sources": sources,
        "hops_used": hop,
        "graph_used": is_graph_available(),
    }