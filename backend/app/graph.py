"""
LangGraph RAG with:
  Phase 1 — query→entity→section graph search (even if tree seeds empty)
  Phase 2 — robust tree-walk (in retrieval_agent)
  Phase 3 — short missing-info queries + graph_added logging
  Phase 4 — weak context → 2 rewritten queries → re-retrieve
"""
from __future__ import annotations

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, TypedDict

from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END

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
from app.core.config import settings

logger = logging.getLogger("pageindex.langgraph")

MAX_TOTAL_CONTEXT_NODES = settings.max_total_context_nodes
CONTEXT_CHARS_PER_NODE = 3000
MAX_HOPS = settings.max_hops

DECOMPOSE_SYSTEM_PROMPT = (
    "You are an expert query analyzer. Your job is to break down a user query into concise, independent "
    "sub-queries required to search across a document database. If the query asks for comparisons or multi-part "
    "information, break it into individual sub-questions. If the query is simple, reply with just the original query.\n"
    "Respond ONLY with a JSON array of string queries. Example:\n"
    '["What is Company X revenue for Q3?", "What is Company Y revenue for Q3?"]'
)

EVALUATE_SYSTEM_PROMPT = (
    "You are an objective auditor evaluating retrieved document excerpts against a user query.\n"
    "Reply in EXACTLY ONE of these formats:\n"
    "COMPLETE\n"
    "or\n"
    "INCOMPLETE: <short search phrase for the missing fact, max 12 words>\n\n"
    "Only mark INCOMPLETE if crucial facts requested by the user are missing.\n"
    "When incomplete, the phrase after the colon must be a SHORT search query, "
    "not a full sentence. Example: INCOMPLETE: sick leave entitlement days"
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are an elite research analyst. Answer using ONLY the provided document excerpts.\n"
    "Rules:\n"
    "1. Focus strictly on what the user asked — ignore unrelated leave types or sections.\n"
    "2. Write a clear Markdown answer:\n"
    "   - One-line direct summary first\n"
    "   - Then ## headings and bullet points for rules / entitlements / conditions\n"
    "   - Short paragraphs only; no wall of text\n"
    "   - Tables only when comparing numbers\n"
    "3. Mention the source filename once in parentheses where helpful.\n"
    "4. Do not invent facts beyond the excerpts."
)

# Only this many top-reranked sections are sent to the LLM and returned as citations
FINAL_CONTEXT_NODES = 4


class RAGAgentState(TypedDict):
    db: Session
    batch_id: str
    document_id: Optional[str]
    original_query: str
    sub_queries: List[str]
    accumulated_context: List[Dict[str, Any]]
    reranked_context: List[Dict[str, Any]]
    missing_info_query: Optional[str]
    hop_count: int
    is_complete: bool
    final_answer: str
    sources: List[Dict[str, Any]]
    rewrite_attempted: bool


def _fetch_nodes_by_document(db: Session, document_ids: List[str]) -> Dict[str, List[TreeNode]]:
    if not document_ids:
        return {}
    all_nodes = db.query(TreeNode).filter(TreeNode.document_id.in_(document_ids)).all()
    by_doc: Dict[str, List[TreeNode]] = {doc_id: [] for doc_id in document_ids}
    for node in all_nodes:
        by_doc.setdefault(node.document_id, []).append(node)
    return by_doc


async def _collect_for_doc(db: Session, doc: Document, query: str, nodes: List[TreeNode]):
    if not nodes:
        return doc, []
    collected = await collect_relevant_nodes(db, doc.id, query, nodes=nodes)
    return doc, collected


def _merge_items(
    existing: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen = {item["node"].id for item in existing}
    merged = list(existing)
    for item in new_items:
        nid = item["node"].id
        if nid not in seen:
            seen.add(nid)
            merged.append(item)
    return merged


def _build_context_text(nodes_with_docs: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    context_blocks = []
    sources = []
    for item in nodes_with_docs:
        doc: Document = item["document"]
        node: TreeNode = item["node"]
        text = (node.raw_text or "").strip() or "(no direct text)"
        context_blocks.append(
            f"### [{doc.filename}] {node.title} (pages {node.page_start}-{node.page_end})\n"
            f"{text[:CONTEXT_CHARS_PER_NODE]}"
        )
        sources.append({
            "document_id": doc.id,
            "filename": doc.filename,
            "node_id": node.id,
            "title": node.title,
            "page_start": node.page_start,
            "page_end": node.page_end,
        })
    return "\n\n".join(context_blocks), sources


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
    if len(text) < 3:
        text = original
    return text


async def _tree_gather(
    db: Session,
    batch_id: str,
    doc_id: Optional[str],
    sub_queries: List[str],
) -> List[Dict[str, Any]]:
    new_gathered: List[Dict[str, Any]] = []

    if doc_id:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return []
        nodes = db.query(TreeNode).filter(TreeNode.document_id == doc_id).all()
        for sq in sub_queries:
            _, collected = await _collect_for_doc(db, doc, sq, nodes)
            for n in collected:
                new_gathered.append({"document": doc, "node": n})
        return new_gathered

    for sq in sub_queries:
        candidates = await narrow_candidate_documents(db, sq, batch_id=batch_id)
        if not candidates:
            continue
        cand_ids = [c.id for c in candidates]
        nodes_by_doc = _fetch_nodes_by_document(db, cand_ids)
        tasks = [_collect_for_doc(db, c, sq, nodes_by_doc.get(c.id, [])) for c in candidates]
        results = await asyncio.gather(*tasks)
        for d, collected_nodes in results:
            for n in collected_nodes:
                new_gathered.append({"document": d, "node": n})
    return new_gathered


async def decompose_query_node(state: RAGAgentState) -> Dict[str, Any]:
    query = state["original_query"]
    logger.info("--- [LangGraph] Decomposing Query (batch=%s): %r ---", state.get("batch_id"), query)

    if state.get("document_id"):
        return {
            "sub_queries": [query],
            "hop_count": 0,
            "accumulated_context": [],
            "rewrite_attempted": False,
        }

    prompt = f"User Query: {query}"
    response = await chat_completion(prompt=prompt, system=DECOMPOSE_SYSTEM_PROMPT)

    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            sub_queries = json.loads(match.group(0))
        else:
            sub_queries = [query]
    except Exception as e:
        logger.warning("Failed to parse sub-queries JSON: %s. Using original query.", e)
        sub_queries = [query]

    if not isinstance(sub_queries, list) or not sub_queries:
        sub_queries = [query]

    logger.info("Generated Sub-Queries: %s", sub_queries)
    return {
        "sub_queries": sub_queries,
        "hop_count": 0,
        "accumulated_context": [],
        "rewrite_attempted": False,
    }


async def parallel_retrieval_node(state: RAGAgentState) -> Dict[str, Any]:
    db = state["db"]
    doc_id = state.get("document_id")
    batch_id = state["batch_id"]
    sub_queries = state.get("sub_queries") or [state["original_query"]]
    original = state["original_query"]
    current_hop = state.get("hop_count", 0) + 1
    rewrite_attempted = bool(state.get("rewrite_attempted"))

    logger.info(
        "--- [LangGraph] Retrieval Pass Hop %d/%d (batch=%s) graph=%s ---",
        current_hop, MAX_HOPS, batch_id, is_graph_available(),
    )

    tree_items = await _tree_gather(db, batch_id, doc_id, sub_queries)
    merged = _merge_items(state.get("accumulated_context") or [], tree_items)
    logger.info("Tree-walk gathered=%d accumulated=%d", len(tree_items), len(merged))

    if is_graph_available():
        try:
            search_q = sub_queries[0] if len(sub_queries) == 1 else original
            entity_items = sections_from_query_entities(db, search_q, batch_id)
            before = len(merged)
            merged = _merge_items(merged, entity_items)
            logger.info(
                "Query-entity graph hits=%d graph_query_added=%d total=%d",
                len(entity_items),
                len(merged) - before,
                len(merged),
            )
        except Exception:
            logger.exception("Query-entity search failed (non-fatal)")

    if is_graph_available() and merged:
        try:
            extra = expand_context_via_graph(
                db,
                seed_items=merged,
                batch_id=batch_id,
                limit=settings.graph_expand_limit,
            )
            before = len(merged)
            merged = _merge_items(merged, extra)
            logger.info(
                "Graph expand: graph_added=%d total=%d",
                len(merged) - before,
                len(merged),
            )
        except Exception:
            logger.exception("Graph expansion failed (non-fatal)")

    if is_weak_context(len(merged)) and not rewrite_attempted and not doc_id:
        logger.info(
            "Weak context (n=%d) — generating 2 rewritten search queries",
            len(merged),
        )
        try:
            rewritten = await generate_two_search_queries(db, batch_id, original)
            rewrite_attempted = True
            extra_tree = await _tree_gather(db, batch_id, None, rewritten)
            merged = _merge_items(merged, extra_tree)

            if is_graph_available():
                for rq in rewritten:
                    ent = sections_from_query_entities(db, rq, batch_id)
                    merged = _merge_items(merged, ent)
                if merged:
                    extra = expand_context_via_graph(
                        db, seed_items=merged, batch_id=batch_id,
                        limit=settings.graph_expand_limit,
                    )
                    merged = _merge_items(merged, extra)

            logger.info(
                "After query-rewrite retrieval: total_context=%d queries=%s",
                len(merged),
                rewritten,
            )
        except Exception:
            logger.exception("Query rewrite recovery failed (non-fatal)")
            rewrite_attempted = True

    logger.info("Total accumulated context nodes after Hop %d: %d", current_hop, len(merged))
    return {
        "accumulated_context": merged,
        "hop_count": current_hop,
        "rewrite_attempted": rewrite_attempted,
    }


async def evaluate_and_rerank_node(state: RAGAgentState) -> Dict[str, Any]:
    accumulated = state.get("accumulated_context", [])
    query = state["original_query"]
    hop_count = state.get("hop_count", 1)

    if not accumulated:
        logger.warning("No context gathered yet.")
        return {
            "is_complete": False,
            "missing_info_query": query,
            "sub_queries": [query],
            "reranked_context": [],
        }

    raw_pairs = [(item["document"], item["node"]) for item in accumulated]
    reranked_pairs = await rerank_nodes(query, raw_pairs, top_n=MAX_TOTAL_CONTEXT_NODES)
    reranked_items = [{"document": d, "node": n} for d, n in reranked_pairs]
    context_text, _ = _build_context_text(reranked_items)

    if hop_count >= MAX_HOPS:
        logger.info("Max hops (%d) reached. Proceeding to answer.", MAX_HOPS)
        return {"is_complete": True, "reranked_context": reranked_items, "missing_info_query": None}

    eval_prompt = f"User Question: {query}\n\nGathered Excerpts:\n{context_text}\n\nIs this sufficient?"
    response = await chat_completion(prompt=eval_prompt, system=EVALUATE_SYSTEM_PROMPT)

    stripped = response.strip()
    match = re.match(r"INCOMPLETE\s*:\s*(.+)", stripped, re.IGNORECASE | re.DOTALL)

    if match:
        missing_raw = match.group(1).strip()
        missing_query = _shorten_missing_query(missing_raw, query)
        logger.info(
            "Evaluation: INCOMPLETE. raw=%r short=%r",
            missing_raw[:120],
            missing_query,
        )
        return {
            "is_complete": False,
            "missing_info_query": missing_query,
            "sub_queries": [missing_query],
            "reranked_context": reranked_items,
        }

    logger.info("Evaluation: COMPLETE context verified.")
    return {"is_complete": True, "missing_info_query": None, "reranked_context": reranked_items}



def _prefer_title_matches(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable re-order: nodes whose title hits query tokens float up (keeps relative order otherwise)."""
    tokens = set()
    for m in re.finditer(r"[A-Za-z0-9]+", query or ""):
        t = m.group(0)
        if len(t) >= 3 or (len(t) >= 2 and t.isupper()):
            tokens.add(t.lower())
    if not tokens:
        return items

    def score(item: Dict[str, Any]) -> int:
        title = (item["node"].title or "").lower()
        return sum(1 for t in tokens if t in title)

    return sorted(items, key=score, reverse=True)


async def synthesize_answer_node(state: RAGAgentState) -> Dict[str, Any]:
    reranked = state.get("reranked_context", [])
    query = state["original_query"]

    if not reranked:
        return {
            "final_answer": "I could not find relevant information in the uploaded documents to answer your query.",
            "sources": [],
        }

    # Prefer title matches, then take top N for synthesis + citations
    ordered = _prefer_title_matches(query, reranked)
    focused = ordered[:FINAL_CONTEXT_NODES]
    context_text, sources = _build_context_text(focused)
    final_prompt = (
        f"Document Excerpts:\n\n{context_text}\n\n"
        f"User Question: {query}\n\n"
        "Write a structured Markdown answer focused only on this question."
    )

    logger.info(
        "Synthesizing from %d/%d context nodes (citations=%d)...",
        len(focused), len(reranked), len(sources),
    )
    final_answer = await chat_completion(prompt=final_prompt, system=SYNTHESIS_SYSTEM_PROMPT)

    return {"final_answer": final_answer, "sources": sources}


def completion_router(state: RAGAgentState) -> str:
    if state.get("is_complete") or state.get("hop_count", 0) >= MAX_HOPS:
        return "synthesize"
    return "retrieve"


def build_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGAgentState)

    workflow.add_node("decompose", decompose_query_node)
    workflow.add_node("retrieve", parallel_retrieval_node)
    workflow.add_node("evaluate", evaluate_and_rerank_node)
    workflow.add_node("synthesize", synthesize_answer_node)

    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        completion_router,
        {"synthesize": "synthesize", "retrieve": "retrieve"},
    )
    workflow.add_edge("synthesize", END)
    return workflow.compile()


langgraph_app = build_rag_graph()


async def run_langgraph_chat(
    db: Session,
    query: str,
    batch_id: str,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    initial_state: RAGAgentState = {
        "db": db,
        "batch_id": batch_id,
        "document_id": document_id,
        "original_query": query,
        "sub_queries": [],
        "accumulated_context": [],
        "reranked_context": [],
        "missing_info_query": None,
        "hop_count": 0,
        "is_complete": False,
        "final_answer": "",
        "sources": [],
        "rewrite_attempted": False,
    }

    result_state = await langgraph_app.ainvoke(initial_state)

    return {
        "answer": result_state.get("final_answer"),
        "sources": result_state.get("sources", []),
        "hops_used": result_state.get("hop_count", 1),
        "graph_used": is_graph_available(),
    }