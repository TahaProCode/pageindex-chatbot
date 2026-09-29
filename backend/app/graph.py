"""
backend/app/graph.py

LangGraph-powered Autonomous Multi-Agent Framework for PageIndex RAG
with optional Knowledge-Graph expansion.

Key Capabilities:
1. Decomposes complex/cross-document queries into precise sub-queries.
2. Parallel context retrieval using Tree-Walking & Corpus Narrowing,
   scoped to a single upload-batch (batch_id).
3. Graph expansion: related sections via shared entities / relations.
4. Reranks & evaluates context completeness dynamically.
5. Auto-correcting feedback loop for missing context (Max Hops control).
6. Synthesizes accurate answers with complete source attribution.
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
from app.services.graph_retrieval_service import expand_context_via_graph
from app.services.graph_db import is_graph_available
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
    "INCOMPLETE: <short description of missing information to search next>\n\n"
    "Only mark INCOMPLETE if crucial facts requested by the user are missing."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are an elite research analyst. Synthesize a complete, structured, and 100% factual answer to the "
    "user query using ONLY the provided document excerpts. Explicitly attribute facts to their source document filename.\n"
    "If the query involves comparisons or tables, format the answer cleanly using Markdown tables.\n"
    "Do not extrapolate or invent facts beyond the excerpts."
)


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


async def decompose_query_node(state: RAGAgentState) -> Dict[str, Any]:
    query = state["original_query"]
    logger.info("--- [LangGraph] Decomposing Query (batch=%s): %r ---", state.get("batch_id"), query)

    if state.get("document_id"):
        return {"sub_queries": [query], "hop_count": 0, "accumulated_context": []}

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
    return {"sub_queries": sub_queries, "hop_count": 0, "accumulated_context": []}


async def parallel_retrieval_node(state: RAGAgentState) -> Dict[str, Any]:
    db = state["db"]
    doc_id = state.get("document_id")
    batch_id = state["batch_id"]
    sub_queries = state.get("sub_queries", [state["original_query"]])
    current_hop = state.get("hop_count", 0) + 1

    logger.info("--- [LangGraph] Retrieval Pass Hop %d/%d (batch=%s) ---", current_hop, MAX_HOPS, batch_id)

    new_gathered: List[Dict[str, Any]] = []

    if doc_id:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        nodes = db.query(TreeNode).filter(TreeNode.document_id == doc_id).all()
        for sq in sub_queries:
            _, collected = await _collect_for_doc(db, doc, sq, nodes)
            for n in collected:
                new_gathered.append({"document": doc, "node": n})
    else:
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

    existing_items = state.get("accumulated_context", [])
    seen_node_ids = {item["node"].id for item in existing_items}

    merged = list(existing_items)
    for item in new_gathered:
        if item["node"].id not in seen_node_ids:
            seen_node_ids.add(item["node"].id)
            merged.append(item)

    # --- Knowledge Graph expansion (hybrid) ---
    if is_graph_available() and merged:
        try:
            extra = expand_context_via_graph(
                db,
                seed_items=merged,
                batch_id=batch_id,
                limit=settings.graph_expand_limit,
            )
            for item in extra:
                if item["node"].id not in seen_node_ids:
                    seen_node_ids.add(item["node"].id)
                    merged.append(item)
            if extra:
                logger.info(
                    "Graph expansion added %d nodes (total context now %d)",
                    len(extra),
                    len(merged),
                )
        except Exception:
            logger.exception("Graph expansion failed (non-fatal)")

    logger.info("Total accumulated context nodes after Hop %d: %d", current_hop, len(merged))
    return {"accumulated_context": merged, "hop_count": current_hop}


async def evaluate_and_rerank_node(state: RAGAgentState) -> Dict[str, Any]:
    accumulated = state.get("accumulated_context", [])
    query = state["original_query"]
    hop_count = state.get("hop_count", 1)

    if not accumulated:
        logger.warning("No context gathered yet.")
        return {"is_complete": False, "missing_info_query": query, "reranked_context": []}

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
        missing_query = match.group(1).strip()
        logger.info("Evaluation: INCOMPLETE. Gap identified: %r", missing_query)
        return {
            "is_complete": False,
            "missing_info_query": missing_query,
            "sub_queries": [missing_query],
            "reranked_context": reranked_items,
        }

    logger.info("Evaluation: COMPLETE context verified.")
    return {"is_complete": True, "missing_info_query": None, "reranked_context": reranked_items}


async def synthesize_answer_node(state: RAGAgentState) -> Dict[str, Any]:
    reranked = state.get("reranked_context", [])
    query = state["original_query"]

    if not reranked:
        return {
            "final_answer": "I could not find relevant information in the uploaded documents to answer your query.",
            "sources": [],
        }

    context_text, sources = _build_context_text(reranked)
    final_prompt = f"Document Excerpts:\n\n{context_text}\n\nUser Question: {query}"

    logger.info("Synthesizing final response from %d context nodes...", len(reranked))
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
        {
            "synthesize": "synthesize",
            "retrieve": "retrieve",
        },
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
    }

    result_state = await langgraph_app.ainvoke(initial_state)

    return {
        "answer": result_state.get("final_answer"),
        "sources": result_state.get("sources", []),
        "hops_used": result_state.get("hop_count", 1),
        "graph_used": is_graph_available(),
    }