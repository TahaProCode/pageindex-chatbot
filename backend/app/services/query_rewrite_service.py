"""
Unclear / weak-retrieval recovery:

When tree + graph yield 0 or very few sections, extract keywords from the
query, pull matching tree titles/summaries as hints, ask the LLM for exactly
TWO clearer search queries, then the caller re-runs retrieval on those.
"""
from __future__ import annotations

import json
import logging
import re

from sqlalchemy.orm import Session

from app.models.models import Document, DocumentStatus, TreeNode
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.query_rewrite")

# Treat retrieval as weak below this many context nodes
WEAK_CONTEXT_THRESHOLD = 2

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user question into clearer document-search queries.\n"
    "You are given the original question, keywords, and a few matching "
    "section titles from the document library.\n"
    "Reply with ONLY a JSON array of exactly 2 short search queries "
    "(no markdown, no explanation). Each query should be specific enough "
    "to find policy/rules sections. Example:\n"
    '["Sick Leave SL entitlement and rules", "Sick Leave notice proof carry forward"]'
)


def is_weak_context(n_nodes: int) -> bool:
    return n_nodes < WEAK_CONTEXT_THRESHOLD


def _query_keywords(query: str) -> list[str]:
    kws: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9]+", query or ""):
        t = m.group(0)
        if len(t) >= 3 or (len(t) >= 2 and t.isupper()):
            key = t.lower()
            if key not in seen and key not in {
                "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
            }:
                seen.add(key)
                kws.append(t)
    return kws


def find_keyword_tree_hints(
    db: Session,
    batch_id: str,
    keywords: list[str],
    limit: int = 12,
) -> list[str]:
    """Match keywords against tree node titles/summaries in the batch."""
    if not keywords:
        return []

    docs = (
        db.query(Document)
        .filter(Document.batch_id == batch_id, Document.status == DocumentStatus.ready)
        .all()
    )
    if not docs:
        return []

    doc_ids = [d.id for d in docs]
    nodes = db.query(TreeNode).filter(TreeNode.document_id.in_(doc_ids)).all()

    kw_lower = [k.lower() for k in keywords]
    scored: list[tuple[int, str]] = []
    for n in nodes:
        title = (n.title or "")
        summary = (n.summary or "")
        blob = f"{title} {summary}".lower()
        hits = sum(1 for k in kw_lower if k in blob)
        if hits:
            label = title.strip() or "(untitled)"
            if n.page_start:
                label = f"{label} (p{n.page_start})"
            scored.append((hits, label))

    scored.sort(key=lambda x: x[0], reverse=True)
    # unique titles preserve order
    seen: set[str] = set()
    hints: list[str] = []
    for _, label in scored:
        if label not in seen:
            seen.add(label)
            hints.append(label)
        if len(hints) >= limit:
            break
    return hints


async def generate_two_search_queries(
    db: Session,
    batch_id: str,
    original_query: str,
) -> list[str]:
    """
    Returns up to 2 rewritten search queries. Falls back to [original] on failure.
    """
    keywords = _query_keywords(original_query)
    hints = find_keyword_tree_hints(db, batch_id, keywords)
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "(no matching section titles)"

    prompt = (
        f"Original question: {original_query}\n"
        f"Keywords: {', '.join(keywords) if keywords else '(none)'}\n"
        f"Matching section titles from documents:\n{hints_text}\n\n"
        "Produce exactly 2 clearer search queries as a JSON array."
    )

    try:
        raw = await chat_completion(prompt=prompt, system=REWRITE_SYSTEM_PROMPT)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                queries = [str(q).strip() for q in data if str(q).strip()]
                queries = queries[:2]
                if queries:
                    logger.info(
                        "Query rewrite: original=%r keywords=%s hints=%d -> %s",
                        original_query, keywords, len(hints), queries,
                    )
                    return queries
    except Exception:
        logger.exception("Query rewrite failed for %r", original_query)

    # Fallback: original + keyword-expanded variant
    if keywords:
        expanded = f"{original_query} {' '.join(keywords)} policy rules entitlement"
        return [original_query, expanded]
    return [original_query]