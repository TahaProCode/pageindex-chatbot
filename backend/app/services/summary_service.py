"""
Phase 3 (extended in Phase 6): generates a short summary AND keywords for
every TreeNode of a document, then rolls those up into one document-level
summary+keywords (used later for corpus-wide candidate narrowing).

Runs bottom-up (children before parents) so a parent node with no direct
text of its own gets summarized from its children's summaries instead of
being left blank.
"""
import asyncio
import json
import logging
import re

from sqlalchemy.orm import Session

from app.models.models import TreeNode, Document
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.summary")

SUMMARY_SYSTEM_PROMPT = (
    "You summarize one section of a document for a retrieval system. "
    "Given the section title and its text, reply in EXACTLY this format, nothing else:\n"
    "SUMMARY: <one short sentence, max 25 words, describing what this section covers>\n"
    "KEYWORDS: <5-8 comma-separated keywords or key phrases capturing the important "
    "terms, names, and numbers mentioned, so nothing important is missed>"
)

DOC_ROLLUP_SYSTEM_PROMPT = (
    "You write a short overview of an entire document for a search index, based on "
    "summaries of its top-level sections. Reply in EXACTLY this format, nothing else:\n"
    "SUMMARY: <2-3 sentences describing what this document is about overall>\n"
    "KEYWORDS: <8-12 comma-separated keywords covering the document's main topics, "
    "entities, and terms>"
)

MAX_CONTENT_CHARS = 4000
MAX_ROOTS_FOR_ROLLUP = 20


def _parse_summary_response(response: str) -> tuple[str, list[str]]:
    """
    Robust line-based parser for the SUMMARY:/KEYWORDS: format above.
    Falls back to treating the whole response as the summary if the model
    doesn't follow the format at all.
    """
    summary = ""
    keywords: list[str] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("summary"):
            _, _, rest = line.partition(":")
            summary = rest.strip()
        elif lower.startswith("keyword"):
            _, _, rest = line.partition(":")
            raw = rest.strip()
            parts = re.split(r"[;,]", raw)
            keywords = [p.strip() for p in parts if p.strip()]
    if not summary:
        summary = response.strip()
    return summary, keywords


def _build_children_map(nodes: list[TreeNode]) -> dict[str | None, list[TreeNode]]:
    children_map: dict[str | None, list[TreeNode]] = {}
    for n in nodes:
        children_map.setdefault(n.parent_id, []).append(n)
    for key in children_map:
        children_map[key].sort(key=lambda n: n.order_index)
    return children_map


async def _summarize_node(node: TreeNode, children_summaries: list[str]) -> dict:
    if node.raw_text and node.raw_text.strip():
        content = node.raw_text[:MAX_CONTENT_CHARS]
    elif children_summaries:
        content = "\n".join(f"- {s}" for s in children_summaries)
    else:
        content = "(no content)"

    prompt = f"Section title: {node.title}\n\nContent:\n{content}"
    response = await chat_completion(prompt=prompt, system=SUMMARY_SYSTEM_PROMPT)
    summary, keywords = _parse_summary_response(response)
    return {"summary": summary or node.title, "keywords": keywords}


async def _summarize_tree_async(db: Session, document_id: str) -> None:
    nodes = db.query(TreeNode).filter(TreeNode.document_id == document_id).all()
    if not nodes:
        return

    children_map = _build_children_map(nodes)
    by_id = {n.id: n for n in nodes}
    summaries: dict[str, str] = {}

    def post_order(parent_id: str | None) -> list[str]:
        order = []
        for child in children_map.get(parent_id, []):
            order.extend(post_order(child.id))
            order.append(child.id)
        return order

    ordered_ids = post_order(None)

    for node_id in ordered_ids:
        node = by_id[node_id]
        child_summaries = [summaries[c.id] for c in children_map.get(node_id, [])]
        result = await _summarize_node(node, child_summaries)
        summaries[node_id] = result["summary"]
        node.summary = result["summary"]
        node.keywords = json.dumps(result["keywords"])
        db.add(node)

    db.commit()


async def _generate_document_rollup(db: Session, document_id: str) -> None:
    roots = (
        db.query(TreeNode)
        .filter(TreeNode.document_id == document_id, TreeNode.parent_id.is_(None))
        .order_by(TreeNode.order_index)
        .limit(MAX_ROOTS_FOR_ROLLUP)
        .all()
    )
    if not roots:
        return

    listing = "\n".join(f"- {n.title}: {n.summary or '(no summary)'}" for n in roots)
    prompt = f"Section summaries:\n{listing}"

    response = await chat_completion(prompt=prompt, system=DOC_ROLLUP_SYSTEM_PROMPT)
    summary, keywords = _parse_summary_response(response)

    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        document.summary = summary or document.filename
        document.keywords = json.dumps(keywords)
        db.add(document)
        db.commit()
        logger.info("Document %s rollup: summary=%r keywords=%s", document_id, summary, keywords)


def generate_summaries_for_document(db: Session, document_id: str) -> None:
    """Sync wrapper - safe to call from a sync context (e.g. a background task).
    Generates per-node summaries+keywords, then rolls them up into one
    document-level summary+keywords."""
    async def _run():
        await _summarize_tree_async(db, document_id)
        await _generate_document_rollup(db, document_id)

    asyncio.run(_run())