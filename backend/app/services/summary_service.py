"""
Phase 3 (extended): generates a short summary AND keywords for every
TreeNode of a document, then rolls those up into one document-level
summary+keywords (used later for corpus-wide candidate narrowing).

Improvements:
- Parallel summarization of independent nodes via asyncio.gather +
  semaphore (max_concurrent_summaries) so a 30-node document no longer
  takes 30 sequential LLM round-trips.
- Robust keyword cleaning (strips bullets, newlines, special chars).
- Sub-batched document rollup when there are more than MAX_ROOTS_FOR_ROLLUP
  top-level sections (no more silent truncation of later sections).
"""
import asyncio
import json
import logging
import re

from sqlalchemy.orm import Session

from app.core.config import settings
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
ROLLUP_BATCH_SIZE = 15


def _clean_keyword(raw: str) -> str:
    """Strip bullets, numbering, quotes and excess whitespace from one keyword."""
    k = raw.strip()
    k = re.sub(r"^[\-\*\u2022\d\.\)\(]+\s*", "", k)
    k = k.strip(" \t\"'`")
    # Drop empty or pure-punctuation tokens
    if not k or re.fullmatch(r"[\W_]+", k):
        return ""
    return k[:80]


def _parse_summary_response(response: str) -> tuple[str, list[str]]:
    """
    Robust line-based parser for the SUMMARY:/KEYWORDS: format.
    Falls back to treating the whole response as the summary if the model
    doesn't follow the format at all. Keywords are aggressively cleaned.
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
            # Split on commas, semicolons, or newlines
            parts = re.split(r"[;,\n]", raw)
            keywords = [k for p in parts if (k := _clean_keyword(p))]
    if not summary:
        summary = response.strip()[:300]
    # Deduplicate while preserving order
    seen = set()
    unique_kw = []
    for k in keywords:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            unique_kw.append(k)
    return summary, unique_kw[:12]


def _build_children_map(nodes: list[TreeNode]) -> dict[str | None, list[TreeNode]]:
    children_map: dict[str | None, list[TreeNode]] = {}
    for n in nodes:
        children_map.setdefault(n.parent_id, []).append(n)
    for key in children_map:
        children_map[key].sort(key=lambda n: n.order_index)
    return children_map


async def _summarize_node(
    node: TreeNode,
    children_summaries: list[str],
    semaphore: asyncio.Semaphore,
) -> dict:
    if node.raw_text and node.raw_text.strip():
        content = node.raw_text[:MAX_CONTENT_CHARS]
    elif children_summaries:
        content = "\n".join(f"- {s}" for s in children_summaries)
    else:
        content = "(no content)"

    prompt = f"Section title: {node.title}\n\nContent:\n{content}"

    async with semaphore:
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
    semaphore = asyncio.Semaphore(getattr(settings, "max_concurrent_summaries", 4))

    def post_order(parent_id: str | None) -> list[str]:
        order = []
        for child in children_map.get(parent_id, []):
            order.extend(post_order(child.id))
            order.append(child.id)
        return order

    ordered_ids = post_order(None)

    # Process level-by-level in post-order batches of independent nodes.
    # Nodes whose children are already summarized can run in parallel.
    # We still respect post-order: a parent only runs after all its children.
    remaining = set(ordered_ids)
    completed: set[str] = set()

    while remaining:
        # A node is ready when all its children (that are in the tree) are done
        ready = []
        for nid in list(remaining):
            child_ids = [c.id for c in children_map.get(nid, [])]
            if all(cid in completed for cid in child_ids):
                ready.append(nid)

        if not ready:
            # Safety: if dependency cycle somehow exists, finish remaining sequentially
            ready = list(remaining)[: settings.max_concurrent_summaries]

        async def _run_one(nid: str) -> tuple[str, dict]:
            node = by_id[nid]
            child_summaries = [
                summaries[c.id]
                for c in children_map.get(nid, [])
                if c.id in summaries
            ]
            result = await _summarize_node(node, child_summaries, semaphore)
            return nid, result

        results = await asyncio.gather(*[_run_one(nid) for nid in ready])

        for nid, result in results:
            summaries[nid] = result["summary"]
            node = by_id[nid]
            node.summary = result["summary"]
            node.keywords = json.dumps(result["keywords"])
            db.add(node)
            completed.add(nid)
            remaining.discard(nid)

        db.commit()
        logger.info(
            "Document %s: summarized %d/%d nodes",
            document_id, len(completed), len(ordered_ids),
        )


async def _rollup_batch(section_listings: list[str]) -> tuple[str, list[str]]:
    """Run one rollup LLM call over a list of 'title: summary' lines."""
    listing = "\n".join(section_listings)
    prompt = f"Section summaries:\n{listing}"
    response = await chat_completion(prompt=prompt, system=DOC_ROLLUP_SYSTEM_PROMPT)
    return _parse_summary_response(response)


async def _generate_document_rollup(db: Session, document_id: str) -> None:
    roots = (
        db.query(TreeNode)
        .filter(TreeNode.document_id == document_id, TreeNode.parent_id.is_(None))
        .order_by(TreeNode.order_index)
        .all()
    )
    if not roots:
        return

    listings = [
        f"- {n.title}: {n.summary or '(no summary)'}"
        for n in roots
    ]

    if len(listings) <= MAX_ROOTS_FOR_ROLLUP:
        summary, keywords = await _rollup_batch(listings)
    else:
        # Sub-batch: roll up groups of ROLLUP_BATCH_SIZE, then roll up the intermediates
        logger.info(
            "Document %s has %d roots (> %d) - using sub-batched rollup",
            document_id, len(listings), MAX_ROOTS_FOR_ROLLUP,
        )
        intermediate: list[str] = []
        for i in range(0, len(listings), ROLLUP_BATCH_SIZE):
            chunk = listings[i : i + ROLLUP_BATCH_SIZE]
            s, _ = await _rollup_batch(chunk)
            intermediate.append(f"- Part {i // ROLLUP_BATCH_SIZE + 1}: {s}")

        summary, keywords = await _rollup_batch(intermediate)

    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        document.summary = summary or document.filename
        document.keywords = json.dumps(keywords)
        db.add(document)
        db.commit()
        logger.info(
            "Document %s rollup: summary=%r keywords=%s",
            document_id, summary, keywords,
        )


def generate_summaries_for_document(db: Session, document_id: str) -> None:
    """Sync wrapper - safe to call from a sync context (e.g. a background task).
    Generates per-node summaries+keywords, then rolls them up into one
    document-level summary+keywords."""
    async def _run():
        await _summarize_tree_async(db, document_id)
        await _generate_document_rollup(db, document_id)

    asyncio.run(_run())