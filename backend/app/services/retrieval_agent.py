"""
Phase 4: tree-search retrieval agent (+ Phase 2 robustness).

Improvements:
- Sibling subtrees concurrent via asyncio.gather.
- visited_ids prevent duplicate collection.
- Root-level keyword boost includes 2-letter ALL-CAPS tokens (SL, CL, EL).
- Root NONE / weak selection → force keyword-matched roots (title/summary).
- Optional cheap leaf-title scan when root still empty.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from app.models.models import TreeNode
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.retrieval")

MAX_SELECTIONS_PER_LEVEL = 3
MAX_LEAF_NODES = 6
MAX_DEPTH = 6
CONTEXT_CHARS_PER_NODE = 3000

SELECTION_SYSTEM_PROMPT = (
    "You are navigating a document's outline to find sections relevant to a "
    "question. You are shown a numbered list of section titles and short "
    "summaries. Reply with ONLY the numbers of sections that might help answer "
    "the question, separated by commas (example: 1,3). If none of them seem "
    "relevant, reply with exactly: NONE. Do not add any explanation."
)

ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the document excerpts provided "
    "below. If the excerpts don't fully answer the question, say what you "
    "found and note what's missing. Do not use outside knowledge. Keep the "
    "answer concise and clear."
)

NO_CONTEXT_SYSTEM_PROMPT = (
    "Answer the question honestly. If you don't have enough information to "
    "answer it, say so plainly rather than guessing."
)


def _build_children_map(nodes: list[TreeNode]) -> dict[str | None, list[TreeNode]]:
    children_map: dict[str | None, list[TreeNode]] = {}
    for n in nodes:
        children_map.setdefault(n.parent_id, []).append(n)
    for key in children_map:
        children_map[key].sort(key=lambda n: n.order_index)
    return children_map


def _parse_selection_numbers(response: str, n_candidates: int) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for token in re.findall(r"\b\d+\b", response):
        idx = int(token)
        if 1 <= idx <= n_candidates and idx not in seen:
            seen.add(idx)
            numbers.append(idx)
    return numbers


def _query_tokens(question: str) -> set[str]:
    """
    Phase 2: include normal tokens len>=3 and ALL-CAPS abbreviations len>=2.
    """
    tokens: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9]+", question or ""):
        t = m.group(0)
        if len(t) >= 3:
            tokens.add(t.lower())
        elif len(t) >= 2 and t.isupper():
            tokens.add(t.lower())
    return tokens


def _keyword_overlap_score(question: str, node: TreeNode) -> float:
    q_tokens = _query_tokens(question)
    if not q_tokens:
        return 0.0

    blob_parts = [node.title or ""]
    if node.summary:
        blob_parts.append(node.summary)
    if node.keywords:
        try:
            kws = json.loads(node.keywords)
            if isinstance(kws, list):
                blob_parts.extend(str(k) for k in kws)
        except Exception:
            blob_parts.append(node.keywords)

    blob = " ".join(blob_parts).lower()
    # Also index individual tokens of title for partial matches (sick, leave)
    hits = sum(1 for t in q_tokens if t in blob)
    return hits / max(len(q_tokens), 1)


def _force_keyword_roots(
    question: str,
    roots: list[TreeNode],
    already: list[TreeNode],
    max_total: int = MAX_SELECTIONS_PER_LEVEL,
) -> list[TreeNode]:
    """
    Phase 2: if LLM picked nothing / few, force roots whose title/summary
    overlap the query (e.g. Leave Policy when query has leave/sick/SL).
    """
    selected = list(already)
    have = {n.id for n in selected}
    scored = sorted(
        ((_keyword_overlap_score(question, n), n) for n in roots),
        key=lambda x: x[0],
        reverse=True,
    )
    for score, node in scored:
        if score < 0.12:
            break
        if node.id not in have:
            selected.append(node)
            have.add(node.id)
            logger.info(
                "Root keyword-force: %r score=%.2f",
                node.title,
                score,
            )
        if len(selected) >= max_total:
            break
    return selected


def _leaf_title_scan(
    question: str,
    nodes: list[TreeNode],
    limit: int = MAX_SELECTIONS_PER_LEVEL,
) -> list[TreeNode]:
    """
    Phase 2 optional: cheap scan of all node titles when root path fails.
    Prefer leaves / high title overlap (e.g. '5. Sick Leave (SL)').
    """
    q_tokens = _query_tokens(question)
    if not q_tokens:
        return []

    scored: list[tuple[float, TreeNode]] = []
    for n in nodes:
        title = (n.title or "").lower()
        if not title:
            continue
        hits = sum(1 for t in q_tokens if t in title)
        if hits <= 0:
            continue
        # Prefer denser match on short titles
        score = hits / max(len(q_tokens), 1)
        if re.search(r"\bsl\b", title) and "sl" in q_tokens:
            score += 0.35
        scored.append((score, n))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [n for sc, n in scored[:limit] if sc >= 0.15]
    if picked:
        logger.info(
            "Leaf-title scan picked: %s",
            [n.title for n in picked],
        )
    return picked


async def _select_relevant(
    question: str,
    candidates: list[TreeNode],
    level_label: str,
    *,
    is_root: bool = False,
) -> list[TreeNode]:
    if not candidates:
        return []

    listing = "\n".join(
        f"{i + 1}. {n.title} - {n.summary or '(no summary yet)'}"
        for i, n in enumerate(candidates)
    )
    prompt = (
        f"Question: {question}\n\nSections:\n{listing}\n\n"
        "Which section numbers might contain the answer?"
    )

    logger.info(
        "[%s] Asking LLM to pick among %d candidates: %s",
        level_label, len(candidates), [n.title for n in candidates],
    )

    response = await chat_completion(prompt=prompt, system=SELECTION_SYSTEM_PROMPT)
    numbers = _parse_selection_numbers(response, len(candidates))
    selected = [candidates[n - 1] for n in numbers]
    selected = selected[:MAX_SELECTIONS_PER_LEVEL]

    if is_root:
        if len(selected) < MAX_SELECTIONS_PER_LEVEL:
            selected = _force_keyword_roots(question, candidates, selected)
        # Still empty → force any weak overlap rather than die
        if not selected:
            selected = _force_keyword_roots(
                question, candidates, [], max_total=MAX_SELECTIONS_PER_LEVEL
            )

    logger.info(
        "[%s] LLM raw reply=%r -> parsed_numbers=%s -> selected=%s",
        level_label, response, numbers, [n.title for n in selected],
    )
    return selected


async def _walk(
    question: str,
    node: TreeNode,
    children_map: dict[str | None, list[TreeNode]],
    depth: int,
    collected: list[TreeNode],
    visited_ids: set[str],
    lock: asyncio.Lock,
) -> None:
    async with lock:
        if len(collected) >= MAX_LEAF_NODES or depth > MAX_DEPTH or node.id in visited_ids:
            return
        visited_ids.add(node.id)

    children = children_map.get(node.id, [])
    logger.info("[depth=%d] Visiting node %r (%d children)", depth, node.title, len(children))

    if not children:
        async with lock:
            if len(collected) < MAX_LEAF_NODES and node.id not in {n.id for n in collected}:
                logger.info("[depth=%d] Leaf node - collecting %r", depth, node.title)
                collected.append(node)
        return

    selected_children = await _select_relevant(
        question, children, level_label=f"depth={depth}"
    )

    if not selected_children:
        # Keyword force on children before falling back to parent text only
        selected_children = _force_keyword_roots(
            question, children, [], max_total=MAX_SELECTIONS_PER_LEVEL
        )

    if not selected_children:
        async with lock:
            if (
                node.raw_text
                and node.raw_text.strip()
                and len(collected) < MAX_LEAF_NODES
                and node.id not in {n.id for n in collected}
            ):
                logger.info(
                    "[depth=%d] No relevant children - falling back to own text of %r",
                    depth, node.title,
                )
                collected.append(node)
            else:
                logger.info(
                    "[depth=%d] No relevant children and no own text for %r - dropping branch",
                    depth, node.title,
                )
        return

    tasks = [
        _walk(question, child, children_map, depth + 1, collected, visited_ids, lock)
        for child in selected_children
    ]
    await asyncio.gather(*tasks)


async def collect_relevant_nodes(
    db: Session,
    document_id: str,
    question: str,
    nodes: list[TreeNode] | None = None,
) -> list[TreeNode]:
    if nodes is None:
        nodes = db.query(TreeNode).filter(TreeNode.document_id == document_id).all()

    if not nodes:
        logger.info("Document %s has no tree nodes - nothing to collect", document_id)
        return []

    children_map = _build_children_map(nodes)
    roots = children_map.get(None, [])

    selected_roots = await _select_relevant(
        question, roots, level_label=f"doc={document_id} root", is_root=True
    )

    collected: list[TreeNode] = []
    visited_ids: set[str] = set()
    lock = asyncio.Lock()

    if selected_roots:
        tasks = [
            _walk(question, root, children_map, depth=1, collected=collected,
                  visited_ids=visited_ids, lock=lock)
            for root in selected_roots
        ]
        await asyncio.gather(*tasks)

    # Phase 2: still empty → leaf title scan across whole doc
    if not collected:
        title_hits = _leaf_title_scan(question, nodes)
        for n in title_hits:
            if n.id not in {x.id for x in collected} and len(collected) < MAX_LEAF_NODES:
                collected.append(n)
                logger.info("Collected via leaf-title scan: %r", n.title)

    seen = set()
    unique: list[TreeNode] = []
    for n in collected:
        if n.id not in seen:
            seen.add(n.id)
            unique.append(n)

    logger.info(
        "Document %s: collected %d node(s) for question %r: %s",
        document_id, len(unique), question, [n.title for n in unique],
    )
    return unique


async def answer_question(db: Session, document_id: str, question: str) -> dict:
    logger.info("=== New question for document %s: %r ===", document_id, question)

    nodes = db.query(TreeNode).filter(TreeNode.document_id == document_id).all()

    if not nodes:
        answer = await chat_completion(
            prompt=question,
            system="No document content is available yet for this document.",
        )
        return {"answer": answer, "cited_node_ids": []}

    collected = await collect_relevant_nodes(db, document_id, question, nodes=nodes)

    if not collected:
        answer = await chat_completion(
            prompt=f"Question: {question}\n\n(No relevant section of the document was found.)",
            system=NO_CONTEXT_SYSTEM_PROMPT,
        )
        return {"answer": answer, "cited_node_ids": []}

    context_blocks = []
    for n in collected:
        text = (n.raw_text or "").strip() or "(no direct text - see subsections)"
        context_blocks.append(
            f"### {n.title} (pages {n.page_start}-{n.page_end})\n"
            f"{text[:CONTEXT_CHARS_PER_NODE]}"
        )
    context = "\n\n".join(context_blocks)

    final_prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
    answer = await chat_completion(prompt=final_prompt, system=ANSWER_SYSTEM_PROMPT)

    return {"answer": answer, "cited_node_ids": [n.id for n in collected]}