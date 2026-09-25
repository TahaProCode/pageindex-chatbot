"""
Phase 4: tree-search retrieval agent.

Given a question, walks a single document's PageIndex tree top-down:
  1. At the root level, select relevant sections (LLM + keyword boost).
  2. For each selected section, recurse into children in PARALLEL.
  3. Collect leaf (or fallback) raw text as context.
  4. Deduplicate by node id so the same section is never collected twice.

Improvements over sequential version:
- Sibling subtrees are traversed concurrently via asyncio.gather.
- visited_ids set prevents duplicate collection.
- Robust number parsing from LLM replies.
- Root-level selection blends LLM picks with simple keyword overlap
  so broad questions are less likely to drop a relevant branch.
"""
import asyncio
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
    """
    Robustly extract 1-based indices from an LLM reply.
    Accepts '1,3', '1. 3', '[1, 2]', 'sections 1 and 3', etc.
    Bounds-checks and preserves first-seen order.
    """
    numbers: list[int] = []
    seen: set[int] = set()
    for token in re.findall(r"\b\d+\b", response):
        idx = int(token)
        if 1 <= idx <= n_candidates and idx not in seen:
            seen.add(idx)
            numbers.append(idx)
    return numbers


def _keyword_overlap_score(question: str, node: TreeNode) -> float:
    """
    Cheap lexical overlap between question tokens and node title+summary+keywords.
    Used as a safety net at root level so the LLM cannot completely drop
    a section that clearly shares terms with the query.
    """
    q_tokens = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    if not q_tokens:
        return 0.0

    blob_parts = [node.title or ""]
    if node.summary:
        blob_parts.append(node.summary)
    if node.keywords:
        try:
            import json
            kws = json.loads(node.keywords)
            if isinstance(kws, list):
                blob_parts.extend(str(k) for k in kws)
        except Exception:
            blob_parts.append(node.keywords)

    blob = " ".join(blob_parts).lower()
    hits = sum(1 for t in q_tokens if t in blob)
    return hits / max(len(q_tokens), 1)


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

    # Root-level safety net: if LLM returned few/none, boost by keyword overlap
    if is_root and len(selected) < MAX_SELECTIONS_PER_LEVEL:
        scored = sorted(
            (( _keyword_overlap_score(question, n), n) for n in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        already = {n.id for n in selected}
        for score, node in scored:
            if score < 0.15:
                break
            if node.id not in already:
                selected.append(node)
                already.add(node.id)
                if len(selected) >= MAX_SELECTIONS_PER_LEVEL:
                    break
        if scored and scored[0][0] > 0:
            logger.info(
                "[%s] Keyword boost added sections: %s",
                level_label,
                [n.title for n in selected if n.id not in {candidates[i-1].id for i in numbers}],
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

    # Parallel traversal of sibling branches
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
    """
    Runs the tree-walk against ONE document and returns the collected
    TreeNode rows relevant to `question` - no LLM synthesis.
    """
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

    # Parallel walk from each selected root
    tasks = [
        _walk(question, root, children_map, depth=1, collected=collected,
              visited_ids=visited_ids, lock=lock)
        for root in selected_roots
    ]
    await asyncio.gather(*tasks)

    # Final dedup (defensive)
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
    """
    Single-document entry point: collects relevant nodes then synthesizes
    one answer from them.
    """
    logger.info("=== New question for document %s: %r ===", document_id, question)

    nodes = db.query(TreeNode).filter(TreeNode.document_id == document_id).all()

    if not nodes:
        logger.info("No tree nodes found for document %s - falling back to plain answer", document_id)
        answer = await chat_completion(
            prompt=question,
            system="No document content is available yet for this document.",
        )
        return {"answer": answer, "cited_node_ids": []}

    collected = await collect_relevant_nodes(db, document_id, question, nodes=nodes)

    logger.info("Final collected context nodes: %s", [n.title for n in collected])

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