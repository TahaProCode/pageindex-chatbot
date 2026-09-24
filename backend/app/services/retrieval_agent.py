### retrieval_agent.py

"""
Phase 4: tree-search retrieval agent.

Given a question, this walks a single document's PageIndex tree top-down:
  1. At the root level, ask the LLM which top-level sections (by title +
     summary) might contain the answer.
  2. For each selected section, recurse into its children and ask again -
     narrowing down level by level instead of scanning the whole document.
  3. Once a selected node has no children (or none of its children seem
     relevant), its raw text is collected as context.
  4. Once enough context is collected, ask the LLM to answer using ONLY
     that collected text.

This is deliberately plain recursive Python (no LangGraph) per the
project plan.

Two entry points are exported:
  - collect_relevant_nodes(): runs steps 1-3 only and returns the raw
    TreeNode list, with NO LLM synthesis. This is the primitive the
    multi-document orchestrators (multi_doc_service.py, graph.py) call
    once per candidate document, so they can merge nodes across many
    documents before doing ONE final synthesis over all of them.
  - answer_question(): the original single-document entry point (steps
    1-4) - collects nodes via collect_relevant_nodes() and then also
    synthesizes a final answer, for callers that just want one document
    answered on its own.
"""
import logging
import re

from sqlalchemy.orm import Session

from app.models.models import TreeNode
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.retrieval")

MAX_SELECTIONS_PER_LEVEL = 3
MAX_LEAF_NODES = 5
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


async def _select_relevant(question: str, candidates: list[TreeNode], level_label: str) -> list[TreeNode]:
    if not candidates:
        return []

    listing = "\n".join(
        f"{i + 1}. {n.title} - {n.summary or '(no summary yet)'}"
        for i, n in enumerate(candidates)
    )
    prompt = f"Question: {question}\n\nSections:\n{listing}\n\nWhich section numbers might contain the answer?"

    logger.info(
        "[%s] Asking LLM to pick among %d candidates: %s",
        level_label, len(candidates), [n.title for n in candidates],
    )

    response = await chat_completion(prompt=prompt, system=SELECTION_SYSTEM_PROMPT)

    numbers = sorted(set(int(x) for x in re.findall(r"\d+", response)))
    selected = [candidates[n - 1] for n in numbers if 1 <= n <= len(candidates)]
    selected = selected[:MAX_SELECTIONS_PER_LEVEL]

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
) -> None:
    if len(collected) >= MAX_LEAF_NODES or depth > MAX_DEPTH:
        return

    children = children_map.get(node.id, [])
    logger.info("[depth=%d] Visiting node %r (%d children)", depth, node.title, len(children))

    if not children:
        logger.info("[depth=%d] Leaf node - collecting %r", depth, node.title)
        collected.append(node)
        return

    selected_children = await _select_relevant(question, children, level_label=f"depth={depth}")
    if not selected_children:
        if node.raw_text and node.raw_text.strip():
            logger.info("[depth=%d] No relevant children - falling back to own text of %r", depth, node.title)
            collected.append(node)
        else:
            logger.info("[depth=%d] No relevant children and no own text for %r - dropping branch", depth, node.title)
        return

    for child in selected_children:
        if len(collected) >= MAX_LEAF_NODES:
            break
        await _walk(question, child, children_map, depth + 1, collected)


async def collect_relevant_nodes(
    db: Session,
    document_id: str,
    question: str,
    nodes: list[TreeNode] | None = None,
) -> list[TreeNode]:
    """
    Runs the tree-walk (steps 1-3 above) against ONE document and returns
    the collected TreeNode rows relevant to `question` - no LLM synthesis,
    no DB writes. This is what multi_doc_service.py and graph.py call once
    per candidate document; they merge the results across every candidate
    before doing a single final synthesis over all of them.

    `nodes` lets a caller pass in an already-fetched list of this
    document's TreeNode rows (avoids a duplicate query when the caller
    already bulk-fetched them, e.g. multi_doc_service's
    _fetch_nodes_by_document). If omitted, they're queried here.
    """
    if nodes is None:
        nodes = db.query(TreeNode).filter(TreeNode.document_id == document_id).all()

    if not nodes:
        logger.info("Document %s has no tree nodes - nothing to collect", document_id)
        return []

    children_map = _build_children_map(nodes)
    roots = children_map.get(None, [])

    selected_roots = await _select_relevant(question, roots, level_label=f"doc={document_id} root")

    collected: list[TreeNode] = []
    for root in selected_roots:
        if len(collected) >= MAX_LEAF_NODES:
            break
        await _walk(question, root, children_map, depth=1, collected=collected)

    logger.info(
        "Document %s: collected %d node(s) for question %r: %s",
        document_id, len(collected), question, [n.title for n in collected],
    )
    return collected


async def answer_question(db: Session, document_id: str, question: str) -> dict:
    """
    Single-document entry point: collects relevant nodes for document_id
    via collect_relevant_nodes(), then synthesizes one answer from them.
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
        context_blocks.append(f"### {n.title} (pages {n.page_start}-{n.page_end})\n{text[:CONTEXT_CHARS_PER_NODE]}")
    context = "\n\n".join(context_blocks)

    final_prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
    answer = await chat_completion(prompt=final_prompt, system=ANSWER_SYSTEM_PROMPT)

    return {"answer": answer, "cited_node_ids": [n.id for n in collected]}