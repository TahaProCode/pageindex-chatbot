### backend/app/services/reranker_service.py
"""
Phase 6 (Chunk 4), Step 4a: lightweight embedding-based reranker.

The multi-document orchestrator (multi_doc_service.py) can end up with
more collected nodes than we want to stuff into the final answer prompt
(e.g. 18 nodes from 8 candidate documents). Previously these were simply
truncated in whatever order the candidate documents happened to be
processed in - not based on actual relevance.

This module replaces that naive cap with a real (but cheap) relevance
ranking: embed the question once, embed each candidate node's
title+summary (not the full raw_text - keeps embedding calls fast and
node summaries are already a concise stand-in for the node's content),
score by cosine similarity, and keep only the top N.

Deliberately has NO LLM /api/chat call in it - only /api/embed calls,
which are much cheaper and faster. This keeps reranking viable even when
there are dozens of candidate nodes.
"""
import asyncio
import logging
import math

from app.models.models import Document, TreeNode
from app.services.ollama_service import get_embedding

logger = logging.getLogger("pageindex.reranker")

DEFAULT_TOP_N = 8


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _node_embedding_text(node: TreeNode) -> str:
    """
    What we actually embed for each node - title + summary, not the full
    raw_text. Keeps embedding calls fast and consistent in size regardless
    of how long a section's raw text is; the summary (from summary_service.py)
    is already a compact stand-in for what the section covers.
    """
    summary = node.summary or ""
    return f"{node.title}. {summary}".strip()


async def rerank_nodes(
    question: str,
    nodes_with_documents: list[tuple[Document, TreeNode]],
    top_n: int = DEFAULT_TOP_N,
) -> list[tuple[Document, TreeNode]]:
    """
    Scores each (document, node) pair by cosine similarity between the
    question's embedding and the node's title+summary embedding, and
    returns the top_n highest-scoring pairs, sorted best-first.

    If there are already <= top_n pairs, returns them unchanged (no need
    to spend embedding calls ranking something that isn't being cut).
    """
    if len(nodes_with_documents) <= top_n:
        logger.info(
            "Only %d nodes collected (<= top_n=%d) - skipping reranking, keeping all",
            len(nodes_with_documents), top_n,
        )
        return nodes_with_documents

    logger.info("Reranking %d nodes down to top %d via embeddings", len(nodes_with_documents), top_n)

    question_embedding = await get_embedding(question)

    node_embeddings = await asyncio.gather(*[
        get_embedding(_node_embedding_text(node))
        for _, node in nodes_with_documents
    ])

    scored = [
        (_cosine_similarity(question_embedding, node_emb), document, node)
        for (document, node), node_emb in zip(nodes_with_documents, node_embeddings)
    ]

    scored.sort(key=lambda triple: triple[0], reverse=True)

    top = scored[:top_n]

    logger.info(
        "Reranked top %d: %s",
        len(top),
        [(round(score, 3), doc.filename, node.title) for score, doc, node in top],
    )

    return [(document, node) for _, document, node in top]