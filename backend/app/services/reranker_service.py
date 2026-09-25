"""
Lightweight embedding-based reranker.

Scores collected (document, node) pairs by cosine similarity between the
question embedding and each node's title+summary embedding, keeps top_n.

Uses batched /api/embed calls so ranking 30-40 nodes is one (or a few)
HTTP requests instead of dozens of sequential ones.
"""
import logging
import math
from typing import List, Tuple

from app.models.models import Document, TreeNode
from app.services.ollama_service import get_embedding, get_embeddings_batch

logger = logging.getLogger("pageindex.reranker")

DEFAULT_TOP_N = 10


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _node_embedding_text(node: TreeNode) -> str:
    """
    What we embed for each node - title + summary, not the full raw_text.
    Keeps embedding calls fast and consistent in size.
    """
    summary = node.summary or ""
    return f"{node.title}. {summary}".strip()


async def rerank_nodes(
    question: str,
    nodes_with_documents: List[Tuple[Document, TreeNode]],
    top_n: int = DEFAULT_TOP_N,
) -> List[Tuple[Document, TreeNode]]:
    """
    Scores each (document, node) pair by cosine similarity and returns
    the top_n highest-scoring pairs, sorted best-first.

    If there are already <= top_n pairs, returns them unchanged.
    """
    if len(nodes_with_documents) <= top_n:
        logger.info(
            "Only %d nodes collected (<= top_n=%d) - skipping reranking, keeping all",
            len(nodes_with_documents), top_n,
        )
        return nodes_with_documents

    logger.info("Reranking %d nodes down to top %d via batch embeddings", len(nodes_with_documents), top_n)

    question_embedding = await get_embedding(question)

    # Single batch call for all node texts
    node_texts = [_node_embedding_text(node) for _, node in nodes_with_documents]
    node_embeddings = await get_embeddings_batch(node_texts)

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