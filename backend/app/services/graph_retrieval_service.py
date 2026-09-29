"""
Graph-based context expansion.

Given TreeNode ids already collected by the tree-walk agent, pull related
sections via shared entities / multi-hop relations inside the same batch.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Document, TreeNode
from app.services.graph_db import is_graph_available, run_query

logger = logging.getLogger("pageindex.graph_retrieval")


def find_related_section_ids(
    seed_node_ids: list[str],
    batch_id: str,
    limit: Optional[int] = None,
) -> list[str]:
    """
    From seed sections, find other sections in the same batch that:
      1. Share an entity (MENTIONS), or
      2. Are linked via Entity -RELATES_TO- Entity paths (1 hop).

    Returns node_ids that are NOT in seed_node_ids.
    """
    if not is_graph_available() or not seed_node_ids:
        return []

    limit = limit or settings.graph_expand_limit

    rows = run_query(
        """
        // Sections that share entities with seed sections
        MATCH (seed:Section)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Section)
        WHERE seed.node_id IN $seed_ids
          AND related.batch_id = $batch_id
          AND NOT related.node_id IN $seed_ids
        WITH related, count(DISTINCT e) AS shared
        ORDER BY shared DESC
        LIMIT $limit

        RETURN related.node_id AS node_id, shared AS score

        UNION

        // Sections reachable via a relation between entities mentioned in seed
        MATCH (seed:Section)-[:MENTIONS]->(e1:Entity)-[r:RELATES_TO]-(e2:Entity)
              <-[:MENTIONS]-(related:Section)
        WHERE seed.node_id IN $seed_ids
          AND related.batch_id = $batch_id
          AND NOT related.node_id IN $seed_ids
        WITH related, count(DISTINCT r) AS rel_count
        ORDER BY rel_count DESC
        LIMIT $limit

        RETURN related.node_id AS node_id, rel_count AS score
        """,
        {
            "seed_ids": seed_node_ids,
            "batch_id": batch_id,
            "limit": limit,
        },
    )

    # Deduplicate preserving order (higher scores first from each branch)
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        nid = row.get("node_id")
        if nid and nid not in seen:
            seen.add(nid)
            ordered.append(nid)
        if len(ordered) >= limit:
            break

    logger.info(
        "Graph expansion from %d seeds (batch=%s) -> %d related sections",
        len(seed_node_ids),
        batch_id,
        len(ordered),
    )
    return ordered


def expand_context_via_graph(
    db: Session,
    seed_items: list[dict],
    batch_id: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    seed_items: list of {"document": Document, "node": TreeNode}
    Returns additional items in the same format (not already in seed).
    """
    if not is_graph_available() or not seed_items:
        return []

    seed_ids = [item["node"].id for item in seed_items]
    related_ids = find_related_section_ids(seed_ids, batch_id, limit=limit)
    if not related_ids:
        return []

    nodes = (
        db.query(TreeNode)
        .filter(TreeNode.id.in_(related_ids))
        .all()
    )
    if not nodes:
        return []

    by_id = {n.id: n for n in nodes}
    doc_ids = {n.document_id for n in nodes}
    docs = {
        d.id: d
        for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()
    }

    extra: list[dict] = []
    # Preserve graph ranking order
    for nid in related_ids:
        node = by_id.get(nid)
        if not node:
            continue
        doc = docs.get(node.document_id)
        if not doc:
            continue
        extra.append({"document": doc, "node": node})

    logger.info("Graph expansion added %d context nodes", len(extra))
    return extra


def search_entities_in_batch(
    query: str,
    batch_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    Simple entity name search within a batch (case-insensitive contains).
    Useful for debugging / future entity-centric UI.
    """
    if not is_graph_available() or not query.strip():
        return []

    rows = run_query(
        """
        MATCH (e:Entity {batch_id: $batch_id})
        WHERE toLower(e.name) CONTAINS toLower($q)
        OPTIONAL MATCH (s:Section)-[:MENTIONS]->(e)
        RETURN e.name AS name,
               e.type AS type,
               e.document_id AS document_id,
               collect(DISTINCT s.node_id) AS section_ids
        LIMIT $limit
        """,
        {"batch_id": batch_id, "q": query.strip(), "limit": limit},
    )
    return rows