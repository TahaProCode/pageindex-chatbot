"""
Graph-based retrieval:

1) Seed expansion — related sections via shared entities (existing).
2) Query → Entity → Section — primary graph search from the user query
   (Phase 1: works even when tree-walk returns zero seeds).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Document, TreeNode
from app.services.graph_db import is_graph_available, run_query

logger = logging.getLogger("pageindex.graph_retrieval")


def _query_tokens(query: str) -> list[str]:
    """
    Tokens for entity matching.
    - Normal words length >= 3
    - ALL-CAPS abbreviations length >= 2 (SL, CL, EL, PR, ...)
    """
    if not query:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9]+", query):
        t = m.group(0)
        if len(t) >= 3 or (len(t) >= 2 and t.isupper()):
            key = t.lower()
            if key not in seen:
                seen.add(key)
                tokens.append(t)
    return tokens


def find_related_section_ids(
    seed_node_ids: list[str],
    batch_id: str,
    limit: Optional[int] = None,
) -> list[str]:
    if not is_graph_available() or not seed_node_ids:
        return []

    limit = limit or settings.graph_expand_limit

    rows = run_query(
        """
        MATCH (seed:Section)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Section)
        WHERE seed.node_id IN $seed_ids
          AND related.batch_id = $batch_id
          AND NOT related.node_id IN $seed_ids
        WITH related, count(DISTINCT e) AS shared
        ORDER BY shared DESC
        LIMIT $limit
        RETURN related.node_id AS node_id, shared AS score

        UNION

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
    if not is_graph_available() or not seed_items:
        logger.info(
            "Graph expand skipped (available=%s seeds=%d)",
            is_graph_available(),
            len(seed_items or []),
        )
        return []

    seed_ids = [item["node"].id for item in seed_items]
    related_ids = find_related_section_ids(seed_ids, batch_id, limit=limit)
    if not related_ids:
        logger.info("Graph expand: graph_added=0 (no related sections)")
        return []

    extra = _nodes_to_items(db, related_ids)
    logger.info("Graph expand: graph_added=%d", len(extra))
    return extra


def sections_from_query_entities(
    db: Session,
    query: str,
    batch_id: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Phase 1: Query → Entity name match → MENTIONS → Section → Postgres nodes.
    Works with zero tree-walk seeds.
    """
    if not is_graph_available():
        logger.info("Query-entity search skipped (graph unavailable)")
        return []

    tokens = _query_tokens(query)
    if not tokens:
        return []

    limit = limit or max(settings.graph_expand_limit, 8)

    rows = run_query(
        """
        UNWIND $tokens AS tok
        MATCH (e:Entity {batch_id: $batch_id})
        WHERE toLower(e.name) CONTAINS toLower(tok)
        WITH e, count(*) AS token_hits
        MATCH (s:Section)-[:MENTIONS]->(e)
        WHERE s.batch_id = $batch_id
        WITH s.node_id AS node_id, sum(token_hits) AS score, collect(DISTINCT e.name) AS entities
        ORDER BY score DESC
        LIMIT $limit
        RETURN node_id, score, entities
        """,
        {"tokens": tokens, "batch_id": batch_id, "limit": limit},
    )

    if not rows:
        q = query.strip()[:80]
        if len(q) >= 3:
            rows = run_query(
                """
                MATCH (e:Entity {batch_id: $batch_id})
                WHERE toLower(e.name) CONTAINS toLower($q)
                MATCH (s:Section)-[:MENTIONS]->(e)
                WHERE s.batch_id = $batch_id
                RETURN DISTINCT s.node_id AS node_id, 1 AS score, collect(DISTINCT e.name) AS entities
                LIMIT $limit
                """,
                {"batch_id": batch_id, "q": q, "limit": limit},
            )

    node_ids = [r["node_id"] for r in rows if r.get("node_id")]
    items = _nodes_to_items(db, node_ids)

    logger.info(
        "Query-entity search (batch=%s tokens=%s): graph_query_hits=%d entities_sample=%s",
        batch_id,
        tokens,
        len(items),
        [r.get("entities") for r in rows[:3]],
    )
    return items


def search_entities_in_batch(
    query: str,
    batch_id: str,
    limit: int = 10,
) -> list[dict]:
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


def _nodes_to_items(db: Session, node_ids: list[str]) -> list[dict]:
    if not node_ids:
        return []

    nodes = db.query(TreeNode).filter(TreeNode.id.in_(node_ids)).all()
    by_id = {n.id: n for n in nodes}
    doc_ids = {n.document_id for n in nodes}
    docs = {
        d.id: d
        for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()
    } if doc_ids else {}

    items: list[dict] = []
    for nid in node_ids:
        node = by_id.get(nid)
        if not node:
            continue
        doc = docs.get(node.document_id)
        if not doc:
            continue
        items.append({"document": doc, "node": node})
    return items