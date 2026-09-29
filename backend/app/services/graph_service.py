"""
High-level Knowledge Graph operations.

Writes Document / Section / Entity nodes and RELATES_TO / MENTIONS edges
into Neo4j, scoped by document_id and batch_id.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.services.graph_db import is_graph_available, run_query

logger = logging.getLogger("pageindex.graph_service")


def upsert_document_node(
    document_id: str,
    batch_id: str,
    filename: str,
    summary: Optional[str] = None,
) -> None:
    if not is_graph_available():
        return
    run_query(
        """
        MERGE (d:Document {document_id: $document_id})
        SET d.batch_id = $batch_id,
            d.filename = $filename,
            d.summary = $summary
        """,
        {
            "document_id": document_id,
            "batch_id": batch_id,
            "filename": filename or "",
            "summary": (summary or "")[:2000],
        },
        write=True,
    )


def upsert_section_node(
    node_id: str,
    document_id: str,
    batch_id: str,
    title: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    summary: Optional[str] = None,
) -> None:
    if not is_graph_available():
        return
    run_query(
        """
        MERGE (s:Section {node_id: $node_id})
        SET s.document_id = $document_id,
            s.batch_id = $batch_id,
            s.title = $title,
            s.page_start = $page_start,
            s.page_end = $page_end,
            s.summary = $summary
        WITH s
        MATCH (d:Document {document_id: $document_id})
        MERGE (d)-[:HAS_SECTION]->(s)
        """,
        {
            "node_id": node_id,
            "document_id": document_id,
            "batch_id": batch_id,
            "title": (title or "Untitled")[:500],
            "page_start": page_start,
            "page_end": page_end,
            "summary": (summary or "")[:1500],
        },
        write=True,
    )


def write_kg_for_section(
    node_id: str,
    document_id: str,
    batch_id: str,
    extraction: dict[str, Any],
) -> None:
    """
    Persist entities + relations for one section.
    - Entity nodes are unique per (name, document_id)
    - Section -[:MENTIONS]-> Entity
    - Entity -[:RELATES_TO {type, confidence, section_id}]-> Entity
    """
    if not is_graph_available():
        return

    entities = extraction.get("entities") or []
    relations = extraction.get("relations") or []
    if not entities and not relations:
        return

    # Upsert entities + MENTIONS edges
    for ent in entities:
        run_query(
            """
            MERGE (e:Entity {name: $name, document_id: $document_id})
            SET e.batch_id = $batch_id,
                e.type = $etype
            WITH e
            MATCH (s:Section {node_id: $node_id})
            MERGE (s)-[:MENTIONS]->(e)
            """,
            {
                "name": ent["name"],
                "document_id": document_id,
                "batch_id": batch_id,
                "etype": ent.get("type", "OTHER"),
                "node_id": node_id,
            },
            write=True,
        )

    # Relations
    for rel in relations:
        run_query(
            """
            MATCH (a:Entity {name: $source, document_id: $document_id})
            MATCH (b:Entity {name: $target, document_id: $document_id})
            MERGE (a)-[r:RELATES_TO {relation: $relation, section_id: $node_id}]->(b)
            SET r.confidence = $confidence,
                r.batch_id = $batch_id
            """,
            {
                "source": rel["source"],
                "target": rel["target"],
                "relation": rel["relation"],
                "confidence": rel.get("confidence", 0.5),
                "node_id": node_id,
                "document_id": document_id,
                "batch_id": batch_id,
            },
            write=True,
        )

    logger.info(
        "Wrote KG for section %s: %d entities, %d relations",
        node_id,
        len(entities),
        len(relations),
    )


def delete_document_graph(document_id: str) -> None:
    """Remove all graph data for a document (sections, entities, edges)."""
    if not is_graph_available():
        return
    run_query(
        """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
        DETACH DELETE d, s, e
        """,
        {"document_id": document_id},
        write=True,
    )
    # Orphan entities that only belonged to this document
    run_query(
        """
        MATCH (e:Entity {document_id: $document_id})
        DETACH DELETE e
        """,
        {"document_id": document_id},
        write=True,
    )


def delete_batch_graph(batch_id: str) -> None:
    if not is_graph_available():
        return
    run_query(
        """
        MATCH (n)
        WHERE n.batch_id = $batch_id
        DETACH DELETE n
        """,
        {"batch_id": batch_id},
        write=True,
    )


def build_document_graph(
    document_id: str,
    batch_id: str,
    filename: str,
    summary: Optional[str],
    sections: list[dict],
) -> None:
    """
    Full graph write for one document.

    sections: list of {
        "node_id", "title", "page_start", "page_end",
        "summary", "extraction": {"entities", "relations"}
    }
    """
    if not is_graph_available():
        logger.info("Graph unavailable - skipping KG write for document %s", document_id)
        return

    # Clear previous graph for this document (re-index safe)
    delete_document_graph(document_id)
    upsert_document_node(document_id, batch_id, filename, summary)

    for sec in sections:
        upsert_section_node(
            node_id=sec["node_id"],
            document_id=document_id,
            batch_id=batch_id,
            title=sec.get("title") or "Untitled",
            page_start=sec.get("page_start"),
            page_end=sec.get("page_end"),
            summary=sec.get("summary"),
        )
        extraction = sec.get("extraction") or {}
        if extraction.get("entities") or extraction.get("relations"):
            write_kg_for_section(
                node_id=sec["node_id"],
                document_id=document_id,
                batch_id=batch_id,
                extraction=extraction,
            )

    logger.info(
        "Document graph built: document=%s sections=%d",
        document_id,
        len(sections),
    )