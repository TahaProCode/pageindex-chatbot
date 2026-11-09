"""
Runs as a background worker task after upload. Creates its own DB
session since the request-scoped session is already closed.

Pipeline:
  1. Parse PDF → tree
  2. Persist tree nodes
  3. Generate summaries + keywords (per-node + document rollup)
  4. Extract Knowledge Graph entities/relations and write to Neo4j
"""
from __future__ import annotations

import asyncio
import logging
import traceback

from app.db.database import SessionLocal
from app.models.models import Document, DocumentStatus, TreeNode
from app.services.pdf_service import parse_pdf_to_tree
from app.services.tree_service import persist_tree
from app.services.summary_service import generate_summaries_for_document
from app.services.kg_extraction_service import extract_kg_from_text
from app.services.graph_service import build_document_graph
from app.services.graph_db import is_graph_available
from app.core.config import settings

logger = logging.getLogger("pageindex.indexing")


async def _build_kg_for_document(db, document_id: str) -> None:
    """
    For every TreeNode that has usable text/summary, extract entities
    and relations, then write the full document graph to Neo4j.
    """
    if not is_graph_available():
        logger.info("Graph disabled/unavailable - skipping KG for %s", document_id)
        return

    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return

    nodes = (
        db.query(TreeNode)
        .filter(TreeNode.document_id == document_id)
        .order_by(TreeNode.order_index)
        .all()
    )
    if not nodes:
        return

    # Limit concurrent LLM extraction calls (same semaphore style as summaries)
    sem = asyncio.Semaphore(getattr(settings, "max_concurrent_summaries", 4))

    async def _extract_one(node: TreeNode) -> dict:
        text = (node.raw_text or "").strip()
        summary = (node.summary or "").strip()
        if len(text) < 40 and len(summary) < 20:
            return {
                "node_id": node.id,
                "title": node.title,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "summary": node.summary,
                "extraction": {"entities": [], "relations": []},
            }

        async with sem:
            extraction = await extract_kg_from_text(
                title=node.title or "Untitled",
                text=text,
                summary=summary or None,
            )

        return {
            "node_id": node.id,
            "title": node.title,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "summary": node.summary,
            "extraction": extraction,
        }

    sections = await asyncio.gather(*[_extract_one(n) for n in nodes])

    build_document_graph(
        document_id=doc.id,
        batch_id=doc.batch_id,
        filename=doc.filename,
        summary=doc.summary,
        sections=list(sections),
    )


def process_document(document_id: str) -> None:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return

        doc.status = DocumentStatus.parsing
        db.commit()

        result = parse_pdf_to_tree(doc.file_path)

        doc.page_count = result["page_count"]
        db.commit()

        persist_tree(db, document_id, result["tree"])
        db.commit()

        doc.status = DocumentStatus.indexing
        db.commit()

        # Phase 3: summaries + keywords
        generate_summaries_for_document(db, document_id)

        # Phase 4: Knowledge Graph extraction + Neo4j write
        try:
            asyncio.run(_build_kg_for_document(db, document_id))
        except Exception:
            # KG failure must not mark the whole document as failed
            logger.exception(
                "KG build failed for document %s (document still marked ready)",
                document_id,
            )

        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = DocumentStatus.ready
            db.commit()

    except Exception as e:
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = DocumentStatus.failed
            doc.error_message = f"{e}\n{traceback.format_exc()}"[:2000]
            db.commit()
    finally:
        db.close()