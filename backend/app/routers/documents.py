# documents.py
"""
Uploading now happens via POST /api/batches/upload (see routers/batches.py),
which accepts multiple files at once and groups them into one Batch. This
router is read-only: listing documents and inspecting a single document's
tree (used by the tree sidebar in the chat UI).
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import Document, TreeNode
from app.schemas.schemas import DocumentOut, TreeNodeOut

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger("pageindex.documents")


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _build_document_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        batch_id=document.batch_id,
        filename=document.filename,
        status=document.status.value if hasattr(document.status, "value") else document.status,
        page_count=document.page_count,
        created_at=document.created_at,
        summary=document.summary,
        keywords=_parse_keywords(document.keywords),
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [_build_document_out(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _build_document_out(doc)


@router.get("/{document_id}/tree", response_model=list[TreeNodeOut])
def get_document_tree(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = (
        db.query(TreeNode)
        .filter(TreeNode.document_id == document_id)
        .order_by(TreeNode.order_index)
        .all()
    )

    by_id: dict[str, TreeNodeOut] = {
        row.id: TreeNodeOut(
            id=row.id,
            parent_id=row.parent_id,
            title=row.title,
            summary=row.summary,
            keywords=_parse_keywords(row.keywords),
            page_start=row.page_start,
            page_end=row.page_end,
            children=[],
        )
        for row in rows
    }

    roots: list[TreeNodeOut] = []
    for row in rows:
        node_out = by_id[row.id]
        if row.parent_id and row.parent_id in by_id:
            by_id[row.parent_id].children.append(node_out)
        else:
            roots.append(node_out)

    return roots