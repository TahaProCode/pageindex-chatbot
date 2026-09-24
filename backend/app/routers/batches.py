# batches.py
"""
A Batch is a group of documents uploaded together (e.g. 10 or 200 PDFs
selected in one go). Uploading now happens here, not in documents.py -
every document created by /upload belongs to exactly one Batch, and every
chat session is scoped to a Batch (see routers/chat.py), so a query made
against a 10-document batch only ever searches those 10 documents.
"""
import logging
import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.models.models import Batch, Document, DocumentStatus
from app.routers.documents import _build_document_out
from app.schemas.schemas import BatchOut
from app.services.indexing_queue import enqueue_document

router = APIRouter(prefix="/api/batches", tags=["batches"])
logger = logging.getLogger("pageindex.batches")


def _build_batch_out(batch: Batch) -> BatchOut:
    return BatchOut(
        id=batch.id,
        name=batch.name,
        created_at=batch.created_at,
        documents=[_build_document_out(d) for d in batch.documents],
    )


@router.post("/upload", response_model=BatchOut)
async def upload_batch(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    Accepts one or more PDFs in a single request, creates one Batch row,
    and creates a Document row per file under that batch. Every document
    is then enqueued for indexing - the indexing queue already processes
    strictly one document at a time (see indexing_queue.py), so a 200-file
    batch is safe to submit in one call; it just takes a while to finish.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported ({f.filename}).")

    os.makedirs(settings.upload_dir, exist_ok=True)

    batch_name = files[0].filename if len(files) == 1 else f"Batch of {len(files)} documents"
    batch = Batch(name=batch_name)
    db.add(batch)
    db.commit()
    db.refresh(batch)

    created_docs = []
    for f in files:
        stored_name = f"{uuid.uuid4()}_{f.filename}"
        dest_path = os.path.join(settings.upload_dir, stored_name)

        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(f.file, out_file)

        doc = Document(
            batch_id=batch.id,
            filename=f.filename,
            file_path=dest_path,
            status=DocumentStatus.uploaded,
        )
        db.add(doc)
        created_docs.append(doc)

    db.commit()

    for doc in created_docs:
        db.refresh(doc)
        queue_position = enqueue_document(doc.id)
        logger.info(
            "Document %s (batch %s) enqueued for indexing (queue size: %d)",
            doc.id, batch.id, queue_position,
        )

    db.refresh(batch)
    return _build_batch_out(batch)


@router.get("", response_model=List[BatchOut])
def list_batches(db: Session = Depends(get_db)):
    batches = db.query(Batch).order_by(Batch.created_at.desc()).all()
    return [_build_batch_out(b) for b in batches]


@router.get("/{batch_id}", response_model=BatchOut)
def get_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return _build_batch_out(batch)