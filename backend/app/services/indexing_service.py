### indexing_service.py
"""
Runs as a FastAPI background task right after upload. Creates its own DB
session since the request-scoped session from the upload endpoint is
already closed by the time this runs
"""
import traceback
from app.db.database import SessionLocal
from app.models.models import Document, DocumentStatus
from app.services.pdf_service import parse_pdf_to_tree
from app.services.tree_service import persist_tree
from app.services.summary_service import generate_summaries_for_document

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

        # Phase 3: generate a short LLM summary for every node, bottom-up
        # This is the slow step (one Ollama call per node) - expect
        # indexing to take a while on larger documents with a 4B model
        generate_summaries_for_document(db, document_id)

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