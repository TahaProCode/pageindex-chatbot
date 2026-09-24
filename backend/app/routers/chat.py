# chat.py
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import ChatSession, Message, MessageRole, Batch, Document, DocumentStatus, TreeNode
from app.schemas.schemas import ChatSessionOut, MessageIn, MessageOut, CitedSection

# Try importing LangGraph agent runner; fallback to the plain async
# multi-doc orchestrator if LangGraph isn't installed. Both are now
# batch-scoped - there is no single-document chat path anymore.
try:
    from app.graph import run_langgraph_chat
    HAS_LANGGRAPH = True
except ImportError:
    from app.services.multi_doc_service import answer_question_multi_doc
    HAS_LANGGRAPH = False

logger = logging.getLogger("pageindex.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _build_message_out(db: Session, message: Message) -> MessageOut:
    """Enriches a stored Message with the actual filenames/titles/pages of
    any cited TreeNodes, since cited_node_ids in the DB is just a JSON
    list of ids. Filename is included because a batch answer can cite
    sections from several different documents at once."""
    cited_sections: list[CitedSection] = []
    if message.cited_node_ids:
        try:
            ids = json.loads(message.cited_node_ids)
        except (TypeError, ValueError):
            ids = []
        if ids:
            nodes = db.query(TreeNode).filter(TreeNode.id.in_(ids)).all()
            by_id = {n.id: n for n in nodes}

            doc_ids = {n.document_id for n in nodes}
            docs_by_id = {
                d.id: d for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()
            } if doc_ids else {}

            for node_id in ids:
                n = by_id.get(node_id)
                if n:
                    doc = docs_by_id.get(n.document_id)
                    cited_sections.append(
                        CitedSection(
                            id=n.id,
                            filename=doc.filename if doc else "unknown",
                            title=n.title,
                            page_start=n.page_start,
                            page_end=n.page_end,
                        )
                    )

    return MessageOut(
        id=message.id,
        role=message.role.value if hasattr(message.role, "value") else message.role,
        content=message.content,
        created_at=message.created_at,
        cited_sections=cited_sections,
    )


@router.post("/sessions/{batch_id}", response_model=ChatSessionOut)
def create_session(batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    session = ChatSession(batch_id=batch_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.post("/sessions/{session_id}/messages", response_model=MessageOut)
async def send_message(session_id: str, body: MessageIn, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    user_msg = Message(session_id=session_id, role=MessageRole.user, content=body.content)
    db.add(user_msg)
    db.commit()

    documents = db.query(Document).filter(Document.batch_id == session.batch_id).all()
    not_ready = [d for d in documents if d.status != DocumentStatus.ready]

    if not documents or not_ready:
        if not documents:
            wait_message = "This batch has no documents."
        else:
            statuses = ", ".join(sorted({d.status.value for d in not_ready}))
            wait_message = (
                f"This batch is still being processed ({len(not_ready)} of {len(documents)} "
                f"document(s) still in progress: {statuses}). Please wait for indexing to "
                "finish on all documents before asking questions - answers could otherwise "
                "miss information from the documents that aren't ready yet."
            )
        assistant_msg = Message(session_id=session_id, role=MessageRole.assistant, content=wait_message)
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)
        return _build_message_out(db, assistant_msg)

    # Invoke LangGraph engine (batch-scoped, cross-document retrieval) or
    # fall back seamlessly to the plain async multi-doc orchestrator.
    if HAS_LANGGRAPH:
        logger.info("Processing message via LangGraph state machine agent (batch=%s)...", session.batch_id)
        graph_result = await run_langgraph_chat(
            db=db,
            query=body.content,
            batch_id=session.batch_id,
        )
        answer_text = graph_result.get("answer", "")
        cited_ids = [s["node_id"] for s in graph_result.get("sources", []) if "node_id" in s]
    else:
        logger.info("Processing message via multi-doc orchestrator (batch=%s)...", session.batch_id)
        result = await answer_question_multi_doc(db, body.content, batch_id=session.batch_id)
        answer_text = result["answer"]
        cited_ids = [s["node_id"] for s in result.get("sources", []) if "node_id" in s]

    assistant_msg = Message(
        session_id=session_id,
        role=MessageRole.assistant,
        content=answer_text,
        cited_node_ids=json.dumps(cited_ids),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return _build_message_out(db, assistant_msg)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def list_messages(session_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [_build_message_out(db, m) for m in messages]