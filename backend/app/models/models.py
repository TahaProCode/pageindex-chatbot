"""
ORM models for the PageIndex chatbot.

Batch       -> one upload-batch (a group of documents uploaded together).
               A ChatSession is scoped to a Batch, so a single query can
               retrieve context across every document in that batch.
Document    -> one uploaded PDF, belongs to exactly one Batch
TreeNode    -> one node of the hierarchical PageIndex tree for a document
               (self-referencing parent_id builds the tree)
ChatSession -> one chat thread tied to a Batch
Message     -> one message (user or assistant) within a chat session
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, Enum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class DocumentStatus(str, enum.Enum):
    uploaded = "uploaded"
    parsing = "parsing"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"


class Batch(Base):
    __tablename__ = "batches"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="batch", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="batch", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    batch_id = Column(UUID(as_uuid=False), ForeignKey("batches.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.uploaded, nullable=False)
    page_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)     # document-level rollup summary (Phase 6)
    keywords = Column(Text, nullable=True)    # JSON-encoded list (Phase 6)
    created_at = Column(DateTime, default=datetime.utcnow)

    batch = relationship("Batch", back_populates="documents")
    nodes = relationship("TreeNode", back_populates="document", cascade="all, delete-orphan")


class TreeNode(Base):
    __tablename__ = "tree_nodes"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    document_id = Column(UUID(as_uuid=False), ForeignKey("documents.id"), nullable=False)
    parent_id = Column(UUID(as_uuid=False), ForeignKey("tree_nodes.id"), nullable=True)

    title = Column(String, nullable=False)
    summary = Column(Text, nullable=True)       # LLM-generated short summary
    keywords = Column(Text, nullable=True)      # JSON-encoded list (Phase 6)
    raw_text = Column(Text, nullable=True)       # full text belonging to this node
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    order_index = Column(Integer, default=0)     # sibling ordering

    document = relationship("Document", back_populates="nodes")
    children = relationship(
        "TreeNode",
        backref="parent",
        remote_side=[id],
        cascade="all, delete-orphan",
        single_parent=True,
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    batch_id = Column(UUID(as_uuid=False), ForeignKey("batches.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    batch = relationship("Batch", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    session_id = Column(UUID(as_uuid=False), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    cited_node_ids = Column(Text, nullable=True)  # JSON-encoded list of TreeNode ids used
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")