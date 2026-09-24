"""
Pydantic schemas used for request/response validation.
Kept separate from the SQLAlchemy models (app/models/models.py).
"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    id: str
    batch_id: str
    filename: str
    status: str
    page_count: Optional[int] = None
    created_at: datetime
    summary: Optional[str] = None
    keywords: List[str] = []

    model_config = ConfigDict(from_attributes=True)


class BatchOut(BaseModel):
    id: str
    name: Optional[str] = None
    created_at: datetime
    documents: List[DocumentOut] = []

    model_config = ConfigDict(from_attributes=True)


class TreeNodeOut(BaseModel):
    id: str
    parent_id: Optional[str]
    title: str
    summary: Optional[str] = None
    keywords: List[str] = []
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    children: List["TreeNodeOut"] = []

    model_config = ConfigDict(from_attributes=True)


class ChatSessionOut(BaseModel):
    id: str
    batch_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageIn(BaseModel):
    content: str


class CitedSection(BaseModel):
    id: str
    filename: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    cited_sections: List[CitedSection] = []

    model_config = ConfigDict(from_attributes=True)


class HealthOut(BaseModel):
    api: str
    database: str
    ollama: str
    ollama_model: str