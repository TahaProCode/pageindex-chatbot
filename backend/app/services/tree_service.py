"""
Persists the nested tree produced by pdf_service.parse_pdf_to_tree()
into the `tree_nodes` table, preserving parent/child relationships and
sibling order.

Performance: generates UUIDs on the application side and prepares all
TreeNode objects in memory, then adds them in one pass so SQLAlchemy
can batch the INSERTs. This avoids the previous per-node db.flush()
which issued one SQL round-trip per node on large trees.
"""
import re
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.models.models import TreeNode

MAX_TITLE_LENGTH = 500


def _normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Minimal whitespace cleanup before storing raw_text:
    - Collapse runs of 3+ newlines into a double newline (paragraph break)
    - Collapse runs of spaces/tabs into a single space
    - Strip leading/trailing whitespace
    Does NOT destroy meaningful structure (tables, lists).
    """
    if not text:
        return text
    # Normalize Windows/Mac newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs (but keep single spaces)
    text = re.sub(r"[ \t]+", " ", text)
    # Clean spaces around newlines
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _prepare_nodes(
    nodes: list[dict],
    document_id: str,
    parent_id: Optional[str] = None,
) -> list[TreeNode]:
    """
    Recursively walk the in-memory tree dicts and build TreeNode ORM
    objects with pre-generated UUIDs so parent_id links are known
    before any DB write.
    """
    prepared: list[TreeNode] = []

    for order_index, node in enumerate(nodes):
        node_id = str(uuid.uuid4())
        title = (node.get("title") or "Untitled")[:MAX_TITLE_LENGTH]
        raw_text = _normalize_text(node.get("raw_text"))

        db_node = TreeNode(
            id=node_id,
            document_id=document_id,
            parent_id=parent_id,
            title=title,
            summary=node.get("summary"),
            raw_text=raw_text,
            page_start=node.get("page_start"),
            page_end=node.get("page_end"),
            order_index=order_index,
        )
        prepared.append(db_node)

        children = node.get("children") or []
        if children:
            prepared.extend(
                _prepare_nodes(children, document_id, parent_id=node_id)
            )

    return prepared


def persist_tree(
    db: Session,
    document_id: str,
    nodes: list[dict],
    parent_id: Optional[str] = None,
) -> None:
    """
    Convert the nested tree into TreeNode rows and insert them in a
    single bulk operation (one commit by the caller is enough).
    """
    all_nodes = _prepare_nodes(nodes, document_id, parent_id=parent_id)
    if not all_nodes:
        return

    # add_all lets SQLAlchemy batch the inserts efficiently
    db.add_all(all_nodes)
    db.flush()  # one flush for the whole tree, not per node