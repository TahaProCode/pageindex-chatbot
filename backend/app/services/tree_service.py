### tree_service.py
"""
Persists the nested tree produced by pdf_service.parse_pdf_to_tree()
into the `tree_nodes` table, preserving parent/child relationships and
sibling order.
"""
from sqlalchemy.orm import Session

from app.models.models import TreeNode

MAX_TITLE_LENGTH = 500


def persist_tree(
    db: Session,
    document_id: str,
    nodes: list[dict],
    parent_id: str | None = None,
) -> None:
    for order_index, node in enumerate(nodes):
        db_node = TreeNode(
            document_id=document_id,
            parent_id=parent_id,
            title=(node.get("title") or "Untitled")[:MAX_TITLE_LENGTH],
            summary=node.get("summary"),
            raw_text=node.get("raw_text"),
            page_start=node.get("page_start"),
            page_end=node.get("page_end"),
            order_index=order_index,
        )
        db.add(db_node)
        db.flush()

        children = node.get("children") or []
        if children:
            persist_tree(db, document_id, children, parent_id=db_node.id)