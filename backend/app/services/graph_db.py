"""
Low-level Neo4j connection manager.

- Single shared driver (lazy init)
- Graceful no-op when GRAPH_ENABLED=false or Neo4j is unreachable
- Schema constraints created once on first successful connect
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from app.core.config import settings

logger = logging.getLogger("pageindex.graph_db")

_driver = None
_schema_ready = False
_connection_failed = False


def is_graph_available() -> bool:
    """True only when graph is enabled and a live driver exists."""
    if not settings.graph_enabled:
        return False
    if _connection_failed:
        return False
    try:
        return get_driver() is not None
    except Exception:
        return False


def get_driver():
    """Lazy-create the Neo4j driver. Returns None if disabled or unreachable."""
    global _driver, _connection_failed

    if not settings.graph_enabled:
        return None

    if _connection_failed:
        return None

    if _driver is not None:
        return _driver

    try:
        from neo4j import GraphDatabase

        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        # Verify connectivity once
        _driver.verify_connectivity()
        logger.info("Neo4j connected: %s", settings.neo4j_uri)
        _ensure_schema(_driver)
        return _driver
    except Exception as e:
        logger.warning(
            "Neo4j unavailable (%s). Graph features disabled until restart. "
            "Error: %s",
            settings.neo4j_uri,
            e,
        )
        _connection_failed = True
        _driver = None
        return None


def _ensure_schema(driver) -> None:
    """Idempotent constraints / indexes for Entity and Section nodes."""
    global _schema_ready
    if _schema_ready:
        return

    statements = [
        # Unique entity per (name, document) so same name in different docs stays distinct
        """
        CREATE CONSTRAINT entity_name_doc IF NOT EXISTS
        FOR (e:Entity) REQUIRE (e.name, e.document_id) IS UNIQUE
        """,
        """
        CREATE CONSTRAINT section_id IF NOT EXISTS
        FOR (s:Section) REQUIRE s.node_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document) REQUIRE d.document_id IS UNIQUE
        """,
        """
        CREATE INDEX entity_name IF NOT EXISTS
        FOR (e:Entity) ON (e.name)
        """,
        """
        CREATE INDEX entity_batch IF NOT EXISTS
        FOR (e:Entity) ON (e.batch_id)
        """,
        """
        CREATE INDEX section_batch IF NOT EXISTS
        FOR (s:Section) ON (s.batch_id)
        """,
    ]

    try:
        with driver.session() as session:
            for stmt in statements:
                session.run(stmt)
        _schema_ready = True
        logger.info("Neo4j schema constraints ready")
    except Exception:
        logger.exception("Failed to create Neo4j schema (non-fatal)")


@contextmanager
def graph_session() -> Generator[Any, None, None]:
    """
    Yield a Neo4j session, or None when graph is unavailable.
    Caller must check for None.
    """
    driver = get_driver()
    if driver is None:
        yield None
        return
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def close_driver() -> None:
    global _driver, _schema_ready, _connection_failed
    if _driver is not None:
        try:
            _driver.close()
        except Exception:
            pass
    _driver = None
    _schema_ready = False
    _connection_failed = False


def reset_connection_state() -> None:
    """Allow retry after a previous connection failure (e.g. Neo4j started late)."""
    global _connection_failed
    _connection_failed = False
    close_driver()


def run_query(
    cypher: str,
    parameters: Optional[dict] = None,
    write: bool = False,
) -> list[dict]:
    """
    Run a Cypher query and return list of record dicts.
    Returns [] if graph is unavailable (never raises for connectivity).
    """
    driver = get_driver()
    if driver is None:
        return []

    parameters = parameters or {}
    try:
        with driver.session() as session:
            if write:
                result = session.execute_write(
                    lambda tx: list(tx.run(cypher, parameters))
                )
            else:
                result = session.execute_read(
                    lambda tx: list(tx.run(cypher, parameters))
                )
            return [dict(r) for r in result]
    except Exception:
        logger.exception("Neo4j query failed: %s", cypher[:120])
        return []