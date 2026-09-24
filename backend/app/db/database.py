"""
Database setup: engine, session factory, and the declarative Base
that all ORM models inherit from.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables if they don't exist yet.
    Fine for early development; once the schema stabilizes, switch to
    Alembic migrations instead of relying on create_all.
    """
    from app.models import models  # noqa: F401  (ensure models are registered)
    Base.metadata.create_all(bind=engine)
