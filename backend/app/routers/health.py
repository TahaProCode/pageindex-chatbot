# health.py
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.core.config import settings
from app.services.ollama_service import ping_ollama
from app.services.graph_db import is_graph_available, reset_connection_state, get_driver
from app.schemas.schemas import HealthOut

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthOut)
async def health_check(db: Session = Depends(get_db)):
    # Check Postgres
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    # Check Ollama
    ollama_ok = await ping_ollama()
    ollama_status = "ok" if ollama_ok else "unreachable or model not pulled"

    # Check Neo4j
    if not settings.graph_enabled:
        graph_status = "disabled"
    else:
        # Allow one retry if previous startup failed (Neo4j still booting)
        if not is_graph_available():
            reset_connection_state()
            get_driver()
        graph_status = "ok" if is_graph_available() else "unreachable"

    return HealthOut(
        api="ok",
        database=db_status,
        ollama=ollama_status,
        ollama_model=settings.ollama_model,
        graph_db=graph_status,
    )