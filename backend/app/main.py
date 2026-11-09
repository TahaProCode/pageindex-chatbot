import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.database import init_db
from app.routers import health, documents, batches, chat
from app.services.indexing_queue import start_worker
from app.services.graph_db import get_driver, close_driver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="PageIndex Chatbot API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(batches.router)
app.include_router(chat.router)


@app.on_event("startup")
def on_startup():
    init_db()
    start_worker()
    # Best-effort Neo4j connect (non-blocking for app start)
    if settings.graph_enabled:
        get_driver()


@app.on_event("shutdown")
def on_shutdown():
    close_driver()


# Serve the built React frontend as static files, so the whole app runs
# on one port/origin. STATIC_DIR only exists in the Docker image (built by
# the multi-stage Dockerfile) - in local dev without Docker, this is
# skipped and you run Vite separately on its own port as before.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {
            "message": "PageIndex Chatbot API is running. No frontend build found at "
                       "./static - run the frontend separately via `npm run dev`, or "
                       "build the full stack with `docker compose up -d --build`."
        }