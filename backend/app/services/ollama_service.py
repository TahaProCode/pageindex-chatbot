### ollama_service.py
"""
Thin wrapper around the local Ollama server.
Kept dependency-free (plain httpx) so we're not locked into a framework
while the core tree-search logic is still being validated (see project plan).
"""
import logging
import time

import httpx

from app.core.config import settings

logger = logging.getLogger("pageindex.ollama")


async def get_embedding(text: str) -> list[float]:
    """
    Gets a vector embedding for a piece of text from Ollama's embedding
    endpoint (separate from /api/chat - this uses a dedicated embedding
    model, e.g. nomic-embed-text, not the chat model qwen3:4b).

    Used by reranker_service.py for cheap, LLM-free relevance scoring -
    no /api/chat call involved here, so this is fast even for many nodes.
    """
    payload = {
        "model": settings.ollama_embedding_model,
        "input": text,
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception("Ollama EMBED REQUEST FAILED after %.1fs", elapsed)
        raise

    elapsed = time.monotonic() - start
    # Ollama's /api/embed returns {"embeddings": [[...]]} - a list of vectors,
    # one per input. We send one input at a time, so take the first.
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise ValueError(f"Ollama returned no embeddings for input of {len(text)} chars")

    vector = embeddings[0]
    logger.info(
        "Ollama EMBED: elapsed=%.1fs text_chars=%d vector_dim=%d",
        elapsed, len(text), len(vector),
    )
    return vector


def _strip_thinking(content: str) -> str:
    """
    Some Ollama versions don't fully honor `"think": false` for reasoning
    models like qwen3, and still emit a hidden chain-of-thought ending in
    `</think>` before the real answer. Strip everything up to and
    including the last `</think>` so callers only see the final answer.
    """
    if "</think>" in content:
        content = content.split("</think>")[-1]
    return content.strip()


async def ping_ollama() -> bool:
    """Quick check that Ollama is up and the configured model is pulled."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = [m.get("name", "") for m in tags]
            return any(settings.ollama_model in n for n in names)
    except Exception:
        logger.exception("ping_ollama failed")
        return False


async def chat_completion(prompt: str, system: str | None = None) -> str:
    """
    Send a single-turn prompt to the local model and return the text response,
    with any hidden reasoning stripped out.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }

    logger.info(
        "Ollama REQUEST: model=%s prompt_chars=%d system_preview=%r",
        settings.ollama_model, len(prompt), (system or "")[:60],
    )
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception("Ollama REQUEST FAILED after %.1fs", elapsed)
        raise

    elapsed = time.monotonic() - start
    raw_content = data.get("message", {}).get("content", "")
    had_thinking = "</think>" in raw_content
    final_content = _strip_thinking(raw_content)

    logger.info(
        "Ollama RESPONSE: elapsed=%.1fs raw_chars=%d had_thinking=%s "
        "eval_count=%s eval_ms=%s final_preview=%r",
        elapsed,
        len(raw_content),
        had_thinking,
        data.get("eval_count"),
        (data.get("eval_duration") or 0) // 1_000_000,
        final_content[:150],
    )

    return final_content