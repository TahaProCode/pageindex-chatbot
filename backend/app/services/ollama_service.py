"""
Thin wrapper around the local Ollama server.
Kept dependency-free (plain httpx) so we're not locked into a framework.
Supports single and batch embedding calls for high-throughput reranking.

Handles both modern (/api/embed) and legacy (/api/embeddings) Ollama endpoints.
"""
import logging
import time
from typing import List

import httpx

from app.core.config import settings

logger = logging.getLogger("pageindex.ollama")


async def get_embedding(text: str) -> list[float]:
    """
    Gets a vector embedding for a single piece of text.
    Prefer get_embeddings_batch() when you have multiple texts.
    """
    vectors = await get_embeddings_batch([text])
    return vectors[0]


async def get_embeddings_batch(texts: List[str]) -> List[list[float]]:
    """
    Batch embedding: prefers /api/embed (modern, supports real batching).
    Falls back to sequential /api/embeddings calls if the server returns 404
    (older Ollama versions).
    """
    if not texts:
        return []

    # Ollama can handle reasonably large batches; chunk if extremely large
    MAX_BATCH = 64
    if len(texts) > MAX_BATCH:
        results: List[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i : i + MAX_BATCH]
            results.extend(await get_embeddings_batch(chunk))
        return results

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            # --- Prefer modern endpoint ---
            payload = {
                "model": settings.ollama_embedding_model,
                "input": texts,
            }
            resp = await client.post(
                f"{settings.ollama_base_url}/api/embed", json=payload
            )

            if resp.status_code == 404:
                # Older Ollama – fall back to legacy single-item endpoint
                logger.warning(
                    "Ollama /api/embed returned 404 – falling back to legacy "
                    "/api/embeddings (consider upgrading Ollama)"
                )
                return await _legacy_embeddings_batch(client, texts)

            resp.raise_for_status()
            data = resp.json()
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception(
            "Ollama BATCH EMBED REQUEST FAILED after %.1fs (n=%d)",
            elapsed, len(texts),
        )
        raise

    elapsed = time.monotonic() - start
    embeddings = data.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
        )

    logger.info(
        "Ollama BATCH EMBED: elapsed=%.1fs n=%d total_chars=%d dim=%d",
        elapsed,
        len(texts),
        sum(len(t) for t in texts),
        len(embeddings[0]) if embeddings else 0,
    )
    return embeddings


async def _legacy_embeddings_batch(
    client: httpx.AsyncClient, texts: List[str]
) -> List[list[float]]:
    """
    Sequential fallback using the old /api/embeddings endpoint
    (expects "prompt" and returns "embedding").
    """
    results: List[list[float]] = []
    for text in texts:
        payload = {
            "model": settings.ollama_embedding_model,
            "prompt": text,
        }
        resp = await client.post(
            f"{settings.ollama_base_url}/api/embeddings", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embedding")
        if emb is None:
            raise ValueError("Legacy /api/embeddings response missing 'embedding'")
        results.append(emb)
    return results


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
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat", json=payload
            )
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