"""
LLM-based Knowledge Graph extraction from a single TreeNode.

Extracts:
  - Entities (PERSON, ORG, PRODUCT, CONCEPT, LOCATION, DATE, METRIC, OTHER)
  - Relations between entities (typed edges)

Designed for local small models (qwen3:4b / qwen2.5:7b):
  - Strict JSON-only prompt
  - Robust parsing with fallbacks
  - Confidence filtering
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import settings
from app.services.ollama_service import chat_completion

logger = logging.getLogger("pageindex.kg_extract")

MAX_TEXT_CHARS = 3500

KG_EXTRACT_SYSTEM_PROMPT = (
    "You extract a knowledge graph from one document section.\n"
    "Reply with ONLY valid JSON, no markdown, no explanation.\n"
    "Schema:\n"
    "{\n"
    '  "entities": [{"name": "string", "type": "PERSON|ORG|PRODUCT|CONCEPT|LOCATION|DATE|METRIC|OTHER"}],\n'
    '  "relations": [{"source": "entity name", "relation": "short verb phrase", "target": "entity name", "confidence": 0.0}]\n'
    "}\n"
    "Rules:\n"
    "- Extract only entities that actually appear in the text.\n"
    "- Keep names short (1-5 words). Prefer proper nouns and key concepts.\n"
    "- Max 12 entities, max 15 relations.\n"
    "- confidence is 0.0-1.0; use >=0.5 only when the relation is clearly stated.\n"
    "- If almost nothing useful is present, return {\"entities\": [], \"relations\": []}."
)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove ```json ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json_loads(raw: str) -> dict[str, Any]:
    raw = _strip_code_fences(raw)
    # Try direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find outermost { ... }
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("KG extract: could not parse JSON from LLM reply: %r", raw[:200])
    return {"entities": [], "relations": []}


def _normalize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name[:120]


def _normalize_type(t: str) -> str:
    allowed = {
        "PERSON", "ORG", "PRODUCT", "CONCEPT",
        "LOCATION", "DATE", "METRIC", "OTHER",
    }
    t = (t or "OTHER").upper().strip()
    return t if t in allowed else "OTHER"


def _clean_extraction(data: dict[str, Any]) -> dict[str, Any]:
    entities_raw = data.get("entities") or []
    relations_raw = data.get("relations") or []

    entities: list[dict] = []
    seen_names: set[str] = set()
    for e in entities_raw:
        if not isinstance(e, dict):
            continue
        name = _normalize_name(str(e.get("name", "")))
        if not name or len(name) < 2:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        entities.append({
            "name": name,
            "type": _normalize_type(str(e.get("type", "OTHER"))),
        })
        if len(entities) >= 12:
            break

    name_set = {e["name"].lower() for e in entities}
    min_conf = settings.kg_min_confidence

    relations: list[dict] = []
    for r in relations_raw:
        if not isinstance(r, dict):
            continue
        source = _normalize_name(str(r.get("source", "")))
        target = _normalize_name(str(r.get("target", "")))
        relation = _normalize_name(str(r.get("relation", "")))
        if not source or not target or not relation:
            continue
        if source.lower() == target.lower():
            continue
        # Only keep relations whose endpoints we extracted
        if source.lower() not in name_set or target.lower() not in name_set:
            continue
        try:
            conf = float(r.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        if conf < min_conf:
            continue
        relations.append({
            "source": source,
            "relation": relation[:80],
            "target": target,
            "confidence": round(min(max(conf, 0.0), 1.0), 2),
        })
        if len(relations) >= 15:
            break

    return {"entities": entities, "relations": relations}


async def extract_kg_from_text(
    title: str,
    text: str,
    *,
    summary: str | None = None,
) -> dict[str, Any]:
    """
    Extract entities + relations from one section.
    Returns {"entities": [...], "relations": [...]}.
    Never raises — returns empty structure on failure.
    """
    body = (text or "").strip()
    if not body and summary:
        body = summary
    if not body or len(body) < 30:
        return {"entities": [], "relations": []}

    body = body[:MAX_TEXT_CHARS]
    prompt = (
        f"Section title: {title}\n\n"
        f"Text:\n{body}\n\n"
        "Extract entities and relations as JSON."
    )

    try:
        response = await chat_completion(prompt=prompt, system=KG_EXTRACT_SYSTEM_PROMPT)
        data = _safe_json_loads(response)
        cleaned = _clean_extraction(data)
        logger.info(
            "KG extract for %r: %d entities, %d relations",
            title[:60],
            len(cleaned["entities"]),
            len(cleaned["relations"]),
        )
        return cleaned
    except Exception:
        logger.exception("KG extraction failed for section %r", title[:60])
        return {"entities": [], "relations": []}