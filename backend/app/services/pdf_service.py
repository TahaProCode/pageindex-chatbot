"""
PDF parsing + structure detection.

Produces a nested tree of sections from a PDF path:

    {
      "page_count": int,
      "tree": [
        {
          "title": str, "summary": None, "raw_text": str,
          "page_start": int, "page_end": int,
          "children": [ ... ]
        },
        ...
      ]
    }

Improvements for complex / scanned / large PDFs:
1. OCR fallback (PyMuPDF built-in OCR) when a page has almost no extractable text.
2. Table-aware text extraction that preserves row structure where possible.
3. Character-based chunking fallback when heading detection fails or produces
   a single giant leaf (in addition to the existing page-based expansion).
4. Oversized-leaf expansion so retrieval can drill into multi-page blobs.
"""
from collections import Counter
from typing import Optional

import pymupdf as fitz

MAX_HEADING_LEVELS = 4
MAX_HEADING_WORDS = 20
BOLD_HEADING_MAX_WORDS = 12
PAGE_CHUNK_SIZE = 5
MAX_LEAF_TEXT_CHARS = 4000
# If a page yields fewer characters than this, try OCR
MIN_CHARS_PER_PAGE_FOR_TEXT = 40
# Character-based chunk size when we fall back to pure text chunking
CHAR_CHUNK_SIZE = 2500
CHAR_CHUNK_OVERLAP = 200


def _is_bold_span(font_name: str, flags: int) -> bool:
    return bool(flags & (1 << 4)) or "bold" in (font_name or "").lower()


def _extract_page_text_with_tables(page) -> str:
    """
    Prefer structured text that keeps table-like alignment.
    Falls back to plain text if find_tables is unavailable or fails.
    """
    parts: list[str] = []

    # Try table extraction first (PyMuPDF 1.23+)
    try:
        tabs = page.find_tables()
        table_areas = []
        if tabs and tabs.tables:
            for tab in tabs.tables:
                try:
                    data = tab.extract()
                    if data:
                        rows = []
                        for row in data:
                            cells = [str(c).strip() if c is not None else "" for c in row]
                            rows.append(" | ".join(cells))
                        parts.append("\n".join(rows))
                        if hasattr(tab, "bbox"):
                            table_areas.append(tab.bbox)
                except Exception:
                    continue
    except Exception:
        table_areas = []

    # Regular text blocks, skipping regions already covered by tables when possible
    try:
        blocks = page.get_text("blocks")
        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            if not text or not str(text).strip():
                continue
            # Skip if this block is mostly inside a detected table
            inside_table = False
            for bbox in table_areas:
                try:
                    bx0, by0, bx1, by1 = bbox
                    if x0 >= bx0 - 2 and y0 >= by0 - 2 and x1 <= bx1 + 2 and y1 <= by1 + 2:
                        inside_table = True
                        break
                except Exception:
                    pass
            if not inside_table:
                parts.append(str(text).strip())
    except Exception:
        # Ultimate fallback
        plain = page.get_text("text") or ""
        if plain.strip():
            parts.append(plain.strip())

    return "\n".join(parts).strip()


def _ocr_page(page) -> str:
    """
    Attempt OCR on a page that has little or no extractable text.
    Uses PyMuPDF's built-in OCR support when available (requires tesseract).
    Returns empty string if OCR is not available or fails.
    """
    try:
        # PyMuPDF OCR: get_textpage_ocr is available in recent versions
        tp = page.get_textpage_ocr(language="eng", dpi=150, full=True)
        text = page.get_text("text", textpage=tp) or ""
        return text.strip()
    except Exception:
        try:
            # Older / alternative path
            text = page.get_text("text") or ""
            return text.strip()
        except Exception:
            return ""


def extract_lines(file_path: str) -> tuple[int, list[dict]]:
    """
    Extract per-line metadata (text, size, bold, page) for heading detection.
    Also performs OCR fallback and table-aware extraction for content.
    """
    doc = fitz.open(file_path)
    page_count = doc.page_count
    lines: list[dict] = []
    # Also keep full page texts for character-chunk fallback later
    page_texts: dict[int, str] = {}

    for page_index in range(page_count):
        page = doc[page_index]
        page_num = page_index + 1

        # Structured / table-aware text for content
        page_text = _extract_page_text_with_tables(page)
        if len(page_text) < MIN_CHARS_PER_PAGE_FOR_TEXT:
            ocr_text = _ocr_page(page)
            if len(ocr_text) > len(page_text):
                page_text = ocr_text

        page_texts[page_num] = page_text

        # Span-level extraction for heading detection (font size / bold)
        try:
            page_dict = page.get_text("dict")
        except Exception:
            page_dict = {"blocks": []}

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text_parts, sizes, bold_flags = [], [], []
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    text_parts.append(text)
                    sizes.append(span.get("size", 0.0))
                    bold_flags.append(
                        _is_bold_span(span.get("font", ""), span.get("flags", 0))
                    )

                line_text = "".join(text_parts).strip()
                if not line_text:
                    continue

                lines.append({
                    "text": line_text,
                    "size": (sum(sizes) / len(sizes)) if sizes else 0.0,
                    "bold": all(bold_flags) if bold_flags else False,
                    "page": page_num,
                })

        # If dict extraction produced no lines but we have page_text (e.g. OCR),
        # synthesize one line so page-chunk fallback still has content.
        if not any(l["page"] == page_num for l in lines) and page_text:
            for para in page_text.split("\n"):
                para = para.strip()
                if para:
                    lines.append({
                        "text": para,
                        "size": 11.0,
                        "bold": False,
                        "page": page_num,
                    })

    doc.close()
    # Stash page_texts on a module-level attribute so parse_pdf_to_tree can use them
    extract_lines._last_page_texts = page_texts  # type: ignore[attr-defined]
    return page_count, lines


def detect_headings(lines: list[dict]) -> list[dict]:
    if not lines:
        return []

    rounded_sizes = [round(l["size"]) for l in lines]
    body_size = Counter(rounded_sizes).most_common(1)[0][0]

    larger_sizes = sorted({s for s in rounded_sizes if s > body_size}, reverse=True)
    size_to_level = {size: i + 1 for i, size in enumerate(larger_sizes[:MAX_HEADING_LEVELS])}

    headings = []
    for idx, line in enumerate(lines):
        size = round(line["size"])
        word_count = len(line["text"].split())

        if size in size_to_level:
            if word_count > MAX_HEADING_WORDS:
                continue
            level = size_to_level[size]
        elif (
            not larger_sizes
            and line["bold"]
            and size >= body_size
            and word_count <= BOLD_HEADING_MAX_WORDS
            and not line["text"].endswith((".", ",", ";", ":"))
        ):
            level = 1
        else:
            continue

        headings.append({
            "line_index": idx,
            "level": level,
            "title": line["text"],
            "page": line["page"],
        })

    return headings


def _attach_direct_text(lines: list[dict], headings: list[dict]) -> None:
    for i, h in enumerate(headings):
        start = h["line_index"] + 1
        end = headings[i + 1]["line_index"] if i + 1 < len(headings) else len(lines)
        text_lines = [lines[j]["text"] for j in range(start, end)]
        h["raw_text"] = "\n".join(text_lines).strip()
        h["page_end"] = lines[end - 1]["page"] if end > start else h["page"]
        h["_own_lines"] = lines[start:end]


def _nest_headings(headings: list[dict]) -> list[dict]:
    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []

    for h in headings:
        node = {
            "title": h["title"],
            "summary": None,
            "raw_text": h["raw_text"],
            "page_start": h["page"],
            "page_end": h["page_end"],
            "children": [],
            "_own_lines": h["_own_lines"],
        }
        while stack and stack[-1][0] >= h["level"]:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            roots.append(node)
        stack.append((h["level"], node))

    return roots


def _build_page_chunks(lines: list[dict], page_count: int) -> list[dict]:
    if not lines or page_count == 0:
        return []

    nodes = []
    for start_page in range(1, page_count + 1, PAGE_CHUNK_SIZE):
        end_page = min(start_page + PAGE_CHUNK_SIZE - 1, page_count)
        chunk_text = "\n".join(
            l["text"] for l in lines if start_page <= l["page"] <= end_page
        ).strip()
        title = (
            f"Pages {start_page}-{end_page}"
            if end_page > start_page
            else f"Page {start_page}"
        )
        nodes.append({
            "title": title,
            "summary": None,
            "raw_text": chunk_text,
            "page_start": start_page,
            "page_end": end_page,
            "children": [],
        })
    return nodes


def _build_char_chunks(
    text: str,
    page_start: int,
    page_end: int,
    title_prefix: str = "Section",
) -> list[dict]:
    """
    Split a long text blob into overlapping character windows.
    Used when heading detection fails or produces oversized leaves.
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= CHAR_CHUNK_SIZE:
        return [{
            "title": title_prefix,
            "summary": None,
            "raw_text": text,
            "page_start": page_start,
            "page_end": page_end,
            "children": [],
        }]

    chunks = []
    start = 0
    idx = 1
    while start < len(text):
        end = min(start + CHAR_CHUNK_SIZE, len(text))
        # Prefer to break at a paragraph or sentence boundary
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at < start + CHAR_CHUNK_SIZE // 2:
                break_at = text.rfind(". ", start, end)
            if break_at >= start + CHAR_CHUNK_SIZE // 2:
                end = break_at + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append({
                "title": f"{title_prefix} (part {idx})",
                "summary": None,
                "raw_text": piece,
                "page_start": page_start,
                "page_end": page_end,
                "children": [],
            })
            idx += 1

        if end >= len(text):
            break
        start = max(end - CHAR_CHUNK_OVERLAP, start + 1)

    return chunks


def _build_page_chunks_from_lines(
    own_lines: list[dict], page_start: int, page_end: int
) -> list[dict]:
    if not own_lines or page_end < page_start:
        return []

    nodes = []
    for chunk_start in range(page_start, page_end + 1, PAGE_CHUNK_SIZE):
        chunk_end = min(chunk_start + PAGE_CHUNK_SIZE - 1, page_end)
        chunk_text = "\n".join(
            l["text"] for l in own_lines if chunk_start <= l["page"] <= chunk_end
        ).strip()
        if not chunk_text:
            continue
        title = (
            f"Pages {chunk_start}-{chunk_end}"
            if chunk_end > chunk_start
            else f"Page {chunk_start}"
        )
        # If even a page-chunk is still huge, sub-split by characters
        if len(chunk_text) > MAX_LEAF_TEXT_CHARS:
            nodes.extend(
                _build_char_chunks(chunk_text, chunk_start, chunk_end, title_prefix=title)
            )
        else:
            nodes.append({
                "title": title,
                "summary": None,
                "raw_text": chunk_text,
                "page_start": chunk_start,
                "page_end": chunk_end,
                "children": [],
            })
    return nodes


def _expand_oversized_leaves(nodes: list[dict]) -> None:
    """
    Recursively expand leaf nodes whose raw_text exceeds MAX_LEAF_TEXT_CHARS
    into page-based (or character-based) children so retrieval can drill in.
    """
    for node in nodes:
        children = node.get("children") or []
        if children:
            _expand_oversized_leaves(children)
            continue

        raw_text = node.get("raw_text") or ""
        own_lines = node.pop("_own_lines", None)

        if len(raw_text) <= MAX_LEAF_TEXT_CHARS:
            continue

        if own_lines:
            sub_chunks = _build_page_chunks_from_lines(
                own_lines, node["page_start"], node["page_end"]
            )
        else:
            sub_chunks = _build_char_chunks(
                raw_text,
                node["page_start"],
                node["page_end"],
                title_prefix=node.get("title") or "Section",
            )

        if len(sub_chunks) > 1:
            node["children"] = sub_chunks


def _strip_internal_keys(nodes: list[dict]) -> None:
    for node in nodes:
        node.pop("_own_lines", None)
        if node.get("children"):
            _strip_internal_keys(node["children"])


def parse_pdf_to_tree(file_path: str) -> dict:
    page_count, lines = extract_lines(file_path)
    headings = detect_headings(lines)

    if headings:
        _attach_direct_text(lines, headings)
        tree = _nest_headings(headings)
        _expand_oversized_leaves(tree)
        _strip_internal_keys(tree)
    else:
        # No headings detected - try page chunks first; if the whole doc
        # is still one giant blob, character-chunk it.
        tree = _build_page_chunks(lines, page_count)
        total_chars = sum(len(n.get("raw_text") or "") for n in tree)
        if len(tree) <= 1 and total_chars > MAX_LEAF_TEXT_CHARS:
            full_text = "\n".join(l["text"] for l in lines)
            tree = _build_char_chunks(full_text, 1, page_count, title_prefix="Document")

    return {"page_count": page_count, "tree": tree}