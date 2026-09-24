# pdf_service.py
"""
Phase 2: PDF parsing + structure detection.

This module is pure logic (no DB, no FastAPI) so it's easy to test on its
own. It takes a PDF file path and returns a nested tree of sections:

    {
      "page_count": int,
      "tree": [
        {
          "title": str, "summary": None, "raw_text": str,
          "page_start": int, "page_end": int,
          "children": [ ... same shape ... ]
        },
        ...
      ]
    }

`summary` is always None here - Phase 3 fills it in with per-node LLM
summaries. This module only figures out the document's structure.

IMPORTANT (added after observing degenerate trees on SEC filings like
10-K/10-Q): font-size-based heading detection can fail to find ANY
section headers inside a document body (e.g. a filing whose only
distinctly-larger text is the cover-page title). When that happens, the
"heading" swallows the ENTIRE rest of the document into one giant
raw_text blob with zero children - a single leaf node covering 50+ pages.
Since retrieval only ever sends the first ~3000 chars of a leaf node's
raw_text to the LLM, everything past the first page or two of that blob
becomes invisible to retrieval, even though it's sitting right there in
the DB.

To fix this, _expand_oversized_leaves() walks the finished tree and, for
any leaf node whose raw_text is larger than a threshold, replaces it with
page-based sub-chunk children (same shape as the "no headings at all"
fallback below) - so the tree-search can drill into the actual pages
that matter instead of only ever seeing the start of a giant blob.
"""
from collections import Counter

import pymupdf as fitz  # PyMuPDF (import name kept as `fitz` for readability below)

MAX_HEADING_LEVELS = 4
MAX_HEADING_WORDS = 20
BOLD_HEADING_MAX_WORDS = 12
PAGE_CHUNK_SIZE = 5

# If a node's own raw_text is longer than this, it gets split into
# page-based sub-chunks instead of being left as one giant leaf. Set
# somewhat above CONTEXT_CHARS_PER_NODE (3000, in retrieval_agent.py /
# multi_doc_service.py) so only genuinely oversized sections get split -
# a normal-sized section that already fits in one context window is left
# alone.
MAX_LEAF_TEXT_CHARS = 4000


def _is_bold_span(font_name: str, flags: int) -> bool:
    return bool(flags & (1 << 4)) or "bold" in font_name.lower()


def extract_lines(file_path: str) -> tuple[int, list[dict]]:
    doc = fitz.open(file_path)
    page_count = doc.page_count
    lines: list[dict] = []

    for page_index in range(page_count):
        page = doc[page_index]
        page_dict = page.get_text("dict")

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
                    bold_flags.append(_is_bold_span(span.get("font", ""), span.get("flags", 0)))

                line_text = "".join(text_parts).strip()
                if not line_text:
                    continue

                lines.append({
                    "text": line_text,
                    "size": (sum(sizes) / len(sizes)) if sizes else 0.0,
                    "bold": all(bold_flags) if bold_flags else False,
                    "page": page_index + 1,
                })

    doc.close()
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
        # Kept only for _expand_oversized_leaves() below - not part of the
        # public tree shape, stripped before returning from parse_pdf_to_tree.
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
        title = f"Pages {start_page}-{end_page}" if end_page > start_page else f"Page {start_page}"
        nodes.append({
            "title": title,
            "summary": None,
            "raw_text": chunk_text,
            "page_start": start_page,
            "page_end": end_page,
            "children": [],
        })
    return nodes


def _build_page_chunks_from_lines(own_lines: list[dict], page_start: int, page_end: int) -> list[dict]:
    """
    Same idea as _build_page_chunks(), but scoped to one oversized node's
    own slice of lines/pages rather than the whole document. Used by
    _expand_oversized_leaves() to break up a giant leaf into page-sized
    children so retrieval can drill into it level by level.
    """
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
        title = f"Pages {chunk_start}-{chunk_end}" if chunk_end > chunk_start else f"Page {chunk_start}"
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
    Recursively walks a tree (in place) and, for any LEAF node (no
    children) whose raw_text is longer than MAX_LEAF_TEXT_CHARS, replaces
    its children with page-based sub-chunks built from that node's own
    lines. This is what prevents a single detected heading from silently
    swallowing an entire multi-page document into one untouchable blob -
    see the module docstring for why this matters.

    The parent node's own raw_text is left as-is (still usable as a
    fallback in retrieval_agent._walk() if none of the new children get
    selected), only its children list is populated.
    """
    for node in nodes:
        children = node.get("children") or []
        if children:
            _expand_oversized_leaves(children)
            continue

        raw_text = node.get("raw_text") or ""
        own_lines = node.pop("_own_lines", None)
        if len(raw_text) > MAX_LEAF_TEXT_CHARS and own_lines:
            sub_chunks = _build_page_chunks_from_lines(
                own_lines, node["page_start"], node["page_end"]
            )
            # Only worth it if this actually produces more than one chunk -
            # a single-chunk result would just be the same blob again.
            if len(sub_chunks) > 1:
                node["children"] = sub_chunks


def _strip_internal_keys(nodes: list[dict]) -> None:
    """Removes the temporary _own_lines key left on any node that never
    got expanded (e.g. it had children already, or wasn't oversized)."""
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
        tree = _build_page_chunks(lines, page_count)

    return {"page_count": page_count, "tree": tree}