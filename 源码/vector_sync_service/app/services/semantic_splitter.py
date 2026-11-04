"""Semantic paragraph splitter with LaTeX protection.

Ports the boundary-priority search algorithm from the demo module into
the production chunking pipeline.  Returns (start, end, text) triples
compatible with the existing ``_sliding_windows()`` signature so it can
drop in as a replacement.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

_LATEX_INLINE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
_LATEX_DISPLAY = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$")
_LATEX_PAREN = re.compile(r"\\\((.+?)\\\)")
_LATEX_BRACKET = re.compile(r"\\\[(.+?)\\\]")
_SENTENCE_END = re.compile(r"[.!?](?=\s+[A-Z])")
_CLAUSE_BOUNDARY = re.compile(r"[,;:](?=\s)")
_SPACE_BOUNDARY = re.compile(r"\s(?=\S)")

_MIN_CHUNK_RATIO = 0.3


def _find_latex_zones(text: str) -> List[Tuple[int, int]]:
    """Detect LaTeX math zones that must not be split inside.

    Returns a list of ``(start, end)`` intervals sorted by start, with
    overlapping intervals merged.
    """
    zones: List[Tuple[int, int]] = []
    for pattern in [_LATEX_DISPLAY, _LATEX_INLINE, _LATEX_BRACKET, _LATEX_PAREN]:
        for m in pattern.finditer(text):
            zones.append((m.start(), m.end()))
    zones.sort()

    merged: List[Tuple[int, int]] = []
    for start, end in zones:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_in_latex_zone(pos: int, zones: List[Tuple[int, int]]) -> bool:
    """Return True when *pos* falls strictly inside a LaTeX zone."""
    for start, end in zones:
        if start < pos < end:
            return True
    return False


def _find_nearest_latex_boundary(
    pos: int, zones: List[Tuple[int, int]], text_len: int
) -> Optional[int]:
    """Find the nearest LaTeX-zone boundary to escape from inside a zone."""
    candidates: List[int] = []
    for start, end in zones:
        if start <= pos <= end:
            if start > 0:
                candidates.append(start)
            if end < text_len:
                candidates.append(end)
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - pos))


def find_semantic_split(
    text: str, target_pos: int, search_window: int = 200
) -> int:
    """Find the best semantic boundary within *search_window* chars before *target_pos*.

    Priority (highest first):

    1. Sentence end: ``[.!?](?=\\s+[A-Z])``
    2. Clause boundary: ``[,;:](?=\\s)``
    3. Word boundary: ``\\s(?=\\S)``
    4. Hard fallback to *target_pos*

    All candidate positions are verified to not fall inside LaTeX math zones.
    """
    text_len = len(text)
    if target_pos >= text_len:
        return text_len
    if target_pos <= 0:
        return target_pos

    latex_zones = _find_latex_zones(text)

    # Escape LaTeX zone first
    if _is_in_latex_zone(target_pos, latex_zones):
        boundary = _find_nearest_latex_boundary(target_pos, latex_zones, text_len)
        if boundary is not None:
            target_pos = boundary

    search_start = max(0, target_pos - search_window)
    search_text = text[search_start:target_pos]

    # Priority 1 — sentence end
    for m in reversed(list(_SENTENCE_END.finditer(search_text))):
        cut_pos = search_start + m.end()
        if not _is_in_latex_zone(cut_pos, latex_zones):
            return cut_pos

    # Priority 2 — clause boundary
    for m in reversed(list(_CLAUSE_BOUNDARY.finditer(search_text))):
        cut_pos = search_start + m.end()
        if not _is_in_latex_zone(cut_pos, latex_zones):
            return cut_pos

    # Priority 3 — word boundary
    for m in reversed(list(_SPACE_BOUNDARY.finditer(search_text))):
        cut_pos = search_start + m.end()
        if not _is_in_latex_zone(cut_pos, latex_zones):
            return cut_pos

    # Priority 4 — hard fallback
    return target_pos


def semantic_sliding_windows(
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    search_window: int = 200,
) -> Iterable[Tuple[int, int, str]]:
    """Semantic-aware sliding window — same signature as ``_sliding_windows()``.

    Yields ``(start, end, segment)`` triples where *end* is aligned to a
    sentence / clause / word boundary whenever possible.
    """
    if len(text) <= max_chars:
        yield 0, len(text), text
        return

    pos = 0
    text_len = len(text)
    min_length = int(max_chars * _MIN_CHUNK_RATIO)

    while pos < text_len:
        raw_end = min(pos + max_chars, text_len)

        if raw_end >= text_len:
            yield pos, text_len, text[pos:]
            break

        best_end = find_semantic_split(text, raw_end, search_window)

        # Minimum chunk length guard
        if best_end - pos < min_length:
            best_end = raw_end

        yield pos, best_end, text[pos:best_end]

        if best_end >= text_len:
            break

        next_pos = best_end - overlap_chars
        if next_pos <= pos:
            next_pos = best_end  # dead-loop guard
        pos = next_pos
