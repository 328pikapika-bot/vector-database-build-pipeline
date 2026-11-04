"""Full-text `.mmd` parser for canonical text and chunk input blocks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Literal, Optional, Sequence, Tuple


BlockType = Literal["section", "paragraph", "table", "reference"]


@dataclass(frozen=True)
class ParsedBlock:
    """A block in canonical full-text order."""

    block_type: BlockType
    text: str
    char_start: int
    char_end: int
    section_path: Optional[str] = None


@dataclass(frozen=True)
class ParsedFullText:
    """Cleaned full text plus structured blocks for chunking."""

    canonical_text: str
    blocks: List[ParsedBlock]


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+?)\s*$")
_CHINESE_NUMBERED_HEADING_RE = re.compile(r"^[\u4e00-\u9fff]+[\u3001.\uff0e]\s*(.+?)\s*$")
_PIPE_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_HTML_TABLE_START_RE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*(?:figures|tables)[^>]*>", re.IGNORECASE)
_MARKDOWN_REF_RE = re.compile(r"!?\[[^\]]*\]\(([^)]*)\)")

_REFERENCE_TITLES = {
    "reference",
    "references",
    "bibliography",
    "literature cited",
    "works cited",
    "\u53c2\u8003\u6587\u732e",
    "\u53c3\u8003\u6587\u737b",
}

_COMMON_HEADING_TITLES = {
    "abstract",
    "introduction",
    "materials and methods",
    "methods",
    "method",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "\u6458\u8981",
    "\u5f15\u8a00",
    "\u524d\u8a00",
    "\u65b9\u6cd5",
    "\u6750\u6599\u4e0e\u65b9\u6cd5",
    "\u7ed3\u679c",
    "\u8ba8\u8bba",
    "\u7ed3\u8bba",
    "\u53c2\u8003\u6587\u732e",
}


def parse_mmd(mmd_text: str) -> ParsedFullText:
    """Parse raw `.mmd` text into canonical text and structured blocks."""

    cleaned_text = clean_mmd_noise(mmd_text)
    raw_lines = cleaned_text.split("\n")
    pending_blocks: List[Tuple[BlockType, str, Optional[str]]] = []
    section_stack: List[Tuple[int, str]] = []
    in_references = False
    reference_level: Optional[int] = None

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        heading = _parse_heading(line)
        if heading:
            level, title = heading
            is_reference = _is_reference_title(title)
            if in_references and reference_level is not None and level <= reference_level and not is_reference:
                in_references = False
                reference_level = None

            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            section_path = _format_section_path(section_stack)

            if is_reference:
                in_references = True
                reference_level = level
                block_type: BlockType = "reference"
            else:
                block_type = "section"

            pending_blocks.append((block_type, title, section_path))
            i += 1
            continue

        if _is_table_start(raw_lines, i):
            table_lines, next_index = _collect_table(raw_lines, i)
            section_path = _format_section_path(section_stack)
            pending_blocks.append(("table", "\n".join(table_lines), section_path))
            i = next_index
            continue

        paragraph_lines, next_index = _collect_paragraph(raw_lines, i)
        text = _join_paragraph_lines(paragraph_lines)
        if text:
            block_type = "reference" if in_references else "paragraph"
            section_path = _format_section_path(section_stack)
            pending_blocks.append((block_type, text, section_path))
        i = next_index

    return _build_parsed_fulltext(pending_blocks)


def clean_mmd_noise(mmd_text: str) -> str:
    """Normalize `.mmd` text and remove image/table file reference noise."""

    if not mmd_text:
        return ""

    text = mmd_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines: List[str] = []
    previous_blank = False

    for raw_line in text.split("\n"):
        line = _strip_noise_references(raw_line).rstrip()
        if _is_bare_asset_reference(line):
            line = ""

        if not line.strip():
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue

        lines.append(line)
        previous_blank = False

    return "\n".join(lines).strip()


def _build_parsed_fulltext(blocks: Sequence[Tuple[BlockType, str, Optional[str]]]) -> ParsedFullText:
    canonical_parts: List[str] = []
    parsed_blocks: List[ParsedBlock] = []
    cursor = 0

    for block_type, text, section_path in blocks:
        text = text.strip()
        if not text:
            continue
        if canonical_parts:
            canonical_parts.append("\n\n")
            cursor += 2

        start = cursor
        end = start + len(text)
        canonical_parts.append(text)
        parsed_blocks.append(
            ParsedBlock(
                block_type=block_type,
                text=text,
                char_start=start,
                char_end=end,
                section_path=section_path or None,
            )
        )
        cursor = end

    return ParsedFullText(canonical_text="".join(canonical_parts), blocks=parsed_blocks)


def _strip_noise_references(line: str) -> str:
    line = _HTML_IMAGE_RE.sub("", line)

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip().lower()
        if _is_asset_target(target):
            return ""
        return match.group(0)

    return _MARKDOWN_REF_RE.sub(replace, line)


def _is_asset_target(target: str) -> bool:
    normalized = target.split("#", 1)[0].split("?", 1)[0].replace("\\", "/").lower()
    return (
        "figures/" in normalized
        or normalized.startswith("figures/")
        or "/figures/" in normalized
        or "tables/" in normalized
        or normalized.startswith("tables/")
        or "/tables/" in normalized
    )


def _is_bare_asset_reference(line: str) -> bool:
    stripped = line.strip().replace("\\", "/").lower()
    return bool(stripped) and (
        stripped.startswith("figures/")
        or stripped.startswith("./figures/")
        or stripped.startswith("tables/")
        or stripped.startswith("./tables/")
    )


def _parse_heading(line: str) -> Optional[Tuple[int, str]]:
    markdown_match = _MARKDOWN_HEADING_RE.match(line)
    if markdown_match:
        return len(markdown_match.group(1)), _clean_heading_title(markdown_match.group(2))

    numbered_match = _NUMBERED_HEADING_RE.match(line)
    if numbered_match and len(line) <= 120:
        number = numbered_match.group(1).rstrip(".")
        level = min(number.count(".") + 1, 6)
        title = _clean_heading_title(f"{numbered_match.group(1)} {numbered_match.group(2)}")
        return level, title

    chinese_match = _CHINESE_NUMBERED_HEADING_RE.match(line)
    if chinese_match and len(line) <= 80:
        return 1, _clean_heading_title(line)

    normalized = _normalize_title(line)
    if normalized in _COMMON_HEADING_TITLES:
        return 1, _clean_heading_title(line)

    return None


def _clean_heading_title(title: str) -> str:
    return title.strip().strip("#").strip()


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().strip(":\uff1a").lower())


def _is_reference_title(title: str) -> bool:
    normalized = _normalize_title(re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title))
    return normalized in _REFERENCE_TITLES


def _format_section_path(stack: Sequence[Tuple[int, str]]) -> Optional[str]:
    if not stack:
        return None
    return "/".join(title for _, title in stack)


def _is_table_start(lines: Sequence[str], index: int) -> bool:
    line = lines[index]
    if _HTML_TABLE_START_RE.search(line):
        return True
    if not _PIPE_TABLE_RE.match(line):
        return False
    next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
    return bool(_TABLE_SEPARATOR_RE.match(next_line) or _PIPE_TABLE_RE.match(next_line))


def _collect_table(lines: Sequence[str], index: int) -> Tuple[List[str], int]:
    collected: List[str] = []
    i = index

    if _HTML_TABLE_START_RE.search(lines[i]):
        while i < len(lines):
            line = lines[i].strip()
            if line:
                collected.append(line)
            i += 1
            if _HTML_TABLE_END_RE.search(line):
                break
        return collected, i

    while i < len(lines):
        line = lines[i].strip()
        if not line or not _PIPE_TABLE_RE.match(line):
            break
        collected.append(line)
        i += 1

    return collected, i


def _collect_paragraph(lines: Sequence[str], index: int) -> Tuple[List[str], int]:
    collected: List[str] = []
    i = index

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            break
        if _parse_heading(line) or _is_table_start(lines, i):
            break
        collected.append(line)
        i += 1

    return collected, i


def _join_paragraph_lines(lines: Sequence[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(line.strip() for line in lines if line.strip())).strip()
