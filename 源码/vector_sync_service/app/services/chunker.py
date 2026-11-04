"""Chunk builder for parsed full-text data."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from app.config import settings
from app.models.chunk import Chunk
from app.services.fulltext_parser import ParsedBlock, ParsedFullText, parse_mmd
from app.services.semantic_splitter import semantic_sliding_windows


TextValue = Optional[Union[str, Sequence[str]]]


@dataclass(frozen=True)
class _ChunkDraft:
    chunk_type: str
    text: str
    char_start: int
    char_end: int
    section_path: Optional[str] = None
    dynamic_fields: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


def build_document_chunks(
    doc_id: str,
    title: TextValue = None,
    abstract: TextValue = None,
    mmd_text: Optional[str] = None,
    parsed: Optional[ParsedFullText] = None,
    max_chars: int = settings.CHUNK_MAX_CHARS,
    overlap_chars: int = settings.CHUNK_OVERLAP_CHARS,
    min_chars: int = settings.CHUNK_MIN_CHARS,
    max_chunks_per_doc: int = settings.CHUNK_MAX_PER_DOC,
    enable_semantic_split: bool = settings.CHUNK_SEMANTIC_SPLIT_ENABLED,
    semantic_search_window: int = settings.CHUNK_SEMANTIC_SEARCH_WINDOW,
    enable_paragraph_merge: bool = settings.CHUNK_PARAGRAPH_MERGE_ENABLED,
    enable_context_overlap: bool = settings.CHUNK_CONTEXT_OVERLAP_ENABLED,
    context_overlap_chars: int = settings.CHUNK_CONTEXT_OVERLAP_CHARS,
    target_chars: int = settings.CHUNK_TARGET_CHARS,
    tail_merge_chars: int = settings.CHUNK_TAIL_MERGE_CHARS,
    attach_section_headings: bool = settings.CHUNK_SECTION_TITLE_ATTACH_ENABLED,
) -> List[Chunk]:
    """Build deterministic document chunks from title/abstract and optional full text.

    Paragraph-level optimisations (all enabled by default, toggle via config):

    * **Semantic splitting** — split long paragraphs at sentence/clause/word
      boundaries instead of arbitrary character offsets.
    * **Section-aware aggregation** — merge adjacent paragraphs in the same
      section toward *target_chars* without crossing structural boundaries.
    * **Cross-paragraph context** — prepend/append a few characters from
      neighbouring paragraphs so that paragraph-boundary semantics are not lost.
    """

    if not doc_id or not str(doc_id).strip():
        raise ValueError("doc_id is required")

    _validate_split_config(
        max_chars,
        overlap_chars,
        min_chars,
        max_chunks_per_doc,
        target_chars,
        tail_merge_chars,
    )
    effective_target_chars = min(target_chars, max_chars)

    drafts: List[_ChunkDraft] = []
    title_abstract = _join_title_abstract(title, abstract)
    if title_abstract:
        drafts.extend(
            _split_draft(
                block_type="title_abstract",
                text=title_abstract,
                base_start=0,
                section_path="title_abstract",
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )
        )

    parsed_fulltext = parsed
    if parsed_fulltext is None and mmd_text:
        parsed_fulltext = parse_mmd(mmd_text)

    if parsed_fulltext:
        blocks = list(parsed_fulltext.blocks)

        # Stage A: aggregate adjacent paragraphs within the same section
        # before splitting. Section boundaries and non-paragraph blocks are
        # hard boundaries; section headings are attached to the first body
        # block instead of becoming tiny standalone vectors.
        merge_map: Dict[int, List[int]] = {}
        if enable_paragraph_merge:
            blocks, merge_map = _aggregate_section_blocks(
                blocks,
                target_chars=effective_target_chars,
                max_chars=max_chars,
                tail_merge_chars=tail_merge_chars,
                attach_section_headings=attach_section_headings,
            )

        for new_idx, block in enumerate(blocks):
            block_dynamic: Dict[str, Any] = {}
            if new_idx in merge_map:
                block_dynamic["merged_from_blocks"] = merge_map[new_idx]

            drafts.extend(
                _split_block(
                    block,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                    min_chars=min_chars,
                    dynamic_fields=block_dynamic if block_dynamic else None,
                    enable_semantic_split=enable_semantic_split,
                    search_window=semantic_search_window,
                )
            )

        # Stage B: cross-paragraph context overlap (after splitting)
        if enable_context_overlap and context_overlap_chars > 0:
            drafts = _add_paragraph_context(drafts, context_overlap_chars, max_chars=max_chars)

    drafts = _limit_drafts(drafts, max_chunks_per_doc)
    return [_build_chunk(doc_id=str(doc_id), index=index, draft=draft) for index, draft in enumerate(drafts)]


def _join_title_abstract(title: TextValue, abstract: TextValue) -> str:
    parts = [*_coerce_text_parts(title), *_coerce_text_parts(abstract)]
    return "\n\n".join(parts).strip()


def _coerce_text_parts(value: TextValue) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    parts: List[str] = []
    for item in value:
        if item is None:
            continue
        stripped = str(item).strip()
        if stripped:
            parts.append(stripped)
    return parts


def _split_block(
    block: ParsedBlock,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    dynamic_fields: Optional[Dict[str, Any]] = None,
    enable_semantic_split: bool = False,
    search_window: int = 200,
) -> List[_ChunkDraft]:
    return _split_draft(
        block_type=block.block_type,
        text=block.text,
        base_start=block.char_start,
        section_path=block.section_path,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        dynamic_fields=dynamic_fields,
        enable_semantic_split=enable_semantic_split,
        search_window=search_window,
    )


def _split_draft(
    block_type: str,
    text: str,
    base_start: int,
    section_path: Optional[str],
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    dynamic_fields: Optional[Dict[str, Any]] = None,
    enable_semantic_split: bool = False,
    search_window: int = 200,
) -> List[_ChunkDraft]:
    stripped = text.strip()
    if not stripped:
        return []

    offset_delta = text.find(stripped)
    base = base_start + max(offset_delta, 0)

    # Choose splitter: semantic for paragraphs (when enabled), flat for everything else
    if block_type == "paragraph" and enable_semantic_split:
        windows = semantic_sliding_windows(
            stripped, max_chars, overlap_chars, min_chars, search_window
        )
    else:
        windows = _sliding_windows(stripped, max_chars, overlap_chars, min_chars)

    base_fields = dict(dynamic_fields or {})

    drafts: List[_ChunkDraft] = []
    for local_start, local_end, segment in windows:
        drafts.append(
            _ChunkDraft(
                chunk_type=block_type,
                text=segment,
                char_start=base + local_start,
                char_end=base + local_end,
                section_path=section_path,
                dynamic_fields=dict(base_fields),
            )
        )
    return drafts


def _sliding_windows(
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> Iterable[Tuple[int, int, str]]:
    if len(text) <= max_chars:
        yield 0, len(text), text
        return

    step = max_chars - overlap_chars
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        next_start = start + step
        if end < len(text) and len(text) - next_start < min_chars:
            end = len(text)
        yield start, end, text[start:end]
        if end == len(text):
            break
        start += step


def _aggregate_section_blocks(
    blocks: List[ParsedBlock],
    *,
    target_chars: int,
    max_chars: int,
    tail_merge_chars: int,
    attach_section_headings: bool,
) -> Tuple[List[ParsedBlock], Dict[int, List[int]]]:
    """Aggregate body paragraphs without crossing section boundaries.

    ``target_chars`` is a soft target. A paragraph is appended while the
    aggregate remains within ``max_chars``; otherwise the current aggregate
    is emitted and a new one starts. Tables, references, and section changes
    flush the aggregate. Section headings are attached to the first following
    block, which keeps their retrieval context without creating tiny vectors.
    """

    if not blocks:
        return [], {}

    result: List[ParsedBlock] = []
    merge_map: Dict[int, List[int]] = {}
    paragraph_buffer: List[Tuple[List[int], ParsedBlock]] = []
    pending_headings: List[Tuple[int, ParsedBlock]] = []
    pending_reference_heading: Optional[Tuple[int, ParsedBlock]] = None

    def _block_length(items: Sequence[ParsedBlock]) -> int:
        return sum(len(item.text) for item in items) + max(len(items) - 1, 0) * 2

    def _flush_paragraphs() -> None:
        if not paragraph_buffer:
            return

        indices = [index for block_indices, _ in paragraph_buffer for index in block_indices]
        items = [item for _, item in paragraph_buffer]
        combined = ParsedBlock(
            block_type="paragraph",
            text="\n\n".join(item.text for item in items),
            char_start=items[0].char_start,
            char_end=items[-1].char_end,
            section_path=items[0].section_path or items[-1].section_path,
        )
        _append_result(combined, indices)
        paragraph_buffer.clear()

    def _append_result(block: ParsedBlock, indices: List[int]) -> None:
        # Merge a short final aggregate into the preceding paragraph only when
        # it is still inside the same section and the hard maximum is kept.
        if (
            block.block_type == "paragraph"
            and len(block.text) < tail_merge_chars
            and result
            and result[-1].block_type == "paragraph"
            and result[-1].section_path == block.section_path
        ):
            previous = result[-1]
            merged_text = previous.text + "\n\n" + block.text
            if len(merged_text) <= max_chars:
                result[-1] = ParsedBlock(
                    block_type="paragraph",
                    text=merged_text,
                    char_start=previous.char_start,
                    char_end=block.char_end,
                    section_path=previous.section_path,
                )
                merge_map[len(result) - 1] = merge_map.get(len(result) - 1, []) + indices
                return

        result.append(block)
        if len(indices) > 1:
            merge_map[len(result) - 1] = list(indices)

    def _attach_pending_heading(block: ParsedBlock, indices: List[int]) -> Tuple[ParsedBlock, List[int]]:
        if not pending_headings or not attach_section_headings:
            return block, indices

        heading_items = [item for _, item in pending_headings]
        heading_text = "\n".join(item.text for item in heading_items if item.text.strip())
        pending_indices = [index for index, _ in pending_headings]
        pending_headings.clear()
        if not heading_text:
            return block, indices

        return (
            ParsedBlock(
                block_type=block.block_type,
                text=f"{heading_text}\n\n{block.text}",
                char_start=heading_items[0].char_start,
                char_end=block.char_end,
                section_path=block.section_path or heading_items[-1].section_path,
            ),
            pending_indices + indices,
        )

    def _flush_pending_headings() -> None:
        if not pending_headings:
            return
        if attach_section_headings:
            for index, heading in pending_headings:
                _append_result(heading, [index])
        pending_headings.clear()

    def _flush_reference_heading() -> None:
        nonlocal pending_reference_heading
        if pending_reference_heading is not None:
            index, heading = pending_reference_heading
            _append_result(heading, [index])
            pending_reference_heading = None

    def _is_reference_heading(block: ParsedBlock) -> bool:
        if block.block_type != "reference" or len(block.text) > 120:
            return False
        normalized = re.sub(r"\s+", " ", block.text.strip().strip(":：").lower())
        return normalized in {
            "reference",
            "references",
            "bibliography",
            "literature cited",
            "works cited",
            "参考文献",
            "參考文獻",
        }

    for index, block in enumerate(blocks):
        if block.block_type == "section":
            _flush_paragraphs()
            _flush_reference_heading()
            if attach_section_headings:
                pending_headings.append((index, block))
            else:
                _append_result(block, [index])
            continue

        if block.block_type == "paragraph":
            _flush_reference_heading()
            if pending_headings:
                block, indices = _attach_pending_heading(block, [index])
            else:
                indices = [index]

            if paragraph_buffer:
                current_section = paragraph_buffer[0][1].section_path
                if current_section != block.section_path:
                    _flush_paragraphs()

            candidate_length = _block_length([item for _, item in paragraph_buffer] + [block])
            if paragraph_buffer and candidate_length > max_chars:
                _flush_paragraphs()
            paragraph_buffer.append((indices, block))

            # Keep accumulating until the soft target; after reaching it, the
            # next block may still fit up to the hard maximum.
            if _block_length([item for _, item in paragraph_buffer]) >= target_chars:
                continue
            continue

        _flush_paragraphs()

        if _is_reference_heading(block):
            _flush_reference_heading()
            pending_reference_heading = (index, block)
            continue

        if pending_reference_heading is not None and block.block_type == "reference":
            heading_index, heading = pending_reference_heading
            pending_reference_heading = None
            block = ParsedBlock(
                block_type="reference",
                text=f"{heading.text}\n\n{block.text}",
                char_start=heading.char_start,
                char_end=block.char_end,
                section_path=block.section_path or heading.section_path,
            )
            _append_result(block, [heading_index, index])
            continue

        _flush_reference_heading()
        block, indices = _attach_pending_heading(block, [index])
        _append_result(block, indices)

    _flush_paragraphs()
    _flush_reference_heading()
    _flush_pending_headings()
    return result, merge_map


def _add_paragraph_context(
    drafts: List[_ChunkDraft],
    context_chars: int,
    max_chars: Optional[int] = None,
) -> List[_ChunkDraft]:
    """Add cross-paragraph micro-context to paragraph chunks.

    For each paragraph draft, prepend the last *context_chars* of the
    nearest preceding paragraph draft and append the first *context_chars*
    of the nearest following paragraph draft.  ``charStart`` / ``charEnd``
    are kept unchanged (they still point to the original paragraph range).

    Metadata is recorded in ``dynamic_fields["context_overlap"]``.
    """
    if context_chars <= 0 or not drafts:
        return list(drafts)

    # Pre-compute neighbour indices to avoid O(n²) scanning
    paragraph_indices = [idx for idx, d in enumerate(drafts) if d.chunk_type == "paragraph"]

    prev_map: Dict[int, Optional[int]] = {}
    next_map: Dict[int, Optional[int]] = {}
    for pi, idx in enumerate(paragraph_indices):
        prev_map[idx] = paragraph_indices[pi - 1] if pi > 0 else None
        next_map[idx] = paragraph_indices[pi + 1] if pi + 1 < len(paragraph_indices) else None

    result: List[_ChunkDraft] = []
    for idx, draft in enumerate(drafts):
        if draft.chunk_type != "paragraph":
            result.append(draft)
            continue

        prev_idx = prev_map.get(idx)
        next_idx = next_map.get(idx)

        prefix_chars = 0
        suffix_chars = 0
        prefix = ""
        suffix = ""

        available_context = (
            max(max_chars - len(draft.text), 0)
            if max_chars is not None
            else context_chars * 2
        )
        prev_budget = min(context_chars, available_context // 2)
        next_budget = min(context_chars, available_context - prev_budget)

        if prev_idx is not None and prev_budget > 0:
            if drafts[prev_idx].section_path == draft.section_path:
                prev_text = drafts[prev_idx].text
                take = min(prev_budget, len(prev_text))
                if take > 0:
                    prefix_chars = take
                    prefix = prev_text[-take:]

        if next_idx is not None and next_budget > 0:
            if drafts[next_idx].section_path == draft.section_path:
                next_text = drafts[next_idx].text
                take = min(next_budget, len(next_text))
                if take > 0:
                    suffix_chars = take
                    suffix = next_text[:take]

        new_text = prefix + draft.text + suffix if (prefix or suffix) else draft.text
        new_dynamic = dict(draft.dynamic_fields)
        new_dynamic["context_overlap"] = {"prev_chars": prefix_chars, "next_chars": suffix_chars}

        result.append(
            _ChunkDraft(
                chunk_type=draft.chunk_type,
                text=new_text,
                char_start=draft.char_start,
                char_end=draft.char_end,
                section_path=draft.section_path,
                dynamic_fields=new_dynamic,
            )
        )

    return result


def _build_chunk(doc_id: str, index: int, draft: _ChunkDraft) -> Chunk:
    return Chunk(
        id=f"{doc_id}_{index}",
        doc_id=doc_id,
        chunkIndex=index,
        chunkType=draft.chunk_type,
        text=draft.text,
        charStart=draft.char_start,
        charEnd=draft.char_end,
        sectionPath=draft.section_path,
        tokenCount=_estimate_tokens(draft.text),
        text_embed=None,
        dynamic_fields=draft.dynamic_fields,
    )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text))
    symbols = len(re.findall(r"[^\s\w\u4e00-\u9fff]", text))
    return cjk_chars + latin_words + symbols


def _limit_drafts(drafts: List[_ChunkDraft], max_chunks_per_doc: int) -> List[_ChunkDraft]:
    if max_chunks_per_doc <= 0 or len(drafts) <= max_chunks_per_doc:
        return drafts
    return drafts[:max_chunks_per_doc]


def _validate_split_config(
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    max_chunks_per_doc: int,
    target_chars: int,
    tail_merge_chars: int,
) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    if min_chars < 0:
        raise ValueError("min_chars must not be negative")
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if tail_merge_chars < 0:
        raise ValueError("tail_merge_chars must not be negative")
    if max_chunks_per_doc < 0:
        raise ValueError("max_chunks_per_doc must not be negative")
