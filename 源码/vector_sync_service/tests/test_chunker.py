from app.services.chunker import build_document_chunks
from app.services.fulltext_parser import ParsedBlock, ParsedFullText


def test_build_document_chunks_without_fulltext_returns_title_abstract_only():
    chunks = build_document_chunks(
        "DOC-001",
        title=["A Study on Stable Chunks"],
        abstract=["This abstract is kept as display text."],
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunkId == "DOC-001_0"
    assert chunk.docId == "DOC-001"
    assert chunk.chunkIndex == 0
    assert chunk.chunkType == "title_abstract"
    assert chunk.text == "A Study on Stable Chunks\n\nThis abstract is kept as display text."
    assert chunk.charStart == 0
    assert chunk.charEnd == len(chunk.text)
    assert chunk.sectionPath == "title_abstract"
    assert chunk.embedding is None


def test_build_document_chunks_from_mmd_preserves_types_paths_and_stable_ids():
    mmd = """
# Article Title

Intro paragraph.

## Methods

| Name | Value |
| --- | --- |
| alpha | beta |

## References

[1] Smith J. Example reference.
"""

    chunks = build_document_chunks(
        "DOC-002",
        title="Article Title",
        abstract="Short abstract.",
        mmd_text=mmd,
    )

    assert [chunk.chunkIndex for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunkId for chunk in chunks] == [f"DOC-002_{i}" for i in range(len(chunks))]
    assert all(chunk.docId == "DOC-002" for chunk in chunks)
    assert all(chunk.embedding is None for chunk in chunks)

    chunk_types = [chunk.chunkType for chunk in chunks]
    assert chunk_types[0] == "title_abstract"
    assert "section" not in chunk_types
    assert "paragraph" in chunk_types
    assert "table" in chunk_types
    assert "reference" in chunk_types

    table_chunk = next(chunk for chunk in chunks if chunk.chunkType == "table")
    assert table_chunk.sectionPath == "Article Title/Methods"
    assert "| alpha | beta |" in table_chunk.text

    reference_chunks = [chunk for chunk in chunks if chunk.chunkType == "reference"]
    assert [chunk.text for chunk in reference_chunks] == [
        "References\n\n[1] Smith J. Example reference.",
    ]


def test_section_aware_aggregation_targets_two_thousand_chars_without_crossing_sections():
    paragraphs = [
        "A" * 700,
        "B" * 700,
        "C" * 700,
        "D" * 700,
    ]
    parsed = ParsedFullText(
        canonical_text="",
        blocks=[
            ParsedBlock(block_type="section", text="Methods", char_start=0, char_end=7, section_path="Methods"),
            ParsedBlock(block_type="paragraph", text=paragraphs[0], char_start=9, char_end=709, section_path="Methods"),
            ParsedBlock(block_type="paragraph", text=paragraphs[1], char_start=711, char_end=1411, section_path="Methods"),
            ParsedBlock(block_type="paragraph", text=paragraphs[2], char_start=1413, char_end=2113, section_path="Methods"),
            ParsedBlock(block_type="section", text="Results", char_start=2115, char_end=2122, section_path="Results"),
            ParsedBlock(block_type="paragraph", text=paragraphs[3], char_start=2124, char_end=2824, section_path="Results"),
        ],
    )

    chunks = build_document_chunks(
        "DOC-AGG",
        parsed=parsed,
        target_chars=2000,
        max_chars=2500,
        overlap_chars=200,
        min_chars=300,
        enable_semantic_split=False,
        enable_context_overlap=False,
    )

    assert [chunk.chunkType for chunk in chunks] == ["paragraph", "paragraph"]
    assert chunks[0].sectionPath == "Methods"
    assert chunks[1].sectionPath == "Results"
    assert "Methods" in chunks[0].text
    assert len(chunks[0].text) <= 2500
    assert len(chunks[0].text) >= 1500
    assert "Results" in chunks[1].text
    assert chunks[0].dynamic_fields["merged_from_blocks"] == [0, 1, 2, 3]


def test_section_heading_without_body_is_preserved():
    parsed = ParsedFullText(
        canonical_text="Title",
        blocks=[
            ParsedBlock(block_type="section", text="Title", char_start=0, char_end=5, section_path="Title"),
        ],
    )

    chunks = build_document_chunks("DOC-AGG-EMPTY", parsed=parsed, enable_context_overlap=False)

    assert len(chunks) == 1
    assert chunks[0].chunkType == "section"
    assert chunks[0].text == "Title"


def test_build_document_chunks_splits_long_blocks_with_overlap_and_offsets():
    text = "A" * 3400
    parsed = ParsedFullText(
        canonical_text=text,
        blocks=[
            ParsedBlock(
                block_type="paragraph",
                text=text,
                char_start=0,
                char_end=len(text),
                section_path="Long Section",
            )
        ],
    )

    chunks = build_document_chunks(
        "DOC-003",
        parsed=parsed,
        max_chars=1500,
        overlap_chars=200,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=False,
    )

    assert len(chunks) == 3
    assert [chunk.chunkId for chunk in chunks] == ["DOC-003_0", "DOC-003_1", "DOC-003_2"]
    assert [chunk.charStart for chunk in chunks] == [0, 1300, 2600]
    assert [chunk.charEnd for chunk in chunks] == [1500, 2800, 3400]
    assert all(len(chunk.text) <= 1500 for chunk in chunks)
    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert chunks[1].text[-200:] == chunks[2].text[:200]
    assert all(chunk.chunkType == "paragraph" for chunk in chunks)
    assert all(chunk.sectionPath == "Long Section" for chunk in chunks)
    assert all(chunk.embedding is None for chunk in chunks)


def test_build_document_chunks_merges_tiny_tail_when_min_chars_is_set():
    text = "A" * 2850
    parsed = ParsedFullText(
        canonical_text=text,
        blocks=[
            ParsedBlock(
                block_type="paragraph",
                text=text,
                char_start=0,
                char_end=len(text),
                section_path="Long Section",
            )
        ],
    )

    chunks = build_document_chunks(
        "DOC-003B",
        parsed=parsed,
        max_chars=1500,
        overlap_chars=200,
        min_chars=300,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=False,
    )

    assert len(chunks) == 2
    assert [chunk.charStart for chunk in chunks] == [0, 1300]
    assert [chunk.charEnd for chunk in chunks] == [1500, 2850]
    assert len(chunks[-1].text) == 1550


def test_build_document_chunks_limits_total_chunks_per_doc():
    text = "A" * 12500
    parsed = ParsedFullText(
        canonical_text=text,
        blocks=[
            ParsedBlock(
                block_type="paragraph",
                text=text,
                char_start=0,
                char_end=len(text),
                section_path="Long Section",
            )
        ],
    )

    chunks = build_document_chunks(
        "DOC-003C",
        parsed=parsed,
        max_chars=1000,
        overlap_chars=0,
        min_chars=0,
        max_chunks_per_doc=10,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=False,
    )

    assert len(chunks) == 10
    assert [chunk.chunkId for chunk in chunks] == [f"DOC-003C_{i}" for i in range(10)]
    assert chunks[-1].charStart == 9000
    assert chunks[-1].charEnd == 10000
    assert all(len(chunk.text) <= 1000 for chunk in chunks)


def test_build_document_chunks_can_disable_total_chunk_limit():
    text = "A" * 12500
    parsed = ParsedFullText(
        canonical_text=text,
        blocks=[
            ParsedBlock(
                block_type="paragraph",
                text=text,
                char_start=0,
                char_end=len(text),
                section_path="Long Section",
            )
        ],
    )

    chunks = build_document_chunks(
        "DOC-003D",
        parsed=parsed,
        max_chars=1000,
        overlap_chars=0,
        min_chars=0,
        max_chunks_per_doc=0,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=False,
    )

    assert len(chunks) == 13
    assert chunks[-1].charEnd == len(text)


def test_build_document_chunks_empty_input_returns_empty_list():
    assert build_document_chunks("DOC-004") == []


# ---------------------------------------------------------------------------
# Short-paragraph merging (Stage A)
# ---------------------------------------------------------------------------


def test_short_paragraphs_merged_into_one():
    """Two paragraphs below min_chars merge into a single chunk."""
    parsed = ParsedFullText(
        canonical_text="Short A.\n\nShort B.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="Short A.", char_start=0, char_end=8, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="Short B.", char_start=10, char_end=18, section_path="S1"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-M1", parsed=parsed,
        max_chars=5000, min_chars=300,
        enable_semantic_split=False,
        enable_paragraph_merge=True,
        enable_context_overlap=False,
    )
    assert len(chunks) == 1
    assert chunks[0].chunkType == "paragraph"
    assert "Short A." in chunks[0].text
    assert "Short B." in chunks[0].text
    assert chunks[0].charStart == 0
    assert chunks[0].charEnd == 18


def test_short_paragraph_merge_respects_max_chars():
    """Two short paragraphs that would exceed max_chars when merged stay separate."""
    big_a = "X" * 4000
    big_b = "Y" * 4000
    parsed = ParsedFullText(
        canonical_text=f"{big_a}\n\n{big_b}",
        blocks=[
            ParsedBlock(block_type="paragraph", text=big_a, char_start=0, char_end=4000, section_path="S1"),
            ParsedBlock(block_type="paragraph", text=big_b, char_start=4002, char_end=8002, section_path="S1"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-M2", parsed=parsed,
        max_chars=5000, min_chars=300,
        enable_semantic_split=False,
        enable_paragraph_merge=True,
        enable_context_overlap=False,
    )
    assert len(chunks) == 2


def test_merge_skips_non_paragraph_blocks():
    """Only paragraph blocks participate in merging."""
    parsed = ParsedFullText(
        canonical_text="Intro.\n\n# Section Title\n\nShort paragraph.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="Short paragraph.", char_start=0, char_end=15, section_path="S1"),
            ParsedBlock(block_type="section", text="Section Title", char_start=17, char_end=30, section_path="S1/Section Title"),
            ParsedBlock(block_type="paragraph", text="Another short.", char_start=32, char_end=45, section_path="S1/Section Title"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-M3", parsed=parsed,
        max_chars=5000, min_chars=300,
        enable_semantic_split=False,
        enable_paragraph_merge=True,
        enable_context_overlap=False,
    )
    # The two paragraphs are separated by a section block — they should NOT merge
    chunk_types = [c.chunkType for c in chunks]
    assert chunk_types.count("paragraph") == 2


def test_merged_paragraphs_carry_merge_metadata():
    """Merged chunks record original block indices in dynamic_fields."""
    parsed = ParsedFullText(
        canonical_text="A.\n\nB.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="A.", char_start=0, char_end=2, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="B.", char_start=4, char_end=6, section_path="S1"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-M4", parsed=parsed,
        max_chars=5000, min_chars=300,
        enable_semantic_split=False,
        enable_paragraph_merge=True,
        enable_context_overlap=False,
    )
    assert len(chunks) == 1
    assert "merged_from_blocks" in chunks[0].dynamic_fields
    assert chunks[0].dynamic_fields["merged_from_blocks"] == [0, 1]


# ---------------------------------------------------------------------------
# Paragraph context overlap (Stage B)
# ---------------------------------------------------------------------------


def test_paragraph_context_adds_neighbor_text():
    """Paragraph chunk text includes prefix/suffix from neighbours, charStart/charEnd unchanged."""
    parsed = ParsedFullText(
        canonical_text="First real paragraph text here.\n\nMiddle paragraph content.\n\nLast paragraph text.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="First real paragraph text here.", char_start=0, char_end=31, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="Middle paragraph content.", char_start=33, char_end=58, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="Last paragraph text.", char_start=60, char_end=79, section_path="S1"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-C1", parsed=parsed,
        max_chars=5000, min_chars=0,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=True,
        context_overlap_chars=50,
    )
    assert len(chunks) == 3

    # Middle chunk should have context from both sides
    middle = chunks[1]
    assert middle.chunkType == "paragraph"
    # charStart/charEnd must remain unchanged (point to original paragraph range)
    assert middle.charStart == 33
    assert middle.charEnd == 58
    # Text must contain the original content
    assert "Middle paragraph content." in middle.text

    # Check context_overlap metadata
    assert "context_overlap" in middle.dynamic_fields
    assert middle.dynamic_fields["context_overlap"]["prev_chars"] > 0
    assert middle.dynamic_fields["context_overlap"]["next_chars"] > 0


def test_paragraph_context_no_neighbor_for_boundary_chunks():
    """First paragraph gets only suffix, last gets only prefix."""
    parsed = ParsedFullText(
        canonical_text="First.\n\nLast.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="First.", char_start=0, char_end=6, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="Last.", char_start=8, char_end=13, section_path="S1"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-C2", parsed=parsed,
        max_chars=5000, min_chars=0,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=True,
        context_overlap_chars=50,
    )
    assert len(chunks) == 2
    first = chunks[0]
    last = chunks[1]
    assert first.dynamic_fields["context_overlap"]["prev_chars"] == 0
    assert first.dynamic_fields["context_overlap"]["next_chars"] > 0
    assert last.dynamic_fields["context_overlap"]["prev_chars"] > 0
    assert last.dynamic_fields["context_overlap"]["next_chars"] == 0


def test_context_overlap_respects_short_neighbor():
    """When a neighbour is shorter than context_chars, use whatever is available."""
    parsed = ParsedFullText(
        canonical_text="Hi.\n\nA much longer paragraph with more content here.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="Hi.", char_start=0, char_end=3, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="A much longer paragraph with more content here.", char_start=5, char_end=55, section_path="S1"),
        ],
    )
    chunks = build_document_chunks(
        "DOC-C3", parsed=parsed,
        max_chars=5000, min_chars=0,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=True,
        context_overlap_chars=50,
    )
    # The first chunk's prev neighbour text is only 3 chars
    long_chunk = chunks[1]
    assert long_chunk.dynamic_fields["context_overlap"]["prev_chars"] == 3


def test_context_overlap_does_not_cross_section_boundaries():
    parsed = ParsedFullText(
        canonical_text="A paragraph.\n\nB paragraph.",
        blocks=[
            ParsedBlock(block_type="paragraph", text="A paragraph.", char_start=0, char_end=12, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="B paragraph.", char_start=14, char_end=26, section_path="S2"),
        ],
    )

    chunks = build_document_chunks(
        "DOC-C4", parsed=parsed,
        max_chars=5000, min_chars=0,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=True,
        context_overlap_chars=50,
    )

    assert chunks[0].dynamic_fields["context_overlap"]["next_chars"] == 0
    assert chunks[1].dynamic_fields["context_overlap"]["prev_chars"] == 0


def test_context_overlap_respects_hard_max_chars():
    parsed = ParsedFullText(
        canonical_text="",
        blocks=[
            ParsedBlock(block_type="paragraph", text="A" * 2000, char_start=0, char_end=2000, section_path="S1"),
            ParsedBlock(block_type="paragraph", text="B" * 2000, char_start=2002, char_end=4002, section_path="S1"),
        ],
    )

    chunks = build_document_chunks(
        "DOC-C5", parsed=parsed,
        target_chars=2000, max_chars=2500, min_chars=0,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=True,
        context_overlap_chars=50,
    )

    assert all(len(chunk.text) <= 2500 for chunk in chunks)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_all_features_off_matches_current_behavior():
    """With all new features disabled, output matches pre-optimization behaviour."""
    text = "A" * 3400
    parsed = ParsedFullText(
        canonical_text=text,
        blocks=[
            ParsedBlock(
                block_type="paragraph", text=text, char_start=0, char_end=len(text), section_path="S1"
            )
        ],
    )
    chunks = build_document_chunks(
        "DOC-BC1", parsed=parsed,
        max_chars=1500, overlap_chars=200,
        enable_semantic_split=False,
        enable_paragraph_merge=False,
        enable_context_overlap=False,
    )
    assert len(chunks) == 3
    assert [c.charStart for c in chunks] == [0, 1300, 2600]
    assert [c.charEnd for c in chunks] == [1500, 2800, 3400]
    assert all(len(c.text) <= 1500 for c in chunks)
    assert chunks[0].text[-200:] == chunks[1].text[:200]


def test_semantic_split_enabled_uses_boundary_detection():
    """When enabled, semantic splitting produces chunks at sentence boundaries."""
    base = "A" * 200 + ". " + "B" * 200 + ". " + "C" * 200 + ". " + "D" * 200 + ". " + "E" * 5000
    parsed = ParsedFullText(
        canonical_text=base,
        blocks=[
            ParsedBlock(block_type="paragraph", text=base, char_start=0, char_end=len(base), section_path="S1")
        ],
    )
    chunks = build_document_chunks(
        "DOC-SS1", parsed=parsed,
        max_chars=1500, overlap_chars=200, min_chars=300,
        enable_semantic_split=True,
        enable_paragraph_merge=False,
        enable_context_overlap=False,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 1500
        assert c.text == base[c.charStart:c.charEnd]
