from app.services.fulltext_parser import parse_mmd


def test_parse_mmd_cleans_asset_references_and_keeps_markdown_structure():
    mmd = """
# Article Title

![](figures/figure-1.jpg)

This introduction paragraph should remain.

## Methods

![](tables/table-1.html)

| Name | Value |
| --- | --- |
| alpha | beta |

## References

[1] Smith J. Example reference.
"""

    parsed = parse_mmd(mmd)

    assert "figures/figure-1.jpg" not in parsed.canonical_text
    assert "tables/table-1.html" not in parsed.canonical_text
    assert "Article Title" in parsed.canonical_text
    assert "This introduction paragraph should remain." in parsed.canonical_text
    assert "| alpha | beta |" in parsed.canonical_text
    assert "[1] Smith J. Example reference." in parsed.canonical_text

    table_blocks = [block for block in parsed.blocks if block.block_type == "table"]
    assert len(table_blocks) == 1
    assert table_blocks[0].section_path == "Article Title/Methods"
    assert "| Name | Value |" in table_blocks[0].text

    reference_blocks = [block for block in parsed.blocks if block.block_type == "reference"]
    assert len(reference_blocks) == 2
    assert reference_blocks[0].text == "References"
    assert reference_blocks[1].text == "[1] Smith J. Example reference."
    assert all(block.section_path == "Article Title/References" for block in reference_blocks)


def test_parse_mmd_tracks_nested_section_paths_and_offsets():
    parsed = parse_mmd(
        """
# Root

Root paragraph.

## Child

Child paragraph.
"""
    )

    child_section = next(block for block in parsed.blocks if block.text == "Child")
    child_paragraph = next(block for block in parsed.blocks if block.text == "Child paragraph.")

    assert child_section.block_type == "section"
    assert child_section.section_path == "Root/Child"
    assert child_paragraph.block_type == "paragraph"
    assert child_paragraph.section_path == "Root/Child"
    assert parsed.canonical_text[child_paragraph.char_start:child_paragraph.char_end] == child_paragraph.text
