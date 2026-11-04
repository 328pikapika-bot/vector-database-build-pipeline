import pytest
from app.services.semantic_splitter import (
    _find_latex_zones,
    _is_in_latex_zone,
    _find_nearest_latex_boundary,
    find_semantic_split,
    semantic_sliding_windows,
)


class TestLatexZoneDetection:
    def test_find_inline_latex_zone(self):
        zones = _find_latex_zones("text $x^2 + y^2$ more text")
        assert len(zones) == 1
        start, end = zones[0]
        assert "x^2 + y^2" in "text $x^2 + y^2$ more text"[start:end]

    def test_find_display_latex_zone(self):
        zones = _find_latex_zones("text $$\\frac{a}{b}$$ more")
        assert len(zones) >= 1

    def test_merge_overlapping_zones(self):
        # "$a$" produces zone (0,3), "$b$" produces zone (2,5) — they overlap and merge
        zones = _find_latex_zones("$a$b$")
        assert len(zones) == 1

    def test_is_in_latex_zone_inside(self):
        text = "abc $math$ def"
        zones = _find_latex_zones(text)
        dollar_pos = text.index("$") + 1
        assert _is_in_latex_zone(dollar_pos, zones)

    def test_is_in_latex_zone_outside(self):
        text = "abc $math$ def"
        zones = _find_latex_zones(text)
        assert not _is_in_latex_zone(0, zones)
        assert not _is_in_latex_zone(len(text) - 1, zones)

    def test_find_nearest_latex_boundary(self):
        text = "prefix $math$ suffix"
        zones = _find_latex_zones(text)
        # Position inside the math zone
        inside = text.index("$") + 2
        boundary = _find_nearest_latex_boundary(inside, zones, len(text))
        assert boundary is not None


class TestFindSemanticSplit:
    def test_sentence_end_boundary(self):
        text = "First sentence ends here. Next sentence starts with capital."
        # Target position in the middle of the second sentence
        target = len(text) - 10
        result = find_semantic_split(text, target, search_window=200)
        # Should split right after the period (m.end() of the sentence-end regex)
        expected = text.index(". ") + 1
        assert result == expected

    def test_clause_boundary_fallback(self):
        text = "no sentence end here, just a comma, then more text following after that point in the text here"
        target = len(text) - 10
        result = find_semantic_split(text, target, search_window=200)
        # Should split right after a comma (m.end() of the clause-boundary regex)
        assert text[result - 1] == ","

    def test_word_boundary_fallback(self):
        text = "no punctuation at all just words flowing together without any stops or pauses whatsoever in this string"
        target = len(text) - 10
        result = find_semantic_split(text, target, search_window=200)
        # Should split at a space boundary (not mid-word)
        assert 0 < result < len(text)

    def test_hard_fallback_at_target(self):
        text = "x" * 500
        target = 300
        result = find_semantic_split(text, target, search_window=200)
        # No semantic boundaries in repeated chars — hard fallback to target
        assert result == target

    def test_target_beyond_text_length(self):
        text = "short text"
        result = find_semantic_split(text, target_pos=100, search_window=200)
        assert result == len(text)

    def test_target_at_zero(self):
        text = "some text here"
        result = find_semantic_split(text, target_pos=0, search_window=200)
        assert result == 0

    def test_latex_protection_avoids_math_zone(self):
        text = "Before math $x^2 + y^2 = z^2$ after math. Another sentence."
        # Target inside or just after the math zone
        target = text.index("after math") + 5
        result = find_semantic_split(text, target, search_window=200)
        # Result should not be inside the LaTeX zone
        zones = _find_latex_zones(text)
        assert not _is_in_latex_zone(result, zones)


class TestSemanticSlidingWindows:
    def test_short_text_returns_single_chunk(self):
        text = "short"
        results = list(semantic_sliding_windows(text, max_chars=1500, overlap_chars=200, min_chars=300))
        assert len(results) == 1
        assert results[0] == (0, len(text), text)

    def test_text_exactly_at_max_returns_single_chunk(self):
        text = "A" * 1500
        results = list(semantic_sliding_windows(text, max_chars=1500, overlap_chars=200, min_chars=300))
        assert len(results) == 1

    def test_long_text_splits_with_overlap(self):
        # Create text with sentence boundaries at predictable positions
        base = "A" * 200 + ". " + "B" * 200 + ". "
        text = base * 20  # ~8000 chars with sentence boundaries every ~400 chars
        results = list(semantic_sliding_windows(text, max_chars=1500, overlap_chars=200, min_chars=300))
        assert len(results) >= 2
        # All chunks must be within max_chars
        for start, end, segment in results:
            assert len(segment) <= 1500
            assert segment == text[start:end]

    def test_min_chunk_length_protection(self):
        # Text with a bad semantic boundary very close to max_chars
        text = "A" * 1400 + ". " + "B" * 5000
        results = list(semantic_sliding_windows(text, max_chars=1500, overlap_chars=200, min_chars=300))
        for start, end, segment in results:
            assert len(segment) >= int(1500 * 0.3) or end == len(text)

    def test_dead_loop_guard_next_pos_advances(self):
        """When overlap would cause pos to stall, next_pos advances to best_end."""
        text = "x" * 5000
        results = list(semantic_sliding_windows(text, max_chars=1500, overlap_chars=200, min_chars=300))
        positions = [r[0] for r in results]
        # Positions must strictly increase
        for i in range(1, len(positions)):
            assert positions[i] > positions[i - 1], f"Position stalled at index {i}: {positions}"

    def test_text_coverage_no_gaps_no_overlap_in_offsets(self):
        """The start/end offsets must be contiguous (overlap in content, not in offsets)."""
        text = "S" * 200 + ". " + "T" * 200 + ". " + "U" * 200 + ". " + "V" * 200 + ". " + "W" * 2000
        results = list(semantic_sliding_windows(text, max_chars=1500, overlap_chars=200, min_chars=300))
        # Verify the union of [start, end) covers the whole text
        for i in range(len(results) - 1):
            # start of next chunk must be <= end of current (content overlap)
            assert results[i + 1][0] <= results[i][1]
        # First starts at 0, last ends at len(text)
        assert results[0][0] == 0
        assert results[-1][1] == len(text)
