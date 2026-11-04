from app.services.image_index_quality import (
    NOT_INDEXABLE_MMD_PENDING,
    READY_CAPTION_CONTEXT,
    evaluate_image_indexability,
    image_search_quality_filter,
)


def test_verified_image_requires_mmd_match_caption_and_adjacent_text():
    quality = evaluate_image_indexability(
        {
            "title": "Figure 1: Sample result",
            "context_before": "Paragraph before image.",
            "context_after": "Figure 1: Sample result\n\nParagraph after image.",
        },
        {
            "mmd_context_located": True,
            "mmd_image_path": "figures/sample.png",
        },
    )

    assert quality.status == READY_CAPTION_CONTEXT
    assert quality.indexable is True
    assert quality.caption_text == "Figure 1: Sample result"
    assert quality.context_text == (
        "Figure 1: Sample result\n\nParagraph before image.\n\nParagraph after image."
    )
    assert quality.caption_input_hash
    assert quality.context_input_hash


def test_background_pending_image_is_not_indexable_even_with_document_title():
    quality = evaluate_image_indexability(
        {
            "doc_title": "A document title is not image context",
            "context_before": "A document title is not image context",
            "context_after": "No caption",
        },
        {"mmd_context_located": False, "mmd_backfill_status": "pending"},
    )

    assert quality.status == NOT_INDEXABLE_MMD_PENDING
    assert quality.indexable is False
    assert quality.caption_text == ""


def test_image_text_search_filter_only_allows_verified_records():
    expected = 'img_meta["quality_status"] == "READY_CAPTION_CONTEXT"'
    assert image_search_quality_filter("caption_embed") == expected
    assert image_search_quality_filter("context_embed") == expected
    assert image_search_quality_filter("visual_embed") == ""
