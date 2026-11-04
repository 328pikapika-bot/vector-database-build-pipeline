from __future__ import annotations

from app.services.image_context_extractor import (
    compose_context_after,
    extract_mmd_image_contexts,
    find_mmd_context_for_image,
)


def test_extract_mmd_image_contexts_uses_one_before_two_after_and_filename_meta():
    mmd_text = "\n\n".join(
        [
            "Paragraph A.",
            "Paragraph B.",
            "![Figure](figures/123-456-whxyp_0.1000_0.2000_0.3000_0.4000_5.jpg)",
            "Paragraph C.",
            "Paragraph D.",
            "Paragraph E.",
        ]
    )

    contexts = extract_mmd_image_contexts(mmd_text)

    assert len(contexts) == 1
    context = contexts[0]
    assert context.filename == "123-456-whxyp_0.1000_0.2000_0.3000_0.4000_5.jpg"
    assert context.context_before == "Paragraph B."
    assert context.context_after == "Paragraph C.\n\nParagraph D."
    assert context.context_for_model == "Paragraph B.\n\nParagraph C.\n\nParagraph D."
    assert context.img_meta["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert context.img_meta["page"] == 5
    assert context.img_meta["mmd_paragraph_index"] == 2


def test_extract_mmd_image_contexts_stops_at_neighboring_image_paragraphs():
    mmd_text = "\n\n".join(
        [
            "Intro paragraph.",
            "![A](figures/a.jpg)",
            "![B](figures/b.jpg)",
            "After paragraph.",
        ]
    )

    contexts = extract_mmd_image_contexts(mmd_text, window=2)

    first = find_mmd_context_for_image(contexts, image_name="a.jpg")
    second = find_mmd_context_for_image(contexts, relative_url="/download/figures/b.jpg")
    assert first is not None
    assert second is not None
    assert first.context_before == "Intro paragraph."
    assert first.context_after == ""
    assert second.context_before == ""
    assert second.context_after == "After paragraph."


def test_find_mmd_context_for_image_matches_filename_or_path_case_insensitively():
    contexts = extract_mmd_image_contexts(
        "Before\n\n![A](figures/Fig-A.JPG)\n\nAfter",
        window=1,
    )

    assert find_mmd_context_for_image(contexts, image_name="fig-a.jpg") is not None
    assert find_mmd_context_for_image(contexts, relative_url="/x/figures/FIG-A.JPG") is not None
    assert find_mmd_context_for_image(contexts, image_name="other.jpg") is None


def test_compose_context_after_prepends_caption_before_after_paragraphs():
    assert (
        compose_context_after("Figure 1 caption", "After one.\n\nAfter two.")
        == "Figure 1 caption\n\nAfter one.\n\nAfter two."
    )
    assert compose_context_after("Figure 1 caption", "") == "Figure 1 caption"
