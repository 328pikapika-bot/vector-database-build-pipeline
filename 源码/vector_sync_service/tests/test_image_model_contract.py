from app.models.image import ImageRecord


def test_image_record_accepts_img_id_and_image_id_alias():
    canonical = ImageRecord(img_id="DOC-1_img_0", docId="DOC-1", sequence=0)
    aliased = ImageRecord.model_validate({"imageId": "DOC-1_img_1", "docId": "DOC-1", "sequence": 1})

    assert canonical.img_id == "DOC-1_img_0"
    assert canonical.imageId == "DOC-1_img_0"
    assert aliased.img_id == "DOC-1_img_1"
    assert aliased.imageId == "DOC-1_img_1"


def test_image_record_represents_docx_metadata_and_feature_fields():
    record = ImageRecord(
        img_id="DOC-2_img_3",
        docId="DOC-2",
        url="figures/figure-3.png",
        title="Figure 3. Spectrum",
        name="figure-3.png",
        width=640,
        height=480,
        sequence=3,
        fileType="png",
        size=2048,
        p_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        visual_embed=[1.0] * 4096,
        text_embed=[2.0] * 2560,
        visual_text_embed=[3.0] * 4096,
        text="Figure 3. Spectrum. surrounding context",
        text_sparse_vec={11: 0.5},
        doc_meta={"sysId": "DOC-2"},
        img_meta={"page": 4, "bbox": [1, 2, 3, 4]},
        imageBytes=b"raw-image-bytes",
    )

    assert record.img_id == "DOC-2_img_3"
    assert record.docId == "DOC-2"
    assert record.url == "figures/figure-3.png"
    assert record.title == "Figure 3. Spectrum"
    assert record.name == "figure-3.png"
    assert record.width == 640
    assert record.height == 480
    assert record.sequence == 3
    assert record.fileType == "png"
    assert record.size == 2048
    assert record.p_hash == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert len(record.visual_embed) == 4096
    assert len(record.text_embed) == 2560
    assert len(record.visual_text_embed) == 4096
    assert record.text_sparse_vec == {11: 0.5}
    assert record.doc_meta == {"sysId": "DOC-2"}
    assert record.img_meta == {"page": 4, "bbox": [1, 2, 3, 4]}


def test_image_bytes_is_excluded_from_persistence_dump():
    record = ImageRecord(
        img_id="DOC-3_img_0",
        docId="DOC-3",
        sequence=0,
        imageBytes=b"processing-only",
    )

    dumped = record.model_dump()

    assert "imageBytes" not in dumped
    assert dumped["img_id"] == "DOC-3_img_0"
