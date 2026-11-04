from app.contracts.docx_contract import (
    DOCUMENT_BASE_FIELDS,
    DOCUMENT_SCALAR_INDEX_FIELDS,
    FULLTEXT_FIELDS,
    IMAGE_FIELDS,
    IMAGE_SCALAR_INDEX_FIELDS,
)
from app.services.milvus_chunk_service import CHUNK_SCALAR_INDEX_SPECS
from app.services.milvus_image_service import IMAGE_SCALAR_INDEX_SPECS


def test_v2_primary_keys_and_history_compatible_image_fields_are_fixed():
    assert DOCUMENT_BASE_FIELDS[0] == "id"
    assert FULLTEXT_FIELDS[9] == "id"
    assert IMAGE_FIELDS[:2] == ("id", "doc_id")
    assert "child_chunk_id" not in FULLTEXT_FIELDS
    assert IMAGE_FIELDS == (
        "id",
        "doc_id",
        "doc_meta",
        "raw_img_meta",
        "img_meta",
        "context_before",
        "context_after",
        "p_hash",
        "visual_embed",
        "caption_embed",
        "context_embed",
        "text",
        "text_embed",
        "visual_text_embed",
        "text_sparse_vec",
    )


def test_v2_scalar_index_sets_match_the_agreed_contract():
    assert set(DOCUMENT_SCALAR_INDEX_FIELDS) >= {"id", "doi", "hasAuthorname", "hasCreateOrganizationname"}
    assert [spec["field_name"] for spec in CHUNK_SCALAR_INDEX_SPECS] == [
        "accessType",
        "classificationCode",
        "year",
        "hasAuthorname",
        "hasCreateOrganizationname",
        "language",
    ]
    assert [spec["index_name"] for spec in IMAGE_SCALAR_INDEX_SPECS] == [
        "image_id_inverted_idx",
        "image_doc_id_inverted_idx",
        "image_img_id_json_inverted_idx",
        "image_sys_data_status_inverted_idx",
        "image_year_inverted_idx",
        "image_classification_code_inverted_idx",
        "image_language_inverted_idx",
        "image_access_type_inverted_idx",
        "image_keyword_inverted_idx",
        "image_author_name_inverted_idx",
        "image_create_org_name_inverted_idx",
        "image_source_sys_source_id_inverted_idx",
        "image_source_id_set_inverted_idx",
    ]
    assert set(IMAGE_SCALAR_INDEX_FIELDS) == {
        "id", "doc_id", "img_meta.img_id", "doc_meta.sysDataStatus", "doc_meta.year",
        "doc_meta.classificationCode", "doc_meta.language", "doc_meta.accessType",
        "doc_meta.keyword", "doc_meta.hasAuthorname", "doc_meta.hasCreateOrganizationname",
        "doc_meta.hasSourcesysSourceId", "doc_meta.hasSourceidSet",
    }
