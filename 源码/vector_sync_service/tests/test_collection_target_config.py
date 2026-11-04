from app.config import Settings


def test_empty_collection_names_materialize_unique_run_targets():
    settings = Settings(
        _env_file=None,
        COLLECTION_AUTO_CREATE_ON_EMPTY=True,
        DOCUMENT_COLLECTION_NAME="",
        FULLTEXT_CHUNK_COLLECTION_NAME="",
        IMAGE_COLLECTION_NAME="",
    )

    assert settings.resolved_document_collection_name.startswith("scientific_documents_auto_")
    assert settings.resolved_fulltext_chunk_collection_name.startswith(
        "scientific_fulltext_chunks_auto_"
    )
    assert settings.IMAGE_COLLECTION_NAME.startswith("scientific_images_auto_")


def test_explicit_collection_names_are_preserved():
    settings = Settings(
        _env_file=None,
        COLLECTION_AUTO_CREATE_ON_EMPTY=True,
        DOCUMENT_COLLECTION_NAME="documents_existing",
        FULLTEXT_CHUNK_COLLECTION_NAME="chunks_existing",
        IMAGE_COLLECTION_NAME="images_existing",
    )

    assert settings.resolved_document_collection_name == "documents_existing"
    assert settings.resolved_fulltext_chunk_collection_name == "chunks_existing"
    assert settings.IMAGE_COLLECTION_NAME == "images_existing"
