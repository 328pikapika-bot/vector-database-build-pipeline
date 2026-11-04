from app.models.sync_runtime import ConsumedRecord
from app.services.kafka_consumer import _build_commit_offsets


def test_build_commit_offsets_uses_max_offset_per_partition():
    batch = [
        ConsumedRecord(
            topic="VECTOR_JOURNAL_ARTICLE",
            partition=0,
            offset=3,
            raw_bytes=b"",
            raw_text="",
            event_fingerprint="a",
        ),
        ConsumedRecord(
            topic="VECTOR_JOURNAL_ARTICLE",
            partition=0,
            offset=9,
            raw_bytes=b"",
            raw_text="",
            event_fingerprint="b",
        ),
        ConsumedRecord(
            topic="VECTOR_JOURNAL_ARTICLE",
            partition=1,
            offset=4,
            raw_bytes=b"",
            raw_text="",
            event_fingerprint="c",
        ),
    ]

    offsets = _build_commit_offsets(batch)

    normalized = [(item.topic, item.partition, item.offset) for item in offsets]
    assert normalized == [
        ("VECTOR_JOURNAL_ARTICLE", 0, 10),
        ("VECTOR_JOURNAL_ARTICLE", 1, 5),
    ]
