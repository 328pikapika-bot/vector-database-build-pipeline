"""Default-skipped live probe for the training vector collections."""

import pytest

from app.services.data_plane_smoke import live_smoke_skip_reason


def _skip_unless_live_probe() -> None:
    for component in ("milvus", "embedding"):
        reason = live_smoke_skip_reason(component)
        if reason:
            pytest.skip(reason)


@pytest.mark.asyncio
async def test_live_training_vector_library_probe_is_read_only_and_returns_hits():
    _skip_unless_live_probe()

    from app.services.training_vector_library_probe import probe_training_vector_library

    result = await probe_training_vector_library(top_k=2)

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["mutation_allowed"] is False
    assert result["embedding"]["success"] is True
    assert result["collections"]["document"]["ok"] is True
    assert result["collections"]["chunk"]["ok"] is True
    assert result["collections"]["image"]["ok"] is True
    assert result["collections"]["document"]["searches"]
    assert result["collections"]["chunk"]["searches"]
    assert result["collections"]["image"]["searches"]
