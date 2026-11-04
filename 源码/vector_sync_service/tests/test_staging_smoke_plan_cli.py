import io
import json
from pathlib import Path

import staging_smoke_plan


def _run_cli(args, *, environ=None):
    stdout = io.StringIO()
    exit_code = staging_smoke_plan.main(args, environ=environ or {}, stdout=stdout)
    return exit_code, json.loads(stdout.getvalue()), stdout.getvalue()


def test_cli_default_prints_sanitized_no_live_plan():
    exit_code, plan, output = _run_cli([])

    assert exit_code == 0
    assert plan["live_calls_allowed"] is False
    assert plan["consistency"]["ready"] is False
    assert plan["safe_ids"]["doc_id"]["provided"] is False
    assert "RUN_LIVE_SMOKE" in plan["required_env_names"]
    assert "MongoStateManager" not in output
    assert "MilvusClient" not in output
    assert "preflight_blockers" not in plan


def test_cli_reports_unsafe_doc_id_without_echoing_raw_value():
    raw_id = "PROD_DOC_should_not_be_echoed"
    exit_code, plan, output = _run_cli(["--components", "mongo,milvus", "--doc-id", raw_id])

    assert exit_code == 0
    assert plan["safe_ids"]["doc_id"]["provided"] is True
    assert plan["safe_ids"]["doc_id"]["safe"] is False
    assert plan["safe_ids"]["doc_id"]["value"] is None
    assert raw_id not in output


def test_cli_passes_safe_ids_components_and_expectations():
    env = {
        "RUN_LIVE_SMOKE": "1",
        "RUN_LIVE_MONGO": "1",
        "RUN_LIVE_MILVUS": "1",
    }

    exit_code, plan, _output = _run_cli(
        [
            "--components",
            "mongo,milvus",
            "--doc-id",
            "SMOKE_DOC_round4_001",
            "--sys-id",
            "SMOKE_SYS_round4_001",
            "--expect-document",
            "true",
            "--expect-chunks",
            "false",
            "--expect-images",
            "none",
        ],
        environ=env,
    )

    assert exit_code == 0
    assert plan["planned_components"] == ["mongo", "milvus"]
    assert plan["skip_reasons"] == {}
    assert plan["safe_ids"]["doc_id"]["value"] == "SMOKE_DOC_round4_001"
    assert plan["safe_ids"]["sys_id"]["value"] == "SMOKE_SYS_round4_001"
    assert plan["expectations"] == {"document": True, "chunks": False, "images": None}
    assert plan["consistency"]["ready"] is True


def test_cli_preflight_prints_no_live_preflight_json():
    exit_code, preflight, output = _run_cli(["--preflight", "--components", "mongo"])

    assert exit_code == 0
    assert preflight["read_only"] is True
    assert preflight["mutation_allowed"] is False
    assert preflight["live_calls_allowed"] is False
    assert preflight["would_run"] == []
    assert preflight["has_blocker"] is True
    assert "preflight_blockers" in preflight
    assert "MongoStateManager" not in output
    assert "MilvusClient" not in output


def test_cli_preflight_does_not_echo_unsafe_doc_id():
    raw_id = "PROD_DOC_should_not_be_echoed"
    exit_code, preflight, output = _run_cli(["--preflight", "--components", "mongo", "--doc-id", raw_id])

    assert exit_code == 0
    assert preflight["safe_ids"]["doc_id"]["provided"] is True
    assert preflight["safe_ids"]["doc_id"]["safe"] is False
    assert preflight["safe_ids"]["doc_id"]["value"] is None
    assert raw_id not in output


def test_cli_source_does_not_reference_live_clients_or_providers():
    source = Path(staging_smoke_plan.__file__).read_text(encoding="utf-8")

    forbidden = [
        "MongoStateManager",
        "MilvusClient",
        "EmbeddingClient",
        "get_embedding_client",
        "httpx",
        "requests",
        "openai",
        "allow_live_clients=True",
    ]
    for keyword in forbidden:
        assert keyword not in source
