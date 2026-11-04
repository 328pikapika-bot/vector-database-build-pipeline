import pytest

from app.services.static_resource_url import build_static_resource_url


def test_build_static_resource_url_joins_safe_relative_path():
    assert build_static_resource_url(
        "/fulltext/doc-1/file.mmd",
        base_url="https://static.example.test/",
    ) == "https://static.example.test/fulltext/doc-1/file.mmd"


@pytest.mark.parametrize(
    "relative_url",
    [
        "../secret",
        "fulltext/../secret",
        "fulltext/%2e%2e/secret",
        "fulltext\\..\\secret",
        "https://attacker.example/secret",
        "//attacker.example/secret",
    ],
)
def test_build_static_resource_url_rejects_unsafe_paths(relative_url):
    with pytest.raises(ValueError):
        build_static_resource_url(relative_url, base_url="https://static.example.test/")
