"""Tests for shared metadata store utilities."""

from metaxy.metadata_store.utils import is_local_path


def test_is_local_path() -> None:
    """is_local_path should correctly detect local vs remote URIs."""
    assert is_local_path("./local/path")
    assert is_local_path("/absolute/path")
    assert is_local_path("relative/path")
    assert is_local_path("C:\\Windows\\Path")
    assert is_local_path("file:///absolute/path")
    assert is_local_path("local://path")

    assert not is_local_path("s3://bucket/path")
    assert not is_local_path("db://database-name")
    assert not is_local_path("http://remote-server/db")
    assert not is_local_path("https://remote-server/db")
    assert not is_local_path("gs://bucket/path")
    assert not is_local_path("az://container/path")
