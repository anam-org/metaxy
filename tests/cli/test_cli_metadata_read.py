"""Tests for the metadata read CLI command."""

import json

import polars as pl
import pytest
from metaxy_testing import TempMetaxyProject


def _define_features():
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

    class FilesRoot(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["files_root"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass


def test_metadata_read_basic(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test basic metadata read without options."""
    with metaxy_project.with_features(_define_features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "category": ["A", "B", "A"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 4)],
            }
        )
        with graph.use(), store.open("w"):
            store.write(FeatureKey(["files_root"]), upstream_data)

        result = metaxy_project.run_cli(
            ["metadata", "read", "files_root", "-f", "markdown"], capsys=capsys, check=False
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        assert result.returncode == 0
        assert "sample_uid" in result.stdout
        assert "value" in result.stdout
        assert "val_1" in result.stdout


def test_metadata_read_json(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test metadata read with JSON output."""
    with metaxy_project.with_features(_define_features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "category": ["A", "B", "A"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 4)],
            }
        )
        with graph.use(), store.open("w"):
            store.write(FeatureKey(["files_root"]), upstream_data)

        result = metaxy_project.run_cli(["metadata", "read", "files_root", "-f", "json"], capsys=capsys)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 3
        assert data[0]["value"] == "val_1"


def test_metadata_read_select_and_filter(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test metadata read with --select and --filter."""
    with metaxy_project.with_features(_define_features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "category": ["A", "B", "A"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 4)],
            }
        )
        with graph.use(), store.open("w"):
            store.write(FeatureKey(["files_root"]), upstream_data)

        result = metaxy_project.run_cli(
            ["metadata", "read", "files_root", "--select", "value", "--filter", "category = 'A'", "-f", "json"],
            capsys=capsys,
            check=False,
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 2
        assert "value" in data[0]
        assert "category" not in data[0]
        assert data[0]["value"] == "val_1"


def test_metadata_read_query(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test metadata read with arbitrary SQL query."""
    with metaxy_project.with_features(_define_features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "category": ["A", "B", "A"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 4)],
            }
        )
        with graph.use(), store.open("w"):
            store.write(FeatureKey(["files_root"]), upstream_data)

        result = metaxy_project.run_cli(
            [
                "metadata",
                "read",
                "files_root",
                "--query",
                "SELECT count(*) as cnt FROM metadata WHERE category = 'A'",
                "-f",
                "json",
            ],
            capsys=capsys,
            check=False,
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["cnt"] == 2


def test_metadata_read_output_file(metaxy_project: TempMetaxyProject, tmp_path, capsys: pytest.CaptureFixture[str]):
    """Test output to a file."""
    with metaxy_project.with_features(_define_features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "category": ["A", "B", "A"],
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 4)],
            }
        )
        with graph.use(), store.open("w"):
            store.write(FeatureKey(["files_root"]), upstream_data)

        output_file = tmp_path / "output.csv"
        result = metaxy_project.run_cli(
            [
                "metadata",
                "read",
                "files_root",
                "--select",
                "sample_uid",
                "--select",
                "value",
                "--select",
                "category",
                "-f",
                "csv",
                "-o",
                str(output_file),
            ],
            capsys=capsys,
            check=False,
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "sample_uid,value,category" in content
        assert "1,val_1,A" in content
