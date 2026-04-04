"""Refactored tests for the metadata read CLI command."""

import json

import pytest
from metaxy_testing import TempMetaxyProject


def _define_features():
    """Define test features for the metadata read command."""
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

    class FilesParquet(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["files_parquet"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass


@pytest.fixture
def project_with_data(metaxy_project: TempMetaxyProject):
    """Fixture that initializes a project with sample metadata written to the 'dev' store."""
    with metaxy_project.with_features(_define_features):
        from metaxy.models.types import FeatureKey

        graph = metaxy_project.graph
        store = metaxy_project.stores["dev"]

        import ibis

        # Create sample data using Ibis (avoid Polars as requested)
        data = ibis.memtable(
            {
                "sample_uid": [1, 2, 3],
                "value": ["val_1", "val_2", "val_3"],
                "category": ["A", "B", "A"],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        )

        with graph.use(), store.open("w"):
            store.write(FeatureKey(["files_root"]), data)
            store.write(FeatureKey(["files_parquet"]), data)

        yield metaxy_project


@pytest.mark.parametrize(
    "fmt, check_fn",
    [
        ("markdown", lambda out: "sample_uid" in out),
        ("json", lambda out: "sample_uid" in out and '"sample_uid":1' in out.replace(" ", "")),
        ("csv", lambda out: "sample_uid" in out and "value" in out and "category" in out),
    ],
)
def test_metadata_read_formats(project_with_data, capsys, fmt, check_fn):
    """Test reading metadata in different text formats."""
    result = project_with_data.run_cli(["metadata", "read", "files_root", "-f", fmt], capsys=capsys)
    assert result.returncode == 0
    assert check_fn(result.stdout)


def test_metadata_read_parquet(project_with_data, tmp_path, capsys):
    """Test outputting metadata to a parquet file."""
    output_file = tmp_path / "output.parquet"
    result = project_with_data.run_cli(
        ["metadata", "read", "files_root", "-f", "parquet", "-o", str(output_file)], capsys=capsys
    )
    assert result.returncode == 0
    assert output_file.exists()

    # Verify content using PyArrow (avoid Polars)
    import pyarrow.parquet as pq

    table = pq.read_table(output_file)
    assert table.num_rows == 3
    assert "value" in table.column_names


@pytest.mark.parametrize(
    "options, expected_count, check_fn",
    [
        (
            ["--select", "sample_uid", "--select", "value", "--filter", "category = 'A'"],
            2,
            lambda out: "val_1" in out and "val_3" in out and "val_2" not in out,
        ),
        (
            ["--query", "SELECT count(*) as cnt FROM files_root WHERE category = 'A'"],
            1,
            lambda out: '"cnt":2' in out.replace(" ", ""),
        ),
    ],
)
def test_metadata_read_filtering(project_with_data, capsys, options, expected_count, check_fn):
    """Test reading with selection, filtering, and queries."""
    cmd = ["metadata", "read", "files_root", "-f", "json"] + options
    result = project_with_data.run_cli(cmd, capsys=capsys)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert len(data) == expected_count
    assert check_fn(result.stdout)


def test_metadata_read_invalid_feature(project_with_data, capsys):
    """Test reading a non-existent feature."""
    result = project_with_data.run_cli(["metadata", "read", "non_existent"], capsys=capsys, check=False)
    assert "Feature(s) not found" in (result.stderr + result.stdout) or result.returncode != 0


def test_metadata_read_explicit_store(project_with_data, capsys):
    """Test reading from an explicit store via --store option."""
    result = project_with_data.run_cli(
        ["metadata", "read", "files_root", "--store", "dev", "-f", "json"], capsys=capsys
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data[0]["sample_uid"] == 1


def test_metadata_read_invalid_format(project_with_data, capsys):
    """Test that an unsupported format returns an error."""
    result = project_with_data.run_cli(["metadata", "read", "files_root", "-f", "invalid"], capsys=capsys, check=False)
    assert result.returncode != 0


def test_metadata_read_output_file_csv(project_with_data, tmp_path, capsys):
    """Test redirecting CSV output to a file."""
    output_file = tmp_path / "output.csv"
    result = project_with_data.run_cli(
        ["metadata", "read", "files_root", "-f", "csv", "-o", str(output_file)], capsys=capsys
    )
    assert result.returncode == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "sample_uid" in content
    assert "value" in content
    assert "category" in content
