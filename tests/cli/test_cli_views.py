"""Tests for the views CLI commands."""

import json
from pathlib import Path

import pytest

from metaxy_testing import TempMetaxyProject


def _features():
    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
    from metaxy_testing.models import SampleFeatureSpec

    class VideoFilesRoot(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video", "files_root"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class VideoFiles(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video", "files"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            deps=[VideoFilesRoot],
        ),
    ):
        pass


@pytest.fixture
def project_with_metadata(metaxy_project: TempMetaxyProject):
    """Project where only the root feature has a table in the store."""
    with metaxy_project.with_features(_features):
        metaxy_project.write_sample_metadata("video/files_root")
        yield metaxy_project


def test_views_sync_dry_run_prints_sql(
    project_with_metadata: TempMetaxyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    result = project_with_metadata.run_cli(["views", "sync", "--dry-run"], capsys=capsys)

    assert 'CREATE OR REPLACE VIEW "video__files_root__current" AS SELECT' in result.stdout
    assert 'CREATE OR REPLACE VIEW "metaxy_feature_statistics" AS' in result.stdout
    assert 'CREATE OR REPLACE VIEW "metaxy_views_manifest" AS' in result.stdout
    assert "Skipping video/files: no table in the store yet" in result.stderr

    check = project_with_metadata.run_cli(["views", "sync", "--check"], capsys=capsys, check=False)
    assert check.returncode == 1, "dry run must not create views"


def test_views_sync_then_check(project_with_metadata: TempMetaxyProject, capsys: pytest.CaptureFixture[str]) -> None:
    before = project_with_metadata.run_cli(["views", "sync", "--check", "--format", "json"], capsys=capsys, check=False)
    assert before.returncode == 1
    assert json.loads(before.stdout) == {
        "error": "VIEWS_OUT_OF_DATE",
        "message": "Views are missing or out of date. Run `mx views sync` to update them.",
        "views": ["metaxy_feature_statistics", "metaxy_views_manifest", "video__files_root__current"],
    }

    sync = project_with_metadata.run_cli(["views", "sync"], capsys=capsys)
    assert "Synchronized 3 views" in sync.stderr

    after = project_with_metadata.run_cli(["views", "sync", "--check"], capsys=capsys)
    assert "Views are up to date" in after.stderr


def test_views_sync_writes_manifest(
    project_with_metadata: TempMetaxyProject, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "views.json"

    project_with_metadata.run_cli(["views", "sync", "--manifest", str(manifest_path)], capsys=capsys)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] is None
    current = next(view for view in manifest["views"] if view["kind"] == "current")
    assert current["view_name"] == "video__files_root__current"
    assert current["feature_key"] == "video/files_root"
    assert current["feature_version"] == project_with_metadata.graph.get_feature_version(["video", "files_root"])
    assert current["id_columns"] == ["sample_uid"]
    assert {column["name"] for column in current["columns"]} >= {"sample_uid", "metaxy_data_version"}


def test_views_sync_rejects_unknown_feature(
    project_with_metadata: TempMetaxyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    result = project_with_metadata.run_cli(
        ["views", "sync", "--feature", "does/not/exist", "--format", "json"], capsys=capsys, check=False
    )

    assert result.returncode == 1
    assert json.loads(result.stdout)["features"] == ["does/not/exist"]


def test_views_sync_rejects_dry_run_with_check(
    project_with_metadata: TempMetaxyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    result = project_with_metadata.run_cli(
        ["views", "sync", "--dry-run", "--check", "--format", "json"], capsys=capsys, check=False
    )

    assert result.returncode == 1
    assert json.loads(result.stdout)["error"] == "CONFLICTING_FLAGS"
