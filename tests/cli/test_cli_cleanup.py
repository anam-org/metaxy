"""Basic CLI coverage for `metaxy metadata cleanup`."""

from __future__ import annotations

import json

import pytest

from metaxy._testing import TempMetaxyProject
from tests.cli.conftest import run_cleanup_json, write_sample_data


def test_cleanup_requires_confirm(metaxy_project: TempMetaxyProject):
    """Should refuse to run destructive cleanup without --confirm."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(
            [
                "metadata",
                "cleanup",
                "--feature",
                "video/files",
                "--id",
                "1",
                "--format",
                "json",
            ],
            check=False,
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "MISSING_CONFIRMATION"
        assert "--confirm" in json.dumps(error)


def test_cleanup_requires_deletion_criterion(metaxy_project: TempMetaxyProject):
    """Must provide at least one deletion flag."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(
            [
                "metadata",
                "cleanup",
                "--feature",
                "video/files",
                "--confirm",
                "--format",
                "json",
            ],
            check=False,
        )

        assert result.returncode == 1
        error = json.loads(result.stdout)
        assert error["error"] == "MISSING_CRITERIA"
        for flag in ["--id", "--retention-days", "--primary-key", "--filter"]:
            assert flag in json.dumps(error)


@pytest.mark.parametrize(
    "deletion_mode_args,deletion_mode",
    [
        ([], "soft"),
        (["--hard"], "hard"),
    ],
)
def test_cleanup_deletes_rows_and_reports(
    metaxy_project: TempMetaxyProject, deletion_mode_args, deletion_mode
):
    """Happy-path cleanup reports rows deleted in JSON output."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        write_sample_data(metaxy_project, "video/files")
        result = run_cleanup_json(
            metaxy_project,
            "--feature",
            "video/files",
            "--id",
            "1",
            *deletion_mode_args,
            "--confirm",
        )

        assert result["total_rows_deleted"] == 1
        assert result["deletion_mode"] == deletion_mode
