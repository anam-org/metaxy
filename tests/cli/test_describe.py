"""Tests for metaxy describe commands."""

from __future__ import annotations

import json


class TestDescribeFeature:
    """Tests for metaxy describe feature command."""

    def test_describe_single_feature_plain(self, metaxy_project):
        """Test describing a single feature in plain format."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class MyFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "my_feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                    description="A test feature for validation.",
                ),
            ):
                """Docstring that should be overridden by explicit description."""

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(["describe", "feature", "test/my_feature"])

            assert result.returncode == 0
            output = result.stdout

            # Check key elements are present
            assert "test/my_feature" in output
            assert "test" in output  # Project name
            assert "A test feature for validation" in output
            assert "Fields:" in output
            assert "default" in output
            assert "Dependencies:" in output

    def test_describe_single_feature_json(self, metaxy_project):
        """Test describing a single feature in JSON format."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class JsonFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "json_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="v1")],
                    description="JSON test description.",
                ),
            ):
                pass

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(["describe", "feature", "test/json_feature", "--format", "json"])

            assert result.returncode == 0
            data = json.loads(result.stdout)

            assert data["key"] == "test/json_feature"
            assert data["description"] == "JSON test description."
            assert data["project"] == "test"
            assert len(data["fields"]) == 1
            assert data["fields"][0]["key"] == "field1"
            assert data["fields"][0]["code_version"] == "v1"
            assert data["dependencies"] == []

    def test_describe_feature_with_dependencies(self, metaxy_project):
        """Test describing a feature with dependencies."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec

            class UpstreamFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "upstream"]),
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
                ),
            ):
                pass

            class DownstreamFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "downstream"]),
                    deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],
                    fields=[FieldSpec(key=FieldKey(["processed"]), code_version="1")],
                ),
            ):
                pass

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(["describe", "feature", "test/downstream", "--format", "json"])

            assert result.returncode == 0
            data = json.loads(result.stdout)

            assert data["key"] == "test/downstream"
            assert len(data["dependencies"]) == 1
            assert data["dependencies"][0]["feature"] == "test/upstream"

    def test_describe_multiple_features(self, metaxy_project):
        """Test describing multiple features."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class Feature1(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature1"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

            class Feature2(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature2"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(
                ["describe", "feature", "test/feature1", "test/feature2", "--format", "json"]
            )

            assert result.returncode == 0
            data = json.loads(result.stdout)

            assert "features" in data
            assert len(data["features"]) == 2
            keys = {f["key"] for f in data["features"]}
            assert keys == {"test/feature1", "test/feature2"}

    def test_describe_nonexistent_feature(self, metaxy_project):
        """Test describing a feature that doesn't exist."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class SomeFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "exists"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(
                ["describe", "feature", "test/nonexistent", "--format", "json"],
                check=False,
            )

            assert result.returncode == 1
            data = json.loads(result.stdout)
            assert data["error"] == "FEATURES_NOT_FOUND"

    def test_describe_feature_with_docstring_extraction(self, metaxy_project):
        """Test that docstrings are extracted as descriptions."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class DocstringFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "docstring_feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                """This is extracted from the docstring."""

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(["describe", "feature", "test/docstring_feature", "--format", "json"])

            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert data["description"] == "This is extracted from the docstring."

    def test_describe_feature_with_metadata(self, metaxy_project):
        """Test describing a feature with metadata."""

        def features():
            from metaxy_testing.models import SampleFeatureSpec

            from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

            class MetadataFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "metadata_feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                    metadata={"owner": "team-ml", "priority": "high"},
                ),
            ):
                pass

        with metaxy_project.with_features(features):
            result = metaxy_project.run_cli(["describe", "feature", "test/metadata_feature", "--format", "json"])

            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert data["metadata"] == {"owner": "team-ml", "priority": "high"}
