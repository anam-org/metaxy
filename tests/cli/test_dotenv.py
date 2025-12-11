"""Tests for .env file loading in CLI."""

import json

from metaxy._testing import TempMetaxyProject


def test_cli_loads_dotenv_file(metaxy_project: TempMetaxyProject):
    """Test that CLI loads .env file and environment variables are available."""
    # Create a .env file with a test variable
    env_file = metaxy_project.project_dir / ".env"
    env_file.write_text("METAXY_TEST_VAR=hello_from_dotenv\n")

    def features():
        import os

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        # Use the env var in the code_version to prove it was loaded
        test_var = os.environ.get("METAXY_TEST_VAR", "not_found")

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "dotenv"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=test_var)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli("list", "features", "--format", "json")

        assert result.returncode == 0
        data = json.loads(result.stdout)

        assert data["feature_count"] == 1
        feature = data["features"][0]
        # The code_version should contain the value from .env
        assert feature["fields"][0]["code_version"] == "hello_from_dotenv"


def test_cli_env_vars_override_dotenv(metaxy_project: TempMetaxyProject):
    """Test that explicit environment variables take precedence over .env file."""
    # Create a .env file
    env_file = metaxy_project.project_dir / ".env"
    env_file.write_text("METAXY_TEST_VAR=from_dotenv\n")

    def features():
        import os

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        test_var = os.environ.get("METAXY_TEST_VAR", "not_found")

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "override"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=test_var)],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Pass env var explicitly - should override .env
        result = metaxy_project.run_cli(
            "list",
            "features",
            "--format",
            "json",
            env={"METAXY_TEST_VAR": "from_explicit_env"},
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)

        feature = data["features"][0]
        # Explicit env var should take precedence
        assert feature["fields"][0]["code_version"] == "from_explicit_env"


def test_cli_works_without_dotenv_file(metaxy_project: TempMetaxyProject):
    """Test that CLI works normally when no .env file exists."""
    # Ensure no .env file exists
    env_file = metaxy_project.project_dir / ".env"
    if env_file.exists():
        env_file.unlink()

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "no_dotenv"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli("list", "features", "--format", "json")

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["feature_count"] == 1
