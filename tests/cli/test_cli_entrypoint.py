import pytest

from metaxy_testing import TempMetaxyProject


def test_cli_no_args(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that running the CLI with no arguments does not crash."""
    metaxy_project.run_cli([], capsys=capsys, check=True)


def test_cli_help_flag(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that running the CLI with --help works."""
    metaxy_project.run_cli(["--help"], capsys=capsys, check=True)
