import pytest
from metaxy_testing import TempMetaxyProject

def test_cli_no_args(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that running the CLI with no arguments does not crash."""
    # This should not raise IndexError. We set check=False because Cyclopts
    # might exit with non-zero when no command is provided, which is fine
    # as long as it's not a crash.
    result = metaxy_project.run_cli([], capsys=capsys, check=False)
    
    # Ensure no traceback is present in output (which would indicate a crash)
    assert "IndexError" not in result.stderr
    assert "Traceback" not in result.stderr
    
    # Check that help or usage information is displayed
    output = (result.stdout + result.stderr).lower()
    assert "usage: metaxy" in output or "options" in output or "commands" in output

def test_cli_help_flag(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test that running the CLI with --help works."""
    result = metaxy_project.run_cli(["--help"], capsys=capsys)
    assert result.returncode == 0
    output = (result.stdout + result.stderr).lower()
    assert "usage: metaxy" in output or "options" in output or "commands" in output
