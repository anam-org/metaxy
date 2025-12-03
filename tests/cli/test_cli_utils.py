"""Tests for CLI utility functions."""

from io import StringIO

import pytest
from rich.console import Console

from metaxy.cli.utils import (
    CLIError,
    print_error,
    print_error_item,
    print_error_list,
)


@pytest.fixture
def console_with_output():
    """Create a console that captures output for testing (no ANSI codes)."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True, width=200)
    return console, output


class TestPrintError:
    """Tests for print_error function."""

    def test_simple_error_message(self, console_with_output):
        console, output = console_with_output
        print_error(console, "Something went wrong")
        result = output.getvalue()
        assert "✗" in result
        assert "Something went wrong" in result

    def test_error_with_exception(self, console_with_output):
        console, output = console_with_output
        print_error(console, "Failed to load", ValueError("bad value"))
        result = output.getvalue()
        assert "Failed to load" in result
        assert "bad value" in result

    def test_escapes_brackets_in_error(self, console_with_output):
        """Error messages with brackets should not be interpreted as Rich markup."""
        console, output = console_with_output
        # This would crash Rich without escaping: [/tmp/foo] looks like a closing tag
        error_with_brackets = "[/tmp/metaxy.delta/file.parquet] not found"
        print_error(console, "File error", error_with_brackets)
        result = output.getvalue()
        # Should contain the literal brackets, not crash
        assert "[/tmp/metaxy.delta/file.parquet]" in result
        assert "not found" in result

    def test_custom_prefix(self, console_with_output):
        console, output = console_with_output
        print_error(console, "Warning message", prefix="[yellow]⚠[/yellow]")
        result = output.getvalue()
        assert "Warning message" in result


class TestPrintErrorItem:
    """Tests for print_error_item function."""

    def test_basic_key_error_pair(self, console_with_output):
        console, output = console_with_output
        print_error_item(console, "feature_a", "computation failed")
        result = output.getvalue()
        assert "feature_a" in result
        assert "computation failed" in result

    def test_escapes_key_with_brackets(self, console_with_output):
        """Keys with brackets should be escaped."""
        console, output = console_with_output
        print_error_item(console, "feature[0]", "error")
        result = output.getvalue()
        assert "feature[0]" in result

    def test_escapes_error_with_brackets(self, console_with_output):
        """Error messages with brackets should be escaped."""
        console, output = console_with_output
        print_error_item(console, "my_feature", "Column [id] not found")
        result = output.getvalue()
        assert "Column [id] not found" in result

    def test_with_indent(self, console_with_output):
        console, output = console_with_output
        print_error_item(console, "key", "error", indent="    ")
        result = output.getvalue()
        assert "    " in result

    def test_custom_prefix(self, console_with_output):
        console, output = console_with_output
        print_error_item(console, "key", "error", prefix="•")
        result = output.getvalue()
        assert "•" in result


class TestPrintErrorList:
    """Tests for print_error_list function."""

    def test_multiple_errors(self, console_with_output):
        console, output = console_with_output
        errors = {
            "feature_a": "error 1",
            "feature_b": "error 2",
        }
        print_error_list(console, errors)
        result = output.getvalue()
        assert "feature_a" in result
        assert "error 1" in result
        assert "feature_b" in result
        assert "error 2" in result

    def test_with_header(self, console_with_output):
        console, output = console_with_output
        errors = {"key": "value"}
        print_error_list(console, errors, header="[red]Errors:[/red]")
        result = output.getvalue()
        assert "key" in result

    def test_max_items_truncation(self, console_with_output):
        console, output = console_with_output
        errors = {f"feature_{i}": f"error {i}" for i in range(10)}
        print_error_list(console, errors, max_items=3)
        result = output.getvalue()
        # Should show first 3 and "... and 7 more"
        assert "feature_0" in result
        assert "feature_1" in result
        assert "feature_2" in result
        assert "7 more" in result

    def test_escapes_brackets_in_errors(self, console_with_output):
        """Errors with Rich-like markup should be escaped."""
        console, output = console_with_output
        errors = {
            "raw_video": "[/tmp/data.parquet, ... 3 other sources]",
        }
        print_error_list(console, errors)
        result = output.getvalue()
        # Should not crash and should contain the literal text
        assert "raw_video" in result
        assert "[/tmp/data.parquet" in result


class TestCLIError:
    """Tests for CLIError dataclass."""

    def test_to_json(self):
        error = CLIError(
            code="TEST_ERROR",
            message="Test message",
            details={"key": "value"},
        )
        result = error.to_json()
        assert result["error"] == "TEST_ERROR"
        assert result["message"] == "Test message"
        assert result["key"] == "value"

    def test_to_plain_escapes_message(self):
        """Message with brackets should be escaped in plain output."""
        error = CLIError(
            code="FILE_ERROR",
            message="File [/tmp/test.txt] not found",
        )
        result = error.to_plain()
        # Should contain escaped brackets (Rich will render them literally)
        assert "[/tmp/test.txt]" in result or "\\[" in result

    def test_to_plain_escapes_hint(self):
        """Hint with brackets should be escaped in plain output."""
        error = CLIError(
            code="HINT_ERROR",
            message="Error occurred",
            hint="Check file [config.yaml]",
        )
        result = error.to_plain()
        assert "Error occurred" in result
        assert "config.yaml" in result

    def test_to_plain_with_rich_markup_in_error(self):
        """Error messages that look like Rich markup should be escaped."""
        error = CLIError(
            code="MARKUP_ERROR",
            message="Invalid tag [red] in input",
        )
        # This should not crash when rendered
        result = error.to_plain()
        assert "Invalid tag" in result
