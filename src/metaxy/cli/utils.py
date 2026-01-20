"""Shared utilities for Metaxy CLI commands."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

import cyclopts
import narwhals as nw
from rich.console import Console
from rich.markup import escape as escape_markup

from metaxy.cli.console import data_console
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureGraph

# Standard output format type used across CLI commands
OutputFormat = Literal["plain", "json"]


def _convert_filters(type_: type, tokens: Sequence[cyclopts.Token]) -> list[nw.Expr]:
    """Cyclopts converter for filter arguments.

    Converts SQL WHERE clause strings into Narwhals filter expressions.
    """
    from pydantic import ValidationError

    from metaxy.models.filter_expression import FilterParseError, parse_filter_string

    result = []
    for token in tokens:
        try:
            result.append(parse_filter_string(token.value))
        except (FilterParseError, ValidationError) as e:
            raise cyclopts.ValidationError(f"Invalid filter syntax: {e}") from e
    return result


# Type alias for filter arguments with custom converter
FilterArgs = Annotated[
    list[
        nw.Expr
    ],  # Actually list[nw.Expr], but using Any to avoid import at module level
    cyclopts.Parameter(
        name=["--filter"],
        help="SQL WHERE clause [filter](https://anam-org.github.io/metaxy/main/guide/learn/filters/) applied to all features. Can be repeated.",
        converter=_convert_filters,
        accepts_keys=False,
    ),
]


def print_error(
    console: Console,
    message: str,
    error: str | Exception | None = None,
    *,
    prefix: str = "[red]✗[/red]",
) -> None:
    """Print an error message, safely escaping dynamic content.

    Args:
        console: Rich console to print to
        message: Static message (Rich markup allowed)
        error: Optional exception/error to append (will be escaped)
        prefix: Symbol/text prefix (default: red ✗)
    """
    if error is not None:
        safe_error = escape_markup(str(error))
        console.print(f"{prefix} {message}: {safe_error}")
    else:
        console.print(f"{prefix} {message}")


def print_error_item(
    console: Console,
    key: str,
    error: str | Exception,
    *,
    prefix: str = "  ✗",
    indent: str = "",
) -> None:
    """Print a single error item with key and error message, safely escaping markup.

    Args:
        console: Rich console to print to
        key: The identifier (e.g., feature key) - will be escaped
        error: The error message or exception - will be escaped
        prefix: Symbol/text before the key (default: "  ✗")
        indent: Additional indentation
    """
    safe_key = escape_markup(str(key))
    safe_error = escape_markup(str(error))
    console.print(f"{indent}{prefix} {safe_key}: {safe_error}")


def print_error_list(
    console: Console,
    errors: Mapping[str, str | Exception],
    *,
    header: str | None = None,
    prefix: str = "  ✗",
    indent: str = "",
    max_items: int | None = None,
) -> None:
    """Print a list of errors with optional header, safely escaping all content.

    Args:
        console: Rich console to print to
        errors: Mapping of keys to error messages (dict[str, str] or dict[str, Exception])
        header: Optional header line with Rich markup (not escaped)
        prefix: Symbol/text before each key (default: "  ✗")
        indent: Additional indentation for each error line
        max_items: Maximum number of items to display (None = all)
    """
    if header:
        console.print(header)

    items = list(errors.items())
    if max_items is not None:
        items = items[:max_items]

    for key, error in items:
        print_error_item(console, key, error, prefix=prefix, indent=indent)

    if max_items is not None and len(errors) > max_items:
        remaining = len(errors) - max_items
        console.print(f"{indent}  ... and {remaining} more")


@dataclass
class CLIError:
    """Structured CLI error that can be rendered as JSON or plain text."""

    code: str  # e.g., "MISSING_REQUIRED_FLAG", "CONFLICTING_FLAGS"
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    hint: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"error": self.code, "message": self.message}
        result.update(self.details)
        return result

    def to_plain(self) -> str:
        """Convert to plain text with Rich markup.

        The message and hint are escaped to prevent Rich markup injection
        from error messages containing brackets (e.g., file paths like [/tmp/...]).
        """
        safe_message = escape_markup(self.message)
        lines = [f"[red]Error:[/red] {safe_message}"]
        if self.hint:
            safe_hint = escape_markup(self.hint)
            lines.append(f"[yellow]Hint:[/yellow] {safe_hint}")
        return "\n".join(lines)


def exit_with_error(error: CLIError, output_format: OutputFormat) -> NoReturn:
    """Print error in appropriate format and exit with code 1."""
    if output_format == "json":
        print(json.dumps(error.to_json()))
    else:
        data_console.print(error.to_plain())
    raise SystemExit(1)


@cyclopts.Parameter(name="*")
@dataclass(kw_only=True)
class FeatureSelector:
    """Encapsulates feature selection logic for CLI commands.

    Handles the common pattern of --feature vs --all-features arguments.
    """

    features: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name="--feature",
            help="Feature key (e.g., 'my_feature' or 'namespace/feature'). Can be repeated.",
        ),
    ] = None
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name="--all-features",
            help="Apply to all features in the project's feature graph.",
        ),
    ] = False

    def validate(self, output_format: OutputFormat) -> None:
        """Validate that exactly one selection mode is specified."""
        if not self.all_features and not self.features:
            exit_with_error(
                CLIError(
                    code="MISSING_REQUIRED_FLAG",
                    message="Must specify either --all-features or --feature",
                    details={"required_flags": ["--all-features", "--feature"]},
                ),
                output_format,
            )
        if self.all_features and self.features:
            exit_with_error(
                CLIError(
                    code="CONFLICTING_FLAGS",
                    message="Cannot specify both --all-features and --feature",
                    details={"conflicting_flags": ["--all-features", "--feature"]},
                ),
                output_format,
            )

    def resolve_keys(
        self,
        graph: FeatureGraph,
        output_format: OutputFormat,
    ) -> tuple[list[FeatureKey], list[FeatureKey]]:
        """Resolve feature selection to keys.

        Args:
            graph: The feature graph to resolve against
            output_format: Output format for error messages

        Returns:
            Tuple of (valid_keys, missing_keys) where:
            - valid_keys: Keys that exist in the graph
            - missing_keys: Keys that were requested but don't exist
        """
        if self.all_features:
            return graph.list_features(only_current_project=True), []

        # Parse explicit feature keys
        parsed_keys: list[FeatureKey] = []
        for raw_key in self.features or []:
            try:
                parsed_keys.append(FeatureKey(raw_key))
            except ValueError as exc:
                exit_with_error(
                    CLIError(
                        code="INVALID_FEATURE_KEY",
                        message=f"Invalid feature key '{raw_key}': {exc}",
                        details={"key": raw_key},
                    ),
                    output_format,
                )

        # Check which keys exist in graph
        valid = [k for k in parsed_keys if k in graph.features_by_key]
        missing = [k for k in parsed_keys if k not in graph.features_by_key]

        return valid, missing


def load_graph_for_command(
    context: AppContext,
    snapshot_version: str | None,
    metadata_store: MetadataStore,
    output_format: OutputFormat,
) -> FeatureGraph:
    """Load feature graph from snapshot or use current.

    Args:
        context: CLI application context
        snapshot_version: Optional snapshot version to load from
        metadata_store: Store to load snapshot from
        output_format: Output format for error messages

    Returns:
        FeatureGraph from snapshot or current context
    """
    if snapshot_version is None:
        return context.graph

    from metaxy.metadata_store.system.storage import SystemTableStorage

    storage = SystemTableStorage(metadata_store)
    try:
        return storage.load_graph_from_snapshot(
            snapshot_version=snapshot_version,
            project=context.project,
        )
    except ValueError as e:
        exit_with_error(
            CLIError(code="SNAPSHOT_ERROR", message=str(e)),
            output_format,
        )
    except ImportError as e:
        exit_with_error(
            CLIError(
                code="SNAPSHOT_LOAD_FAILED",
                message=f"Failed to load snapshot: {e}",
                hint="Feature classes may have been moved or deleted.",
            ),
            output_format,
        )
