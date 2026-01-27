"""Shared utilities for Metaxy CLI commands."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
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
# --filter: Applied only to the target feature (target_filters)
FilterArgs = Annotated[
    list[nw.Expr],  # Actually list[nw.Expr], but using Any to avoid import at module level
    cyclopts.Parameter(
        name=["--filter"],
        help="SQL WHERE clause [filter](https://docs.metaxy.io/main/guide/learn/filters/) applied to the result of the status increment. Can be repeated.",
        converter=_convert_filters,
        accepts_keys=False,
    ),
]

# Type alias for global filter arguments
# --global-filter: Applied to all features (including upstream dependencies)
GlobalFilterArgs = Annotated[
    list[nw.Expr],
    cyclopts.Parameter(
        name=["--global-filter"],
        help="SQL WHERE clause [filter](https://docs.metaxy.io/main/guide/learn/filters/) applied to all features being selected (including upstream). Can be repeated.",
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


class CLIErrorCode(str, Enum):
    """Error codes for CLI errors."""

    CONFLICTING_FLAGS = "CONFLICTING_FLAGS"
    FEATURES_NOT_FOUND = "FEATURES_NOT_FOUND"
    GENERIC_ERROR = "GENERIC_ERROR"
    INVALID_FEATURE_KEY = "INVALID_FEATURE_KEY"
    MISSING_CONFIRMATION = "MISSING_CONFIRMATION"
    MISSING_REQUIRED_FLAG = "MISSING_REQUIRED_FLAG"
    NO_FEATURES = "NO_FEATURES"
    SNAPSHOT_ERROR = "SNAPSHOT_ERROR"
    SNAPSHOT_LOAD_FAILED = "SNAPSHOT_LOAD_FAILED"


@dataclass
class CLIError:
    """Structured CLI error that can be rendered as JSON or plain text."""

    code: CLIErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    hint: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"error": self.code.value, "message": self.message}
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
@dataclass
class FeatureSelector:
    """Encapsulates feature selection logic for CLI commands.

    Handles positional feature arguments vs --all-features flag.

    Usage:
        selector.resolve(format, error_missing=True)
        for key in selector:
            ...
    """

    features: Annotated[
        tuple[str, ...],
        cyclopts.Parameter(
            help="Feature keys (e.g., 'my_feature' or 'namespace/feature').",
        ),
    ] = ()
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name="--all-features",
            help="Apply to all features in the project's feature graph.",
        ),
    ] = False

    _resolved_keys: Annotated[
        list[FeatureKey] | None,
        cyclopts.Parameter(parse=False),
    ] = field(default=None, repr=False)
    _missing_keys: Annotated[
        list[FeatureKey],
        cyclopts.Parameter(parse=False),
    ] = field(default_factory=list, repr=False)

    def validate(self, output_format: OutputFormat) -> None:
        """Validate that exactly one selection mode is specified."""
        if not self.all_features and not self.features:
            exit_with_error(
                CLIError(
                    code=CLIErrorCode.MISSING_REQUIRED_FLAG,
                    message="Must specify either --all-features or feature arguments",
                    details={"required_flags": ["--all-features", "<features>"]},
                ),
                output_format,
            )
        if self.all_features and self.features:
            exit_with_error(
                CLIError(
                    code=CLIErrorCode.CONFLICTING_FLAGS,
                    message="Cannot specify both --all-features and feature arguments",
                    details={"conflicting_flags": ["--all-features", "<features>"]},
                ),
                output_format,
            )

    def resolve(
        self,
        output_format: OutputFormat,
        *,
        graph: FeatureGraph | None = None,
        warn_missing: bool = True,
        error_missing: bool = False,
    ) -> FeatureSelector:
        """Validate and resolve feature keys.

        Gets graph from AppContext.get() if not provided.
        """
        from metaxy.cli.context import AppContext

        self.validate(output_format)

        if graph is None:
            graph = AppContext.get().graph

        valid_keys, missing_keys = self._resolve_keys_internal(graph, output_format)

        if missing_keys:
            if error_missing:
                exit_with_error(
                    CLIError(
                        code=CLIErrorCode.FEATURES_NOT_FOUND,
                        message="Feature(s) not found in graph",
                        details={"features": [k.to_string() for k in missing_keys]},
                    ),
                    output_format,
                )
            elif warn_missing and output_format == "plain":
                formatted = ", ".join(k.to_string() for k in missing_keys)
                data_console.print(f"[yellow]Warning:[/yellow] Feature(s) not found: {formatted}")

        self._resolved_keys = valid_keys
        self._missing_keys = missing_keys
        return self

    def __iter__(self) -> Iterator[FeatureKey]:
        if self._resolved_keys is None:
            raise RuntimeError("Must call resolve() before iterating")
        return iter(self._resolved_keys)

    def __len__(self) -> int:
        if self._resolved_keys is None:
            raise RuntimeError("Must call resolve() before checking length")
        return len(self._resolved_keys)

    def __bool__(self) -> bool:
        if self._resolved_keys is None:
            raise RuntimeError("Must call resolve() before boolean check")
        return bool(self._resolved_keys)

    @property
    def missing_keys(self) -> list[FeatureKey]:
        if self._resolved_keys is None:
            raise RuntimeError("Must call resolve() first")
        return self._missing_keys

    def _resolve_keys_internal(
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
                        code=CLIErrorCode.INVALID_FEATURE_KEY,
                        message=f"Invalid feature key '{raw_key}': {exc}",
                        details={"key": raw_key},
                    ),
                    output_format,
                )

        # Check which keys exist in graph
        valid = [k for k in parsed_keys if k in graph.feature_definitions_by_key]
        missing = [k for k in parsed_keys if k not in graph.feature_definitions_by_key]

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
            CLIError(code=CLIErrorCode.SNAPSHOT_ERROR, message=str(e)),
            output_format,
        )
    except ImportError as e:
        exit_with_error(
            CLIError(
                code=CLIErrorCode.SNAPSHOT_LOAD_FAILED,
                message=f"Failed to load snapshot: {e}",
                hint="Feature classes may have been moved or deleted.",
            ),
            output_format,
        )
