"""Shared utilities for Metaxy CLI commands."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

import cyclopts

from metaxy.cli.console import data_console
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureGraph

# Standard output format type used across CLI commands
OutputFormat = Literal["plain", "json"]


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
        """Convert to plain text with Rich markup."""
        lines = [f"[red]Error:[/red] {self.message}"]
        if self.hint:
            lines.append(f"[yellow]Hint:[/yellow] {self.hint}")
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
