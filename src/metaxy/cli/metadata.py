"""Metadata management commands for Metaxy CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
import narwhals as nw

from metaxy.cli.console import console, error_console
from metaxy.models.feature_spec import BaseFeatureSpecWithIDColumns, FeatureSpec
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import BaseFeature, FeatureGraph
    from metaxy.models.feature_spec import IDColumns

# Metadata subcommand app
app = cyclopts.App(
    name="metadata",  # pyrefly: ignore[unexpected-keyword]
    help="Manage Metaxy metadata",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
)


@dataclass
class _SnapshotOverride:
    spec: BaseFeatureSpecWithIDColumns
    feature_version: str
    feature_spec_version: str | None
    feature_tracking_version: str | None


@contextmanager
def _apply_snapshot_overrides(
    graph: FeatureGraph, overrides: Mapping[FeatureKey, _SnapshotOverride]
) -> Any:
    """Temporarily override feature specs/versions when checking historical snapshots."""
    if not overrides:
        yield
        return

    patched: list[
        tuple[
            FeatureKey,
            type[BaseFeature[IDColumns]],
            BaseFeatureSpecWithIDColumns,
            BaseFeatureSpecWithIDColumns,
            Any,
        ]
    ] = []

    try:
        for feature_key, override in overrides.items():
            feature_cls = graph.features_by_key.get(feature_key)
            if feature_cls is None:
                continue

            original_spec = feature_cls._spec  # type: ignore[attr-defined]
            original_graph_spec = graph.feature_specs_by_key.get(
                feature_key, original_spec
            )
            original_feature_version = feature_cls.__dict__.get("feature_version")

            feature_cls._spec = override.spec  # type: ignore[attr-defined]
            graph.feature_specs_by_key[feature_key] = override.spec

            def _version_method(value: str) -> Any:
                def _inner(cls: type[BaseFeature[IDColumns]]) -> str:
                    return value

                return classmethod(_inner)

            feature_cls.feature_version = _version_method(override.feature_version)  # type: ignore[assignment]
            patched.append(
                (
                    feature_key,
                    feature_cls,
                    original_spec,
                    original_graph_spec,
                    original_feature_version,
                )
            )

        yield
    finally:
        for (
            feature_key,
            feature_cls,
            original_spec,
            original_graph_spec,
            original_feature_version,
        ) in patched:
            feature_cls._spec = original_spec  # type: ignore[attr-defined]
            graph.feature_specs_by_key[feature_key] = original_graph_spec
            if original_feature_version is not None:
                feature_cls.feature_version = original_feature_version  # type: ignore[assignment]


def _parse_feature_key(raw: str) -> FeatureKey:
    try:
        return FeatureKey(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid feature key '{raw}': {exc}") from exc


def _count_lazy_rows(lazy_frame: nw.LazyFrame[Any]) -> int:
    """Return row count for a Narwhals LazyFrame."""
    try:
        count_df = lazy_frame.select(nw.len().alias("row_count")).collect()
        count_pl = count_df.to_polars()
        return int(count_pl["row_count"][0])
    except Exception:
        result_df = lazy_frame.collect()
        native = result_df.to_native()
        try:
            return len(native)
        except TypeError:
            return int(native.shape[0])  # type: ignore[attr-defined]


def _load_snapshot_overrides(
    store: MetadataStore,
    snapshot_id: str,
    project: str | None,
) -> dict[FeatureKey, _SnapshotOverride]:
    features_df = store.read_features(
        current=False, snapshot_version=snapshot_id, project=project
    )
    if features_df.height == 0:
        return {}

    overrides: dict[FeatureKey, _SnapshotOverride] = {}
    for row in features_df.iter_rows(named=True):
        feature_key = FeatureKey(row["feature_key"])
        spec_value = row.get("feature_spec")
        spec_dict = (
            json.loads(spec_value) if isinstance(spec_value, str) else spec_value
        )
        spec = FeatureSpec.model_validate(spec_dict)
        overrides[feature_key] = _SnapshotOverride(
            spec=spec,
            feature_version=row["feature_version"],
            feature_spec_version=row.get("feature_spec_version"),
            feature_tracking_version=row.get("feature_tracking_version"),
        )
    return overrides


def _preview_rows(
    lazy_frame: nw.LazyFrame[Any],
    id_columns: Sequence[str] | None,
    limit: int = 5,
) -> list[str]:
    """Return formatted preview rows for verbose output."""
    try:
        preview_df = lazy_frame.collect().to_polars()
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Failed to preview rows: {exc}",
            style="dim",
        )
        return []

    headers = list(id_columns or ())
    if not headers:
        headers = ["sample_uid"]

    available_headers = [col for col in headers if col in preview_df.columns]
    if not available_headers:
        available_headers = list(preview_df.columns)
    if not available_headers:
        return []

    preview_df = preview_df.select(available_headers).head(limit)

    return [
        ", ".join(f"{col}={row[col]}" for col in available_headers)
        for row in preview_df.to_dicts()
    ]


@app.command()
def status(
    feature_keys: Annotated[
        list[str],
        cyclopts.Parameter(
            help="Feature keys to inspect (e.g., namespace/feature). Provide at least one.",
        ),
    ],
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store name (defaults to configured default store).",
        ),
    ] = None,
    snapshot_id: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot-id"],
            help="Check metadata against a specific snapshot ID.",
        ),
    ] = None,
    assert_in_sync: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--assert-in-sync"],
            help="Exit with error if any feature needs updates or metadata is missing.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--verbose"],
            help="Show additional details about samples needing updates.",
        ),
    ] = False,
) -> None:
    """Check metadata completeness and freshness for specified features."""
    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.exceptions import FeatureNotFoundError

    if not feature_keys:
        console.print("[red]✗[/red] At least one FEATURE_KEY argument is required.")
        raise SystemExit(1)

    parsed_keys: list[FeatureKey] = []
    for raw_key in feature_keys:
        try:
            parsed_keys.append(_parse_feature_key(raw_key))
        except ValueError as exc:
            console.print(f"[red]✗[/red] {exc}")
            raise SystemExit(1)

    context = AppContext.get()
    metadata_store = context.get_store(store)
    graph = context.graph

    with metadata_store:
        overrides: dict[FeatureKey, _SnapshotOverride] = {}
        if snapshot_id:
            overrides = _load_snapshot_overrides(
                metadata_store, snapshot_id, context.project
            )
            if not overrides:
                console.print(
                    f"[red]✗[/red] No features recorded for snapshot {snapshot_id}."
                )
                raise SystemExit(1)

        missing_in_graph = [
            key for key in parsed_keys if key not in graph.features_by_key
        ]
        if missing_in_graph:
            formatted = ", ".join(key.to_string() for key in missing_in_graph)
            console.print(
                f"[red]✗[/red] Feature(s) not found in active graph: {formatted}"
            )
            raise SystemExit(1)

        if snapshot_id:
            missing_in_snapshot = [key for key in parsed_keys if key not in overrides]
            if missing_in_snapshot:
                formatted = ", ".join(key.to_string() for key in missing_in_snapshot)
                console.print(
                    f"[red]✗[/red] Feature(s) not present in snapshot {snapshot_id}: {formatted}"
                )
                raise SystemExit(1)

        needs_update = False
        header = (
            f"Metadata status (snapshot {snapshot_id})"
            if snapshot_id
            else "Metadata status"
        )
        console.print(f"\n[bold]{header}[/bold]")

        with _apply_snapshot_overrides(graph, overrides):
            for feature_key in parsed_keys:
                feature_cls = graph.features_by_key[feature_key]

                target_version = (
                    overrides[feature_key].feature_version
                    if snapshot_id
                    else feature_cls.feature_version()
                )

                try:
                    lazy_increment = metadata_store.resolve_update(
                        feature_cls, lazy=True
                    )
                except FeatureNotFoundError:
                    lazy_increment = None
                except ValueError as exc:
                    console.print(f"[red]✗[/red] {feature_key.to_string()}: {exc}")
                    needs_update = True
                    continue
                except Exception as exc:  # pragma: no cover
                    console.print(
                        f"[red]✗[/red] {feature_key.to_string()}: failed to compute diff ({exc})"
                    )
                    needs_update = True
                    continue

                added_count = 0
                changed_count = 0
                added_preview: list[str] = []
                changed_preview: list[str] = []

                id_columns_spec = feature_cls.spec().id_columns  # type: ignore[attr-defined]
                id_columns_seq = (
                    tuple(id_columns_spec) if id_columns_spec is not None else None
                )

                if lazy_increment is not None:
                    added_count = _count_lazy_rows(lazy_increment.added)
                    changed_count = _count_lazy_rows(lazy_increment.changed)

                    if verbose and added_count:
                        added_preview = _preview_rows(
                            lazy_increment.added,
                            id_columns_seq,
                        )

                    if verbose and changed_count:
                        changed_preview = _preview_rows(
                            lazy_increment.changed,
                            id_columns_seq,
                        )

                try:
                    metadata_lazy = metadata_store.read_metadata(
                        feature_cls.spec().key,  # type: ignore[attr-defined]
                        feature_version=target_version,
                        current_only=False,
                        columns=list(id_columns_seq)
                        if id_columns_seq is not None
                        else None,
                    )
                    row_count = _count_lazy_rows(metadata_lazy)
                except FeatureNotFoundError:
                    row_count = 0
                except Exception:
                    row_count = 0

                if lazy_increment is None:
                    status_icon = "[red]✗[/red]"
                    status_text = "missing metadata"
                    needs_update = True
                elif added_count or changed_count:
                    status_icon = "[yellow]⚠[/yellow]"
                    status_text = "needs update"
                    needs_update = True
                else:
                    status_icon = "[green]✓[/green]"
                    status_text = "up-to-date"

                console.print(
                    f"{status_icon} {feature_key.to_string()} "
                    f"(rows: {row_count}, added: {added_count}, changed: {changed_count}) — {status_text}"
                )

                if verbose and added_preview:
                    console.print("    Added samples: " + "; ".join(added_preview))
                if verbose and changed_preview:
                    console.print("    Changed samples: " + "; ".join(changed_preview))

        if needs_update:
            raise SystemExit(1)
        if assert_in_sync:
            raise SystemExit(0)


@app.command()
def copy(
    from_store: Annotated[
        str,
        cyclopts.Parameter(
            name=["--from", "FROM"],
            help="Source store name (must be configured in metaxy.toml)",
        ),
    ],
    to_store: Annotated[
        str,
        cyclopts.Parameter(
            name=["--to", "TO"],
            help="Destination store name (must be configured in metaxy.toml)",
        ),
    ],
    features: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--feature"],
            help="Feature key to copy (e.g., 'my_feature' or 'group/my_feature'). Can be repeated multiple times. If not specified, uses --all-features.",
        ),
    ] = None,
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--all-features"],
            help="Copy all features from source store",
        ),
    ] = False,
    from_snapshot: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--snapshot"],
            help="Snapshot version to copy (defaults to latest in source store). The snapshot_version is preserved in the destination.",
        ),
    ] = None,
    incremental: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--incremental"],
            help="Use incremental copy (compare provenance_by_field to skip existing rows). Disable for better performance if destination is empty or uses deduplication.",
        ),
    ] = True,
):
    """Copy metadata between stores.

    Copies metadata for specified features from one store to another,
    optionally using a historical version. Useful for:
    - Migrating data between environments
    - Backfilling metadata
    - Copying specific feature versions

    Incremental Mode (default):
        By default, performs an anti-join on sample_uid to skip rows that already exist
        in the destination for the same snapshot_version. This prevents duplicate writes.

        Disabling incremental (--no-incremental) may improve performance when:
        - The destination store is empty or has no overlap with source
        - The destination store has eventual deduplication

    Examples:
        # Copy all features from latest snapshot in dev to staging
        $ metaxy metadata copy --from dev --to staging --all-features

        # Copy specific features (repeatable flag)
        $ metaxy metadata copy --from dev --to staging --feature user_features --feature customer_features

        # Copy specific snapshot
        $ metaxy metadata copy --from prod --to staging --all-features --snapshot abc123

        # Non-incremental copy (faster, but may create duplicates)
        $ metaxy metadata copy --from dev --to staging --all-features --no-incremental
    """

    from metaxy.cli.context import AppContext

    context = AppContext.get()
    config = context.config

    # Validate arguments
    if not all_features and not features:
        console.print(
            "[red]Error:[/red] Must specify either --all-features or --feature"
        )
        raise SystemExit(1)

    if all_features and features:
        console.print(
            "[red]Error:[/red] Cannot specify both --all-features and --feature"
        )
        raise SystemExit(1)

    # Parse feature keys
    feature_keys: list[FeatureKey | type[BaseFeature]] | None = None
    if features:
        feature_keys = []
        for feature_str in features:
            # Parse feature key (supports both "feature" and "part1/part2/..." formats)
            if "/" in feature_str:
                parts = feature_str.split("/")
                feature_keys.append(FeatureKey(parts))
            else:
                # Single-part key
                feature_keys.append(FeatureKey([feature_str]))

    from metaxy.metadata_store.types import AccessMode

    # Get stores
    console.print(f"[cyan]Source store:[/cyan] {from_store}")
    console.print(f"[cyan]Destination store:[/cyan] {to_store}")

    source_store = config.get_store(from_store)
    dest_store = config.get_store(to_store)

    # Open both stores and copy
    with source_store.open(), dest_store.open(AccessMode.WRITE):
        console.print("\n[bold]Starting copy operation...[/bold]\n")

        try:
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=feature_keys,
                from_snapshot=from_snapshot,
                incremental=incremental,
            )

            console.print(
                f"\n[green]✓[/green] Copy complete: {stats['features_copied']} features, {stats['rows_copied']} rows"
            )

        except Exception as e:
            console.print(f"\n[red]✗[/red] Copy failed:\n{e}")
            raise SystemExit(1)


@app.command()
def drop(
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Store name to drop metadata from (defaults to configured default store)",
        ),
    ] = None,
    features: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--feature"],
            help="Feature key to drop (e.g., 'my_feature' or 'group/my_feature'). Can be repeated multiple times. If not specified, uses --all-features.",
        ),
    ] = None,
    all_features: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--all-features"],
            help="Drop metadata for all features defined in the current project's feature graph",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--confirm"],
            help="Confirm the drop operation (required to prevent accidental deletion)",
        ),
    ] = False,
):
    """Drop metadata from a store.

    Removes metadata for specified features from the store.
    This is a destructive operation and requires --confirm flag.

    When using --all-features, drops metadata for all features defined in the
    current project's feature graph.

    Useful for:
    - Cleaning up test data
    - Re-computing feature metadata from scratch
    - Removing obsolete features

    Examples:
        # Drop specific feature (requires confirmation)
        $ metaxy metadata drop --feature user_features --confirm

        # Drop multiple features
        $ metaxy metadata drop --feature user_features --feature customer_features --confirm

        # Drop all features defined in current project
        $ metaxy metadata drop --store dev --all-features --confirm
    """
    from metaxy.cli.context import AppContext

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    # Validate arguments
    if not all_features and not features:
        console.print(
            "[red]Error:[/red] Must specify either --all-features or --feature"
        )
        raise SystemExit(1)

    if all_features and features:
        console.print(
            "[red]Error:[/red] Cannot specify both --all-features and --feature"
        )
        raise SystemExit(1)

    if not confirm:
        console.print(
            "[red]Error:[/red] This is a destructive operation. Must specify --confirm flag."
        )
        raise SystemExit(1)

    # Parse feature keys
    feature_keys: list[FeatureKey] = []
    if features:
        for feature_str in features:
            # Parse feature key (supports both "feature" and "part1/part2/..." formats)
            if "/" in feature_str:
                parts = feature_str.split("/")
                feature_keys.append(FeatureKey(parts))
            else:
                # Single-part key
                feature_keys.append(FeatureKey([feature_str]))

    from metaxy.metadata_store.types import AccessMode

    # Get store
    metadata_store = context.get_store(store)

    with metadata_store.open(AccessMode.WRITE):
        # If all_features, get all feature keys from the active feature graph
        if all_features:
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            # Get all feature keys from the graph (features defined in code for current project)
            feature_keys = graph.list_features(only_current_project=True)

            if not feature_keys:
                console.print(
                    "[yellow]Warning:[/yellow] No features found in active graph. "
                    "Make sure your features are imported."
                )
                return

        console.print(
            f"\n[bold]Dropping metadata for {len(feature_keys)} feature(s)...[/bold]\n"
        )

        dropped_count = 0
        for feature_key in feature_keys:
            try:
                metadata_store.drop_feature_metadata(feature_key)
                console.print(f"[green]✓[/green] Dropped: {feature_key.to_string()}")
                dropped_count += 1
            except Exception as e:
                console.print(
                    f"[red]✗[/red] Failed to drop {feature_key.to_string()}: {e}"
                )

        console.print(
            f"\n[green]✓[/green] Drop complete: {dropped_count} feature(s) dropped"
        )
