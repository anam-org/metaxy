"""Views commands for Metaxy CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import cyclopts

from metaxy.cli.console import console, data_console, error_console
from metaxy.cli.utils import CLIError, CLIErrorCode, OutputFormat, exit_with_error

if TYPE_CHECKING:
    from metaxy.ext.ibis.views import ViewPlan

app = cyclopts.App(
    name="views",
    help="Manage SQL views that expose current feature rows and statistics to readers without Metaxy.",
    console=console,
    error_console=error_console,
)


@app.command()
def sync(
    *,
    feature: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name=["--feature"],
            help="Only create or update the current view of these features. "
            "The statistics and manifest views always cover all internal features.",
        ),
    ] = None,
    schema: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--schema"],
            help="Schema (database on ClickHouse) to create the views in. Defaults to the store's database.",
        ),
    ] = None,
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--dry-run"],
            help="Print the SQL statements without executing them.",
        ),
    ] = False,
    check: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--check"],
            help="Exit with an error when a view is missing or out of date, without changing anything.",
        ),
    ] = False,
    manifest: Annotated[
        Path | None,
        cyclopts.Parameter(
            name=["--manifest"],
            help="Write the view manifest (view names, feature versions, id columns and column types) to this JSON file.",
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            name=["--format"],
            help="Output format: 'plain' (default) or 'json'.",
        ),
    ] = "plain",
) -> None:
    """Create or update SQL views of current feature rows and feature statistics.

    Every internal feature with a table in the store gets a `<table>__current` view that returns the
    same rows as `MetadataStore.read()`: the current feature version, the latest row per sample and
    no soft-deleted rows. The `metaxy_feature_statistics` view reports, per feature, the row count and
    the missing, stale and orphaned counts of `mx metadata status`, computed live by the database.
    The `metaxy_views_manifest` view describes all views.

    Views are pinned to the feature versions of the current feature graph. Run this command after
    feature definitions change, for example next to `mx push` in a deployment pipeline.
    """
    from metaxy.cli.context import AppContext
    from metaxy.ext.ibis.metadata_store import IbisMetadataStore
    from metaxy.ext.ibis.views import check_views, plan_views, supports_views, sync_views
    from metaxy.models.types import FeatureKey

    if dry_run and check:
        exit_with_error(
            CLIError(code=CLIErrorCode.CONFLICTING_FLAGS, message="--dry-run and --check cannot be used together."),
            format,
        )

    context = AppContext.get()
    context.raise_command_cannot_override_project()
    metadata_store = context.get_store(store)

    if not isinstance(metadata_store, IbisMetadataStore) or not supports_views(metadata_store):
        exit_with_error(
            CLIError(
                code=CLIErrorCode.GENERIC_ERROR,
                message=f"Store {metadata_store.display()} does not support views. "
                "Views require an Ibis-based store with the native versioning engine, such as ClickHouse or DuckDB.",
            ),
            format,
        )

    graph = context.graph
    internal_keys = [key for key in graph.list_features() if not graph.get_feature_definition(key).is_external]
    selected_keys = [FeatureKey(key) for key in feature] if feature else internal_keys
    unknown_keys = [key for key in selected_keys if key not in internal_keys]
    if unknown_keys:
        exit_with_error(
            CLIError(
                code=CLIErrorCode.FEATURES_NOT_FOUND,
                message="Views can only be created for internal features of the current project.",
                details={"features": [key.to_string() for key in unknown_keys]},
            ),
            format,
        )

    with metadata_store.open("w" if not (dry_run or check) else "r"), graph.use():
        plan = plan_views(metadata_store, graph, internal_keys, schema=schema)
        selected_views = [view for view in plan.views if view.feature_key is None or view.feature_key in selected_keys]

        for key in plan.skipped_features:
            console.print(f"[yellow]Skipping[/yellow] {key.to_string()}: no table in the store yet")

        if manifest is not None:
            manifest.write_text(json.dumps(_manifest_document(plan), indent=2) + "\n")
            console.print(f"[green]✓[/green] Wrote manifest to {manifest}")

        if dry_run:
            for statement in plan.create_statements(selected_views):
                data_console.print(f"{statement};", markup=False, highlight=False, soft_wrap=True)
            return

        if check:
            outdated = check_views(metadata_store, plan, selected_views)
            if outdated:
                exit_with_error(
                    CLIError(
                        code=CLIErrorCode.VIEWS_OUT_OF_DATE,
                        message="Views are missing or out of date. Run `mx views sync` to update them.",
                        details={"views": outdated},
                    ),
                    format,
                )
            _print_result(plan, "Views are up to date", format)
            return

        sync_views(metadata_store, plan, selected_views)
        _print_result(plan, f"Synchronized {len(selected_views) + 1} views", format)


def _manifest_document(plan: ViewPlan) -> dict[str, object]:
    return {"schema": plan.schema, "views": plan.manifest_entries}


def _print_result(plan: ViewPlan, message: str, format: OutputFormat) -> None:
    if format == "json":
        data_console.print_json(json.dumps(_manifest_document(plan)))
    else:
        console.print(f"[green]✓[/green] {message}")
