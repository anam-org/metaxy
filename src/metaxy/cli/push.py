"""Push command for Metaxy CLI."""

from typing import Annotated

import cyclopts

from metaxy.cli.console import console, data_console, error_console

# Push subcommand app
app = cyclopts.App(
    name="push",
    help="Push feature definitions to the metadata store",
    console=console,
    error_console=error_console,
)


@app.default
def push(
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store)",
        ),
    ] = None,
    *,
    tags: Annotated[
        dict[str, str] | None,
        cyclopts.Parameter(
            name=["--tags", "-t"],
            help="Arbitrary key-value pairs to attach to the pushed snapshot. Example: `--tags.git_commit abc123def`.",
        ),
    ] = None,
):
    """Serialize all Metaxy features to the metadata store.

    This is intended to be invoked in a CD pipeline **before** running Metaxy code in production.
    """
    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.system.models import METAXY_TAG
    from metaxy.metadata_store.system.storage import SystemTableStorage

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    # Ensure project is set for push (required to determine which features to include)
    if context.config.project is None:
        console.print(
            "[red]Error:[/red] The 'project' field must be set in metaxy.toml for 'metaxy push'.",
            style="bold",
        )
        raise SystemExit(1)

    metadata_store = context.get_store(store)

    tags = tags or {}

    assert METAXY_TAG not in tags, "`metaxy` tag is reserved for internal use"

    with metadata_store.open("w"):
        storage = SystemTableStorage(metadata_store)
        result = storage.push_graph_snapshot(
            project=context.config.project,
            tags=tags,
        )

        # Log store metadata for the system table
        from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

        store_metadata = metadata_store.get_store_metadata(FEATURE_VERSIONS_KEY)
        if store_metadata:
            console.print(f"[dim]Recorded at: {store_metadata}[/dim]")

        # Scenario 1: New snapshot (computational changes)
        if not result.already_pushed:
            console.print("[green]✓[/green] Recorded feature graph")

        # Scenario 2: Feature info updates to existing snapshot
        elif result.updated_features:
            console.print("[blue]ℹ[/blue] [cyan]Updated feature information[/cyan] (no topological changes)")
            console.print("  [dim]Updated features:[/dim]")
            for feature_key in result.updated_features:
                console.print(f"    [yellow]- {feature_key}[/yellow]")

        # Scenario 3: No changes
        else:
            console.print("[green]✓[/green] [green]Snapshot already recorded[/green] [dim](no changes)[/dim]")

        # Always output the project version to stdout (for scripting)
        # Note: project_version is "empty" when graph has no features
        data_console.print(result.project_version)
