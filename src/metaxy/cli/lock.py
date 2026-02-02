"""Lock command for Metaxy CLI."""

from typing import Annotated

import cyclopts

from metaxy.cli.console import console, error_console

# Lock subcommand app
app = cyclopts.App(
    name="lock",
    help="Generate `metaxy.lock` file with external feature definitions from the metadata store",
    console=console,
    error_console=error_console,
    show=False,
)


@app.default
def lock(
    store: Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--store"],
            help="Metadata store to use (defaults to configured default store)",
        ),
    ] = None,
    *,
    output: Annotated[
        str,
        cyclopts.Parameter(
            name=["--output", "-o"],
            help="Output file path (defaults to metaxy.lock in config directory)",
        ),
    ] = "",
):
    """Fetch feature definitions for features specified in Metaxy configuration and serialize them to metaxy.lock.

    Reads the `features` list from `metaxy.toml` (or `tool.metaxy.features` from `pyproject.toml`) and fetches those feature definitions
    from the metadata store, writing them to a TOML lock file.

    Features belonging to the current project are excluded from the lock file.
    """
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.utils.lock_file import generate_lock_file

    context = AppContext.get()
    context.raise_command_cannot_override_project()

    # Ensure project is set (required to filter out current project's features)
    if context.config.project is None:
        console.print(
            "[red]Error:[/red] The 'project' field must be set in metaxy.toml for 'metaxy lock'.",
            style="bold",
        )
        raise SystemExit(1)

    config = context.config

    # Determine output path
    if output:
        output_path = Path(output)
    elif config.config_file:
        output_path = config.config_file.parent / "metaxy.lock"
    else:
        output_path = Path.cwd() / "metaxy.lock"

    metadata_store = context.get_store(store)

    try:
        count = generate_lock_file(
            metadata_store,
            config.features,
            output_path,
            exclude_project=config.project,
        )
    except FeatureNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    if count == 0:
        console.print(f"[green]✓[/green] Created empty lock file at {output_path}")
    else:
        console.print(f"[green]✓[/green] Locked {count} feature(s) to {output_path}")
