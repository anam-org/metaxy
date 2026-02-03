"""Lock command for Metaxy CLI."""

from typing import Annotated

import cyclopts

from metaxy.cli.console import console, error_console

# Lock subcommand app
app = cyclopts.App(
    name="lock",
    help="Generate `metaxy.lock` file with external feature definitions fetched from the metadata store.",
    console=console,
    error_console=error_console,
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
    """Fetch external feature definitions and serialize them to metaxy.lock.

    Analyzes the current feature graph to find external dependencies, then fetches
    those feature definitions (and their transitive dependencies) from the metadata store.

    This functionality is currently experimental.
    """
    from pathlib import Path

    from metaxy.cli.context import AppContext
    from metaxy.cli.utils import CLIError, CLIErrorCode
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
    elif config.lock_file:
        output_path = config.lock_file
    else:
        output_path = Path.cwd() / config.metaxy_lock_path

    metadata_store = context.get_store(store)

    try:
        count = generate_lock_file(
            metadata_store,
            output_path,
            exclude_project=config.project,
        )
    except FeatureNotFoundError as e:
        error = CLIError(
            code=CLIErrorCode.FEATURES_NOT_FOUND,
            message=str(e),
            details={"features": [k.to_string() for k in e.keys]},
            hint="Run `metaxy push` in the project that defines these features.",
        )
        console.print(error.to_plain())
        raise SystemExit(1) from None

    if count == 0:
        console.print(f"[green]✓[/green] Created empty lock file at {output_path}")
    else:
        console.print(f"[green]✓[/green] Locked {count} feature(s) to {output_path}")
