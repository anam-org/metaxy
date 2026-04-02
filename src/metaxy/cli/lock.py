"""Lock command for Metaxy CLI."""

from functools import reduce
from operator import or_
from typing import Annotated

import cyclopts
from rich.status import Status

from metaxy.cli.console import console, error_console

# Lock subcommand app
app = cyclopts.App(
    name="lock",
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
    """Generate `metaxy.lock` file with [external feature definitions](https://docs.metaxy.io/stable/guide/concepts/definitions/external-features/) fetched from the metadata store.

    Analyzes the current feature graph to find external dependencies, then fetches
    those feature definitions (and their transitive dependencies) from the metadata store.

    Distribution entry points are automatically filtered to only load features from
    the current project and its Python dependencies (including transitive), preventing
    unrelated packages from polluting the lock file.

    This functionality is currently experimental.
    """
    import time
    from collections import defaultdict
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
        output_path = Path.cwd() / (config.metaxy_lock_path or "metaxy.lock")

    metadata_store = context.get_store(store)
    relative_path = output_path.relative_to(Path.cwd()) if output_path.is_relative_to(Path.cwd()) else output_path

    try:
        t0 = time.monotonic()
        with Status(
            f"Updating [link={output_path.as_uri()}]{relative_path}[/link] with external features from {metadata_store}...",
            console=console,
            spinner="dots",
        ):
            result = generate_lock_file(
                metadata_store,
                output_path,
                exclude_project=config.project,
                selection=reduce(or_, config.extra_features) if config.extra_features else None,
            )
        elapsed = time.monotonic() - t0
    except FeatureNotFoundError as e:
        error = CLIError(
            code=CLIErrorCode.FEATURES_NOT_FOUND,
            message=str(e),
            details={"features": [k.to_string() for k in e.keys]},
            hint="Run `metaxy push` in the project that defines these features.",
        )
        console.print(error.to_plain())
        raise SystemExit(1) from None

    # Collect change lines grouped by project, sorted by key within each
    lines_by_project: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

    def _add(project: str, key: str, line: str) -> None:
        lines_by_project[project].append((key, line))

    for change in result.added:
        _add(change.project, change.key, f" [green]+[/green] {change.key} [dim]({change.version})[/dim]")
    for update in result.updated:
        if update.metadata_only:
            _add(
                update.project,
                update.key,
                f" [yellow]~[/yellow] {update.key} [dim]({update.old_version}, non-versioning metadata update)[/dim]",
            )
        else:
            _add(
                update.project,
                update.key,
                f" [yellow]~[/yellow] {update.key} [dim]({update.old_version} -> {update.new_version})[/dim]",
            )
    for change in result.removed:
        _add(change.project, change.key, f" [red]-[/red] {change.key} [dim]({change.version})[/dim]")

    for project, entries in sorted(lines_by_project.items()):
        if len(lines_by_project) > 1:
            console.print(f" [bold]{project}[/bold]")
        for _, line in sorted(entries):
            console.print(line)

    lock_link = f"[link={output_path.as_uri()}]{relative_path}[/link]"
    if result.changed:
        console.print(f"Updated {lock_link} from {metadata_store} in [bold]{elapsed:.2f}s[/bold].")
    else:
        console.print(f"Lock file {lock_link} is up to date [dim]({elapsed:.2f}s)[/dim].")
