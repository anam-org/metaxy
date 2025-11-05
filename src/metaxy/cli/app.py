"""Main Metaxy CLI application."""

from pathlib import Path
from typing import Annotated

import cyclopts

from metaxy import init_metaxy
from metaxy._version import __version__
from metaxy.cli.console import console, error_console

# Main app
app = cyclopts.App(
    name="metaxy",  # pyrefly: ignore[unexpected-keyword]
    help="Metaxy - Feature Metadata Management",  # pyrefly: ignore[unexpected-keyword]
    version=__version__,  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
    config=cyclopts.config.Env(  # pyrefly: ignore[unexpected-keyword,implicit-import]
        "METAXY_",  # Every environment variable for setting the arguments will begin with this.  # pyrefly: ignore[bad-argument-count]
    ),
)


@app.command
def shell():
    """Start interactive shell."""
    app.interactive_shell()


# Meta app for global parameters


@app.meta.default
def launcher(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config_file: Annotated[
        Path | None,
        cyclopts.Parameter(
            None,
            help="Global option. Path to the Metaxy configuration file. Defaults to auto-discovery.",
        ),
    ] = None,
    project: Annotated[
        str | None,
        cyclopts.Parameter(
            None,
            help="Global option. Metaxy project to work with. Some commands may forbid setting this argument.",
        ),
    ] = None,
    all_projects: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--all-projects"],
            help="Global option. Operate on all available Metaxy projects. Some commands may forbid setting this argument.",
        ),
    ] = False,
):
    """Metaxy CLI.

    Auto-discovers config file (metaxy.toml or pyproject.toml) by searching
    current directory and parent directories.

    Environment variables can override config (METAXY_STORE, METAXY_MIGRATIONS_DIR, etc).
    """
    import logging
    import os

    logging.getLogger().setLevel(os.environ.get("METAXY_LOG_LEVEL", "INFO"))

    # Load Metaxy configuration with parent directory search
    # This handles TOML discovery, env vars, and entrypoint loading
    config = init_metaxy(config_file=config_file, search_parents=True)

    # Store config in context for commands to access
    # Commands will instantiate and open store as needed
    from metaxy.cli.context import AppContext

    AppContext.set(config, cli_project=project, all_projects=all_projects)

    # Run the actual command
    app(tokens)


# Register subcommands (lazy loading via import strings)
app.command("metaxy.cli.migrations:app", name="migrations")
app.command("metaxy.cli.graph:app", name="graph")
app.command("metaxy.cli.graph_diff:app", name="graph-diff")
app.command("metaxy.cli.list:app", name="list")
app.command("metaxy.cli.metadata:app", name="metadata")


def main():
    """Entry point for the CLI."""
    app.meta()


if __name__ == "__main__":
    main()
