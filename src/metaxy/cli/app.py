"""Main Metaxy CLI application."""

from pathlib import Path
from typing import Annotated

import cyclopts

from metaxy import init_metaxy
from metaxy._version import __version__
from metaxy.cli.console import console, error_console

# Main app
app = cyclopts.App(
    name="metaxy",
    version=__version__,
    console=console,
    error_console=error_console,
    config=cyclopts.config.Env(  # ty: ignore[invalid-argument-type]
        "METAXY_",  # Every environment variable for setting the arguments will begin with this.
    ),
    help_epilogue="Learn more in [Metaxy docs](https://docs.metaxy.io)",
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
    sync: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--sync"],
            help="Global option. Load external feature definitions from the metadata store before executing the command.",
        ),
    ] = False,
    locked: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--locked"],
            help="Global option. When used with --sync, raise an error if external feature versions don't match the actual versions from the metadata store.",
        ),
    ] = False,
):
    """Metaxy CLI.

    Auto-discovers configuration (`metaxy.toml` or `pyproject.toml`) in current or parent directories.
    Feature definitions are collected via [feature discovery](https://docs.metaxy.io/main/learn/feature-discovery/).
    Supports loading environment variables from a `.env` file in the current directory.
    """
    import logging
    import os

    logging.getLogger().setLevel(os.environ.get("METAXY_LOG_LEVEL", "INFO"))

    # Load Metaxy configuration with parent directory search
    # This handles TOML discovery, env vars, and entrypoint loading
    config = init_metaxy(config_file)

    # Store config in context for commands to access
    # Commands will instantiate and open store as needed
    from metaxy.cli.context import AppContext

    AppContext.set(config, cli_project=project, all_projects=all_projects, sync=sync, locked=locked)

    # Run the actual command
    app(tokens)


# Register subcommands (lazy loading via import strings)
app.command("metaxy.cli.config:app", name="config")
app.command("metaxy.cli.migrations:app", name="migrations")
app.command("metaxy.cli.graph:app", name="graph")
app.command("metaxy.cli.graph_diff:app", name="graph-diff")
app.command("metaxy.cli.list:app", name="list")
app.command("metaxy.cli.metadata:app", name="metadata")
app.command("metaxy.cli.mcp:app", name="mcp")


def main():
    """Entry point for the CLI."""
    from dotenv import load_dotenv

    # Load .env from the current working directory before parsing CLI arguments
    # This allows .env to configure METAXY_* environment variables for the CLI
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    app.meta()


if __name__ == "__main__":
    main()
