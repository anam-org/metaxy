"""Main Metaxy CLI application."""

from typing import Annotated

import cyclopts
from rich.console import Console

from metaxy._version import __version__
from metaxy.config import MetaxyConfig

# Rich console for formatted output
console = Console()

# Main app
app = cyclopts.App(
    name="metaxy",  # pyrefly: ignore[unexpected-keyword]
    help="Metaxy - Feature Metadata Management",  # pyrefly: ignore[unexpected-keyword]
    version=__version__,  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    config=cyclopts.config.Env(  # pyrefly: ignore[unexpected-keyword,implicit-import]
        "METAXY_",  # Every environment variable for setting the arguments will begin with this.  # pyrefly: ignore[bad-argument-count]
    ),
)


def push():
    """Record all feature versions (push graph snapshot).

    This should be called during CD (Continuous Deployment) to record what
    feature versions are being deployed. Call this before the deployment goes live.

    Records a snapshot of the entire feature graph state based on code definitions.
    """
    from metaxy.cli.context import get_store
    from metaxy.models.feature import FeatureGraph

    # Get store from context
    metadata_store = get_store()

    with metadata_store:
        # Get the active graph
        graph = FeatureGraph.get_active()

        # Record all feature versions
        snapshot_id, was_already_recorded = (
            metadata_store.record_feature_graph_snapshot()
        )
        feature_count = len(graph.features_by_key)

        if was_already_recorded:
            app.console.print(
                f"[blue]ℹ[/blue] Snapshot already recorded (skipped): [bold]{snapshot_id}[/bold]"
            )
        else:
            app.console.print(
                f"[green]✓[/green] Recorded graph snapshot: [bold]{snapshot_id}[/bold]"
            )
        app.console.print(f"  {feature_count} features")


@app.command
def shell():
    """Start interactive shell."""
    app.interactive_shell()


# Meta app for global parameters
@app.meta.default
def launcher(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
):
    """Metaxy CLI.

    Auto-discovers config file (metaxy.toml or pyproject.toml) by searching
    current directory and parent directories.

    Environment variables can override config (METAXY_STORE, METAXY_MIGRATIONS_DIR, etc).
    """
    import logging

    logging.getLogger().setLevel(logging.INFO)

    # Load Metaxy configuration with parent directory search
    # This handles TOML discovery, env vars, and entrypoint loading
    metaxy_config = MetaxyConfig.load(search_parents=True)

    # Store config in context for commands to access
    # Commands will instantiate and open store as needed
    from metaxy.cli.context import set_config

    set_config(metaxy_config)

    # Run the actual command
    app(tokens)


# Register subcommands (lazy loading via import strings)
app.command("metaxy.cli.migrations:app", name="migrations")
app.command("metaxy.cli.push:push", name="push")
app.command("metaxy.cli.list:app", name="list")
app.command("metaxy.cli.metadata:app", name="metadata")


def main():
    """Entry point for the CLI."""
    app.meta()


if __name__ == "__main__":
    main()
