"""Push command for recording feature versions."""

from rich.console import Console

console = Console()


def push(store: str | None = None):
    """Record all feature versions (push graph snapshot).

    Records all features in the active graph to the metadata store
    with a deterministic snapshot ID. This should be run after deploying
    new feature definitions.

    Example:
        $ metaxy push

        Recorded 5 features with snapshot ID: abc123def456...

    Args:
        store: The metadata store to use. Defaults to the default store.
    """
    from metaxy.cli.context import get_store

    metadata_store = get_store(store)

    with metadata_store:
        snapshot_id = metadata_store.serialize_feature_graph()

        console.print("[green]✓[/green] Recorded feature graph")
        console.print(f"  Snapshot ID: {snapshot_id}")


if __name__ == "__main__":
    push()
