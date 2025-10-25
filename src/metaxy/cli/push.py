"""Push command for recording feature versions."""

from rich.console import Console

console = Console()


def push(store: str | None = None):
    """Record all feature versions (push graph snapshot).

    Records all features in the active graph to the metadata store
    with a deterministic snapshot version. This should be run after deploying
    new feature definitions.

    Example:
        $ metaxy push

        ✓ Recorded feature graph
          Snapshot version: abc123def456...

        # Or if already recorded:
        ℹ Snapshot already recorded (skipped)
          Snapshot version: abc123def456...

    Args:
        store: The metadata store to use. Defaults to the default store.
    """
    from metaxy.cli.context import get_store

    metadata_store = get_store(store)

    with metadata_store:
        snapshot_version, was_already_recorded = (
            metadata_store.record_feature_graph_snapshot()
        )

        if was_already_recorded:
            console.print("[blue]ℹ[/blue] Snapshot already recorded (skipped)")
            console.print(f"  Snapshot version: {snapshot_version}")
        else:
            console.print("[green]✓[/green] Recorded feature graph")
            console.print(f"  Snapshot version: {snapshot_version}")


if __name__ == "__main__":
    push()
