"""Push command for recording feature versions."""

from rich.console import Console

console = Console()


def push(store: str | None = None):
    """Record all feature versions (push graph snapshot).

    Records all features in the active graph to the metadata store
    with a deterministic snapshot version. This should be run after deploying
    new feature definitions.

    Example:
        $ metaxy graph push

        ✓ Recorded feature graph
          Snapshot version: abc123def456...

        # Or if already recorded:
        ℹ Snapshot already recorded (no changes)
          Snapshot version: abc123def456...

        # Or if metadata-only changes:
        ℹ Updated feature graph metadata (no topological changes)
          Features with metadata changes:
            - video/processing
            - user/profile
          Snapshot version: abc123def456...

    Args:
        store: The metadata store to use. Defaults to the default store.
    """
    from metaxy.cli.context import AppContext
    from metaxy.models.feature import FeatureGraph

    context = AppContext.get()
    metadata_store = context.get_store(store)

    with metadata_store:
        # Get active graph
        active_graph = FeatureGraph.get_active()
        if len(active_graph.features_by_key) == 0:
            console.print("[yellow]⚠[/yellow] No features in active graph")
            return

        # Record feature graph snapshot (idempotent)
        # Returns SnapshotPushResult
        result = metadata_store.record_feature_graph_snapshot()

        # Scenario 1: New snapshot (computational changes)
        if not result.already_recorded:
            console.print("[green]✓[/green] Recorded feature graph")
            console.print(f"  Snapshot version: {result.snapshot_version}")

        # Scenario 2: Metadata-only changes
        elif result.metadata_changed:
            console.print(
                "[blue]ℹ[/blue] Updated feature graph metadata (no topological changes)"
            )
            if result.features_with_spec_changes:
                console.print("  Features with metadata changes:")
                for feature_key in result.features_with_spec_changes:
                    console.print(f"    - {feature_key}")
            console.print(f"  Snapshot version: {result.snapshot_version}")

        # Scenario 3: No changes
        else:
            console.print("[blue]ℹ[/blue] Snapshot already recorded (no changes)")
            console.print(f"  Snapshot version: {result.snapshot_version}")


if __name__ == "__main__":
    push()
