"""Record feature graph snapshot (normally done in CI/CD)."""

from metaxy import init_metaxy

config = init_metaxy()
with config.get_store() as store:
    result = store.record_feature_graph_snapshot()
    snapshot_version = result.snapshot_version
    print(f"📸 Recorded feature graph snapshot: {snapshot_version[:16]}...")
    print(f"   Full ID: {snapshot_version}")
