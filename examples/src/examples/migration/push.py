"""Record feature graph snapshot (normally done in CI/CD)."""

from metaxy import MetaxyConfig

config = MetaxyConfig.load(search_parents=True)
with config.get_store() as store:
    result = store.record_feature_graph_snapshot()
    snapshot_version = result.snapshot_version
    print(f"📸 Recorded feature graph snapshot: {snapshot_version[:16]}...")
    print(f"   Full ID: {snapshot_version}")
