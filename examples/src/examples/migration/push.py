"""Record feature graph snapshot (normally done in CI/CD)."""

from metaxy import MetaxyConfig

config = MetaxyConfig.load(search_parents=True)
with config.get_store() as store:
    snapshot_id = store.record_feature_graph_snapshot()
    print(f"📸 Recorded feature graph snapshot: {snapshot_id[:16]}...")
    print(f"   Full ID: {snapshot_id}")
