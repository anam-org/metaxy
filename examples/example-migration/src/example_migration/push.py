"""Record feature graph snapshot (normally done in CI/CD)."""

from metaxy import init_metaxy
from metaxy.metadata_store.system import SystemTableStorage

config = init_metaxy()
with config.get_store() as store:
    result = SystemTableStorage(store).push_graph_snapshot()
    snapshot_version = result.snapshot_version
    print(f"📸 Recorded feature graph snapshot: {snapshot_version[:16]}...")
    print(f"   Full ID: {snapshot_version}")
