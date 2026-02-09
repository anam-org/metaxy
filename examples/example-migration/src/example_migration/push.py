"""Record feature graph snapshot (normally done in CI/CD)."""

import metaxy as mx
from metaxy.metadata_store.system import SystemTableStorage

config = mx.init()
with config.get_store() as store:
    result = SystemTableStorage(store).push_graph_snapshot()
    project_version = result.project_version
    print(f"📸 Recorded feature graph snapshot: {project_version[:16]}...")
    print(f"   Full ID: {project_version}")
