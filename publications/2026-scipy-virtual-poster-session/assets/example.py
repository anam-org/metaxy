from my_project.features import FaceDetection

import metaxy as mx
from metaxy.metadata_store.duckdb import DuckDBMetadataStore

mx.init()  # load feature definitions

with DuckDBMetadataStore("meta.duckdb") as store:
    diff = store.resolve_update(FaceDetection)
    # diff: new . changed . orphaned

    if diff.added.height or diff.changed.height:
        rows = run_face_detection(diff)  # GPU work
        store.write_metadata(FaceDetection, rows)
