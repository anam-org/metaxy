!!! tip "Flushing Metadata"
    Usually it's desired to write metadata to the metadata store as soon as it becomes available.
    This ensures the pipeline can resume processing after a failure and no data is lost.
    [`BatchedMetadataWriter`][metaxy.utils.BatchedMetadataWriter] to achieve this: it writes metadata in real-time from a background thread.
