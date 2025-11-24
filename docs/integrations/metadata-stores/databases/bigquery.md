# BigQueryMetadataStore

Metaxy implements [`BigQueryMetadataStore`][metaxy.metadata_store.bigquery.BigQueryMetadataStore]. It uses [BigQuery](https://cloud.google.com/bigquery) as metadata storage and versioning engine.

!!! warning

    It's on the user to set up infrastructure for Metaxy correctly. Make sure to have large tables partitioned as appropriate for your use case.

!!! tip

    BigQuery automatically optimizes queries on partitioned tables. When tables are partitioned (e.g., by date or ingestion time with `_PARTITIONTIME`), BigQuery will automatically prune partitions based on WHERE clauses in queries, without needing explicit configuration in the metadata store.

## Installation

```shell
pip install 'metaxy[bigquery]'
```

## Authentication

BigQuery supports multiple authentication methods with the following priority:

1. Explicit service account file via `credentials_path`
2. Explicit credentials object
3. Application Default Credentials (ADC)
4. Google Cloud SDK credentials

### Service Account

```py
store = BigQueryMetadataStore(
    project_id="my-project",
    dataset_id="my_dataset",
    credentials_path="/path/to/service-account.json",
)
```

### Application Default Credentials

```py
# Uses ADC from environment
store = BigQueryMetadataStore(
    project_id="my-project",
    dataset_id="my_dataset",
)
```

## Configuration

### Location

Specify the data location for regional data residency requirements:

```py
store = BigQueryMetadataStore(
    project_id="my-project",
    dataset_id="my_dataset",
    location="EU",  # or "US", "asia-northeast1", etc.
)
```

### Hash Algorithms

BigQuery supports multiple native hash functions:

```py
from metaxy.versioning.types import HashAlgorithm

store = BigQueryMetadataStore(
    project_id="my-project",
    dataset_id="my_dataset",
    hash_algorithm=HashAlgorithm.SHA256,  # or MD5, FARMHASH
)
```

---

## References

Learn more in the [API docs][metaxy.metadata_store.bigquery.BigQueryMetadataStore].
