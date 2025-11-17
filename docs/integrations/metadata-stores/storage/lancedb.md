# LanceDBMetadataStore

## Installation

Install the [`lancedb`](https://lancedb.com/) backend alongside Metaxy using the `lancedb` dependency group.

## Usage

```py
from pathlib import Path

import polars as pl

from metaxy.metadata_store.lancedb import LanceDBMetadataStore

root = Path("/data/metaxy/lancedb")
feature = MyFeature

with LanceDBMetadataStore(root) as store:
    with store.allow_cross_project_writes():
        store.write_metadata(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "provenance_by_field": [
                        {"frames": "hash1", "audio": "hash1"},
                        {"frames": "hash2", "audio": "hash2"},
                    ],
                }
            ),
        )

    metadata = store.read_metadata(feature)
    if metadata is not None:
        df = metadata.collect().to_polars()
        print(df)
```

### Remote Connections

LanceDB supports various remote storage backends and cloud services, not just local filesystem:

#### S3 and S3-Compatible Storage

Connect to S3 or S3-compatible storage (MinIO, DigitalOcean Spaces, etc.) using S3 URIs:

```py
from metaxy.metadata_store.lancedb import LanceDBMetadataStore

# AWS S3
store = LanceDBMetadataStore("s3://my-bucket/metaxy/metadata")

# With fallback to production S3 bucket
prod_store = LanceDBMetadataStore("s3://prod-bucket/metadata")
dev_store = LanceDBMetadataStore(
    "s3://dev-bucket/metadata", fallback_stores=[prod_store]
)
```

**Requirements:**

- AWS credentials configured via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Or IAM role with appropriate S3 permissions
- Ensure the bucket exists and your credentials have read/write access

#### LanceDB Cloud (SaaS)

Connect to [LanceDB Cloud](https://lancedb.com/cloud) using the `db://` URI scheme:

```py
import os
from metaxy.metadata_store.lancedb import LanceDBMetadataStore

# Set your API key
os.environ["LANCEDB_API_KEY"] = "your-api-key-here"

# Connect to cloud database
store = LanceDBMetadataStore("db://my-database-name")

# Or pass credentials directly (avoids relying on environment variables)
store = LanceDBMetadataStore(
    "db://my-database-name",
    connect_kwargs={
        "api_key": "your-api-key-here",
        "region": "us-east-1",
    },
)
```

**Requirements:**

- LanceDB Cloud account and API key
- Either set `LANCEDB_API_KEY` environment variable or supply `connect_kwargs`
- Database must be created via LanceDB Cloud console or API

#### Google Cloud Storage

```py
# GCS bucket
store = LanceDBMetadataStore("gs://my-gcs-bucket/metaxy/metadata")
```

**Requirements:**

- Google Cloud credentials configured via `GOOGLE_APPLICATION_CREDENTIALS`
- Or default credentials via `gcloud auth application-default login`

#### Azure Blob Storage

```py
# Azure blob container
store = LanceDBMetadataStore("az://mycontainer/metaxy/metadata")
```

**Requirements:**

- Azure credentials configured via environment variables or Azure CLI

#### Remote HTTP/HTTPS Endpoints

For self-hosted LanceDB servers:

```py
# Remote LanceDB instance
store = LanceDBMetadataStore("https://lancedb.mycompany.com/databases/prod")
```

### Production Deployment Patterns

#### Multi-Environment Setup

```py
import os
from metaxy.metadata_store.lancedb import LanceDBMetadataStore

# Environment-aware configuration
env = os.getenv("ENV", "dev")

if env == "prod":
    # Production uses LanceDB Cloud
    os.environ["LANCEDB_API_KEY"] = os.getenv("LANCEDB_PROD_API_KEY")
    store = LanceDBMetadataStore("db://prod-database")
elif env == "staging":
    # Staging uses S3 with fallback to production
    with LanceDBMetadataStore("db://prod-database") as prod_store:
        store = LanceDBMetadataStore(
            "s3://staging-bucket/metadata", fallback_stores=[prod_store]
        )
else:
    # Local development
    store = LanceDBMetadataStore("./dev-metadata")
```

#### Cross-Cloud Fallback Chain

```py
# Local dev -> S3 staging -> LanceDB Cloud production
with LanceDBMetadataStore("db://prod-db") as prod_store:
    with LanceDBMetadataStore(
        "s3://staging/metadata", fallback_stores=[prod_store]
    ) as staging_store:
        dev_store = LanceDBMetadataStore(
            "./local-metadata", fallback_stores=[staging_store]
        )
```

### Storage Layout

- Each feature is stored as an individual Lance table inside the database directory.
- LanceDB manages the table schema, updates, and transactions. Metaxy simply appends new rows for each metadata write.

### Hashing

`LanceDBMetadataStore` defaults to `HashAlgorithm.XXHASH64`, identical to other non-SQL backends. You can override the hash algorithm via the constructor if you prefer a different supported variant.
