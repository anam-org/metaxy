---
name: dagster
description: Write Dagster pipelines using correct import patterns, the modern ConfigurableResource API, and proper asset definitions. Covers integration with Metaxy for metadata management.
---

# Dagster - Data Orchestration Framework

Dagster is a data orchestration platform that helps build, test, and monitor data pipelines with strong typing and extensive metadata capabilities.

Docs: https://docs.dagster.io/

## Import Conventions

**ALWAYS use the namespace import pattern:**

```python
import dagster as dg
```

**Reference all Dagster objects from the `dg` namespace:**

```python
@dg.asset
def my_asset(context: dg.AssetExecutionContext) -> int:
    context.log.info("Processing...")
    return 42


defs = dg.Definitions(
    assets=[my_asset],
    resources={...},
)
```

## Modern Resource API: ConfigurableResource

Dagster resources should use the modern `ConfigurableResource` API, not the legacy `@resource` decorator.

### Basic Resource Pattern

```python
import dagster as dg


class DatabaseResource(dg.ConfigurableResource):
    """Database connection resource with configuration."""

    host: str
    port: int = 5432
    database: str

    def connect(self):
        """Connect to the database."""
        return create_connection(self.host, self.port, self.database)

    def query(self, sql: str):
        """Execute a SQL query."""
        conn = self.connect()
        return conn.execute(sql)


# Use in assets
@dg.asset
def data_from_database(
    context: AssetExecutionContext,
    db_conn: DatabaseResource,
) -> dict:
    context.log.info("Querying database...")
    return db_conn.query("SELECT * FROM table")


# Configure in Definitions
defs = dg.Definitions(
    assets=[data_from_database],
    resources={
        "db_conn": DatabaseResource(
            host="localhost",
            port=5432,
            database="mydb",
        ),
    },
)
```

### Resource with Context Manager

Resources that need setup/teardown should implement context manager protocol:

```python
import dagster as dg
from typing import Any


class ConnectionResource(dg.ConfigurableResource):
    connection_string: str

    def __enter__(self):
        self._conn = create_connection(self.connection_string)
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_conn"):
            self._conn.close()


# Usage
@dg.asset
def process_data(conn: ConnectionResource):
    with conn as connection:
        # Connection is managed automatically
        return connection.query("SELECT * FROM data")
```

### Resource Configuration from Config Files

```python
import dagster as dg


class APIResource(dg.ConfigurableResource):
    api_key: str
    base_url: str = "https://api.example.com"
    timeout: int = 30


# In dagster.yaml or environment
# resources:
#   api:
#     config:
#       api_key: ${API_KEY}
#       base_url: https://api.example.com
#       timeout: 60
```

## Asset Definitions

### Basic Asset

```python
import dagster as dg
from dagster import AssetExecutionContext


@dg.asset
def raw_data(context: AssetExecutionContext) -> dict:
    """Root asset that loads raw data."""
    context.log.info("Loading raw data...")
    return {"data": [1, 2, 3]}
```

### Asset with Dependencies

```python
import dagster as dg
from dagster import AssetExecutionContext


@dg.asset
def processed_data(
    context: AssetExecutionContext,
    raw_data: dict,  # Dependency on raw_data asset
) -> dict:
    """Process raw data."""
    context.log.info("Processing data...")
    return {"processed": raw_data["data"]}
```

### Asset with Resource Dependencies

```python
import dagster as dg
from dagster import AssetExecutionContext


@dg.asset
def data_from_api(
    context: AssetExecutionContext,
    api: APIResource,  # Resource dependency
) -> dict:
    """Fetch data from API."""
    response = api.get("/data")
    return response.json()
```

### Asset with Metadata

```python
import dagster as dg
from dagster import AssetExecutionContext, Output


@dg.asset
def enriched_data(context: AssetExecutionContext) -> Output[dict]:
    """Asset that returns metadata."""
    data = {"value": 42}

    return Output(
        data,
        metadata={
            "num_records": len(data),
            "preview": str(data)[:100],
        },
    )
```

## Partitioned Assets

### Static Partitions

```python
import dagster as dg
from dagster import AssetExecutionContext

# Define partitions
video_partitions = dg.StaticPartitionsDefinition(["video_1", "video_2", "video_3"])


@dg.asset(partitions_def=video_partitions)
def video_frames(context: AssetExecutionContext) -> dict:
    """Process one video per partition."""
    video_id = context.partition_key
    context.log.info(f"Processing video: {video_id}")
    return process_video(video_id)
```

### Dynamic Partitions

```python
import dagster as dg
from dagster import AssetExecutionContext

# Define dynamic partitions
videos = dg.DynamicPartitionsDefinition(name="videos")


@dg.asset(partitions_def=videos)
def video_embeddings(context: AssetExecutionContext) -> dict:
    """Generate embeddings for each video."""
    video_id = context.partition_key
    return generate_embeddings(video_id)


# Add partitions dynamically
def add_new_video(video_id: str):
    return dg.AddDynamicPartitionsRequest(
        partitions_def_name="videos",
        partition_keys=[video_id],
    )
```

### Time-based Partitions

```python
import dagster as dg
from dagster import AssetExecutionContext
from datetime import datetime

# Daily partitions
daily_partitions = dg.DailyPartitionsDefinition(start_date="2024-01-01")


@dg.asset(partitions_def=daily_partitions)
def daily_metrics(context: AssetExecutionContext) -> dict:
    """Compute metrics for each day."""
    partition_date = context.partition_key
    context.log.info(f"Processing date: {partition_date}")
    return compute_metrics(partition_date)
```

## IO Managers

IO Managers handle how assets are stored and loaded.

### Custom IO Manager

```python
import dagster as dg
from dagster import AssetExecutionContext, InputContext, OutputContext


class CustomIOManager(dg.ConfigurableIOManager):
    base_path: str

    def handle_output(self, context: OutputContext, obj):
        """Store asset output."""
        path = f"{self.base_path}/{context.asset_key}.json"
        save_json(path, obj)

    def load_input(self, context: InputContext):
        """Load asset input."""
        path = f"{self.base_path}/{context.asset_key}.json"
        return load_json(path)


# Configure in Definitions
defs = dg.Definitions(
    assets=[...],
    resources={
        "io_manager": CustomIOManager(base_path="/data"),
    },
)
```

### IO Manager with Per-Asset Configuration

```python
import dagster as dg
from dagster import AssetExecutionContext


@dg.asset(
    io_manager_key="s3_io_manager",  # Use specific IO manager
    metadata={
        "s3_path": "s3://bucket/data/asset.parquet",
    },
)
def cloud_data(context: AssetExecutionContext) -> dict:
    return {"data": [1, 2, 3]}
```
