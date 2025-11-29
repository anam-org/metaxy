# PostgreSQL Configuration

PostgreSQL metadata store is configured via constructor parameters:

```python
from metaxy import PostgresMetadataStore

store = PostgresMetadataStore(
    connection_string="postgresql://user:pass@host:5432/db",
    # or individual parameters:
    # host="localhost",
    # port=5432,
    # database="metaxy",
    # user="user",
    # password="pass",
    schema="public",  # optional
    enable_pgcrypto=False,  # set True to auto-enable pgcrypto extension
    hash_algorithm="MD5",  # or "SHA256" (requires pgcrypto)
    auto_create_tables=True,
)
```

## Parameters

See [API Reference](api.md) for detailed parameter documentation.
