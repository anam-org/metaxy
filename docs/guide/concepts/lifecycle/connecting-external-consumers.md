---
title: "Connecting External Consumers"
description: "How services outside a Metaxy project read feature definitions and metadata."
---

# Connecting External Consumers

Dashboards, SQL tools, services written in other languages and AI agents sometimes need data from a Metaxy metadata store. This page explains how these external consumers can read feature definitions and feature metadata.

## Reading Feature Definitions

A consumer often needs to know which features exist, which ID columns and fields they have, and which feature version is current. [Metaxy keeps this information in the metadata store](./deployment.md).

Read the [`metaxy_system__feature_versions`](../../../reference/system-tables.md#metaxy_system__feature_versions) table, which [`metaxy push`](../../../reference/cli.md#metaxy-push) writes. For each `feature_key`, the row with the latest `recorded_at` holds the current definition.

## Reading Feature Metadata

It's often desired to read the current rows of a feature. Use one of the methods below to fetch feature metadata correctly, according to Metaxy's [storage layout](../../../metaxy/design.md#storage).

### From Python

If the consumer can run Python, use [`MetadataStore.read`][metaxy.MetadataStore.read]. It returns only the latest rows of the current feature version (see [Reading Metadata](../metadata-stores.md#reading-metadata)).

### From CLI

For a quick check from a shell, use the [`mx metadata`](../../../reference/cli.md#metaxy-metadata) commands. For example, `mx metadata status --format json` prints the number of rows and the number of missing, stale and orphaned samples of each feature.

### From Database Views

Dashboards and SQL tools often can only run SQL queries against the database. Give them a database view. Create the view from the frame that [`MetadataStore.read`][metaxy.MetadataStore.read] returns since it also carries the right SQL query:

<!-- invisible-code-block: python
import tempfile
from pathlib import Path

import duckdb
import polars as pl

from metaxy.ext.duckdb import DuckDBMetadataStore

database = Path(tempfile.mkdtemp()) / "metaxy.duckdb"
store = DuckDBMetadataStore(database=database, auto_create_tables=True)
with store.open("w"):
    store.write(MyFeature, df)
    # A newer row of sample "b" and a soft-deleted sample "c" give the table history that the view must hide.
    store.write(MyFeature, df.filter(pl.col("id") == "b").with_columns(status=pl.lit("done")))
    store.delete(MyFeature, nw.col("id") == "c")
-->

!!! example
    ```py
    with store.open("w"):
        store.conn.create_view(
            "my__feature__current",
            store.read(MyFeature, allow_fallback=False).to_native(),
            overwrite=True,
        )
    ```

    !!! note
        This example only works with Ibis-backed metadata stores.

<!-- invisible-code-block: python
with store:
    expected_rows = sorted(store.read(MyFeature).collect().to_polars().rows())

with duckdb.connect(str(database), read_only=True) as reader:
    view_rows = sorted(reader.sql("SELECT * FROM my__feature__current").fetchall())

assert view_rows == expected_rows
assert [(row[0], row[1]) for row in view_rows] == [("a", "active"), ("b", "done")]
-->

### MCP Server

An AI agent often has to find out which features exist before it can query their metadata. Give it the [MCP server](../../../integrations/ai/mcp.md). Its tools list the features and their specifications, list the configured metadata stores and query the metadata of a feature.
