# Metaxy + Dagster

Metaxy's dependency system has been originally inspired by [Dagster](https://dagster.io/).

Because of this, Metaxy's concepts naturally map into Dagster concepts, which makes wrapping Metaxy code with Dagster effortless.

The only step that has to be taken in order to inject Metaxy into Dagster assets is to annotate Dagster assets with the well-known `"metaxy/feature"` key (1):
{ .annotate }

1. :man_raising_hand: pointing to... the Metaxy feature key!

<!-- dprint-ignore-start -->
!!! example

    ```python
    import dagster as dg

    @dg.asset(metadata={"metaxy/feature": "my/metaxy/feature"})
    def my_asset():
    ...
    ```
<!-- dprint-ignore-end -->

Once this is done, it becomes possible unleash the full power of [`metaxy.ext.dagster.metaxify`][metaxy.ext.dagster.metaxify.metaxify] on Dagster!

<!-- dprint-ignore-start -->
!!! example
    ```python {hl_lines="3"}
    import metaxy.ext.dagster as mxd

    @mxd.metaxify
    @dg.asset(metadata={"metaxy/feature": "my/metaxy/feature"})
    def my_asset():
      ...
    ```
<!-- dprint-ignore-end -->

It will take care of bringing the right lineage, description, metadata, and other transferable properties from the Metaxy feature to the Dagster asset.

## Features

This integration provides:

- [`MetaxyStoreFromConfigResource`][metaxy.ext.dagster.MetaxyStoreFromConfigResource] - a resource that provides access to [`MetadataStore`][metaxy.MetadataStore]

- [`MetaxyIOManager`][metaxy.ext.dagster.io_manager.MetaxyIOManager] - an IO manager that reads and writes Dagster assets that are Metaxy features

- [`metaxify`][metaxy.ext.dagster.metaxify.metaxify] - a decorator that enriches Dagster asset definitions with Metaxy information such as correct upstream dependencies and metadata

## Quick Start

### 1. Define Metaxy Features

```python {title="defs.py"}
import metaxy as mx

# Upstream feature
upstream_spec = mx.FeatureSpec(
    key="audio/embeddings",
    id_columns=["audio_id"],
    fields=["embedding"],
)


class AudioEmbeddings(mx.BaseFeature, spec=upstream_spec):
    audio_id: str


# Downstream feature that depends on upstream
downstream_spec = mx.FeatureSpec(
    key="audio/clusters",
    id_columns=["audio_id"],
    fields=["cluster_id"],
    deps=[AudioEmbeddings],
)


class AudioClusters(mx.BaseFeature, spec=downstream_spec):
    audio_id: str
    mean: float
    std: float
```

### 2. Define Dagster Assets

<!-- dprint-ignore-start -->
!!! example "Root Asset"
    Let's define an asset that doesn't have any upstream Metaxy features.

    ```python {title="defs.py"}
    import dagster as dg
    import polars as pl
    import metaxy.ext.dagster as mxd


    @mxd.metaxify
    @dg.asset(
        metadata={"metaxy/feature": "audio/embeddings"},
        io_manager_key="metaxy_io_manager",
    )
    def audio_embeddings(
        store: dg.ResourceParam[mx.MetadataStore],
    ):
        # somehow, acquire root source data
        samples = pl.DataFrame(
            {
                "audio_id": ["a1", "a2", "a3"],
                "metaxy_provenance_by_field": [
                    {"embedding": "hash1"},
                    {"embedding": "hash2"},
                    {"embedding": "hash3"},
                ],
            }
        )

        # resolve the increment with Metaxy

        with store:
            increment = store.resolve_update("audio/embeddings", samples=samples)

        # Compute embeddings...

        df = ...  # at this point this dataframe should have `mean` and `std` columns set

        # either write embeddings metadata via Metaxy
        # or return a dataframe to write it via MetaxyIOManager

        return df

    ```
<!-- dprint-ignore-end -->

<!-- dprint-ignore-start -->
!!! example "Downstream Asset"

    ```py {title="defs.py"}
    @mxd.metaxify
    @dg.asset(
        metadata={"metaxy/feature": "audio/clusters"},
        io_manager_key="metaxy_io_manager",
        ins={
            "embeddings": dg.AssetIn(
            key=["audio", "embeddings"],
            )
        },
    )
    def audio_clusters(
        store: dg.ResourceParam[mx.MetadataStore],
    ):
        with store:
            # Get IDs that need recomputation
            update = store.resolve_update(AudioClusters)
        ...
    ```
<!-- dprint-ignore-end -->

<!-- dprint-ignore-start -->
!!! example "Non-Metaxy Downstream Asset"

    ```py
    @dg.asset(
        ins={
            "clusters": dg.AssetIn(
                key=["audio", "clusters"],
            )
        },
    )
    def cluster_report(clusters: nw.LazyFrame):
        # clusters is a narwhals LazyFrame loaded via MetaxyIOManager
        df = clusters.collect().to_polars()
        # Generate a report...
        return {"total_clusters": df.select("cluster_id").n_unique()}`
    ```
<!-- dprint-ignore-end -->

### 3. Create Dagster Definitions

```py {title="defs.py"}
store = mxd.MetaxyStoreFromConfigResource(name="dev")
metaxy_io_manager = mxd.MetaxyIOManager(store=store)


@dg.definitions
def definitions():
    mx.init_metaxy()  # (1)!

    return dg.Definitions(
        assets=[
            audio_embeddings,
            audio_clusters,
            cluster_report,
        ],
        resources={
            "store": store,
            "metaxy_io_manager": metaxy_io_manager,
        },
    )
```

1. 🌌 This loads Metaxy configuration and feature definitions

### 4. Start Dagster

```bash
dg dev -f defs.py
```

Materialize your ultra-giga-multi-modal-big-data assets and let Metaxy take care of state and versioning!

## Reference

- Integration [API docs](../../reference/api/ext/dagster.md)
- Dagster [docs](https://docs.dagster.io/)
