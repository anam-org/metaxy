import metaxy as mx
import narwhals as nw

# --8<-- [start:feature-definitions]
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
    deps=[AudioEmbeddings],
)


class AudioClusters(mx.BaseFeature, spec=downstream_spec):
    audio_id: str
    mean: float
    std: float


# --8<-- [end:feature-definitions]


import dagster as dg
import metaxy.ext.dagster as mxd
import polars as pl


# --8<-- [start:root-asset]
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
        store.resolve_update("audio/embeddings", samples=samples)

    # Compute embeddings...

    df = ...  # at this point this dataframe should have `mean` and `std` columns set

    # either write embeddings metadata via Metaxy
    # or return a dataframe to write it via MetaxyIOManager

    return df


# --8<-- [end:root-asset]


# --8<-- [start:downstream-asset]
@mxd.metaxify
@dg.asset(
    metadata={"metaxy/feature": "audio/clusters"},
    io_manager_key="metaxy_io_manager",
)
def audio_clusters(
    store: dg.ResourceParam[mx.MetadataStore],
):
    with store:
        # Get IDs that need recomputation
        store.resolve_update(AudioClusters)
    ...


# --8<-- [end:downstream-asset]


# --8<-- [start:non-metaxy-downstream]
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
    return {"total_clusters": df.select("cluster_id").n_unique()}


# --8<-- [end:non-metaxy-downstream]


# --8<-- [start:dagster-definitions]
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


# --8<-- [end:dagster-definitions]
