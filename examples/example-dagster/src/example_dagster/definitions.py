import narwhals as nw
from pydantic import Field

import metaxy as mx

# Upstream feature
upstream_spec = mx.FeatureSpec(
    key="audio/embeddings",
    id_columns=["audio_id"],
    fields=["embedding"],
)


class AudioEmbeddings(mx.BaseFeature, spec=upstream_spec):
    """Embeddings produced with Whisper"""

    audio_id: str = Field(description="Unique identifier for the audio")
    properties: dict[str, str] = Field(description="Properties of the embedding")


# Downstream feature that depends on upstream
downstream_spec = mx.FeatureSpec(
    key="audio/clusters",
    id_columns=["audio_id"],
    fields=["cluster_id"],
    deps=[AudioEmbeddings],
)


class AudioClusters(mx.BaseFeature, spec=downstream_spec):
    """Audio clusters calculated with UMAP"""

    audio_id: str = Field(description="Unique identifier for the audio")
    mean: float = Field(description="Mean embedding value")
    num_dim: int = Field(description="Number of dimensions in the embedding")


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
        update = store.resolve_update(AudioClusters)
    ...


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


store = mxd.MetaxyStoreFromConfigResource(name="dev")
metaxy_io_manager = mxd.MetaxyIOManager(store=store)


@dg.definitions
def definitions():
    mx.init_metaxy()

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
