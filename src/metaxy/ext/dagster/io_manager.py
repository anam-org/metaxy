from typing import Any

import dagster as dg
import narwhals as nw
import pydantic
from narwhals.typing import IntoFrame

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.ext.dagster.utils import (
    build_partition_filter,
    build_runtime_feature_metadata,
)
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.models.constants import METAXY_MATERIALIZATION_ID
from metaxy.models.types import ValidatedFeatureKey

#: Type alias for MetaxyIOManager output - any narwhals-compatible dataframe or None
MetaxyOutput = IntoFrame | None


class MetaxyIOManager(dg.ConfigurableIOManager):
    """MetaxyIOManager is a Dagster IOManager that reads and writes data to/from Metaxy's [`MetadataStore`][metaxy.MetadataStore].

    It automatically attaches Metaxy feature and store metadata to Dagster materialization events and handles partitioned assets.

    !!! warning "Always set `"metaxy/feature"` Dagster metadata"

        This IOManager is using `"metaxy/feature"` Dagster metadata key to map Dagster assets into Metaxy features.
        It expects it to be set on the assets being loaded or materialized.

        ??? example

            ```py
            import dagster as dg

            @dg.asset(
                metadata={
                    "metaxy/feature": "my/feature/key",
                }
            )
            def my_asset():
                ...
            ```

    !!! tip "Defining Partitioned Assets"

        To tell Metaxy which column to use when filtering partitioned assets, set `"partition_by"` Dagster metadata key.

        ??? example
            ```py
            import dagster as dg

            @dg.asset(
                metadata={
                    "metaxy/feature": "my/feature/key",
                    "partition_by": "date",
                }
            )
            def my_partitioned_asset():
                ...
            ```

        This key is commonly used to configure partitioning behavior by various Dagster IO managers.

    """

    store: dg.ResourceDependency[MetaxyStoreFromConfigResource] = pydantic.Field(
        default_factory=MetaxyStoreFromConfigResource(name="dev")
    )

    @property
    def metadata_store(
        self,
    ) -> mx.MetadataStore:  # this property mostly exists to fix the type annotation
        return self.store  # pyright: ignore[reportReturnType]

    def _feature_key_from_context(
        self, context: dg.InputContext | dg.OutputContext
    ) -> ValidatedFeatureKey:
        if isinstance(context, dg.InputContext):
            assert context.upstream_output is not None
            assert context.upstream_output.metadata is not None
            return mx.ValidatedFeatureKeyAdapter.validate_python(
                context.upstream_output.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
            )
        elif isinstance(context, dg.OutputContext):
            return mx.ValidatedFeatureKeyAdapter.validate_python(
                context.definition_metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
            )
        else:
            raise ValueError(f"Unexpected context type: {type(context)}")

    def load_input(self, context: "dg.InputContext") -> nw.LazyFrame[Any]:
        """Load feature metadata from [`MetadataStore`][metaxy.MetadataStore].

        Reads metadata for the feature specified in the asset's `"metaxy/feature"` metadata.
        For partitioned assets, filters to the current partition using the column specified
        in `"partition_by"` metadata.

        Args:
            context: Dagster input context containing asset metadata.

        Returns:
            A narwhals LazyFrame with the feature metadata.
        """
        with self.metadata_store:
            context.log.debug(
                f"Reading metadata for Metaxy feature {self._feature_key_from_context(context).to_string()} from {self.metadata_store.display()}"
            )

            # Build partition filter if applicable
            partition_col = (
                context.definition_metadata.get(DAGSTER_METAXY_PARTITION_KEY)
                if context.has_asset_partitions
                else None
            )
            partition_key = (
                context.asset_partition_key if context.has_asset_partitions else None
            )
            filters = build_partition_filter(
                partition_col,  # pyright: ignore[reportArgumentType]
                partition_key,
            )

            return self.metadata_store.read_metadata(
                feature=self._feature_key_from_context(context),
                filters=filters,
            )

    def handle_output(self, context: "dg.OutputContext", obj: MetaxyOutput) -> None:
        """Write feature metadata to [`MetadataStore`][metaxy.MetadataStore].

        Writes the output dataframe to the metadata store for the feature specified
        in the asset's `"metaxy/feature"` metadata. Also logs metadata about the
        feature and store to Dagster's materialization events.

        If `obj` is `None`, only metadata logging is performed (no data is written).

        Args:
            context: Dagster output context containing asset metadata.
            obj: A narwhals-compatible dataframe to write, or None to skip writing.
        """
        assert DAGSTER_METAXY_FEATURE_METADATA_KEY in context.definition_metadata, (
            f'Missing `"{DAGSTER_METAXY_FEATURE_METADATA_KEY}"` key in asset metadata'
        )
        key = self._feature_key_from_context(context)
        feature = mx.get_feature_by_key(key)

        if obj is not None:
            context.log.debug(
                f'Writing metadata for Metaxy feature "{key.to_string()}" into {self.metadata_store.display()}'
            )
            with self.metadata_store.open("write"):
                self.metadata_store.write_metadata(feature=feature, df=obj)
            context.log.debug(
                f'Metadata written for Metaxy feature "{key.to_string()}" into {self.metadata_store.display()}'
            )
        else:
            context.log.debug(
                f'The output corresponds to Metaxy feature "{key.to_string()}" stored in {self.metadata_store.display()}'
            )

        self._log_output_metadata(context)

    def _log_output_metadata(self, context: dg.OutputContext):
        with self.metadata_store:
            key = self._feature_key_from_context(context)

            try:
                feature = mx.get_feature_by_key(key)

                # Build runtime metadata from data (includes metaxy/feature, metaxy/info,
                # metaxy/store, row count, table preview, etc.)
                lazy_df = self.metadata_store.read_metadata(feature)
                runtime_metadata = build_runtime_feature_metadata(
                    key, self.metadata_store, lazy_df, context
                )
                context.add_output_metadata(runtime_metadata)

                # Get materialized-in-run count
                mat_lazy_df = self.metadata_store.read_metadata(
                    feature,
                    filters=[nw.col(METAXY_MATERIALIZATION_ID) == context.run_id],
                )
                materialized_in_run = (
                    mat_lazy_df.select(feature.spec().id_columns)
                    .unique()
                    .collect()
                    .to_native()
                )
                context.add_output_metadata(
                    {"metaxy/materialized_in_run": len(materialized_in_run)}
                )
            except FeatureNotFoundError:
                pass
