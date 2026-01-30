from typing import Any

import dagster as dg
import narwhals as nw
import pydantic
from narwhals.typing import IntoFrame

import metaxy as mx
from metaxy._decorators import public
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_PARTITION_KEY,
    DAGSTER_METAXY_PARTITION_METADATA_KEY,
)
from metaxy.ext.dagster.resources import MetaxyStoreFromConfigResource
from metaxy.ext.dagster.utils import (
    build_partition_filter_from_input_context,
    build_runtime_feature_metadata,
)
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.models.constants import METAXY_MATERIALIZATION_ID
from metaxy.models.types import ValidatedFeatureKey

#: Type alias for MetaxyIOManager output - any narwhals-compatible dataframe or None
MetaxyOutput = IntoFrame | None


@public
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
            def my_asset(): ...
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
            def my_partitioned_asset(): ...
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
        return self.store  # ty: ignore[invalid-return-type]

    def _feature_key_from_context(self, context: dg.InputContext | dg.OutputContext) -> ValidatedFeatureKey:
        if isinstance(context, dg.InputContext):
            assert context.upstream_output is not None
            assert context.upstream_output.definition_metadata is not None
            return mx.ValidatedFeatureKeyAdapter.validate_python(
                context.upstream_output.definition_metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY]
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
            feature_key = self._feature_key_from_context(context)
            store_metadata = self.metadata_store.get_store_metadata(feature_key)

            # Build input metadata, transforming special keys to dagster standard format
            input_metadata: dict[str, Any] = {}
            for key, value in store_metadata.items():
                if key == "display":
                    input_metadata["metaxy/store"] = value
                elif key == "table_name":
                    input_metadata["dagster/table_name"] = value
                elif key == "uri":
                    input_metadata["dagster/uri"] = dg.MetadataValue.path(value)
                else:
                    input_metadata[key] = value

            # Only add input metadata if we have exactly one partition key
            # (add_input_metadata internally uses asset_partition_key which fails with multiple)
            has_single_partition = context.has_asset_partitions and len(list(context.asset_partition_keys)) == 1
            if input_metadata and (not context.has_asset_partitions or has_single_partition):
                context.add_input_metadata(input_metadata, description="Metadata Store Info")

            # Build partition filters from context (handles partition_by and metaxy/partition)
            filters = build_partition_filter_from_input_context(context)

            return self.metadata_store.read_metadata(
                feature=feature_key,
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
        # Check if expensive runtime metadata was already logged (e.g., via generate_materialize_results)
        # dagster/row_count requires database reads and is always set by build_runtime_feature_metadata,
        # so we use it as a sentinel to detect if metadata was already computed
        # Using step_context.get_output_metadata() to check for existing metadata
        # See: https://github.com/dagster-io/dagster/issues/17923
        existing_metadata = context.step_context.get_output_metadata(context.name)
        if existing_metadata and "dagster/row_count" in existing_metadata:
            context.log.debug("Skipping runtime metadata logging - already logged via MaterializeResult")
            return

        with self.metadata_store:
            key = self._feature_key_from_context(context)

            try:
                feature = mx.get_feature_by_key(key)

                # Get partition column from metadata (for Dagster partitions)
                partition_col = context.definition_metadata.get(DAGSTER_METAXY_PARTITION_KEY)

                # Get metaxy partition from metadata (for multi-asset logical partitions)
                metaxy_partition = context.definition_metadata.get(DAGSTER_METAXY_PARTITION_METADATA_KEY)

                # Build runtime metadata (handles reading and filtering internally)
                runtime_metadata, _ = build_runtime_feature_metadata(
                    key,
                    self.metadata_store,
                    context,
                    partition_col=partition_col,
                    metaxy_partition=metaxy_partition,
                )
                context.add_output_metadata(runtime_metadata)

                mat_lazy_df = self.metadata_store.read_metadata(
                    feature,
                    filters=[nw.col(METAXY_MATERIALIZATION_ID) == context.run_id],
                )
                materialized_in_run = mat_lazy_df.select(feature.id_columns).unique().collect().to_native()
                context.add_output_metadata({"metaxy/materialized_in_run": len(materialized_in_run)})
            except FeatureNotFoundError:
                pass
