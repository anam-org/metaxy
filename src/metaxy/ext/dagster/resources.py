import dagster as dg

import metaxy as mx


class MetaxyStoreFromConfigResource(dg.ConfigurableResource[mx.MetadataStore]):
    """This resource creates a [`metaxy.MetadataStore`][metaxy.MetadataStore] based on the current Metaxy configuration (`metaxy.toml`)."""

    name: str

    def create_resource(self, context: dg.InitResourceContext) -> mx.MetadataStore:
        """Create a MetadataStore from the Metaxy configuration.

        Args:
            context: Dagster resource initialization context.

        Returns:
            A MetadataStore configured with the Dagster run ID as the materialization ID.
        """
        assert context.run is not None
        return mx.MetaxyConfig.get().get_store(
            self.name, materialization_id=context.run.run_id
        )
