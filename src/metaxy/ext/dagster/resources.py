import dagster as dg

import metaxy as mx
from metaxy._decorators import public


@public
class MetaxyStoreFromConfigResource(dg.ConfigurableResource[mx.MetadataStore]):
    """This resource creates a [`metaxy.MetadataStore`](https://docs.metaxy.io/main/guide/learn/metadata-stores/) based on the current Metaxy configuration (`metaxy.toml`).

    If `name` is not provided, the default store will be used.
    The default store name can be set with `store = "my_name"` in `metaxy.toml` or with` $METAXY_STORE` environment variable.
    """

    name: str | None = None

    def create_resource(self, context: dg.InitResourceContext) -> mx.MetadataStore:
        """Create a MetadataStore from the Metaxy configuration.

        Args:
            context: Dagster resource initialization context.

        Returns:
            A MetadataStore configured with the Dagster run ID as the materialization ID.
        """
        assert context.run is not None
        return mx.MetaxyConfig.get().get_store(self.name, materialization_id=context.run.run_id)
