import dagster as dg

import metaxy as mx


class MetaxyStoreFromConfigResource(dg.ConfigurableResource[mx.MetadataStore]):
    """This resource creates a [`metaxy.MetadataStore`][metaxy.MetadataStore] based on the current Metaxy configuration (`metaxy.toml`)."""

    name: str

    def create_resource(self, context: dg.InitResourceContext) -> mx.MetadataStore:
        return mx.MetaxyConfig.get().get_store(
            self.name, materialization_id=context.run_id
        )
