---
title: "Local Development"
description: "Local development patterns and best practices with Metaxy."
---

# Local Development

--8<-- "initialization.md"

Metaxy supports local-first development workflows.

It all starts from the [metadata store](../metadata-stores.md). The default metadata store name in Metaxy [configuration](/reference/configuration.md/#store) is `"dev"` (1). Configure it in the config file:
{ .annotate }

1. of course, it can be tweaked to something like `"local"`

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "${HOME}/.metaxy/dev"
```



Metaxy APIs and CLI commands will automatically use the default store unless specified otherwise. It is a good practice to rely on the default store detection in order to easily swap it to another store in other environments:

<!-- skip next -->
```py
import metaxy as mx

store = mx.init().get_store()
```

### Using the Metaxy CLI

Metaxy provides a [CLI](/reference/cli.md) which is useful for local development.
Here are some of the things you can do with it:

- `mx list features` - view the features available via feature discovery

### Fallback Store

You'll probably want to configure the `dev` store to [pull missing data from production](../metadata-stores.md#fallback-stores). Configure `fallback_stores` in order to achieve this:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "${HOME}/.metaxy/dev"
fallback_stores = ["prod"]

[stores.prod]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "s3://my-prod-bucket/metaxy"
```
