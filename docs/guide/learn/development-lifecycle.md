---
title: "Development Lifecycle"
description: "Development patterns and best practices with Metaxy."
---

# Development Patterns and Best Practices

In all environments, Metaxy must be explicitly initialized via:

<!-- skip next -->
```py
import metaxy as mx

mx.init_metaxy()
```

This triggers feature discovery and config discovery.

## Local Development

Metaxy supports local-first development workflows.

It all starts from the [metadata store](./metadata-stores.md). The default metadata store name in Metaxy [configuration](/reference/configuration.md/#store) is `"dev"` (1). Configure it in the config file:
{ .annotate }

1. of course, it can be tweaked to something like `"local"`

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "${HOME}/.metaxy/dev"
```

Metaxy APIs and CLI commands will automatically use the default store unless specified otherwise. It is a good practice to rely on the default store detection in order to easily swap it to another store in other environments:

<!-- skip next -->
```py
import metaxy as mx

store = mx.init_metaxy().get_store()
```

### Using the Metaxy CLI

Metaxy provides a [CLI](/reference/cli.md) which is useful for local development.
Here are some of the things you can do with it:

- `mx list features` - view the features available via feature discovery

### Fallback Store

You'll probably want to configure the `dev` store to [pull missing data from production](./metadata-stores.md#fallback-stores). Configure `fallback_stores` in order to achieve this:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "${HOME}/.metaxy/dev"
fallback_stores = ["prod"]

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "s3://my-prod-bucket/metaxy"
```

## Production

1. Make sure to set `METAXY_STORE` to `"prod"` in your production environment. This will make it the default Metaxy metadata store without any code changes.

2. Add the following step to your production deployment pipeline:

    ```shell
    mx push --store prod
    ```

    This will persist feature definitions in the metadata store, enabling feature history tracking and multi-project setups.

## Branch Deployments

Branch Deployments, also known as Preview (1) Deployments, are ephemeral environments typically created to test changes in a production-like setting.
They are usually created by CI/CD for Pull Requests.
Some tooling like [Dagster](https://dagster.io/) has built-in support for Branch Deployments, and so does Metaxy.
{ .annotate }

1. or Review Environments, or Feature Branches, or whatever

In order to benefit from Branch Deployments, it is recommended to configure a separate store, templated by some kind of a deployment identifier.
For example:

```toml title="metaxy.toml"
[stores.branch]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "s3://branch-bucket/${PULL_REQUEST_ID}"
fallback_stores = ["prod"]

[stores.prod] title="metaxy.toml"
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "s3://my-prod-bucket/metaxy"
```

It is beneficial to have it fallback to `prod` store to avoid having to materialize all upstream features to a given one of interest (that's being tested).

Of course, the Branch Deployment CD needs to do:

```shell
mx push --store branch
```

And set `METAXY_STORE` to `"branch"`.
