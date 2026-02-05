---
title: "Deployment"
description: "Learn how to deploy Metaxy code to Production and Branch Deployments."
---

## Production

--8<-- "initialization.md"

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
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "s3://branch-bucket/${PULL_REQUEST_ID}"
fallback_stores = ["prod"]

[stores.prod] title="metaxy.toml"
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "s3://my-prod-bucket/metaxy"
```

It is beneficial to have it fallback to `prod` store to avoid having to materialize all upstream features to a given one of interest (that's being tested).

Of course, the Branch Deployment CD needs to do:

```shell
mx push --store branch
```

And set `METAXY_STORE` to `"branch"`.
