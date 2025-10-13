# Metaxy Overview

## Components

Metaxy consists of several components that work together to provide a comprehensive feature management solution:

- metadata store: This component is responsible for storing metadata about features, containers, and other entities in Metaxy. It provides a centralized repository for managing and retrieving metadata information.
- feature: This component represents a feature in Metaxy. A feature is a collection of containers.
- container: each feature must have at least one container (defaults to `default` container). Code versions and dependencies are set on container level.

As a quick example, when working with `.mp4` files, the file itself can be represented as a feature with `frames` and `audio` containers.

## Versioning

Metaxy has two concepts related to versioning: code version and data version. Code versions are defined by the user for each container. Data versions are derived from upstream data versions and the current code version at runtime.

Versions on container level are scalars (strings), while versions on feature or sample level are mappings (container->data_version).

Here is how Metaxy calculates the data version for a given sample.

For each container defined for the sample's feature, Metaxy:
  1. loads the relevant data versions for the upstream samples
  2. hashes them together (this is called a Merkle tree)
  3. adds the current container's code version to the hash

Because this process is recursive, it ensures that changes to any upstream container will trigger a re-calculation of the data version for the current container.

All these operations are vectorized, meaning they can be efficiently applied to multiple samples at once.
