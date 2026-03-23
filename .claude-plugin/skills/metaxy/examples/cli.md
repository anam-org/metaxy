# Metaxy CLI

Metaxy provides a CLI (`metaxy` or `mx`) for managing features, metadata, and migrations. Explore the CLI with `--help` for more information.

See full documentation: https://docs.metaxy.io/latest/reference/cli/

## Common Commands

### List Features

```bash
mx list features           # List all features
mx list features --verbose # Show field dependencies
```

### Push Feature Graph

```bash
mx push                    # Push feature graph to store
```

### Graph Operations

```bash
mx graph render            # Visualize feature graph in terminal
mx graph render --format mermaid -o graph.mmd  # Export as Mermaid
mx history                 # Show snapshot history
```

### Feature Locking (Multi-Environment)

For multi-environment setups where projects can't be pip-installed into each other, push definitions to a shared store and lock them:

```bash
mx push              # Publish feature definitions to store
mx lock              # Fetch external deps into metaxy.lock
```

Both commands accept a `--store` argument.

### Metadata Operations

```bash
mx metadata status --all-features  # Check metadata freshness (expensive!)
mx metadata copy my/feature --from prod --to dev  # Copy between stores
```
