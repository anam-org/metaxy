# Metaxy CLI

Metaxy provides a CLI (`metaxy` or `mx`) for managing features, metadata, and migrations. Explore the CLI with `--help` for more information.

See full documentation: https://anam-org.github.io/metaxy/reference/cli/

## Common Commands

### List Features

```bash
mx list features           # List all features
mx list features --verbose # Show field dependencies
```

### Graph Operations

```bash
mx graph push              # Push feature graph to store
mx graph render            # Visualize feature graph in terminal
mx graph render --format mermaid -o graph.mmd  # Export as Mermaid
mx graph history           # Show snapshot history
```

### Metadata Operations

```bash
mx metadata status --all-features  # Check metadata freshness (expensive!)
mx metadata copy my/feature --from prod --to dev  # Copy between stores
```
