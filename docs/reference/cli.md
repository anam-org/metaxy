# CLI Commands

This section provides a comprehensive reference for all Metaxy CLI commands.

::: cyclopts
    :module: metaxy.cli.app:app
    :heading-level: 2
    :recursive: true
    :flatten-commands: false
    :generate-toc: true

## Examples

### Recording a graph snapshot

```bash
# Push the current feature graph to the metadata store
metaxy graph push
```

The recommendation is to run this command in your CD pipeline.

### Generating and applying migrations

```bash
# Generate a migration for detected changes
metaxy migrations generate --op metaxy.migrations.ops.DataVersionReconciliation

# Apply pending migrations
metaxy migrations apply
```

### Visualizing the feature graph

```bash
# Render as terminal tree view
metaxy graph render

# Render as Mermaid diagram
metaxy graph render --format mermaid

# Compare two snapshots
metaxy graph-diff render <snapshot-id> current --format mermaid
```
