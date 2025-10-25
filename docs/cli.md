# metaxy

Metaxy - Feature Metadata Management

## Table of Contents

- [`shell`](#metaxy-shell)
- [`migrations`](#metaxy-migrations)
  - [`generate`](#metaxy-migrations-generate)
  - [`scaffold`](#metaxy-migrations-scaffold)
  - [`apply`](#metaxy-migrations-apply)
  - [`status`](#metaxy-migrations-status)
- [`graph`](#metaxy-graph)
  - [`push`](#metaxy-graph-push)
  - [`history`](#metaxy-graph-history)
  - [`describe`](#metaxy-graph-describe)
  - [`render`](#metaxy-graph-render)
- [`list`](#metaxy-list)
  - [`features`](#metaxy-list-features)
- [`metadata`](#metaxy-metadata)
  - [`copy`](#metaxy-metadata-copy)
  - [`drop`](#metaxy-metadata-drop)

**Usage**:

```console
$ metaxy COMMAND
```

**Arguments**:


**Commands**:

* `graph`: Manage feature graphs
* `list`: List Metaxy entities
* `metadata`: Manage Metaxy metadata
* `migrations`: Metadata migration commands
* `shell`: Start interactive shell.

## `metaxy shell`

Start interactive shell.

**Usage**:

```console
$ metaxy shell
```

## `metaxy migrations`

Metadata migration commands

**Usage**:

```console
$ metaxy migrations COMMAND
```

**Commands**:

* `apply`: Apply migration(s) up to specified revision.
* `generate`: Generate migration file from detected feature changes.
* `scaffold`: Create an empty migration scaffold for user-defined operations.
* `status`: Show migration status.

## `metaxy migrations generate`

Generate migration file from detected feature changes.

Two modes: 1. Default (no snapshots): Compare store's latest snapshot vs current code 2. Historical (both snapshots): 
Compare two historical snapshots

Automatically detects features that need migration and generates explicit operations for ALL affected features.

Example (default):
    $ metaxy migrations generate  Detected 1 root feature change(s):  ✓ video_processing: abc12345 → def67890  
Generating explicit operations for 3 downstream features:  ✓ feature_c (current: xyz111) ✓ feature_d (current: aaa222) ✓
feature_e (current: bbb333)  Generated 4 total operations (1 root + 3 downstream)
      
Example (historical):
    $ metaxy migrations generate --from-snapshot abc123... --to-snapshot def456...

**Usage**:

```console
$ metaxy migrations generate [OPTIONS]
```

**Options**:

* `--migrations-dir`: Directory for migration files (uses config if not specified)
* `--from-snapshot`: Compare from this historical snapshot version (optional)
* `--to-snapshot`: Compare to this historical snapshot version (optional)


## `metaxy migrations scaffold`

Create an empty migration scaffold for user-defined operations.

Generates a migration file template with: - Snapshot versions from current store state - Empty operations list for manual 
editing - Proper structure and metadata

Use this when you need to write custom migration operations that can't be auto-generated (e.g., complex data 
transformations, backfills).

**Usage**:

```console
$ metaxy migrations scaffold [OPTIONS]
```

**Options**:

* `--migrations-dir`: Directory for migration files (uses config if not specified)
* `--description`: Migration description (optional)
* `--from-snapshot`: Use this as from_snapshot_version (defaults to latest in store)
* `--to-snapshot`: Use this as to_snapshot_version (defaults to current graph)


## `metaxy migrations apply`

Apply migration(s) up to specified revision.

Applies all migrations in dependency order up to the target revision. If no revision specified, applies all migrations.

Migrations are applied with parent validation - parent migrations must be completed before applying child migrations. 
Already-completed migrations are skipped.

Errors if there are multiple heads (migrations with no children) and no revision is specified.

**Usage**:

```console
$ metaxy migrations apply [OPTIONS] [ARGS]
```

**Options**:

* `REVISION, --revision`: Migration ID to apply up to (applies all if not specified)
* `--dry-run, --no-dry-run`: Preview changes without executing  *[default: --no-dry-run]*
* `--force, --no-force`: Re-apply even if already completed  *[default: --no-force]*
* `--migrations-dir`: Directory containing migration files (uses config if not specified)


## `metaxy migrations status`

Show migration status.

Displays all registered migrations and their completion status. Status is derived from system tables (migrations, ops, 
steps).

**Usage**:

```console
$ metaxy migrations status
```


## `metaxy graph`

Manage feature graphs

**Usage**:

```console
$ metaxy graph COMMAND
```

**Commands**:

* `describe`: Describe a graph snapshot.
* `history`: Show history of recorded graph snapshots.
* `push`: Record all feature versions (push graph snapshot).
* `render`: Render feature graph visualization.

## `metaxy graph push`

Record all feature versions (push graph snapshot).

Records all features in the active graph to the metadata store with a deterministic snapshot version. This should be run 
after deploying new feature definitions.

**Usage**:

```console
$ metaxy graph push [ARGS]
```

**Options**:

* `STORE, --store`: Metadata store to use (defaults to configured default store)


## `metaxy graph history`

Show history of recorded graph snapshots.

Displays all recorded graph snapshots from the metadata store, showing snapshot versions, when they were recorded, and 
feature counts.

**Usage**:

```console
$ metaxy graph history [ARGS]
```

**Options**:

* `STORE, --store`: Metadata store to use (defaults to configured default store)
* `LIMIT, --limit`: Limit number of snapshots to show (defaults to all)


## `metaxy graph describe`

Describe a graph snapshot.

Shows detailed information about a graph snapshot including: - Feature count - Graph depth (longest dependency chain) - 
Root features (features with no dependencies)

**Usage**:

```console
$ metaxy graph describe [ARGS]
```

**Options**:

* `SNAPSHOT, --snapshot`: Snapshot version to describe (defaults to current graph from code)
* `STORE, --store`: Metadata store to use (defaults to configured default store)


## `metaxy graph render`

Render feature graph visualization.

Visualize the feature graph in different formats: - terminal: Terminal rendering with two types:

    graph (default): Hierarchical tree view  cards: Panel/card-based view with dependency edges


 • mermaid: Mermaid flowchart markup
 • graphviz: Graphviz DOT format
╭───────────────────────────────── System Message: ERROR/3 (<rst-document>, line 5); ──────────────────────────────────╮
│ <rst-document>:5: (ERROR/3) Unexpected indentation.                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── System Message: WARNING/2 (<rst-document>, line 7); ─────────────────────────────────╮
│ <rst-document>:7: (WARNING/2) Block quote ends without a blank line; unexpected unindent.                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

**Usage**:

```console
$ metaxy graph render [ARGS]
```

**Options**:

* `SHOW-FIELDS, --show-fields, --no-show-fields`: Render configuration  *[default: --show-fields]*
* `SHOW-FEATURE-VERSIONS, --show-feature-versions, --no-show-feature-versions`: Render configuration  *[default: --show-feature-versions]*
* `SHOW-FIELD-VERSIONS, --show-field-versions, --no-show-field-versions`: Render configuration  *[default: --show-field-versions]*
* `SHOW-CODE-VERSIONS, --show-code-versions, --no-show-code-versions`: Render configuration  *[default: --no-show-code-versions]*
* `SHOW-SNAPSHOT-VERSION, --show-snapshot-version, --no-show-snapshot-version`: Render configuration  *[default: --show-snapshot-version]*
* `HASH-LENGTH, --hash-length`: Render configuration  *[default: 8]*
* `DIRECTION, --direction`: Render configuration  *[default: TB]*
* `FEATURE, --feature`: Render configuration
* `UP, --up`: Render configuration
* `DOWN, --down`: Render configuration
* `-f, --format`: Output format: terminal, mermaid, or graphviz  *[default: terminal]*
* `-t, --type`: Terminal rendering type: graph or cards (only for --format terminal)  *[choices: graph, cards]*  *[default: graph]*
* `-o, --output`: Output file path (default: stdout)
* `SNAPSHOT, --snapshot`: Snapshot version to render (default: current graph from code)
* `STORE, --store`: Metadata store to use (for loading historical snapshots)
* `MINIMAL, --minimal, --no-minimal`: Minimal output: only feature keys and dependencies  *[default: --no-minimal]*
* `VERBOSE, --verbose, --no-verbose`: Verbose output: show all available information  *[default: --no-verbose]*


## `metaxy list`

List Metaxy entities

**Usage**:

```console
$ metaxy list COMMAND
```

**Commands**:

* `features`: List Metaxy features

## `metaxy list features`

List Metaxy features

**Usage**:

```console
$ metaxy list features
```


## `metaxy metadata`

Manage Metaxy metadata

**Usage**:

```console
$ metaxy metadata COMMAND
```

**Commands**:

* `copy`: Copy metadata between stores.
* `drop`: Drop metadata from a store.

## `metaxy metadata copy`

Copy metadata between stores.

Copies metadata for specified features from one store to another, optionally using a historical version. Useful for: - 
Migrating data between environments - Backfilling metadata - Copying specific feature versions

Incremental Mode (default):
    By default, performs an anti-join on sample_uid to skip rows that already exist in the destination for the same 
snapshot_version. This prevents duplicate writes.  Disabling incremental (--no-incremental) may improve performance 
when: - The destination store is empty or has no overlap with source - The destination store has eventual deduplication

**Usage**:

```console
$ metaxy metadata copy FROM TO [ARGS]
```

**Arguments**:

* `FROM`: Source store name (must be configured in metaxy.toml)  **[required]**
* `TO`: Destination store name (must be configured in metaxy.toml)  **[required]**

**Options**:

* `FEATURE, --feature, --empty-feature`: Feature key to copy (e.g., 'my_feature' or 'group/my_feature'). Can be repeated multiple times. If not specified, uses 
--all-features.
* `ALL-FEATURES, --all-features, --no-all-features`: Copy all features from source store  *[default: --no-all-features]*
* `SNAPSHOT, --snapshot`: Snapshot version to copy (defaults to latest in source store). The snapshot_version is preserved in the destination.
* `INCREMENTAL, --incremental, --no-incremental`: Use incremental copy (compare data_version to skip existing rows). Disable for better performance if destination is 
empty or uses deduplication.  *[default: --incremental]*


## `metaxy metadata drop`

Drop metadata from a store.

Removes metadata for specified features from the store. This is a destructive operation and requires --confirm flag.

Useful for: - Cleaning up test data - Re-computing feature metadata from scratch - Removing obsolete features

**Usage**:

```console
$ metaxy metadata drop [ARGS]
```

**Options**:

* `STORE, --store`: Store name to drop metadata from (defaults to configured default store)
* `FEATURE, --feature, --empty-feature`: Feature key to drop (e.g., 'my_feature' or 'group/my_feature'). Can be repeated multiple times. If not specified, uses 
--all-features.
* `ALL-FEATURES, --all-features, --no-all-features`: Drop metadata for all features in the store  *[default: --no-all-features]*
* `CONFIRM, --confirm, --no-confirm`: Confirm the drop operation (required to prevent accidental deletion)  *[default: --no-confirm]*
