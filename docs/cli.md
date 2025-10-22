# metaxy

Metaxy - Feature Metadata Management

## Table of Contents

- [`shell`](#metaxy-shell)
- [`migrations`](#metaxy-migrations)
  - [`generate`](#metaxy-migrations-generate)
  - [`scaffold`](#metaxy-migrations-scaffold)
  - [`apply`](#metaxy-migrations-apply)
  - [`status`](#metaxy-migrations-status)
- [`push`](#metaxy-push)

**Usage**:

```console
$ metaxy COMMAND
```

**Arguments**:


**Commands**:

* `migrations`: Metadata migration commands
* `push`: Record all feature versions (push graph snapshot).
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
* `--from-snapshot`: Compare from this historical snapshot ID (optional)
* `--to-snapshot`: Compare to this historical snapshot ID (optional)


## `metaxy migrations scaffold`

Create an empty migration scaffold for user-defined operations.

Generates a migration file template with: - Snapshot IDs from current store state - Empty operations list for manual 
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
* `--from-snapshot`: Use this as from_snapshot_id (defaults to latest in store)
* `--to-snapshot`: Use this as to_snapshot_id (defaults to current registry)


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


## `metaxy push`

Record all feature versions (push graph snapshot).

Records all features in the active registry to the metadata store with a deterministic snapshot ID. This should be run 
after deploying new feature definitions.

**Usage**:

```console
$ metaxy push [ARGS]
```

**Options**:

### Options

* `STORE, --store`: The metadata store to use. Defaults to the default store.
