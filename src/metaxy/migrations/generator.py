"""Migration file generation."""

import os
from datetime import datetime
from typing import TYPE_CHECKING

from metaxy.migrations.detector import detect_feature_changes
from metaxy.migrations.models import FeatureVersionMigration, Migration

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


def generate_migration(
    store: "MetadataStore",
    *,
    output_dir: str = "migrations",
) -> str | None:
    """Generate migration file from detected feature changes.

    Automatically detects all features that have changed (comparing code vs metadata)
    and generates a YAML migration file.

    Args:
        store: Metadata store to check
        output_dir: Directory to write migration file (default: "migrations")

    Returns:
        Path to generated migration file, or None if no changes detected

    Example:
        >>> migration_file = generate_migration(store)

        Detected 2 feature changes:
          ✓ video_processing: abc12345 → def67890
          ✓ audio_processing: xyz11111 → xyz22222

        Will propagate to 5 downstream features:
          - feature_c
          - feature_d
          - feature_e
          - feature_f
          - feature_g

        Generated: migrations/20250113_103000_update_video_processing_audio_processing.yaml

        >>> print(migration_file)
        migrations/20250113_103000_update_video_processing_audio_processing.yaml
    """
    # Detect changes
    changes = detect_feature_changes(store)

    if not changes:
        print("No feature changes detected. All features up to date!")
        return None

    # Generate migration ID and timestamp
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    migration_id = f"migration_{timestamp_str}"

    # Build filename from feature names
    feature_count = len(changes)
    feature_names = "_".join(
        change.feature_key.to_string().replace("/", "_") for change in changes[:3]
    )
    if feature_count > 3:
        feature_names += f"_and_{feature_count - 3}_more"

    filename = f"{output_dir}/{timestamp_str}_update_{feature_names}.yaml"

    # Build operations
    operations = [
        FeatureVersionMigration(
            feature_key=list(change.feature_key),
            from_=change.from_version,
            to=change.to_version,
            change_type=change.change_type,
            reason="TODO: Describe what changed and why",
        )
        for change in changes
    ]

    # Create migration
    migration = Migration(
        version=1,
        id=migration_id,
        description=f"Auto-generated migration for {feature_count} changed feature(s)",
        created_at=timestamp,
        operations=operations,
    )

    # Show detected changes
    print(f"\nDetected {feature_count} feature change(s):")
    for change in changes:
        print(
            f"  ✓ {change.feature_key.to_string()}: "
            f"{change.from_version} → {change.to_version}"
        )

    # Show downstream impact
    source_keys = [change.feature_key for change in changes]
    downstream_keys = store.registry.get_downstream_features(source_keys)

    if downstream_keys:
        print(f"\nWill propagate to {len(downstream_keys)} downstream feature(s):")
        for dk in downstream_keys:
            print(f"  - {dk.to_string()}")

    # Write YAML file
    os.makedirs(output_dir, exist_ok=True)
    migration.to_yaml(filename)

    # Print next steps
    print(f"\nGenerated: {filename}")
    print("\nNEXT STEPS:")
    print("1. Review the migration file")
    print("2. Edit the 'reason' fields to document what changed")
    print("3. Run 'metaxy migrations apply --dry-run' to preview")
    print("4. Run 'metaxy migrations apply' to execute")
    print("5. Commit the migration file to git")

    return filename
