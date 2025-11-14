"""Example 4: PythonMigration with Pydantic validation.

This example demonstrates how to use Pydantic validators to add custom
validation logic to your migrations, ensuring data quality and catching
configuration errors early.

Use Case:
---------
Add validation to ensure:
- Migration configuration is valid before execution
- Required external resources exist (S3 buckets, databases, APIs)
- Feature dependencies are satisfied
- Data constraints are met
- Environment is properly configured

Pydantic provides several validation mechanisms:
- Field validators: Validate individual fields
- Model validators: Validate entire model after construction
- Pre-validation: Transform data before validation
- Custom validators: Implement complex validation logic

When to Use:
------------
- Complex migration configuration that needs validation
- External dependencies that should be checked upfront
- Data quality requirements
- Environment prerequisite checks
- Fail-fast validation before expensive operations

Example Usage:
--------------
Place this file in .metaxy/migrations/ directory and run:
    metaxy migrations apply

The migration will validate configuration before execution, raising clear
errors if validation fails.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import field_validator, model_validator

from metaxy.migrations.models import MigrationResult, PythonMigration

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


class ValidatedDatabaseBackfillMigration(PythonMigration):
    """Migration with comprehensive Pydantic validation.

    This migration demonstrates various validation techniques:
    1. Field validators for individual field constraints
    2. Model validators for cross-field validation
    3. Pre-validation data transformation
    4. Custom validation logic for external resources
    """

    migration_id: str = "20250101_150000_custom_validation"
    created_at: datetime = datetime(2025, 1, 1, 15, 0, 0, tzinfo=timezone.utc)

    # Configuration fields with validation
    database_url: str  # PostgreSQL connection string
    table_name: str  # Source table
    batch_size: int = 1000  # Process in batches
    max_retries: int = 3  # Retry failed batches
    target_feature_key: list[str]  # Feature to backfill

    # Optional filters
    date_filter_start: str | None = None  # ISO format: YYYY-MM-DD
    date_filter_end: str | None = None
    min_confidence: float | None = None  # Filter by confidence score

    # Validation flags
    require_database_connectivity: bool = True
    validate_schema: bool = True

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format.

        Args:
            v: Database URL string

        Returns:
            Validated URL

        Raises:
            ValueError: If URL format is invalid
        """
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                f"database_url must be a PostgreSQL connection string "
                f"(postgres:// or postgresql://), got: {v[:20]}..."
            )

        # Check URL contains required components
        if "@" not in v or "/" not in v.split("@")[1]:
            raise ValueError(
                "database_url must include host and database name: "
                "postgresql://user:pass@host:port/dbname"
            )

        return v

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """Validate table name (no SQL injection).

        Args:
            v: Table name

        Returns:
            Validated table name

        Raises:
            ValueError: If table name contains invalid characters
        """
        import re

        # Only allow alphanumeric, underscore, and dot (for schema.table)
        if not re.match(r"^[a-zA-Z0-9_.]+$", v):
            raise ValueError(
                f"table_name contains invalid characters. "
                f"Only alphanumeric, underscore, and dot allowed. Got: {v}"
            )

        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is reasonable.

        Args:
            v: Batch size

        Returns:
            Validated batch size

        Raises:
            ValueError: If batch size is invalid
        """
        if v < 1:
            raise ValueError(f"batch_size must be positive, got: {v}")
        if v > 100_000:
            raise ValueError(
                f"batch_size too large (max 100,000), got: {v}. "
                f"Large batches may cause memory issues."
            )
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        """Validate retry count.

        Args:
            v: Max retries

        Returns:
            Validated retry count

        Raises:
            ValueError: If retry count is invalid
        """
        if v < 0:
            raise ValueError(f"max_retries must be non-negative, got: {v}")
        if v > 10:
            raise ValueError(
                f"max_retries too high (max 10), got: {v}. "
                f"Consider fixing the root issue instead of excessive retries."
            )
        return v

    @field_validator("date_filter_start", "date_filter_end")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        """Validate date strings are ISO format.

        Args:
            v: Date string or None

        Returns:
            Validated date string

        Raises:
            ValueError: If date format is invalid
        """
        if v is None:
            return v

        # Try to parse as ISO date
        try:
            datetime.fromisoformat(v)
        except ValueError as e:
            raise ValueError(
                f"Date must be in ISO format (YYYY-MM-DD), got: {v}. Error: {e}"
            ) from e

        return v

    @field_validator("min_confidence")
    @classmethod
    def validate_confidence(cls, v: float | None) -> float | None:
        """Validate confidence score is in valid range.

        Args:
            v: Confidence score or None

        Returns:
            Validated confidence score

        Raises:
            ValueError: If confidence is out of range
        """
        if v is None:
            return v

        if not 0.0 <= v <= 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got: {v}")

        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "ValidatedDatabaseBackfillMigration":
        """Validate date range is logical (start before end).

        Returns:
            Validated model

        Raises:
            ValueError: If date range is invalid
        """
        if self.date_filter_start is not None and self.date_filter_end is not None:
            start = datetime.fromisoformat(self.date_filter_start)
            end = datetime.fromisoformat(self.date_filter_end)

            if start > end:
                raise ValueError(
                    f"date_filter_start ({self.date_filter_start}) must be "
                    f"before date_filter_end ({self.date_filter_end})"
                )

        return self

    @model_validator(mode="after")
    def validate_database_connectivity(
        self,
    ) -> "ValidatedDatabaseBackfillMigration":
        """Validate database is accessible (if enabled).

        Returns:
            Validated model

        Raises:
            ValueError: If database is not accessible
        """
        if not self.require_database_connectivity:
            return self

        # Try to connect to database
        try:
            import psycopg2  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

            # Parse connection string
            # Note: In production, use proper connection pooling
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()

            # Verify table exists (if schema validation enabled)
            if self.validate_schema:
                cursor.execute(
                    "SELECT to_regclass(%s)",
                    (self.table_name,),
                )
                result = cursor.fetchone()
                if result is not None and result[0] is None:
                    raise ValueError(
                        f"Table '{self.table_name}' does not exist in database"
                    )

            cursor.close()
            conn.close()

        except ImportError:
            # psycopg2 not available - skip validation
            # In production, you might want to raise an error here
            pass
        except Exception as e:
            raise ValueError(
                f"Failed to connect to database: {e}. "
                f"Check database_url and ensure database is accessible. "
                f"Set require_database_connectivity=False to skip this check."
            ) from e

        return self

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
        """Return affected feature keys.

        Args:
            store: Metadata store
            project: Project name

        Returns:
            List of affected feature keys
        """
        return ["/".join(self.target_feature_key)]

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Execute database backfill with validation.

        Since all validation is done in Pydantic validators, this method
        can assume the configuration is valid and focus on the actual
        backfill logic.

        Args:
            store: Metadata store
            project: Project name
            dry_run: If True, validate without executing

        Returns:
            Migration result
        """
        start_time = datetime.now(timezone.utc)
        feature_key_str = "/".join(self.target_feature_key)

        try:
            # Configuration is already validated, so we can proceed confidently
            # Implementation would load from database, filter, and write to store
            # (Similar to Example 2, but loading from database instead of S3)

            # For this example, we'll return a success result
            # In a real migration, you'd implement the actual backfill logic here

            return MigrationResult(
                migration_id=self.migration_id,
                status="completed" if not dry_run else "skipped",
                features_completed=1,
                features_failed=0,
                features_skipped=0,
                affected_features=[feature_key_str],
                errors={},
                rows_affected=0,
                duration_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                timestamp=start_time,
            )

        except Exception as e:
            return MigrationResult(
                migration_id=self.migration_id,
                status="failed",
                features_completed=0,
                features_failed=1,
                features_skipped=0,
                affected_features=[],
                errors={feature_key_str: str(e)},
                rows_affected=0,
                duration_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                timestamp=start_time,
            )


# Alternative Example: File-based validation
# -------------------------------------------
class FileBasedMigration(PythonMigration):
    """Migration that validates file paths and permissions.

    Demonstrates validation of filesystem resources.
    """

    migration_id: str = "20250101_150000_file_validation"
    created_at: datetime = datetime(2025, 1, 1, 15, 0, 0, tzinfo=timezone.utc)

    input_directory: Path
    output_directory: Path
    file_pattern: str = "*.parquet"

    @field_validator("input_directory", "output_directory")
    @classmethod
    def validate_directory_exists(cls, v: Path) -> Path:
        """Validate directory exists and is accessible.

        Args:
            v: Directory path

        Returns:
            Validated path

        Raises:
            ValueError: If directory doesn't exist or isn't accessible
        """
        if not v.exists():
            raise ValueError(f"Directory does not exist: {v}")

        if not v.is_dir():
            raise ValueError(f"Path is not a directory: {v}")

        return v

    @model_validator(mode="after")
    def validate_directories_differ(self) -> "FileBasedMigration":
        """Validate input and output directories are different.

        Returns:
            Validated model

        Raises:
            ValueError: If directories are the same
        """
        if self.input_directory.resolve() == self.output_directory.resolve():
            raise ValueError(
                "input_directory and output_directory must be different. "
                f"Both point to: {self.input_directory}"
            )

        return self

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
        """Return affected features."""
        return []

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Execute with validated file paths."""
        # Implementation here
        return MigrationResult(
            migration_id=self.migration_id,
            status="completed",
            features_completed=0,
            features_failed=0,
            features_skipped=0,
            affected_features=[],
            errors={},
            rows_affected=0,
            duration_seconds=0.0,
            timestamp=datetime.now(timezone.utc),
        )


# Key Takeaways:
# --------------
# 1. Use @field_validator for individual field validation
# 2. Use @model_validator for cross-field validation
# 3. Validation runs automatically when migration is instantiated
# 4. Fail early with clear error messages
# 5. Validate external resources (databases, files, APIs) upfront
# 6. Use mode="after" for model validators that need access to all fields
# 7. Pydantic validators are type-safe and well-documented

# Validation Best Practices:
# ---------------------------
# 1. Validate as early as possible (fail fast)
# 2. Provide clear, actionable error messages
# 3. Check external resources are accessible before expensive operations
# 4. Validate data ranges and constraints
# 5. Use field validators for single-field checks
# 6. Use model validators for multi-field relationships
# 7. Consider adding skip flags for optional validation (useful in dev)

# Common Validation Patterns:
# ----------------------------
# - URL format and connectivity
# - File paths and permissions
# - Date ranges and time windows
# - Numeric ranges and constraints
# - Enum/choice validation
# - External resource existence (S3 buckets, databases, APIs)
# - Data quality constraints
# - Environment prerequisites
