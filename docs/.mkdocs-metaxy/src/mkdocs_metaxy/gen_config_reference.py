"""Generate configuration reference documentation."""

import sys
from pathlib import Path

# Add project source to path so we can import metaxy
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import mkdocs_gen_files

from metaxy.config import MetaxyConfig
from mkdocs_metaxy.config_generator import (  # pyright: ignore
    extract_field_info,
    generate_individual_field_doc,
    generate_toml_example,
)


def generate_configuration_docs():
    """Generate the configuration reference page."""

    # Extract field information from MetaxyConfig
    fields = extract_field_info(MetaxyConfig)

    # Build the documentation content
    lines = [
        "# Configuration",
        "",
        "Metaxy can be configured using TOML configuration files or environment variables.",
        "",
        "## Default Configuration",
        "",
        "Here is the complete default configuration with all available options:",
        "",
        '=== "!metaxy.toml"',
        "",
        "    ```toml",
    ]

    # Generate metaxy.toml example
    metaxy_toml = generate_toml_example(fields, include_tool_section=False)
    for line in metaxy_toml.split("\n"):
        lines.append(f"    {line}")

    lines.extend(
        [
            "    ```",
            "",
            '=== "pyproject.toml"',
            "",
            "    ```toml",
        ]
    )

    # Generate pyproject.toml example
    pyproject_toml = generate_toml_example(fields, include_tool_section=True)
    for line in pyproject_toml.split("\n"):
        lines.append(f"    {line}")

    lines.extend(
        [
            "    ```",
            "",
            "## Configuration Fields",
            "",
            "Each field can be set via TOML configuration or environment variables.",
            "",
        ]
    )

    # Generate individual field documentation
    # Group fields by section
    top_level_fields = [f for f in fields if len(f["path"]) == 1 and not f["is_nested"]]
    nested_sections: dict[str, list[dict]] = {}  # pyright: ignore

    for field in fields:
        if len(field["path"]) > 1 and not field["is_nested"]:
            section = ".".join(field["path"][:-1])
            if section not in nested_sections:
                nested_sections[section] = []
            nested_sections[section].append(field)

    # Document top-level fields
    for field in top_level_fields:
        field_doc = generate_individual_field_doc(field)
        lines.append(field_doc)

    # Document nested sections
    for section, section_fields in nested_sections.items():
        section_title = section.replace(".", " > ").title()
        lines.append(f"## {section_title} Configuration")
        lines.append("")

        for field in section_fields:
            field_doc = generate_individual_field_doc(field)
            lines.append(field_doc)

    # Add configuration types documentation
    lines.extend(
        [
            "",
            "## Configuration Types",
            "",
            "### StoreConfig",
            "",
            "Configuration for a single metadata store backend.",
            "",
            "**Fields:**",
            "",
            "- `type` (str): Full import path to the store class",
            "- `config` (dict[str, Any]): Store-specific configuration options",
            "",
            "### ExtConfig",
            "",
            "Configuration for Metaxy integrations with third-party tools.",
            "",
            "**Fields:**",
            "",
            "- `sqlmodel` ([SQLModelConfig](#sqlmodelconfig)): SQLModel integration configuration",
            "",
            "### SQLModelConfig",
            "",
            "Configuration for SQLModel integration.",
            "",
            "**Fields:**",
            "",
            "- `enable` (bool): Whether to enable the plugin (default: `false`)",
            "- `infer_db_table_names` (bool): Whether to automatically use `FeatureKey.table_name` for sqlalchemy's `__tablename__` value (default: `true`)",
            "- `system_tables` (bool): Whether to use SQLModel definitions for system tables (default: `true`)",
            "",
        ]
    )

    # Add stores configuration example
    lines.extend(
        [
            "## Store Configuration",
            "",
            "The `stores` field configures metadata store backends. Each store is defined by:",
            "",
            "- **`type`**: Full import path to the store class (e.g., `metaxy.metadata_store.duckdb.DuckDBMetadataStore`)",
            "- **`config`**: Dictionary of store-specific configuration options",
            "",
            "### Example: Multiple Stores with Fallback Chain",
            "",
            '=== "!metaxy.toml"',
            "",
            "    ```toml",
            "    # Default store to use",
            '    store = "dev"',
            "",
            "    # Development store (in-memory) with fallback to production",
            "    [stores.dev]",
            '    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"',
            "    [stores.dev.config]",
            '    db_path = ":memory:"',
            '    fallback_stores = ["prod"]',
            "",
            "    # Production store",
            "    [stores.prod]",
            '    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"',
            "    [stores.prod.config]",
            '    db_path = "s3://my-bucket/metadata.duckdb"',
            "    ```",
            "",
            '=== "pyproject.toml"',
            "",
            "    ```toml",
            "    [tool.metaxy]",
            '    store = "dev"',
            "",
            "    [tool.metaxy.stores.dev]",
            '    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"',
            "    [tool.metaxy.stores.dev.config]",
            '    db_path = ":memory:"',
            '    fallback_stores = ["prod"]',
            "",
            "    [tool.metaxy.stores.prod]",
            '    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"',
            "    [tool.metaxy.stores.prod.config]",
            '    db_path = "s3://my-bucket/metadata.duckdb"',
            "    ```",
            "",
            "### Available Store Types",
            "",
            "| Store Type | Import Path | Description |",
            "|------------|-------------|-------------|",
            "| DuckDB | `metaxy.metadata_store.duckdb.DuckDBMetadataStore` | File-based or in-memory DuckDB backend |",
            "| ClickHouse | `metaxy.metadata_store.clickhouse.ClickHouseMetadataStore` | ClickHouse database backend |",
            "| In-Memory | `metaxy.metadata_store.memory.InMemoryMetadataStore` | In-memory backend for testing |",
            "",
            "### Getting a Store Instance",
            "",
            "```python",
            "from metaxy.config import MetaxyConfig",
            "",
            "config = MetaxyConfig.load()",
            "",
            "# Get the default store",
            "with config.get_store() as store:",
            "    # Use store",
            "    pass",
            "",
            "# Get a specific store by name",
            'with config.get_store("prod") as store:',
            "    # Use store",
            "    pass",
            "```",
            "",
            "## Configuration Priority",
            "",
            "When the same setting is defined in multiple places, Metaxy uses the following priority order (highest to lowest):",
            "",
            "1. **Explicit arguments** - Values passed directly to `MetaxyConfig()`",
            "2. **Environment variables** - Values from `METAXY_*` environment variables",
            "3. **Configuration files** - Values from `metaxy.toml` or `pyproject.toml`",
            "",
            "This allows you to override file-based configuration with environment variables, which is useful for CI/CD pipelines and different deployment environments.",
            "",
            "## Loading Configuration",
            "",
            "### Auto-Discovery",
            "",
            "```python",
            "from metaxy.config import MetaxyConfig",
            "",
            "# Auto-discover config file in current or parent directories",
            "config = MetaxyConfig.load()",
            "```",
            "",
            "### Explicit File",
            "",
            "```python",
            "# Load from specific file",
            'config = MetaxyConfig.load("path/to/metaxy.toml")',
            "```",
            "",
            "### Programmatic Configuration",
            "",
            "```python",
            "# Create configuration programmatically",
            "config = MetaxyConfig(",
            '    store="prod",',
            '    migrations_dir=".migrations",',
            ")",
            "```",
            "",
        ]
    )

    # Write the generated documentation
    content = "\n".join(lines)

    with mkdocs_gen_files.open("reference/configuration.md", "w") as f:
        f.write(content)


# Generate the configuration docs when the module is imported by mkdocs-gen-files
generate_configuration_docs()

if __name__ == "__main__":
    # This won't be called by mkdocs-gen-files, but allows manual testing
    pass
