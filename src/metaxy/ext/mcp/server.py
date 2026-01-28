"""FastMCP server for Metaxy.

Exposes Metaxy's feature graph and metadata store operations to AI assistants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from metaxy import FeatureGraph, MetaxyConfig, coerce_to_feature_key, init_metaxy

if TYPE_CHECKING:
    import narwhals as nw


@dataclass
class MetaxyContext:
    """Holds initialized Metaxy state for the MCP server.

    Note:
        The context is reloaded on each tool call to pick up code changes.
        This is intentional for development workflows where feature definitions
        may change between tool invocations.
    """

    config: MetaxyConfig
    graph: FeatureGraph


def create_server() -> FastMCP:
    """Create a FastMCP server.

    The server reloads configuration and feature graph on each tool call,
    allowing it to pick up code changes without restart.

    For testing, use MetaxyConfig.set() and FeatureGraph.set_active()
    (or their context manager equivalents) to inject test dependencies
    before making MCP calls.

    Returns:
        Configured FastMCP server instance.
    """
    server = FastMCP("metaxy")
    _register_tools(server)
    return server


def _get_metaxy_context() -> MetaxyContext:
    """Get MetaxyContext, loading config and graph on each call.

    If MetaxyConfig is already set (e.g., by tests), uses the existing config.
    Otherwise, initializes Metaxy from configuration files.

    This allows the MCP server to pick up code changes without restart in
    development, while also respecting test fixtures that set up custom configs.
    """
    if MetaxyConfig.is_set():
        config = MetaxyConfig.get()
    else:
        config = init_metaxy()

    graph = FeatureGraph.get_active()

    return MetaxyContext(config=config, graph=graph)


def _register_tools(server: FastMCP) -> None:
    """Register all MCP tools on the server."""

    @server.tool()
    def get_config() -> dict[str, Any]:
        """Get the current Metaxy configuration as JSON.

        Returns:
            The full Metaxy configuration serialized as JSON, including all
            settings like project, store, entrypoints, stores, migrations_dir, etc.
        """
        metaxy_ctx = _get_metaxy_context()
        config = metaxy_ctx.config

        # Serialize the full config, handling Path and class objects
        config_dict = config.model_dump(mode="json")

        # Add config_file which is a private attr not included in model_dump
        config_dict["config_file"] = str(config.config_file) if config.config_file else None

        return config_dict

    @server.tool()
    def list_features(project: str | None = None, verbose: bool = False) -> dict[str, Any]:
        """List all registered features with their metadata.

        Matches the output format of `mx list features --format json`.

        Args:
            project: Filter by project name (optional)
            verbose: Include detailed field information and dependencies (default: False)

        Returns:
            Dictionary containing:
            - feature_count: Total number of features
            - features: List of feature dictionaries with key, version, is_root,
              project, import_path, field_count, fields, and optionally deps
        """
        from metaxy.models.plan import FQFieldKey

        metaxy_ctx = _get_metaxy_context()
        graph = metaxy_ctx.graph

        features_data: list[dict[str, Any]] = []

        for feature_key, definition in graph.feature_definitions_by_key.items():
            # Filter by project if specified
            if project and definition.project != project:
                continue

            feature_spec = definition.spec
            version = graph.get_feature_version(feature_key)
            is_root = not feature_spec.deps
            import_path = definition.feature_class_path

            # Get the feature plan for resolved field dependencies
            feature_plan = graph.get_feature_plan(feature_key) if verbose else None

            # Build field info
            fields_info: list[dict[str, Any]] = []
            for field_key, field_spec in feature_spec.fields_by_key.items():
                field_version = graph.get_field_version(FQFieldKey(feature=feature_key, field=field_key))
                field_data: dict[str, Any] = {
                    "key": field_spec.key.to_string(),
                    "code_version": field_spec.code_version,
                    "version": field_version,
                }

                # In verbose mode, get resolved field dependencies from the plan
                if verbose and feature_plan and not is_root:
                    resolved_deps = feature_plan.field_dependencies.get(field_key, {})
                    if resolved_deps:
                        deps_list = []
                        for upstream_feature, upstream_fields in resolved_deps.items():
                            deps_list.append(
                                {
                                    "feature": upstream_feature.to_string(),
                                    "fields": [f.to_string() for f in upstream_fields],
                                }
                            )
                        field_data["deps"] = deps_list

                fields_info.append(field_data)

            # Build feature info
            feature_data: dict[str, Any] = {
                "key": feature_key.to_string(),
                "version": version,
                "is_root": is_root,
                "project": definition.project,
                "import_path": import_path,
                "field_count": len(fields_info),
                "fields": fields_info,
            }

            if verbose and feature_spec.deps:
                feature_data["deps"] = [dep.feature.to_string() for dep in feature_spec.deps]

            features_data.append(feature_data)

        return {
            "feature_count": len(features_data),
            "features": features_data,
        }

    @server.tool()
    def get_feature(feature_key: str) -> dict[str, Any]:
        """Get the complete specification for a feature.

        Args:
            feature_key: Feature key in slash notation (e.g., "video/processing")

        Returns:
            Complete feature specification as a dictionary including fields,
            dependencies, id_columns, and other metadata
        """
        metaxy_ctx = _get_metaxy_context()
        graph = metaxy_ctx.graph

        key = coerce_to_feature_key(feature_key)
        definition = graph.get_feature_definition(key)

        return definition.spec.model_dump(mode="json")

    @server.tool()
    def list_stores() -> list[dict[str, str]]:
        """List all configured metadata stores.

        Returns:
            List of dictionaries with 'name' and 'type' for each store
        """
        metaxy_ctx = _get_metaxy_context()
        config = metaxy_ctx.config

        return [
            {
                "name": name,
                "type": store_config.type_path
                if isinstance(store_config.type_path, str)
                else f"{store_config.type_path.__module__}.{store_config.type_path.__qualname__}",
            }
            for name, store_config in config.stores.items()
        ]

    @server.tool()
    def get_store(store_name: str) -> str:
        """Get display information for a metadata store.

        Args:
            store_name: Name of the store to get info for

        Returns:
            Human-readable display string for the store
            (e.g., "DuckDBMetadataStore(database=/tmp/db.duckdb)")
        """
        metaxy_ctx = _get_metaxy_context()
        config = metaxy_ctx.config

        store = config.get_store(store_name)
        return store.display()

    @server.tool()
    def get_metadata(
        feature_key: str,
        store_name: str,
        columns: list[str] | None = None,
        filters: list[str] | None = None,
        current_only: bool = True,
        latest_only: bool = True,
        include_soft_deleted: bool = False,
        allow_fallback: bool = True,
        sort_by: list[str] | None = None,
        descending: bool | list[bool] = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query metadata for a feature from a store.

        Args:
            feature_key: Feature key in slash notation (e.g., "video/processing")
            store_name: Name of the metadata store to read from
            columns: List of columns to select (None for all)
            filters: List of SQL-like filter expressions (e.g., "column > 5", "name == 'foo'")
            current_only: Only return current (non-superseded) rows (default: True)
            latest_only: Only return latest version of each row (default: True)
            include_soft_deleted: Include soft-deleted rows (default: False)
            allow_fallback: If True, check fallback stores when feature is not found
                in the primary store (default: True)
            sort_by: List of column names to sort by
            descending: Sort descending (bool for all columns, or list per column)
            limit: Maximum number of rows to return (default: 50)

        Returns:
            Dictionary containing:
            - columns: List of column names
            - rows: List of row dictionaries
            - total_rows: Number of rows returned
        """
        import polars as pl

        metaxy_ctx = _get_metaxy_context()
        config = metaxy_ctx.config
        graph = metaxy_ctx.graph

        key = coerce_to_feature_key(feature_key)
        definition = graph.get_feature_definition(key)
        store = config.get_store(store_name)

        nw_filters = _parse_filters(filters)

        with store:
            lazy_frame = store.read_metadata(
                definition,
                columns=columns,
                filters=nw_filters if nw_filters else None,
                current_only=current_only,
                latest_only=latest_only,
                include_soft_deleted=include_soft_deleted,
                allow_fallback=allow_fallback,
            )

            if sort_by:
                lazy_frame = lazy_frame.sort(by=sort_by, descending=descending)

            df: pl.DataFrame = lazy_frame.head(limit).collect().to_polars()  # type: ignore[assignment]

            return {
                "columns": list(df.columns),
                "rows": df.to_dicts(),
                "total_rows": len(df),
            }


def _parse_filters(filters: list[str] | None) -> list[nw.Expr]:
    """Parse filter strings into narwhals expressions."""
    if not filters:
        return []

    from metaxy.models.filter_expression import parse_filter_string

    return [parse_filter_string(filter_str) for filter_str in filters]


# Default server instance for production use
mcp = create_server()


def run_server() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run_server()
