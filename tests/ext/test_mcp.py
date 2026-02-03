"""Tests for the MCP server extension.

Uses FastMCP's recommended testing pattern with Client to test tools
through the MCP protocol rather than calling implementation functions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from fastmcp import Client
from metaxy_testing.models import SampleFeatureSpec

from metaxy import FeatureKey, FieldKey, FieldSpec, MetaxyConfig
from metaxy.ext.mcp.server import create_server
from metaxy.models.feature import BaseFeature

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def mcp_client(config: MetaxyConfig) -> AsyncIterator[Client]:
    """Create an MCP client connected to a test server.

    The server uses MetaxyConfig.get() and FeatureGraph.get_active() internally,
    which are set by the conftest.py fixtures (config and graph).
    """
    # config fixture is used to ensure it's set before server starts
    _ = config
    server = create_server()

    async with Client(server) as client:
        yield client


def _create_test_features() -> tuple[type[BaseFeature], type[BaseFeature]]:
    """Create test features in the active graph."""

    class ParentFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["mcp_test", "parent"]),
            fields=[FieldSpec(key=FieldKey(["embedding"]), code_version="1")],
        ),
    ):
        pass

    class ChildFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["mcp_test", "child"]),
            fields=[FieldSpec(key=FieldKey(["prediction"]), code_version="1")],
            deps=[ParentFeature],
        ),
    ):
        pass

    return ParentFeature, ChildFeature


async def _call_tool(client: Client, name: str, **kwargs: Any) -> Any:
    """Call an MCP tool and return the result data.

    Parses the JSON text content directly to avoid issues with FastMCP's
    structured_content having different formats for dict vs list returns.
    For plain string returns, returns the text directly.
    """
    import json

    from mcp.types import TextContent

    result = await client.call_tool(name, kwargs)
    content = result.content[0]
    assert isinstance(content, TextContent)
    text_content = content.text

    # Try to parse as JSON, fall back to raw text for string returns
    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        return text_content


class TestGetConfig:
    """Tests for get_config tool."""

    async def test_returns_config_info(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "get_config")

        assert "project" in result
        assert result["project"] == "test"
        assert "store" in result
        assert "entrypoints" in result
        assert "stores" in result
        assert isinstance(result["stores"], dict)

    async def test_stores_have_type_info(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "get_config")

        assert "dev" in result["stores"]
        # Full config serialization uses type_path as the field name
        assert "type_path" in result["stores"]["dev"]
        assert "DuckDBMetadataStore" in result["stores"]["dev"]["type_path"]


class TestListFeatures:
    """Tests for list_features tool."""

    async def test_lists_all_features(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "list_features")

        # Now returns dict with feature_count and features list
        assert result["feature_count"] == 2
        keys = [f["key"] for f in result["features"]]
        assert "mcp_test/parent" in keys
        assert "mcp_test/child" in keys

    async def test_features_have_import_path(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "list_features")

        for feature in result["features"]:
            assert "import_path" in feature
            assert "test_mcp" in feature["import_path"]

    async def test_features_have_version_and_metadata(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "list_features")

        for feature in result["features"]:
            assert "version" in feature
            assert "is_root" in feature
            assert "project" in feature
            assert "field_count" in feature
            assert "fields" in feature

    async def test_verbose_includes_field_details(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "list_features", verbose=True)

        for feature in result["features"]:
            for field in feature["fields"]:
                assert "key" in field
                assert "code_version" in field
                assert "version" in field


class TestGetFeature:
    """Tests for get_feature tool."""

    async def test_returns_feature_spec(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "get_feature", feature_key="mcp_test/parent")

        assert result["key"] == "mcp_test/parent"
        assert "fields" in result
        assert "id_columns" in result

    async def test_includes_dependencies(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "get_feature", feature_key="mcp_test/child")

        assert "deps" in result
        assert len(result["deps"]) == 1
        assert result["deps"][0]["feature"] == "mcp_test/parent"

    async def test_raises_for_unknown_feature(self, mcp_client: Client) -> None:
        _create_test_features()

        with pytest.raises(Exception):  # MCP wraps errors
            await _call_tool(mcp_client, "get_feature", feature_key="unknown/feature")


class TestListStores:
    """Tests for list_stores tool."""

    async def test_lists_configured_stores(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "list_stores")

        assert len(result) >= 1
        names = [s["name"] for s in result]
        assert "dev" in names

    async def test_stores_have_type(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "list_stores")

        for store in result:
            assert "type" in store
            assert isinstance(store["type"], str)


class TestGetStore:
    """Tests for get_store tool."""

    async def test_returns_display_string(self, mcp_client: Client) -> None:
        _create_test_features()

        result = await _call_tool(mcp_client, "get_store", store_name="dev")

        assert isinstance(result, str)
        assert "DuckDBMetadataStore" in result

    async def test_raises_for_unknown_store(self, mcp_client: Client) -> None:
        _create_test_features()

        with pytest.raises(Exception):  # MCP wraps errors
            await _call_tool(mcp_client, "get_store", store_name="nonexistent")


class TestGetMetadata:
    """Tests for get_metadata tool."""

    async def test_raises_for_feature_not_in_store(self, mcp_client: Client, config: MetaxyConfig) -> None:
        """When feature has no table in store, FeatureNotFoundError is raised."""
        _create_test_features()

        # Feature exists in graph but has no data written to store
        with pytest.raises(Exception) as exc_info:
            await _call_tool(
                mcp_client,
                "get_metadata",
                feature_key="mcp_test/parent",
                store_name="dev",
            )

        # The error should mention the feature wasn't found
        assert "not found" in str(exc_info.value).lower()

    async def test_returns_data_when_present(self, mcp_client: Client, config: MetaxyConfig) -> None:
        parent_cls, _ = _create_test_features()

        store = config.get_store("dev")
        test_data = pl.DataFrame(
            {
                "sample_uid": ["a", "b", "c"],
                "embedding": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "metaxy_provenance_by_field": [
                    {"embedding": "hash1"},
                    {"embedding": "hash2"},
                    {"embedding": "hash3"},
                ],
            }
        )

        with store.open("write"):
            store.write(parent_cls, test_data)

        result = await _call_tool(
            mcp_client,
            "get_metadata",
            feature_key="mcp_test/parent",
            store_name="dev",
            limit=2,
        )

        assert len(result["columns"]) > 0
        assert "sample_uid" in result["columns"]
        assert len(result["rows"]) == 2
        assert result["total_rows"] == 2

    async def test_respects_limit(self, mcp_client: Client, config: MetaxyConfig) -> None:
        parent_cls, _ = _create_test_features()

        store = config.get_store("dev")
        test_data = pl.DataFrame(
            {
                "sample_uid": [f"id_{i}" for i in range(5)],
                "embedding": [[float(i)] for i in range(5)],
                "metaxy_provenance_by_field": [{"embedding": f"hash{i}"} for i in range(5)],
            }
        )

        with store.open("write"):
            store.write(parent_cls, test_data)

        result = await _call_tool(
            mcp_client,
            "get_metadata",
            feature_key="mcp_test/parent",
            store_name="dev",
            limit=3,
        )

        assert result["total_rows"] == 3
        assert len(result["rows"]) == 3

    async def test_select_columns(self, mcp_client: Client, config: MetaxyConfig) -> None:
        parent_cls, _ = _create_test_features()

        store = config.get_store("dev")
        test_data = pl.DataFrame(
            {
                "sample_uid": ["a", "b"],
                "embedding": [[1.0, 2.0], [3.0, 4.0]],
                "metaxy_provenance_by_field": [
                    {"embedding": "hash1"},
                    {"embedding": "hash2"},
                ],
            }
        )

        with store.open("write"):
            store.write(parent_cls, test_data)

        result = await _call_tool(
            mcp_client,
            "get_metadata",
            feature_key="mcp_test/parent",
            store_name="dev",
            columns=["sample_uid"],
        )

        assert result["columns"] == ["sample_uid"]
        assert len(result["rows"]) == 2

    async def test_sort_by(self, mcp_client: Client, config: MetaxyConfig) -> None:
        parent_cls, _ = _create_test_features()

        store = config.get_store("dev")
        test_data = pl.DataFrame(
            {
                "sample_uid": ["c", "a", "b"],
                "embedding": [[1.0], [2.0], [3.0]],
                "metaxy_provenance_by_field": [
                    {"embedding": "hash1"},
                    {"embedding": "hash2"},
                    {"embedding": "hash3"},
                ],
            }
        )

        with store.open("write"):
            store.write(parent_cls, test_data)

        # Sort ascending
        result = await _call_tool(
            mcp_client,
            "get_metadata",
            feature_key="mcp_test/parent",
            store_name="dev",
            sort_by=["sample_uid"],
        )

        uids = [row["sample_uid"] for row in result["rows"]]
        assert uids == ["a", "b", "c"]

        # Sort descending
        result = await _call_tool(
            mcp_client,
            "get_metadata",
            feature_key="mcp_test/parent",
            store_name="dev",
            sort_by=["sample_uid"],
            descending=True,
        )

        uids = [row["sample_uid"] for row in result["rows"]]
        assert uids == ["c", "b", "a"]
