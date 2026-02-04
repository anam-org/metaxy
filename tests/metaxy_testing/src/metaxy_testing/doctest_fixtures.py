"""Pre-defined features and stores for docstring examples.

These features have stable import paths so they work with to_snapshot()/from_snapshot().
"""

from __future__ import annotations

import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec

if TYPE_CHECKING:
    from metaxy.config import MetaxyConfig

# Define spec at module level for stable reference
_MY_FEATURE_SPEC = FeatureSpec(key="my/feature", id_columns=["id"], fields=["part/1", "part/2"])

sample_data = pl.DataFrame(
    {
        "id": ["a", "b", "c"],
        "status": ["active", "pending", "active"],
        "is_valid": [True, False, True],
        "quality": [0.95, 0.72, 0.88],
        "metaxy_provenance_by_field": [
            {"part_1": "hash_a1", "part_2": "hash_a2"},
            {"part_1": "hash_b1", "part_2": "hash_b2"},
            {"part_1": "hash_c1", "part_2": "hash_c2"},
        ],
    }
)


class MyFeature(BaseFeature, spec=None):
    """Pre-populated feature for docstring examples.

    Available in all docstring examples as `MyFeature`.
    Has key "my/feature" and id column "id".
    """

    id: str
    status: str | None = None
    is_valid: bool | None = None
    quality: float | None = None


# Set up class attributes (bypassing metaclass project detection)
MyFeature._spec = _MY_FEATURE_SPEC
MyFeature.__metaxy_project__ = "docs"


def register_doctest_fixtures(graph: FeatureGraph) -> None:
    """Register doctest fixtures to the given graph.

    Args:
        graph: The FeatureGraph to register fixtures to.
    """
    MyFeature.graph = graph
    graph.add_feature(MyFeature)


class DocsStoreFixtures:
    """Manages store fixtures for docstring examples.

    Creates temporary directories and stores that are cleaned up on teardown.
    Uses ExitStack for proper cleanup of all context managers.
    """

    def __init__(self, graph: FeatureGraph):
        self.graph = graph
        self._exit_stack: ExitStack | None = None
        self._store: DeltaMetadataStore | None = None
        self._store_with_data: DeltaMetadataStore | None = None
        self._config: MetaxyConfig | None = None

    def setup(self) -> None:
        """Create temporary stores and config."""
        from metaxy.config import MetaxyConfig, StoreConfig

        self._exit_stack = ExitStack()

        # Create and register temporary directory for cleanup
        temp_dir = self._exit_stack.enter_context(tempfile.TemporaryDirectory())
        temp_path = Path(temp_dir)

        # Empty store
        self._store = DeltaMetadataStore(root_path=temp_path / "store")

        # Store with sample data
        self._store_with_data = DeltaMetadataStore(root_path=temp_path / "store_with_data")

        # Create a config with both stores pre-configured
        self._config = MetaxyConfig(
            store="dev",
            stores={
                "dev": StoreConfig(
                    type="metaxy.metadata_store.delta.DeltaMetadataStore",
                    config={"root_path": str(temp_path / "store")},
                ),
                "prod": StoreConfig(
                    type="metaxy.metadata_store.delta.DeltaMetadataStore",
                    config={"root_path": str(temp_path / "store_with_data")},
                ),
            },
        )
        # Enter config context so MetaxyConfig.get() returns our config
        self._exit_stack.enter_context(self._config.use())

        # Write sample data to store_with_data
        # Include required metaxy_provenance_by_field column with struct matching feature fields
        # Field keys use underscores in struct keys (part/1 -> part_1)

        with self._store_with_data.open("w"):
            self._store_with_data.write(MyFeature, sample_data)

    def teardown(self) -> None:
        """Clean up all resources via ExitStack."""
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
            self._config = None
            self._store = None
            self._store_with_data = None

    @property
    def store(self) -> DeltaMetadataStore:
        """Empty store for write examples."""
        if self._store is None:
            raise RuntimeError("Store fixtures not set up. Call setup() first.")
        return self._store

    @property
    def store_with_data(self) -> DeltaMetadataStore:
        """Store pre-populated with sample data."""
        if self._store_with_data is None:
            raise RuntimeError("Store fixtures not set up. Call setup() first.")
        return self._store_with_data

    @property
    def config(self) -> MetaxyConfig:
        """Pre-configured MetaxyConfig with dev and prod stores."""
        if self._config is None:
            raise RuntimeError("Store fixtures not set up. Call setup() first.")
        return self._config


__all__ = ["MyFeature", "register_doctest_fixtures", "DocsStoreFixtures"]
