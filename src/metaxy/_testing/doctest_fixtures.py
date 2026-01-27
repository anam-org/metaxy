"""Pre-defined features and stores for docstring examples.

These features have stable import paths so they work with to_snapshot()/from_snapshot().
"""

import tempfile
from pathlib import Path

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec

# Define spec at module level for stable reference
_MY_FEATURE_SPEC = FeatureSpec(key="my/feature", id_columns=["id"], fields=["part/1", "part/2"])


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
MyFeature.project = "docs"


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
    """

    def __init__(self, graph: FeatureGraph):
        self.graph = graph
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._store: DeltaMetadataStore | None = None
        self._store_with_data: DeltaMetadataStore | None = None

    def setup(self) -> None:
        """Create temporary stores."""
        self._temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self._temp_dir.name)

        # Empty store
        self._store = DeltaMetadataStore(root_path=temp_path / "store")

        # Store with sample data
        self._store_with_data = DeltaMetadataStore(root_path=temp_path / "store_with_data")

        # Write sample data to store_with_data
        # Include required metaxy_provenance_by_field column with struct matching feature fields
        # Field keys use underscores in struct keys (part/1 -> part_1)
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

        with self._store_with_data.open("write"):
            self._store_with_data.write_metadata(MyFeature, sample_data)

    def teardown(self) -> None:
        """Clean up temporary directories."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

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


__all__ = ["MyFeature", "register_doctest_fixtures", "DocsStoreFixtures"]
