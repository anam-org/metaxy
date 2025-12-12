"""Tests for optional dependencies feature (FeatureDep.optional parameter).

This module tests:
- FeatureDep.optional field serialization and versioning
- FeatureSpec validation (first dep cannot be optional)
- FeaturePlan helper properties (optional_deps, required_deps, main_dep)
"""

from __future__ import annotations

import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature import FeatureGraph


class TestFeatureDepOptional:
    """Test FeatureDep.optional field."""

    def test_optional_defaults_to_false(self) -> None:
        """Test that optional parameter defaults to False."""
        dep = FeatureDep(feature=FeatureKey(["test", "feature"]))
        assert dep.optional is False

    def test_optional_true_serializes_correctly(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Test that optional=True serializes and deserializes correctly."""
        dep = FeatureDep(
            feature=FeatureKey(["test", "feature"]),
            optional=True,
        )
        assert dep.optional is True

        # Test serialization
        serialized = dep.model_dump(mode="json")
        assert serialized == snapshot

        # Test deserialization
        deserialized = FeatureDep.model_validate(serialized)
        assert deserialized.optional is True
        assert deserialized.feature == FeatureKey(["test", "feature"])

    def test_optional_false_serializes_correctly(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Test that optional=False serializes correctly (default behavior)."""
        dep = FeatureDep(
            feature=FeatureKey(["test", "feature"]),
            optional=False,
        )

        # Test serialization
        serialized = dep.model_dump(mode="json")
        assert serialized == snapshot

    def test_feature_spec_version_changes_when_optional_changes(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Test that feature_spec_version changes when optional changes."""
        # Create two specs - one with optional=False, one with optional=True
        spec_required = SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=FeatureKey(["test", "upstream1"])),
                FeatureDep(feature=FeatureKey(["test", "upstream2"]), optional=False),
            ],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        )

        spec_optional = SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=FeatureKey(["test", "upstream1"])),
                FeatureDep(feature=FeatureKey(["test", "upstream2"]), optional=True),
            ],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        )

        # Versions should differ
        assert spec_required.feature_spec_version != spec_optional.feature_spec_version

        # Snapshot both versions
        assert {
            "required": spec_required.feature_spec_version,
            "optional": spec_optional.feature_spec_version,
        } == snapshot


class TestFeatureSpecValidation:
    """Test FeatureSpec validation for optional dependencies."""

    def test_first_dep_cannot_be_optional(
        self, graph: FeatureGraph, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that first dependency cannot be optional=True."""
        # Use upstream1 from fixture
        _ = upstream_features["upstream1"]

        # Should raise error when first dep is optional
        with pytest.raises(
            ValueError,
            match=r"The first dependency cannot be optional.*test/upstream1.*optional=True.*defines the sample universe",
        ):
            _ = SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]),
                        optional=True,  # First dep cannot be optional!
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            )

    def test_first_dep_required_subsequent_optional_is_valid(
        self, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that first dep required and subsequent deps optional is valid."""
        # Use upstream features from fixture
        _ = upstream_features["upstream1"]
        _ = upstream_features["upstream2"]

        # Should NOT raise error - first is required, second is optional
        spec = SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=FeatureKey(["test", "upstream1"]), optional=False),
                FeatureDep(feature=FeatureKey(["test", "upstream2"]), optional=True),
            ],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        )

        assert len(spec.deps) == 2
        assert spec.deps[0].optional is False
        assert spec.deps[1].optional is True

    def test_all_deps_required_is_valid(
        self, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that all deps being required (optional=False) is valid."""
        # Use upstream features from fixture
        _ = upstream_features["upstream1"]
        _ = upstream_features["upstream2"]

        # Should NOT raise error - all required
        spec = SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=FeatureKey(["test", "upstream1"]), optional=False),
                FeatureDep(feature=FeatureKey(["test", "upstream2"]), optional=False),
            ],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        )

        assert len(spec.deps) == 2
        assert all(not dep.optional for dep in spec.deps)

    def test_single_dependency_cannot_be_optional(
        self, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that single dependency cannot be optional (first dep rule applies)."""
        # Use upstream1 from fixture
        _ = upstream_features["upstream1"]

        # Should raise error - single dep is effectively the first dep
        with pytest.raises(
            ValueError,
            match=r"The first dependency cannot be optional",
        ):
            _ = SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]),
                        optional=True,
                    )
                ],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            )

    def test_no_deps_is_valid(self, graph: FeatureGraph) -> None:  # noqa: ARG002
        """Test that feature with no dependencies is valid (root feature)."""
        spec = SampleFeatureSpec(
            key=FeatureKey(["test", "root"]),
            deps=[],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        )

        assert len(spec.deps) == 0


class TestFeaturePlanHelpers:
    """Test FeaturePlan helper properties for optional dependencies."""

    def test_optional_deps_returns_correct_subset(
        self, graph: FeatureGraph, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that optional_deps returns only deps with optional=True."""
        # Use upstream features from fixture
        _ = upstream_features["upstream1"]
        _ = upstream_features["upstream2"]
        _ = upstream_features["upstream3"]

        class _Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]), optional=False
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]), optional=True
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream3"]), optional=True
                    ),
                ],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Verify downstream registered in graph
        assert FeatureKey(["test", "downstream"]) in graph.features_by_key

        plan = graph.get_feature_plan(FeatureKey(["test", "downstream"]))

        # Should return only deps with optional=True
        optional_deps = plan.optional_deps
        assert len(optional_deps) == 2
        assert all(dep.optional for dep in optional_deps)
        assert {dep.feature for dep in optional_deps} == {
            FeatureKey(["test", "upstream2"]),
            FeatureKey(["test", "upstream3"]),
        }

    def test_required_deps_returns_correct_subset(
        self, graph: FeatureGraph, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that required_deps returns only deps with optional=False."""
        # Use upstream features from fixture
        _ = upstream_features["upstream1"]
        _ = upstream_features["upstream2"]
        _ = upstream_features["upstream3"]

        class _Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]), optional=False
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]), optional=True
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream3"]), optional=False
                    ),
                ],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Verify downstream registered in graph
        assert FeatureKey(["test", "downstream"]) in graph.features_by_key

        plan = graph.get_feature_plan(FeatureKey(["test", "downstream"]))

        # Should return only deps with optional=False
        required_deps = plan.required_deps
        assert len(required_deps) == 2
        assert all(not dep.optional for dep in required_deps)
        assert {dep.feature for dep in required_deps} == {
            FeatureKey(["test", "upstream1"]),
            FeatureKey(["test", "upstream3"]),
        }

    def test_main_dep_returns_first_dependency(
        self, graph: FeatureGraph, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test that main_dep returns the first dependency (sample universe)."""
        # Use upstream features from fixture
        _ = upstream_features["upstream1"]
        _ = upstream_features["upstream2"]

        class _Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]), optional=False
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]), optional=True
                    ),
                ],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Verify downstream registered in graph
        assert FeatureKey(["test", "downstream"]) in graph.features_by_key

        plan = graph.get_feature_plan(FeatureKey(["test", "downstream"]))

        # Should return first dependency
        main_dep = plan.main_dep
        assert main_dep is not None
        assert main_dep.feature == FeatureKey(["test", "upstream1"])
        assert main_dep.optional is False

    def test_main_dep_is_none_when_no_deps(self, graph: FeatureGraph) -> None:
        """Test that main_dep returns None when there are no dependencies."""

        class _RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "root"]),
                deps=[],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Verify feature registered in graph
        assert FeatureKey(["test", "root"]) in graph.features_by_key

        plan = graph.get_feature_plan(FeatureKey(["test", "root"]))

        # Should return None for root features
        assert plan.main_dep is None

    def test_all_helpers_with_all_required_deps(
        self, graph: FeatureGraph, upstream_features: dict[str, type[BaseFeature]]
    ) -> None:
        """Test helper properties when all deps are required (no optional deps)."""
        # Use upstream features from fixture
        _ = upstream_features["upstream1"]
        _ = upstream_features["upstream2"]

        class _Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream1"]), optional=False
                    ),
                    FeatureDep(
                        feature=FeatureKey(["test", "upstream2"]), optional=False
                    ),
                ],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Verify downstream registered in graph
        assert FeatureKey(["test", "downstream"]) in graph.features_by_key

        plan = graph.get_feature_plan(FeatureKey(["test", "downstream"]))

        # All deps are required
        assert len(plan.optional_deps) == 0
        assert len(plan.required_deps) == 2

        # Main dep is still the first one
        assert plan.main_dep is not None
        assert plan.main_dep.feature == FeatureKey(["test", "upstream1"])
