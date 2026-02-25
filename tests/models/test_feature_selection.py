"""Tests for FeatureSelection model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from metaxy.models.feature_selection import FeatureSelection


class TestValidation:
    def test_projects_mode(self):
        sel = FeatureSelection(projects=["my-project"])
        assert sel.projects == ["my-project"]
        assert sel.keys is None
        assert sel.all is None

    def test_keys_mode(self):
        sel = FeatureSelection(keys=["a/b", "c/d"])
        assert sel.keys == ["a/b", "c/d"]

    def test_all_mode(self):
        sel = FeatureSelection(all=True)
        assert sel.all is True

    def test_both_projects_and_keys(self):
        sel = FeatureSelection(projects=["p"], keys=["a/b"])
        assert sel.projects == ["p"]
        assert sel.keys == ["a/b"]

    def test_no_mode_raises(self):
        with pytest.raises(ValidationError, match="At least one"):
            FeatureSelection()

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FeatureSelection(projects=["x"], tags=["v1"])  # type: ignore[call-arg]

    def test_frozen(self):
        sel = FeatureSelection(projects=["x"])
        with pytest.raises(ValidationError, match="frozen"):
            sel.projects = ["y"]


class TestOr:
    def test_merges_projects(self):
        result = FeatureSelection(projects=["a"]) | FeatureSelection(projects=["b"])
        assert result.projects == ["a", "b"]

    def test_merges_keys(self):
        result = FeatureSelection(keys=["x/1"]) | FeatureSelection(keys=["x/2"])
        assert result.keys == ["x/1", "x/2"]

    def test_mixed_fields(self):
        result = FeatureSelection(projects=["a"]) | FeatureSelection(keys=["x/1"])
        assert result.projects == ["a"]
        assert result.keys == ["x/1"]

    def test_all_absorbs(self):
        result = FeatureSelection(projects=["a"]) | FeatureSelection(all=True)
        assert result.all is True

    def test_deduplicates(self):
        result = FeatureSelection(projects=["a", "b"]) | FeatureSelection(projects=["b", "c"])
        assert result.projects == ["a", "b", "c"]


class TestAnd:
    def test_intersects_projects(self):
        result = FeatureSelection(projects=["a", "b"]) & FeatureSelection(projects=["b", "c"])
        assert result.projects == ["b"]

    def test_intersects_keys(self):
        result = FeatureSelection(keys=["x/1", "x/2"]) & FeatureSelection(keys=["x/2", "x/3"])
        assert result.keys == ["x/2"]

    def test_all_passes_through(self):
        sel = FeatureSelection(projects=["a"])
        assert (FeatureSelection(all=True) & sel) == sel
        assert (sel & FeatureSelection(all=True)) == sel

    def test_disjoint_projects_raises(self):
        """Intersecting disjoint sets produces None fields, which fails validation."""
        with pytest.raises(ValidationError, match="At least one"):
            FeatureSelection(projects=["a"]) & FeatureSelection(projects=["b"])


class TestSub:
    def test_subtracts_projects(self):
        result = FeatureSelection(projects=["a", "b", "c"]) - FeatureSelection(projects=["b"])
        assert result.projects == ["a", "c"]

    def test_subtracts_keys(self):
        result = FeatureSelection(keys=["x/1", "x/2"]) - FeatureSelection(keys=["x/1"])
        assert result.keys == ["x/2"]

    def test_sub_all_raises(self):
        """Subtracting all from a selection empties it, which fails validation."""
        with pytest.raises(ValidationError, match="At least one"):
            FeatureSelection(projects=["a"]) - FeatureSelection(all=True)

    def test_non_overlapping_field_unchanged(self):
        result = FeatureSelection(projects=["a"], keys=["x/1"]) - FeatureSelection(keys=["x/1"])
        assert result.projects == ["a"]
        assert result.keys is None
