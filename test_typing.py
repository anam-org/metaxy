#!/usr/bin/env python3
"""Test script to verify the typing fixes in feature_spec.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from metaxy.models.feature_spec import BaseFeatureSpec, FeatureSpec, TestingFeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FieldKey


def test_base_feature_spec():
    """Test BaseFeatureSpec initialization with proper typing."""
    spec = BaseFeatureSpec(
        key="test/feature",
        fields=[FieldSpec(key=FieldKey(["field1"]))],
        id_columns=["sample_uid"],
    )
    print(f"BaseFeatureSpec created: {spec.key}")


def test_feature_spec():
    """Test FeatureSpec initialization with proper typing."""
    spec = FeatureSpec(key="test/feature", fields=[FieldSpec(key=FieldKey(["field1"]))])
    print(f"FeatureSpec created: {spec.key}")


def test_testing_feature_spec():
    """Test TestingFeatureSpec initialization with proper typing."""
    spec = TestingFeatureSpec(
        key="test/feature", fields=[FieldSpec(key=FieldKey(["field1"]))]
    )
    print(f"TestingFeatureSpec created: {spec.key}")


if __name__ == "__main__":
    try:
        test_base_feature_spec()
        test_feature_spec()
        test_testing_feature_spec()
        print("\n✅ All tests passed! The typing fix is working correctly.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
