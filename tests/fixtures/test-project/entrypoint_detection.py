"""Test script to verify entry point detection."""

from test_metaxy_project.features import (  # ty: ignore[unresolved-import]
    TestFeature,
)

# Verify Feature.project is detected from entry points
assert TestFeature.project == "test-metaxy-project", (
    f"Expected 'test-metaxy-project', got {TestFeature.project}"
)

print("SUCCESS: Detected project from entry points")
