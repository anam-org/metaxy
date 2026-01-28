"""Test script to verify package-based project detection."""

from test_metaxy_project.features import (  # ty: ignore[unresolved-import]
    TestFeature,
)

# Verify Feature.metaxy_project() is detected from __metaxy_project__ in package __init__.py
assert TestFeature.metaxy_project() == "test-metaxy-project", (
    f"Expected 'test-metaxy-project', got {TestFeature.metaxy_project()}"
)

print("SUCCESS: Detected project from __metaxy_project__ variable")
