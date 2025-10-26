"""Entrypoint for migration example.

Set VERSION environment variable to load the appropriate feature version (1 or 2).
"""

import importlib
import os


def entrypoint():
    """Load features based on STAGE environment variable."""
    version = os.environ.get("VERSION", "1")
    module_name = f"examples.overview.features_{version}"
    module = importlib.import_module(module_name)
    return module


# Execute the entrypoint function when this module is imported
# This ensures features are registered to the active FeatureGraph
entrypoint()
