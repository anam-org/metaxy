"""Entrypoint for recompute example.

Set STAGE environment variable to load the appropriate feature version (1 or 2).
"""

import importlib
import os


def entrypoint():
    """Load features based on STAGE environment variable."""
    version = os.environ.get("STAGE", "1")
    module_name = f"example_recompute.features_{version}"
    module = importlib.import_module(module_name)
    return module


entrypoint()
