"""Entrypoint for recompute example.

This module imports and registers the feature definitions.
"""

import importlib
import os

# Load features based on STAGE environment variable
version = os.environ.get("STAGE", "1")
module_name = f"example_recompute.features_{version}"
_module = importlib.import_module(module_name)

# Expose all Feature classes from the imported module
# This allows the entrypoint system to access them
for name in dir(_module):
    if not name.startswith("_"):
        globals()[name] = getattr(_module, name)
