"""Entrypoint for recompute example.

Set STAGE environment variable to load the appropriate feature version (1 or 2).
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
