"""Entrypoint for DuckLake example.

Imports feature definitions so they register with the active FeatureGraph.
"""


def entrypoint() -> None:
    """Load the DuckLake feature definitions."""
    # Importing the module has the side-effect of registering features
    from examples.ducklake import features  # noqa: F401


entrypoint()
