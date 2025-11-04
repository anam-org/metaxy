"""Test Metaxy project for testing entry point detection."""


def init():
    """Entry point function that loads features.

    This is called by Metaxy's entry point loading system.
    Simply importing the features module registers them to the active graph.
    """
    from test_metaxy_project import features  # pyright: ignore[reportUnusedImport,reportImplicitRelativeImport]  # noqa: F401, I001
