from metaxy import __version__ as metaxy_version
from metaxy._version import __version__


def test_version() -> None:
    assert isinstance(__version__, str)
    assert __version__
    assert metaxy_version == __version__
